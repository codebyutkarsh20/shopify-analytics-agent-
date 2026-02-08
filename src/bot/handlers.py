"""Message handlers for Telegram bot.

Processes natural language queries and handles bot interactions with users.
Integrates session management, learning, feedback analysis, and channel adaptation.
"""

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from src.config.settings import settings
from src.database.operations import DatabaseOperations
from src.learning.pattern_learner import PatternLearner
from src.learning.preference_manager import PreferenceManager
from src.learning.session_manager import SessionManager
from src.learning.template_manager import TemplateManager
from src.learning.feedback_analyzer import FeedbackAnalyzer
from src.learning.insight_aggregator import InsightAggregator
from src.bot.channel_adapter import ChannelAdapter
from src.bot.telegram_adapter import TelegramAdapter
from src.services.claude_service import ClaudeService
from src.utils.logger import get_logger
from src.utils.formatters import format_error_message, markdown_to_telegram_html

logger = get_logger(__name__)

# Telegram's message length limit
TELEGRAM_MESSAGE_LIMIT = 4096


class MessageHandler:
    """Handles text messages and user queries.

    Integrates session management, intent classification, feedback analysis,
    template learning, recovery learning, and insight aggregation into the
    message processing pipeline.
    """

    def __init__(
        self,
        claude_service: ClaudeService,
        pattern_learner: PatternLearner,
        preference_manager: PreferenceManager,
        db_ops: DatabaseOperations,
        bot_commands=None,
        session_manager: SessionManager = None,
        feedback_analyzer: FeedbackAnalyzer = None,
        template_manager: TemplateManager = None,
        insight_aggregator: InsightAggregator = None,
        telegram_adapter: TelegramAdapter = None,
    ):
        """
        Initialize MessageHandler.

        Args:
            claude_service: ClaudeService instance for processing messages with Claude AI
            pattern_learner: PatternLearner instance for learning from queries
            preference_manager: PreferenceManager instance for user preferences
            db_ops: DatabaseOperations instance for database access
            bot_commands: Optional BotCommands instance for handling connect flow
            session_manager: Optional SessionManager for conversation session tracking
            feedback_analyzer: Optional FeedbackAnalyzer for implicit feedback detection
            template_manager: Optional TemplateManager for query template learning
            insight_aggregator: Optional InsightAggregator for cross-user insights
            telegram_adapter: Optional TelegramAdapter for channel-specific formatting
        """
        self.claude_service = claude_service
        self.pattern_learner = pattern_learner
        self.preference_manager = preference_manager
        self.db_ops = db_ops
        self.bot_commands = bot_commands
        self.session_manager = session_manager
        self.feedback_analyzer = feedback_analyzer
        self.template_manager = template_manager
        self.insight_aggregator = insight_aggregator
        self.telegram_adapter = telegram_adapter or TelegramAdapter()

        # Counter for periodic aggregation
        self._interaction_count = 0
        self._aggregation_interval = getattr(
            settings, "aggregation_interval", 50
        )

        logger.info("MessageHandler initialized with learning components")

    async def handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Main handler for text messages.

        Enhanced flow with learning system integration:
        1. Resolve user (via adapter metadata)
        2. Classify intent (coarse + fine-grained)
        3. Detect/create session (via SessionManager)
        4. Analyze feedback on previous response
        5. Show typing indicator
        6. Process through Claude (session-aware message history)
        7. Learn from query patterns + update preferences
        8. Format & send response (via adapter)
        9. Save conversation (with session_id, channel_type, tool_calls)
        10. Periodic aggregation check

        Args:
            update: The update object
            context: The context object
        """
        telegram_user_id = update.effective_user.id
        message_text = update.message.text
        message_length = len(message_text) if message_text else 0

        logger.info(
            "Message received",
            user_id=telegram_user_id,
            username=update.effective_user.username,
            message_length=message_length,
        )

        try:
            # Check if user is in connect flow (handled by BotCommands)
            if self.bot_commands and hasattr(self.bot_commands, 'handle_connect_response'):
                if await self.bot_commands.handle_connect_response(update, context):
                    logger.debug("Message handled by connect flow", user_id=telegram_user_id)
                    return

            # Step 1: Resolve user
            user = self.db_ops.get_or_create_user(
                telegram_user_id=telegram_user_id,
                telegram_username=update.effective_user.username,
                first_name=update.effective_user.first_name,
            )
            logger.debug("User resolved", user_id=user.id)

            # Update user activity and interaction count
            self.db_ops.update_user_activity(user.id)
            try:
                self.db_ops.increment_user_interaction(user.id)
            except Exception:
                pass  # Non-critical if column doesn't exist yet

            # Step 2: Classify intent
            intent = self.pattern_learner.classify_intent(message_text)
            query_type = intent.coarse
            logger.debug(
                "Intent classified",
                user_id=user.id,
                coarse=intent.coarse,
                fine=intent.fine,
            )

            # Step 3: Detect/create session
            session = None
            session_id = None
            channel_type = "telegram"

            if self.session_manager:
                session = self.session_manager.get_or_create_session(
                    user_id=user.id,
                    channel_type=channel_type,
                    current_intent=intent.coarse,
                )
                session_id = session.id
                logger.debug(
                    "Session resolved",
                    user_id=user.id,
                    session_id=session_id,
                )

            # Step 4: Analyze feedback on previous response
            if self.feedback_analyzer:
                self._analyze_feedback(user.id, message_text, query_type)

            # Step 5: Show typing indicator
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING
            )

            # Check if user has a store connected; auto-connect from .env if possible
            if self.bot_commands and hasattr(self.bot_commands, '_auto_connect_from_env'):
                self.bot_commands._auto_connect_from_env(user.id)

            store = self.db_ops.get_store_by_user(user.id)
            if not store:
                no_store_msg = (
                    "⚠️ <b>No Shopify Store Connected</b>\n\n"
                    "I need your Shopify store credentials to help with analytics.\n\n"
                    "Use /connect to connect your store."
                )
                await update.message.reply_text(
                    no_store_msg,
                    parse_mode="HTML"
                )
                logger.info("No store connected for user", user_id=user.id)
                return

            # Step 6: Process message through Claude service (session-aware)
            logger.debug("Processing message through Claude", user_id=user.id)
            response = await self.claude_service.process_message(
                user_id=user.id,
                message=message_text,
                intent=intent.fine,
                session_id=session_id,
            )
            logger.debug(
                "Claude response received",
                user_id=user.id,
                response_length=len(response) if response else 0,
            )

            if not response:
                response = "Sorry, I couldn't process that query. Please try again."
                logger.warning("Empty response from Claude service", user_id=user.id)

            # Step 7: Learn from query patterns + update preferences
            logger.debug("Learning from query", user_id=user.id)
            self.pattern_learner.learn_from_query(
                user_id=user.id,
                query=message_text,
                query_type=query_type,
            )
            self.preference_manager.update_preferences_from_patterns(user.id)

            # Step 8: Format & send response
            logger.debug("Sending response", user_id=user.id)
            await self._send_message_split(update, response)

            # Step 9: Save conversation (with session_id, channel_type, tool_calls)
            tool_calls_json = self.claude_service.last_tool_calls_json
            self.db_ops.save_conversation(
                user_id=user.id,
                message=message_text,
                response=response,
                query_type=query_type,
                session_id=session_id,
                channel_type=channel_type,
                tool_calls_json=tool_calls_json,
            )
            logger.info(
                "Message processed successfully",
                user_id=user.id,
                query_type=query_type,
                session_id=session_id,
            )

            # Step 10: Periodic aggregation check
            self._check_aggregation()

        except Exception as e:
            logger.error(
                "Error processing message",
                user_id=telegram_user_id,
                error=str(e),
                exc_info=True,
            )
            await self._handle_error(update, e)

    def _analyze_feedback(
        self,
        user_id: int,
        current_message: str,
        current_query_type: str,
    ) -> None:
        """Analyze implicit feedback from this message about previous response.

        Retrieves the latest conversation and compares the current message
        to determine if it represents positive, negative, or neutral feedback.

        Args:
            user_id: User's database ID
            current_message: Current user message
            current_query_type: Current query type classification
        """
        try:
            latest_convs = self.db_ops.get_latest_conversation(user_id)
            if not latest_convs:
                return

            latest_conv = latest_convs[0]
            previous_query_type = latest_conv.query_type or ""

            feedback = self.feedback_analyzer.analyze_follow_up(
                previous_query_type=previous_query_type,
                current_message=current_message,
                current_query_type=current_query_type,
            )

            if feedback and feedback.get("quality_score", 0) != 0:
                # Save feedback to database
                self.db_ops.save_response_feedback(
                    conversation_id=latest_conv.id,
                    user_id=user_id,
                    feedback_type=feedback["feedback_type"],
                    quality_score=feedback["quality_score"],
                    signal_text=feedback.get("signal_text"),
                )

                # Update conversation quality score
                self.db_ops.update_conversation_quality(
                    conversation_id=latest_conv.id,
                    quality_score=feedback["quality_score"],
                )

                # Update template quality if a template was used
                if (
                    self.template_manager
                    and latest_conv.template_id_used
                ):
                    self.template_manager.update_template_quality(
                        template_id=latest_conv.template_id_used,
                        quality_score=feedback["quality_score"],
                    )

                logger.debug(
                    "Feedback recorded",
                    user_id=user_id,
                    feedback_type=feedback["feedback_type"],
                    quality_score=feedback["quality_score"],
                )

        except Exception as e:
            logger.warning(
                "Failed to analyze feedback",
                user_id=user_id,
                error=str(e),
            )

    def _check_aggregation(self) -> None:
        """Check if it's time to run periodic insight aggregation.

        Increments interaction counter and triggers aggregation when
        the configured interval is reached.
        """
        if not self.insight_aggregator:
            return

        self._interaction_count += 1

        if self._interaction_count >= self._aggregation_interval:
            try:
                logger.info(
                    "Running periodic insight aggregation",
                    interaction_count=self._interaction_count,
                )
                self.insight_aggregator.run_aggregation()
                self._interaction_count = 0
            except Exception as e:
                logger.warning(
                    "Insight aggregation failed",
                    error=str(e),
                )

    async def _send_message_split(
        self,
        update: Update,
        message: str
    ) -> None:
        """
        Send a message, splitting into chunks if it exceeds Telegram's limit.

        Uses the TelegramAdapter for formatting if available.

        Args:
            update: The update object
            message: The message text to send
        """
        if not message:
            return

        # Format via adapter (Telegram HTML conversion)
        formatted = self.telegram_adapter.format_response(message)
        msg_limit = self.telegram_adapter.get_message_limit()

        logger.debug(
            "Splitting message if needed",
            message_length=len(formatted),
            limit=msg_limit,
        )

        # Split into chunks
        chunks = []
        current_chunk = ""

        for paragraph in formatted.split("\n\n"):
            if len(current_chunk) + len(paragraph) + 2 > msg_limit:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph

        if current_chunk:
            chunks.append(current_chunk)

        logger.debug("Message split into chunks", chunk_count=len(chunks))

        # Send each chunk
        for idx, chunk in enumerate(chunks):
            try:
                await update.message.reply_text(
                    chunk,
                    parse_mode="HTML"
                )
                logger.debug(
                    "Message chunk sent",
                    chunk_number=idx + 1,
                    total_chunks=len(chunks),
                    chunk_length=len(chunk),
                )
            except Exception as e:
                logger.error(
                    "Error sending message chunk",
                    chunk_number=idx + 1,
                    error=str(e),
                    exc_info=True,
                )
                # Try without HTML parsing if that fails
                try:
                    await update.message.reply_text(chunk)
                except Exception as retry_error:
                    logger.error(
                        "Failed to send message chunk even without HTML",
                        chunk_number=idx + 1,
                        error=str(retry_error),
                    )

    async def _handle_error(
        self,
        update: Update,
        error: Exception
    ) -> None:
        """
        Handle errors and send user-friendly messages.

        Args:
            update: The update object
            error: The exception that occurred
        """
        logger.warning(
            "Handling error",
            user_id=update.effective_user.id,
            error_type=type(error).__name__,
            error_message=str(error),
        )

        try:
            error_msg = format_error_message(error)
            await update.message.reply_text(
                error_msg,
                parse_mode="HTML"
            )
        except Exception as format_error:
            logger.error(
                "Failed to send formatted error message",
                user_id=update.effective_user.id,
                error=str(format_error),
            )
            # Fall back to plain text error
            try:
                fallback_msg = (
                    "Sorry, an error occurred while processing your request. "
                    "Please try again later."
                )
                await update.message.reply_text(fallback_msg)
            except Exception as final_error:
                logger.error(
                    "Failed to send error message",
                    user_id=update.effective_user.id,
                    error=str(final_error),
                )

    async def handle_error(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Global error handler for the bot.

        Logs errors and sends user-friendly messages when the bot encounters
        unexpected exceptions.

        Args:
            update: The update object
            context: The context object (contains the error in context.error)
        """
        error = context.error
        user_id = update.effective_user.id if update and update.effective_user else "unknown"

        logger.error(
            "Global error handler triggered",
            user_id=user_id,
            error_type=type(error).__name__,
            error_message=str(error),
            exc_info=error,
        )

        try:
            if update and update.message:
                error_msg = format_error_message(error)
                await update.message.reply_text(
                    error_msg,
                    parse_mode="HTML"
                )
            else:
                logger.warning(
                    "Cannot send error message - no update.message available",
                    user_id=user_id,
                )

        except Exception as e:
            logger.error(
                "Failed to send error message in global handler",
                user_id=user_id,
                error=str(e),
            )
