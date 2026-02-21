"""Message handlers for Telegram bot.

Processes natural language queries and handles bot interactions with users.
Integrates session management, learning, feedback analysis, and channel adaptation.
"""

import hmac
import os
import re
from typing import List, Tuple, Union

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
from src.utils.rate_limiter import RateLimiter

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
        channel_adapter: ChannelAdapter = None,
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
            telegram_adapter: Optional TelegramAdapter for channel-specific formatting (deprecated, use channel_adapter)
            channel_adapter: Optional ChannelAdapter — preferred generic adapter parameter
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
        # Support both the old telegram_adapter param and the new generic channel_adapter
        self.telegram_adapter = channel_adapter or telegram_adapter or TelegramAdapter()

        # Counter for periodic aggregation
        self._interaction_count = 0
        self._aggregation_interval = getattr(
            settings, "aggregation_interval", 50
        )
        # Security: rate limiter
        self._rate_limiter = RateLimiter(
            max_requests=settings.security.rate_limit_per_minute,
            window_seconds=60,
        )
        # Security: allowed user IDs (empty = allow all)
        self._allowed_users = settings.telegram.allowed_users
        # Security: shared access code (empty = no verification required)
        self._bot_access_code = settings.security.bot_access_code

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

        # Security: user allowlist check
        if self._allowed_users and telegram_user_id not in self._allowed_users:
            logger.warning(
                "Unauthorized access attempt blocked",
                user_id=telegram_user_id,
                username=update.effective_user.username,
            )
            await update.message.reply_text(
                "⛔ This bot is private. Contact the administrator for access."
            )
            return

        logger.info(
            "Message received",
            user_id=telegram_user_id,
            username=update.effective_user.username,
            message_length=message_length,
        )

        # Security: rate limiting
        allowed, retry_after = self._rate_limiter.is_allowed(telegram_user_id)
        if not allowed:
            await update.message.reply_text(
                f"⏳ Too many requests. Please wait {retry_after}s before trying again."
            )
            return

        try:
            # Step 0: Access-code verification gate
            # If BOT_ACCESS_CODE is configured, new users must provide it once.
            if self._bot_access_code:
                user_for_verify = self.db_ops.get_or_create_user(
                    telegram_user_id=telegram_user_id,
                    telegram_username=update.effective_user.username,
                    first_name=update.effective_user.first_name,
                )
                if not user_for_verify.is_verified:
                    # Check if this message IS the access code
                    code_attempt = (message_text or "").strip()
                    if hmac.compare_digest(code_attempt, self._bot_access_code):
                        # Correct code — verify the user
                        self.db_ops.verify_user(user_for_verify.id)
                        logger.info(
                            "User verified via access code",
                            user_id=user_for_verify.id,
                            telegram_user_id=telegram_user_id,
                        )
                        # Delete the message containing the code for security
                        try:
                            await update.message.delete()
                        except Exception:
                            pass
                        await update.message.reply_text(
                            "✅ <b>Access Granted!</b>\n\n"
                            "You're now verified. Welcome aboard!\n"
                            "Type /start to get started or just ask me a question.",
                            parse_mode="HTML",
                        )
                        return
                    else:
                        # Wrong code or regular message — prompt for code
                        logger.warning(
                            "Unverified user attempted access",
                            telegram_user_id=telegram_user_id,
                            username=update.effective_user.username,
                        )
                        await update.message.reply_text(
                            "🔒 <b>Verification Required</b>\n\n"
                            "Please enter the access code to use this bot.\n\n"
                            "If you don't have a code, contact the bot administrator.",
                            parse_mode="HTML",
                        )
                        return

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
            except Exception as e:
                logger.debug("increment_user_interaction skipped", error=str(e))

            # Step 2: Classify intent
            intent = self.pattern_learner.classify_intent(message_text)
            
            # Smart Intent Refinement: Use LLM if intent is ambiguous ("general")
            # or if query seems complex
            if intent.coarse == "general" or self.pattern_learner.assess_query_complexity(message_text) == "complex":
                logger.info("Refining intent with LLM", initial_intent=intent.coarse)
                try:
                    refined_intent = await self.pattern_learner.refine_intent_with_llm(
                        message_text, self.claude_service
                    )
                    if refined_intent:
                        intent = refined_intent
                        logger.info("Intent refined", new_intent=intent.coarse, fine=intent.fine)
                except Exception as e:
                    logger.warning("Intent refinement failed, continuing with original", error=str(e))

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
            
            # Note: We are passing the intent object (Intent class) to the service, 
            # but the service expects a string or similar in `process_message`.
            # Let's pass the fine-grained intent string as before, but ensure the service uses it.
            # Ideally, we should pass the whole intent object if the service supports it, 
            # OR pass the refined intent string.
            
            response = await self.claude_service.process_message(
                user_id=user.id,
                message=message_text,
                intent=intent,  # Passing the full intent object allows explicit access to coarse/fine
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

            # Step 8: Format & send response (with charts if any)
            logger.debug("Sending response", user_id=user.id)
            chart_files = getattr(self.claude_service, "last_chart_files", [])
            await self._send_message_split(update, response, chart_files=chart_files)

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

    # ── Chart marker pattern: [CHART:0], [CHART:1], etc. ──────────
    _CHART_MARKER_RE = re.compile(r"\[CHART:(\d+)\]")

    def _build_interleaved_segments(
        self,
        message: str,
        chart_files: List[str],
    ) -> List[Union[str, Tuple[str, str]]]:
        """Split a response into interleaved text and chart segments.

        Scans the message for [CHART:N] markers and builds an ordered list
        of segments, each being either a text string or a ("chart", filepath)
        tuple.  Any charts that weren't referenced inline are appended at the
        end so they're never lost.

        Args:
            message: The raw LLM response (may contain [CHART:N] markers)
            chart_files: Ordered list of chart file paths (index ↔ marker N)

        Returns:
            List of segments:
                str          → text segment (send via reply_text)
                ("chart", p) → chart segment (send via reply_photo)
        """
        if not chart_files:
            # No charts at all — just return the text
            return [message] if message.strip() else []

        segments: List[Union[str, Tuple[str, str]]] = []
        used_indices: set = set()

        # Split text on [CHART:N] markers, keeping the N values
        parts = self._CHART_MARKER_RE.split(message)

        # parts alternates: [text, index, text, index, text, ...]
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Text segment
                text = part.strip()
                if text:
                    segments.append(text)
            else:
                # Chart index
                chart_idx = int(part)
                if chart_idx < len(chart_files):
                    segments.append(("chart", chart_files[chart_idx]))
                    used_indices.add(chart_idx)
                else:
                    logger.warning(
                        "Chart index out of range",
                        chart_index=chart_idx,
                        available=len(chart_files),
                    )

        # Append any charts that weren't referenced inline (safety net)
        for idx, filepath in enumerate(chart_files):
            if idx not in used_indices:
                logger.debug("Appending unreferenced chart", chart_index=idx)
                segments.append(("chart", filepath))

        return segments

    async def _send_text_chunk(self, update: Update, text: str) -> None:
        """Send a single text chunk, splitting further if over Telegram limit."""
        msg_limit = self.telegram_adapter.get_message_limit()
        formatted = self.telegram_adapter.format_response(text)

        # Split into sub-chunks if needed
        chunks = []
        current = ""
        for paragraph in formatted.split("\n\n"):
            if len(current) + len(paragraph) + 2 > msg_limit:
                if current:
                    chunks.append(current)
                current = paragraph
            else:
                current = f"{current}\n\n{paragraph}" if current else paragraph
        if current:
            chunks.append(current)

        for idx, chunk in enumerate(chunks):
            try:
                await update.message.reply_text(chunk, parse_mode="HTML")
                logger.debug(
                    "Text chunk sent",
                    chunk_number=idx + 1,
                    total_chunks=len(chunks),
                    chunk_length=len(chunk),
                )
            except Exception as e:
                logger.error("Error sending text chunk", error=str(e), exc_info=True)
                try:
                    await update.message.reply_text(chunk)
                except Exception as retry_err:
                    logger.error("Fallback send also failed", error=str(retry_err))

    async def _send_chart_photo(self, update: Update, filepath: str) -> None:
        """Send a chart image and clean up the temp file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, "rb") as photo:
                    await update.message.reply_photo(photo=photo)
                logger.info("Chart image sent", chart_file=filepath)
            else:
                logger.warning("Chart file not found", chart_file=filepath)
        except Exception as e:
            logger.error(
                "Failed to send chart image",
                chart_file=filepath,
                error=str(e),
                exc_info=True,
            )
        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug("Chart file cleaned up", chart_file=filepath)
            except Exception:
                pass

    async def _send_message_split(
        self,
        update: Update,
        message: str,
        chart_files: List[str] = None,
    ) -> None:
        """
        Send a response with dynamically interleaved text and chart images.

        The LLM places [CHART:N] markers in its response to indicate where
        each chart should appear.  This method splits the message at those
        markers and sends text chunks and chart photos in the correct order,
        producing a natural mixed layout in Telegram:

            Text → Image → Text → Image → Text

        If no markers are found, charts are appended after the text.

        Args:
            update: The Telegram update object
            message: The LLM's response text (may contain [CHART:N] markers)
            chart_files: Ordered list of chart image file paths
        """
        if not message:
            return

        segments = self._build_interleaved_segments(
            message, chart_files or []
        )

        logger.debug(
            "Sending interleaved response",
            segment_count=len(segments),
            text_segments=sum(1 for s in segments if isinstance(s, str)),
            chart_segments=sum(1 for s in segments if isinstance(s, tuple)),
        )

        for segment in segments:
            if isinstance(segment, tuple) and segment[0] == "chart":
                await self._send_chart_photo(update, segment[1])
            elif isinstance(segment, str):
                await self._send_text_chunk(update, segment)

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
