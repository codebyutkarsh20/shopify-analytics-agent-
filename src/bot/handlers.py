"""Message handlers for Telegram bot.

Processes natural language queries and handles bot interactions with users.
"""

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from src.database.operations import DatabaseOperations
from src.learning.pattern_learner import PatternLearner
from src.learning.preference_manager import PreferenceManager
from src.services.claude_service import ClaudeService
from src.utils.logger import get_logger
from src.utils.formatters import format_error_message, markdown_to_telegram_html

logger = get_logger(__name__)

# Telegram's message length limit
TELEGRAM_MESSAGE_LIMIT = 4096


class MessageHandler:
    """Handles text messages and user queries."""

    def __init__(
        self,
        claude_service: ClaudeService,
        pattern_learner: PatternLearner,
        preference_manager: PreferenceManager,
        db_ops: DatabaseOperations,
        bot_commands=None,
    ):
        """
        Initialize MessageHandler.

        Args:
            claude_service: ClaudeService instance for processing messages with Claude AI
            pattern_learner: PatternLearner instance for learning from queries
            preference_manager: PreferenceManager instance for user preferences
            db_ops: DatabaseOperations instance for database access
            bot_commands: Optional BotCommands instance for handling connect flow
        """
        self.claude_service = claude_service
        self.pattern_learner = pattern_learner
        self.preference_manager = preference_manager
        self.db_ops = db_ops
        self.bot_commands = bot_commands
        logger.info("MessageHandler initialized")

    async def handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Main handler for text messages.

        Flow:
        1. Get or create user in database
        2. Update user activity
        3. Show typing indicator
        4. Process message through MCP service
        5. Learn from query patterns
        6. Update user preferences
        7. Send response (split if > 4096 chars)
        8. Save conversation to database
        9. Handle errors gracefully

        Args:
            update: The update object
            context: The context object
        """
        user_id = update.effective_user.id
        message_text = update.message.text
        message_length = len(message_text) if message_text else 0

        logger.info(
            "Message received",
            user_id=user_id,
            username=update.effective_user.username,
            message_length=message_length,
        )

        try:
            # Check if user is in connect flow (handled by BotCommands)
            if self.bot_commands and hasattr(self.bot_commands, 'handle_connect_response'):
                if await self.bot_commands.handle_connect_response(update, context):
                    logger.debug("Message handled by connect flow", user_id=user_id)
                    return

            # 1. Get or create user
            user = self.db_ops.get_or_create_user(
                telegram_user_id=user_id,
                telegram_username=update.effective_user.username,
                first_name=update.effective_user.first_name,
            )
            logger.debug("User retrieved/created", user_id=user.id, db_user_id=user.id)

            # 2. Update user activity
            self.db_ops.update_user_activity(user.id)
            logger.debug("User activity updated", user_id=user.id)

            # 3. Show typing indicator
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING
            )
            logger.debug("Typing action sent", user_id=user.id)

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

            # 4. Process message through Claude service
            logger.debug("Processing message through Claude", user_id=user.id)
            response = await self.claude_service.process_message(
                user_id=user.id,
                message=message_text,
            )
            logger.debug(
                "Claude response received",
                user_id=user.id,
                response_length=len(response) if response else 0,
            )

            if not response:
                response = "Sorry, I couldn't process that query. Please try again."
                logger.warning("Empty response from MCP service", user_id=user.id)

            # 5. Learn from query patterns
            logger.debug("Learning from query", user_id=user.id)
            self.pattern_learner.learn_from_query(
                user_id=user.id,
                query=message_text
            )

            # 6. Update preferences from patterns
            logger.debug("Updating preferences", user_id=user.id)
            self.preference_manager.update_preferences_from_patterns(user.id)

            # 7. Send response (split if too long)
            logger.debug("Sending response", user_id=user.id)
            await self._send_message_split(update, response)

            # 8. Save conversation to database
            query_type = self.pattern_learner.detect_query_type(message_text)
            self.db_ops.save_conversation(
                user_id=user.id,
                message=message_text,
                response=response,
                query_type=query_type,
            )
            logger.info(
                "Message processed successfully",
                user_id=user.id,
                query_type=query_type,
            )

        except Exception as e:
            logger.error(
                "Error processing message",
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            # 9. Handle errors gracefully
            await self._handle_error(update, e)

    async def _send_message_split(
        self,
        update: Update,
        message: str
    ) -> None:
        """
        Send a message, splitting into chunks if it exceeds Telegram's limit.

        Telegram has a 4096 character limit per message. This function handles
        splitting long responses into multiple messages.

        Args:
            update: The update object
            message: The message text to send
        """
        if not message:
            return

        # Convert Claude's Markdown to Telegram-compatible HTML
        message = markdown_to_telegram_html(message)

        logger.debug(
            "Splitting message if needed",
            message_length=len(message),
            limit=TELEGRAM_MESSAGE_LIMIT,
        )

        # Split into chunks
        chunks = []
        current_chunk = ""

        for paragraph in message.split("\n\n"):
            if len(current_chunk) + len(paragraph) + 2 > TELEGRAM_MESSAGE_LIMIT:
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
                # Try without markdown if that fails
                try:
                    await update.message.reply_text(chunk)
                except Exception as retry_error:
                    logger.error(
                        "Failed to send message chunk even without markdown",
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
