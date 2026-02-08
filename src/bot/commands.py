"""Bot command handlers for Telegram.

Handles all bot commands like /start, /help, /settings, /connect, /forget, /status, and /verify.
All messages use HTML parse_mode for robust formatting (no escaping headaches).
"""

import hmac
import re
from html import escape
from typing import Optional

from telegram import Update
from telegram.ext import ContextTypes

from src.config.settings import settings
from src.database.operations import DatabaseOperations
from src.learning.preference_manager import PreferenceManager
from src.utils.logger import get_logger
from src.utils.formatters import (
    format_welcome_message,
    format_help_message,
    format_error_message,
)
from src.bot.keyboards import get_forget_confirmation_keyboard

logger = get_logger(__name__)

PARSE_MODE = "HTML"


class BotCommands:
    """Handles all bot commands."""

    def __init__(
        self,
        db_ops: DatabaseOperations,
        preference_manager: PreferenceManager,
    ):
        self.db_ops = db_ops
        self.preference_manager = preference_manager
        self._bot_access_code = settings.security.bot_access_code
        logger.info("BotCommands initialized")

    # ── Access-code helpers ─────────────────────────────────────────

    def _is_user_verified(self, telegram_user_id: int) -> bool:
        """Check whether a user has passed the access-code gate.

        Returns True if no BOT_ACCESS_CODE is configured (open mode)
        or if the user's is_verified flag is True.
        """
        if not self._bot_access_code:
            return True
        user = self.db_ops.get_or_create_user(telegram_user_id=telegram_user_id)
        return bool(user.is_verified)

    async def _require_verification(self, update: Update) -> bool:
        """Send the verification prompt if user is not yet verified.

        Returns True if access was blocked, False if user is clear.
        """
        if self._is_user_verified(update.effective_user.id):
            return False
        await update.message.reply_text(
            "🔒 <b>Verification Required</b>\n\n"
            "Please enter the access code to use this bot.\n"
            "You can send it as a message or use:\n"
            "<code>/verify YOUR_CODE</code>\n\n"
            "If you don't have a code, contact the bot administrator.",
            parse_mode=PARSE_MODE,
        )
        return True

    def _auto_connect_from_env(self, user_id: int) -> bool:
        """
        Auto-connect a store from .env if credentials are configured
        and user has no store yet.

        Returns True if auto-connected, False otherwise.
        """
        store = self.db_ops.get_store_by_user(user_id)
        if store:
            return False  # Already connected

        # Check if .env has valid credentials
        domain = settings.shopify.shop_domain
        token = settings.shopify.access_token
        if domain and token:
            self.db_ops.add_store(
                user_id=user_id,
                shop_domain=domain,
                access_token=token,
            )
            logger.info(
                "Auto-connected store from .env",
                user_id=user_id,
                domain=domain,
            )
            return True
        return False

    async def start_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        logger.info(
            "Start command received",
            user_id=update.effective_user.id,
            username=update.effective_user.username,
        )

        # Gate: access-code verification
        if await self._require_verification(update):
            return

        try:
            user = self.db_ops.get_or_create_user(
                telegram_user_id=update.effective_user.id,
                telegram_username=update.effective_user.username,
                first_name=update.effective_user.first_name,
            )

            # Auto-connect from .env if possible
            self._auto_connect_from_env(user.id)

            store = self.db_ops.get_store_by_user(user.id)

            if store:
                domain = escape(store.shop_domain)
                name = escape(update.effective_user.first_name or "there")
                welcome_msg = (
                    f"👋 <b>Welcome back, {name}!</b>\n\n"
                    f"✅ Connected to <b>{domain}</b>\n\n"
                    f"<b>Quick start:</b>\n"
                    f'• "Show me sales for last 7 days"\n'
                    f'• "What were yesterday\'s orders?"\n'
                    f'• "Top 5 products by revenue"\n\n'
                    f"Type /help for all commands."
                )
            else:
                welcome_msg = format_welcome_message(
                    user_name=update.effective_user.first_name or "there"
                )
                welcome_msg += (
                    "\n\nUse /connect to link your Shopify store."
                )

            await update.message.reply_text(welcome_msg, parse_mode=PARSE_MODE)

        except Exception as e:
            logger.error("Error in start command", error=str(e), exc_info=True)
            error_msg = format_error_message(e)
            await update.message.reply_text(error_msg, parse_mode=PARSE_MODE)

    async def help_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help command."""
        logger.info("Help command received", user_id=update.effective_user.id)

        # Gate: access-code verification
        if await self._require_verification(update):
            return

        try:
            help_msg = format_help_message()
            await update.message.reply_text(help_msg, parse_mode=PARSE_MODE)
        except Exception as e:
            logger.error("Error in help command", error=str(e), exc_info=True)
            error_msg = format_error_message(e)
            await update.message.reply_text(error_msg, parse_mode=PARSE_MODE)

    async def connect_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /connect command."""
        logger.info("Connect command received", user_id=update.effective_user.id)

        # Gate: access-code verification
        if await self._require_verification(update):
            return

        try:
            import secrets
            nonce = secrets.token_hex(16)
            context.user_data["awaiting_credentials"] = True
            context.user_data["connection_step"] = "waiting_for_domain"
            context.user_data["connect_nonce"] = nonce

            instructions = (
                "<b>🔗 Connect Your Shopify Store</b>\n\n"
                "I'll help you connect your Shopify store. Please provide:\n\n"
                "1. Your shop domain (e.g., <code>myshop.myshopify.com</code>)\n"
                "2. Your Shopify API access token\n\n"
                "For security reasons, send them as a single message in this format:\n"
                "<code>domain:token</code>\n\n"
                "<i>Example:</i> <code>myshop.myshopify.com:shpat_abc123xyz</code>\n\n"
                "⚠️ <b>Important:</b> I will delete your message containing the token immediately after reading it for security."
            )

            await update.message.reply_text(instructions, parse_mode=PARSE_MODE)

        except Exception as e:
            logger.error("Error in connect command", error=str(e), exc_info=True)
            error_msg = format_error_message(e)
            await update.message.reply_text(error_msg, parse_mode=PARSE_MODE)

    async def handle_connect_response(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """
        Handle response during the connection flow.
        Returns True if message was handled, False otherwise.
        """
        if not context.user_data.get("awaiting_credentials"):
            return False

        # Verify connect flow nonce for CSRF protection
        if not context.user_data.get("connect_nonce"):
            logger.warning("Missing connect nonce", user_id=update.effective_user.id)
            return False

        user_id = update.effective_user.id
        message_text = update.message.text.strip()

        logger.info(
            "Processing connection response",
            user_id=user_id,
            message_length=len(message_text),
        )

        try:
            if ":" not in message_text:
                error_msg = (
                    "❌ Invalid format. Please use: <code>domain:token</code>\n\n"
                    "<i>Example:</i> <code>myshop.myshopify.com:shpat_abc123xyz</code>"
                )
                await update.message.reply_text(error_msg, parse_mode=PARSE_MODE)
                return True

            parts = message_text.split(":", 1)
            domain = parts[0].strip()
            token = parts[1].strip()

            # Security: delete the message containing the token
            try:
                await update.message.delete()
                logger.info("Credential message deleted for security", user_id=user_id)
            except Exception as del_err:
                logger.warning(
                    "Could not delete credential message — bot may lack delete permission",
                    user_id=user_id,
                    error=str(del_err),
                )

            if not domain or not token:
                await update.message.reply_text(
                    "❌ Domain and token cannot be empty.",
                    parse_mode=PARSE_MODE,
                )
                return True

            if not re.match(r"^[\w\-]+\.myshopify\.com$", domain):
                await update.message.reply_text(
                    "❌ Invalid domain format. "
                    "Expected something like <code>myshop.myshopify.com</code>.",
                    parse_mode=PARSE_MODE,
                )
                return True

            user = self.db_ops.get_or_create_user(
                telegram_user_id=user_id,
                telegram_username=update.effective_user.username,
                first_name=update.effective_user.first_name,
            )

            store = self.db_ops.add_store(
                user_id=user.id,
                shop_domain=domain,
                access_token=token,
            )

            logger.info(
                "Store connected successfully",
                user_id=user.id,
                store_id=store.id,
                domain=domain,
            )

            context.user_data["awaiting_credentials"] = False
            context.user_data.pop("connection_step", None)

            safe_domain = escape(domain)
            success_msg = (
                f"✅ <b>Store Connected!</b>\n\n"
                f"Successfully connected to <b>{safe_domain}</b>.\n\n"
                f"You can now ask me questions about your analytics:\n"
                f'• "Show me sales for last 7 days"\n'
                f'• "What were yesterday\'s orders?"\n'
                f'• "Top 5 products"\n\n'
                f"Type /help for more examples."
            )
            await update.message.reply_text(success_msg, parse_mode=PARSE_MODE)

            return True

        except Exception as e:
            logger.error(
                "Error processing connection response",
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            error_msg = format_error_message(e)
            await update.message.reply_text(error_msg, parse_mode=PARSE_MODE)
            context.user_data["awaiting_credentials"] = False
            return True

    async def settings_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /settings command."""
        logger.info("Settings command received", user_id=update.effective_user.id)

        # Gate: access-code verification
        if await self._require_verification(update):
            return

        try:
            user = self.db_ops.get_or_create_user(
                telegram_user_id=update.effective_user.id,
                telegram_username=update.effective_user.username,
                first_name=update.effective_user.first_name,
            )

            profile = self.preference_manager.get_user_profile(user.id)

            query_count = profile.get('query_count', 0)
            member_since = escape(str(profile.get('member_since', 'N/A')))
            fav_metric = escape(profile.get('favorite_metric', 'revenue'))
            time_range = escape(profile.get('preferred_time_range', 'last_7_days')).replace('_', ' ')
            query_type = escape(profile.get('primary_query_type', 'general'))

            settings_msg = (
                "<b>⚙️ Your Settings</b>\n\n"
                f"📋 <b>Profile</b>\n"
                f"• Member since: <code>{member_since}</code>\n"
                f"• Total queries: <code>{query_count}</code>\n\n"
                f"🎯 <b>Learned Preferences</b>\n"
                f"• Favorite metric: <code>{fav_metric}</code>\n"
                f"• Time range: <code>{time_range}</code>\n"
                f"• Query type: <code>{query_type}</code>\n\n"
                f"<i>Preferences are learned automatically from your queries.</i>\n\n"
                f"• /forget — Reset all learned data\n"
                f"• /connect — Reconnect store\n"
                f"• /help — All commands"
            )

            await update.message.reply_text(settings_msg, parse_mode=PARSE_MODE)

        except Exception as e:
            logger.error("Error in settings command", error=str(e), exc_info=True)
            error_msg = format_error_message(e)
            await update.message.reply_text(error_msg, parse_mode=PARSE_MODE)

    async def forget_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /forget command."""
        logger.info("Forget command received", user_id=update.effective_user.id)

        # Gate: access-code verification
        if await self._require_verification(update):
            return

        try:
            confirmation_msg = (
                "⚠️ <b>Clear All Learned Data</b>\n\n"
                "This will delete:\n"
                "• All query patterns\n"
                "• All preferences\n"
                "• All conversation history\n\n"
                "This action <b>cannot be undone</b>. Continue?"
            )

            keyboard = get_forget_confirmation_keyboard()
            await update.message.reply_text(
                confirmation_msg,
                reply_markup=keyboard,
                parse_mode=PARSE_MODE,
            )

        except Exception as e:
            logger.error("Error in forget command", error=str(e), exc_info=True)
            error_msg = format_error_message(e)
            await update.message.reply_text(error_msg, parse_mode=PARSE_MODE)

    async def handle_forget_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle callback from forget confirmation keyboard."""
        query = update.callback_query
        user_id = update.effective_user.id

        logger.info("Forget callback received", user_id=user_id, action=query.data)

        await query.answer()

        try:
            if query.data == "forget_confirm":
                user = self.db_ops.get_or_create_user(
                    telegram_user_id=user_id,
                    telegram_username=update.effective_user.username,
                    first_name=update.effective_user.first_name,
                )
                self.preference_manager.clear_learned_data(user.id)
                logger.info("User data cleared", user_id=user.id)

                confirm_msg = (
                    "✅ <b>Data Cleared</b>\n\n"
                    "All your learned data has been deleted.\n"
                    "I'll start fresh learning your preferences again."
                )
                await query.edit_message_text(confirm_msg, parse_mode=PARSE_MODE)

            elif query.data == "forget_cancel":
                cancel_msg = "❌ Cancelled. Your data remains intact."
                await query.edit_message_text(cancel_msg, parse_mode=PARSE_MODE)
                logger.info("User cancelled forget operation", user_id=user_id)

        except Exception as e:
            logger.error("Error in forget callback", error=str(e), exc_info=True)
            error_msg = format_error_message(e)
            await query.edit_message_text(error_msg, parse_mode=PARSE_MODE)

    async def status_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /status command."""
        logger.info("Status command received", user_id=update.effective_user.id)

        # Gate: access-code verification
        if await self._require_verification(update):
            return

        try:
            user = self.db_ops.get_or_create_user(
                telegram_user_id=update.effective_user.id,
                telegram_username=update.effective_user.username,
                first_name=update.effective_user.first_name,
            )

            # Auto-connect from .env if possible
            self._auto_connect_from_env(user.id)

            store = self.db_ops.get_store_by_user(user.id)

            if store:
                domain = escape(store.shop_domain)
                installed_at_str = store.installed_at.strftime('%b %d, %Y at %H:%M') if store.installed_at else "unknown"
                status_msg = (
                    f"<b>📡 Bot Status</b>\n\n"
                    f"✅ <b>Store:</b> <code>{domain}</code>\n"
                    f"📅 <b>Connected:</b> {installed_at_str}\n\n"
                    f"Everything looks good — ask me anything about your store!"
                )
            else:
                status_msg = (
                    "<b>📡 Bot Status</b>\n\n"
                    "❌ <b>Store:</b> Not connected\n\n"
                    "Use /connect to link your Shopify store, "
                    "or add credentials to your <code>.env</code> file."
                )

            await update.message.reply_text(status_msg, parse_mode=PARSE_MODE)

        except Exception as e:
            logger.error("Error in status command", error=str(e), exc_info=True)
            error_msg = format_error_message(e)
            await update.message.reply_text(error_msg, parse_mode=PARSE_MODE)

    # ── /verify command ─────────────────────────────────────────────

    async def verify_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /verify <code> command.

        Allows users to explicitly submit the access code.
        Usage: /verify MY_SECRET_CODE
        """
        telegram_user_id = update.effective_user.id
        logger.info("Verify command received", user_id=telegram_user_id)

        # If no access code is configured, verification is not needed
        if not self._bot_access_code:
            await update.message.reply_text(
                "ℹ️ No access code is required for this bot.",
                parse_mode=PARSE_MODE,
            )
            return

        # Already verified?
        user = self.db_ops.get_or_create_user(
            telegram_user_id=telegram_user_id,
            telegram_username=update.effective_user.username,
            first_name=update.effective_user.first_name,
        )
        if user.is_verified:
            await update.message.reply_text(
                "✅ You're already verified! Just ask me a question.",
                parse_mode=PARSE_MODE,
            )
            return

        # Extract code from command arguments
        args = context.args  # list of words after /verify
        if not args:
            await update.message.reply_text(
                "Usage: <code>/verify YOUR_ACCESS_CODE</code>\n\n"
                "Or simply send the code as a regular message.",
                parse_mode=PARSE_MODE,
            )
            return

        code_attempt = " ".join(args).strip()

        # Constant-time comparison to prevent timing attacks
        if hmac.compare_digest(code_attempt, self._bot_access_code):
            self.db_ops.verify_user(user.id)
            logger.info(
                "User verified via /verify command",
                user_id=user.id,
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
                parse_mode=PARSE_MODE,
            )
        else:
            logger.warning(
                "Invalid access code attempt via /verify",
                telegram_user_id=telegram_user_id,
            )
            await update.message.reply_text(
                "❌ <b>Invalid access code.</b>\n\n"
                "Please check the code and try again, or contact the bot administrator.",
                parse_mode=PARSE_MODE,
            )
