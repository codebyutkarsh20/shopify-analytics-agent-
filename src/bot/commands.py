"""Bot command handlers for Telegram.

Handles all bot commands like /start, /help, /settings, /connect, /forget, and /status.
All messages use HTML parse_mode for robust formatting (no escaping headaches).
"""

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
from src.bot.keyboards import get_forget_confirmation_keyboard, get_resetstore_confirmation_keyboard

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
        logger.info("BotCommands initialized")

    def _auto_connect_from_env(self, user_id: int) -> bool:
        """
        Auto-connect a store from .env if credentials are configured
        and user has no store yet.  If the .env credentials differ
        from the currently stored store, update the store record and
        clear all stale learning data automatically.

        Returns True if a new store was connected or credentials were
        updated, False if nothing changed.
        """
        domain = settings.shopify.shop_domain
        token = settings.shopify.access_token
        if not domain or not token:
            return False

        store = self.db_ops.get_store_by_user(user_id)

        if store is None:
            # No store yet — fresh connect
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

        # Store exists — check if credentials changed
        if store.shop_domain == domain and store.access_token == token:
            return False  # Same store, nothing to do

        # Credentials changed → switching stores
        old_domain = store.shop_domain
        logger.info(
            "Store change detected — resetting learning data",
            user_id=user_id,
            old_domain=old_domain,
            new_domain=domain,
        )

        # Clear all learning data tied to the old store
        counts = self.db_ops.reset_store_learning_data(user_id)
        logger.info(
            "Learning data reset complete",
            user_id=user_id,
            deleted_counts=counts,
        )

        # Update the store record with new credentials
        self.db_ops.update_store_credentials(
            store_id=store.id,
            shop_domain=domain,
            access_token=token,
        )
        logger.info(
            "Store credentials updated",
            user_id=user_id,
            old_domain=old_domain,
            new_domain=domain,
        )

        # Re-seed query templates for the new store
        try:
            from src.learning.template_seeds import seed_templates
            seed_templates(self.db_ops)
            logger.info("Re-seeded query templates for new store")
        except Exception as e:
            logger.warning("Template re-seeding failed", error=str(e))

        return True

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

        try:
            context.user_data["awaiting_credentials"] = True
            context.user_data["connection_step"] = "waiting_for_domain"

            instructions = (
                "<b>🔗 Connect Your Shopify Store</b>\n\n"
                "I'll help you connect your Shopify store. Please provide:\n\n"
                "1. Your shop domain (e.g., <code>myshop.myshopify.com</code>)\n"
                "2. Your Shopify API access token\n\n"
                "For security reasons, send them as a single message in this format:\n"
                "<code>domain:token</code>\n\n"
                "<i>Example:</i> <code>myshop.myshopify.com:shpat_abc123xyz</code>\n\n"
                "⚠️ <b>Important:</b> Your credentials are encrypted and stored securely."
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
                f"• /resetstore — Reset learning data (keep store)\n"
                f"• /forget — Reset all data\n"
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

    async def resetstore_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /resetstore command — clear learning data for the current store."""
        logger.info("Resetstore command received", user_id=update.effective_user.id)

        try:
            user = self.db_ops.get_or_create_user(
                telegram_user_id=update.effective_user.id,
                telegram_username=update.effective_user.username,
                first_name=update.effective_user.first_name,
            )

            store = self.db_ops.get_store_by_user(user.id)
            if not store:
                await update.message.reply_text(
                    "❌ No store connected. Use /connect first.",
                    parse_mode=PARSE_MODE,
                )
                return

            domain = escape(store.shop_domain)
            confirmation_msg = (
                "⚠️ <b>Reset Store Learning Data</b>\n\n"
                f"Currently connected to <b>{domain}</b>.\n\n"
                "This will delete:\n"
                "• All conversation history\n"
                "• All query patterns & preferences\n"
                "• All learned templates & recovery patterns\n"
                "• All cached analytics\n\n"
                "Your store connection will be kept.\n"
                "This action <b>cannot be undone</b>. Continue?"
            )

            keyboard = get_resetstore_confirmation_keyboard()
            await update.message.reply_text(
                confirmation_msg,
                reply_markup=keyboard,
                parse_mode=PARSE_MODE,
            )

        except Exception as e:
            logger.error("Error in resetstore command", error=str(e), exc_info=True)
            error_msg = format_error_message(e)
            await update.message.reply_text(error_msg, parse_mode=PARSE_MODE)

    async def handle_resetstore_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle callback from resetstore confirmation keyboard."""
        query = update.callback_query
        user_id = update.effective_user.id

        logger.info("Resetstore callback received", user_id=user_id, action=query.data)

        await query.answer()

        try:
            if query.data == "resetstore_confirm":
                user = self.db_ops.get_or_create_user(
                    telegram_user_id=user_id,
                    telegram_username=update.effective_user.username,
                    first_name=update.effective_user.first_name,
                )
                counts = self.db_ops.reset_store_learning_data(user.id)
                logger.info("Store learning data reset", user_id=user.id, counts=counts)

                # Re-seed query templates
                try:
                    from src.learning.template_seeds import seed_templates
                    seed_templates(self.db_ops)
                except Exception as e:
                    logger.warning("Template re-seeding failed", error=str(e))

                total = sum(counts.values())
                confirm_msg = (
                    "✅ <b>Store Data Reset</b>\n\n"
                    f"Cleared {total} items across {len(counts)} tables.\n"
                    "I'll start fresh learning your preferences again."
                )
                await query.edit_message_text(confirm_msg, parse_mode=PARSE_MODE)

            elif query.data == "resetstore_cancel":
                cancel_msg = "❌ Cancelled. Your data remains intact."
                await query.edit_message_text(cancel_msg, parse_mode=PARSE_MODE)
                logger.info("User cancelled resetstore operation", user_id=user_id)

        except Exception as e:
            logger.error("Error in resetstore callback", error=str(e), exc_info=True)
            error_msg = format_error_message(e)
            await query.edit_message_text(error_msg, parse_mode=PARSE_MODE)
