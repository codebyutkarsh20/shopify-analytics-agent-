"""Telegram-specific channel adapter."""

from typing import Dict, Any

from src.bot.channel_adapter import ChannelAdapter
from src.utils.formatters import markdown_to_telegram_html
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TelegramAdapter(ChannelAdapter):
    """Telegram channel adapter.

    Handles Telegram-specific user resolution, message formatting
    (HTML), and message size limits.
    """

    def get_channel_type(self) -> str:
        return "telegram"

    def get_channel_user_id(self, update) -> str:
        """Extract Telegram user ID as string."""
        return str(update.effective_user.id)

    def get_user_metadata(self, update) -> Dict[str, Any]:
        """Extract Telegram user metadata."""
        user = update.effective_user
        return {
            "telegram_user_id": user.id,
            "telegram_username": user.username,
            "first_name": user.first_name,
            "display_name": user.first_name or user.username or str(user.id),
        }

    def format_response(self, markdown_response: str) -> str:
        """Convert markdown to Telegram HTML."""
        return markdown_to_telegram_html(markdown_response)

    def get_message_limit(self) -> int:
        return 4096

    def get_session_hard_timeout(self) -> int:
        return 7200  # 2 hours

    def get_session_soft_timeout(self) -> int:
        return 1800  # 30 minutes
