"""Channel adapter abstraction for multi-channel support.

Provides a base interface that Telegram, WhatsApp, and future channels
implement.  The core agent logic works with any adapter.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional


class ChannelAdapter(ABC):
    """Base class for all channel adapters."""

    @abstractmethod
    def get_channel_type(self) -> str:
        """Return channel identifier: 'telegram', 'whatsapp', etc."""

    @abstractmethod
    def get_channel_user_id(self, event) -> str:
        """Extract the channel-specific user ID from an event."""

    @abstractmethod
    def get_user_metadata(self, event) -> Dict[str, Any]:
        """Extract user metadata (username, first_name, etc.) from event."""

    @abstractmethod
    def format_response(self, markdown_response: str) -> str:
        """Convert agent markdown to channel-specific format."""

    @abstractmethod
    def get_message_limit(self) -> int:
        """Max characters per message for this channel."""

    def get_session_hard_timeout(self) -> int:
        """Hard session timeout in seconds. Override per channel."""
        return 7200  # 2 hours default

    def get_session_soft_timeout(self) -> int:
        """Soft session timeout in seconds. Override per channel."""
        return 1800  # 30 minutes default
