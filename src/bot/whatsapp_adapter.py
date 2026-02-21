"""WhatsApp-specific channel adapter.

Implements the ChannelAdapter interface for WhatsApp Business API
(supports both Twilio and Meta Cloud API webhook payloads).
"""

import re
from typing import Dict, Any

from src.bot.channel_adapter import ChannelAdapter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def markdown_to_whatsapp(text: str) -> str:
    """Convert agent markdown to WhatsApp-compatible formatting.

    WhatsApp supports a limited subset of formatting:
        *bold*   _italic_   ~strikethrough~   ```monospace```

    This function converts standard Markdown into WhatsApp formatting
    while stripping unsupported constructs (HTML tags, links, headings).
    """
    if not text:
        return text

    # ── Protect code blocks from further processing ──
    protected: dict[str, str] = {}
    counter = [0]

    def _protect(replacement: str) -> str:
        key = f"\x00WA{counter[0]}\x00"
        protected[key] = replacement
        counter[0] += 1
        return key

    # Fenced code blocks → WhatsApp monospace blocks
    def _code_block(m):
        return _protect(f"```{m.group(1).strip()}```")
    text = re.sub(r"```(?:\w*\n)?(.*?)```", _code_block, text, flags=re.DOTALL)

    # Inline code → WhatsApp monospace
    def _inline_code(m):
        return _protect(f"`{m.group(1)}`")
    text = re.sub(r"`([^`]+)`", _inline_code, text)

    # ── Convert Markdown → WhatsApp formatting ──

    # Headings → bold line
    text = re.sub(r"^#{1,6}\s+(.+?)\s*$", r"*\1*", text, flags=re.MULTILINE)

    # Horizontal rules → blank line
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Bold: **text** → *text*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)

    # Italic: *text* → _text_ (only when not at line-start bullets)
    text = re.sub(r"(?<![*\w])\*([^*\n]+?)\*(?![*\w])", r"_\1_", text)

    # Strikethrough: ~~text~~ → ~text~
    text = re.sub(r"~~(.+?)~~", r"~\1~", text)

    # Links: [text](url) → text (url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    # Strip any remaining HTML tags (e.g., <b>, <i>, <a>)
    text = re.sub(r"<[^>]+>", "", text)

    # Bullet points: keep • or convert - to •
    text = re.sub(r"^[-]\s+", "• ", text, flags=re.MULTILINE)

    # ── Restore protected code blocks ──
    for key, value in protected.items():
        text = text.replace(key, value)

    # Clean up excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


class WhatsAppAdapter(ChannelAdapter):
    """WhatsApp channel adapter.

    Handles WhatsApp-specific user resolution, message formatting
    (WhatsApp-flavored markdown), and message size limits.
    """

    def get_channel_type(self) -> str:
        return "whatsapp"

    def get_channel_user_id(self, event: Dict[str, Any]) -> str:
        """Extract WhatsApp phone number as the user identifier.

        Args:
            event: Normalized webhook event dict with at least a ``from_number`` key.
        """
        return str(event.get("from_number", ""))

    def get_user_metadata(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract WhatsApp user metadata from the webhook event."""
        return {
            "whatsapp_phone": event.get("from_number", ""),
            "whatsapp_name": event.get("profile_name", ""),
            "first_name": event.get("profile_name", "").split()[0] if event.get("profile_name") else "",
            "display_name": event.get("profile_name", "") or event.get("from_number", ""),
        }

    def format_response(self, markdown_response: str) -> str:
        """Convert agent markdown to WhatsApp-compatible formatting."""
        return markdown_to_whatsapp(markdown_response)

    def get_message_limit(self) -> int:
        # WhatsApp Business API text message limit
        return 4096

    def get_session_hard_timeout(self) -> int:
        # WhatsApp conversations have a 24-hour window;
        # keep internal session shorter for analytics context.
        return 7200  # 2 hours

    def get_session_soft_timeout(self) -> int:
        return 1800  # 30 minutes
