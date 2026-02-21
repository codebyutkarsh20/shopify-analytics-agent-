"""Cross-channel user linking service.

Allows users to link their Telegram and WhatsApp identities into a
single canonical user.  Inspired by OpenClaw's identity-links concept
but adapted for self-serve multi-user bots.

Flow:
  1. User sends /link on Telegram (or "link" on WhatsApp)
  2. Bot generates a short-lived 6-digit code
  3. User sends "link <code>" on the OTHER channel within 10 minutes
  4. System verifies the code, merges the two User rows into one
     canonical user, and re-points all data (conversations, patterns,
     preferences, sessions, stores, etc.)
  5. Both channels now share the same identity, history, and preferences
"""

import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from src.database.operations import DatabaseOperations
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Linking codes expire after 10 minutes
LINK_CODE_TTL_SECONDS = 600


@dataclass
class PendingLink:
    """A pending cross-channel link request."""
    code: str
    user_id: int                # Internal User.id of the requester
    source_channel: str         # Channel where the code was generated ("telegram" or "whatsapp")
    source_channel_id: str      # Channel-specific ID (telegram_user_id or phone number)
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > LINK_CODE_TTL_SECONDS


class ChannelLinker:
    """Manages cross-channel user identity linking.

    Stores pending link codes in memory (they're short-lived and don't
    need persistence).  When a code is redeemed on the target channel,
    merges the two User records into one canonical identity.
    """

    def __init__(self, db_ops: DatabaseOperations):
        self.db_ops = db_ops
        # In-memory store: code → PendingLink
        self._pending: Dict[str, PendingLink] = {}
        logger.info("ChannelLinker initialized")

    def generate_link_code(
        self,
        user_id: int,
        source_channel: str,
        source_channel_id: str,
    ) -> str:
        """Generate a 6-digit linking code for a user.

        Args:
            user_id: Internal User.id requesting the link.
            source_channel: "telegram" or "whatsapp".
            source_channel_id: The channel-specific identifier.

        Returns:
            A 6-digit code string.
        """
        # Clean up any expired codes first
        self._cleanup_expired()

        # Remove any existing pending code for this user
        self._pending = {
            code: link for code, link in self._pending.items()
            if link.user_id != user_id
        }

        # Generate a unique 6-digit code
        code = self._generate_unique_code()

        self._pending[code] = PendingLink(
            code=code,
            user_id=user_id,
            source_channel=source_channel,
            source_channel_id=source_channel_id,
        )

        logger.info(
            "Link code generated",
            user_id=user_id,
            source_channel=source_channel,
            code=code,
        )

        return code

    def redeem_link_code(
        self,
        code: str,
        target_channel: str,
        target_user_id: int,
        target_channel_id: str,
    ) -> Tuple[bool, str]:
        """Redeem a linking code from the target channel.

        Validates the code and merges the two user accounts.

        Args:
            code: The 6-digit code to redeem.
            target_channel: Channel where the code is being redeemed.
            target_user_id: Internal User.id of the redeemer.
            target_channel_id: Channel-specific ID of the redeemer.

        Returns:
            (success: bool, message: str)
        """
        self._cleanup_expired()

        pending = self._pending.get(code)

        if not pending:
            return False, "Invalid or expired linking code. Please generate a new one."

        if pending.is_expired:
            del self._pending[code]
            return False, "This linking code has expired. Please generate a new one."

        # Cannot link to the same channel
        if pending.source_channel == target_channel:
            return False, (
                f"This code was generated on {target_channel}. "
                f"Please use it on the OTHER channel to link your accounts."
            )

        # Cannot link to yourself (same user_id)
        if pending.user_id == target_user_id:
            del self._pending[code]
            return False, "These accounts are already linked!"

        # Perform the merge
        try:
            source_user_id = pending.user_id
            self._merge_users(
                keep_user_id=source_user_id,
                merge_user_id=target_user_id,
                target_channel=target_channel,
                target_channel_id=target_channel_id,
            )

            # Remove the used code
            del self._pending[code]

            logger.info(
                "Accounts linked successfully",
                source_user_id=source_user_id,
                merged_user_id=target_user_id,
                source_channel=pending.source_channel,
                target_channel=target_channel,
            )

            return True, (
                "Accounts linked successfully! "
                "Your conversation history, preferences, and store connections "
                "are now shared across both channels."
            )

        except Exception as e:
            logger.error(
                "Failed to link accounts",
                error=str(e),
                source_user_id=pending.user_id,
                target_user_id=target_user_id,
                exc_info=True,
            )
            return False, "Failed to link accounts due to an internal error. Please try again."

    def _merge_users(
        self,
        keep_user_id: int,
        merge_user_id: int,
        target_channel: str,
        target_channel_id: str,
    ) -> None:
        """Merge two user accounts into one canonical identity.

        The ``keep_user_id`` survives; the ``merge_user_id`` has all its
        data re-pointed to keep_user_id and is then deleted.

        Also updates the kept user with the channel-specific identifier
        from the merged user (e.g., adds whatsapp_phone if linking from
        WhatsApp, or telegram_user_id if linking from Telegram).
        """
        self.db_ops.merge_users(
            keep_user_id=keep_user_id,
            merge_user_id=merge_user_id,
            target_channel=target_channel,
            target_channel_id=target_channel_id,
        )

    def check_already_linked(self, user_id: int) -> Optional[str]:
        """Check if a user already has both channels linked.

        Returns a message if already linked, None otherwise.
        """
        user = self.db_ops.get_user_by_id(user_id)
        if not user:
            return None

        has_telegram = user.telegram_user_id is not None
        has_whatsapp = user.whatsapp_phone is not None

        if has_telegram and has_whatsapp:
            return (
                "Your accounts are already linked! "
                f"Telegram and WhatsApp ({user.whatsapp_phone}) share the same identity."
            )

        return None

    def _generate_unique_code(self) -> str:
        """Generate a 6-digit code that isn't currently in use."""
        for _ in range(100):  # Safety limit
            code = f"{secrets.randbelow(1000000):06d}"
            if code not in self._pending:
                return code
        raise RuntimeError("Failed to generate unique link code")

    def _cleanup_expired(self) -> None:
        """Remove expired pending links."""
        expired = [
            code for code, link in self._pending.items()
            if link.is_expired
        ]
        for code in expired:
            del self._pending[code]
