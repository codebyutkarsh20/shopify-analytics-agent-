"""Session management for single-thread chat platforms.

Handles session lifecycle for platforms like Telegram and WhatsApp where
explicit conversation boundaries aren't available. Uses time gaps, topic shifts,
and explicit user signals to detect session boundaries.
"""

import json
from datetime import datetime, timedelta
from typing import Optional

from src.database.models import Session
from src.utils.logger import get_logger
from src.utils.timezone import now_ist

logger = get_logger(__name__)


class SessionManager:
    """Manages conversation sessions for single-thread chat platforms.

    Detects session boundaries using:
    - Hard timeout: session expires after N seconds regardless of activity
    - Soft timeout: session expires if topic changes within N seconds
    - Min gap: explicit session signals require minimum gap before triggering
    """

    def __init__(self, db_ops, settings):
        """Initialize SessionManager.

        Args:
            db_ops: DatabaseOperations instance
            settings: Settings object with session configuration
        """
        self.db_ops = db_ops
        self.settings = settings

        # Extract session timeout settings
        self.hard_timeout_seconds = getattr(
            settings, "session_hard_timeout_seconds", 7200
        )
        self.soft_timeout_seconds = getattr(
            settings, "session_soft_timeout_seconds", 1800
        )
        self.min_gap_seconds = getattr(
            settings, "session_min_gap_seconds", 300
        )

        logger.info(
            "SessionManager initialized",
            hard_timeout=self.hard_timeout_seconds,
            soft_timeout=self.soft_timeout_seconds,
            min_gap=self.min_gap_seconds,
        )

    def get_or_create_session(
        self,
        user_id: int,
        channel_type: str,
        current_intent: Optional[str] = None,
    ) -> Session:
        """Get existing session or create new one.

        Implements multi-factor session boundary detection:
        1. Check for active session
        2. Compute time gap since last message
        3. Detect if new session needed
        4. Auto-summarize old session if ended

        Args:
            user_id: The user's ID
            channel_type: Type of channel (telegram, whatsapp, etc.)
            current_intent: Current query intent category

        Returns:
            Active Session object
        """
        # Step 1: Get active session
        active_session = self.db_ops.get_active_session(user_id)

        if not active_session:
            logger.info(
                "Creating new session (no active session)",
                user_id=user_id,
                channel_type=channel_type,
            )
            return self.db_ops.create_session(
                user_id=user_id,
                channel_type=channel_type,
                primary_intent=current_intent,
            )

        # Step 2: Compute time gap
        now = now_ist()
        last_msg = active_session.last_message_at or active_session.started_at
        time_gap = (now - last_msg).total_seconds()

        # Step 3: Get previous intent
        previous_intent = active_session.primary_intent

        # Step 4: Detect if new session needed
        needs_new_session = self._detect_new_session(
            time_gap=time_gap,
            current_intent=current_intent,
            previous_intent=previous_intent,
        )

        if needs_new_session:
            logger.info(
                "Session boundary detected, ending old session",
                user_id=user_id,
                session_id=active_session.id,
                time_gap_seconds=time_gap,
                current_intent=current_intent,
            )

            # Step 5a: Auto-generate summary and end old session
            summary = self._generate_session_summary(active_session.id)
            self.db_ops.end_session(
                session_id=active_session.id,
                summary=summary,
            )

            # Step 5b: Create new session
            new_session = self.db_ops.create_session(
                user_id=user_id,
                channel_type=channel_type,
                primary_intent=current_intent,
            )
            logger.info(
                "New session created",
                user_id=user_id,
                new_session_id=new_session.id,
            )
            return new_session

        # Step 6: Update existing session activity
        logger.debug(
            "Continuing existing session",
            user_id=user_id,
            session_id=active_session.id,
            time_gap_seconds=time_gap,
        )
        self.db_ops.update_session_activity(
            session_id=active_session.id,
        )

        return active_session

    def _detect_new_session(
        self,
        time_gap: float,
        current_intent: Optional[str],
        previous_intent: Optional[str],
    ) -> bool:
        """Detect if a new session is needed.

        Multi-factor decision:
        1. Hard timeout: always start new session if exceeded
        2. Soft timeout + topic change: start new if time gap and intent changed
        3. Explicit signal: start new if user signals new topic

        Args:
            time_gap: Seconds since last message in session
            current_intent: Current query's intent category
            previous_intent: Previous session's primary intent

        Returns:
            True if new session should be created
        """
        # Hard timeout: always end session after this duration
        if time_gap > self.hard_timeout_seconds:
            logger.debug(
                "Hard timeout triggered",
                time_gap=time_gap,
                hard_timeout=self.hard_timeout_seconds,
            )
            return True

        # Soft timeout + topic change
        if time_gap > self.soft_timeout_seconds:
            if current_intent and previous_intent and current_intent != previous_intent:
                logger.debug(
                    "Soft timeout with topic change",
                    time_gap=time_gap,
                    soft_timeout=self.soft_timeout_seconds,
                    prev_intent=previous_intent,
                    current_intent=current_intent,
                )
                return True

        # Explicit signal with minimum gap
        if time_gap > self.min_gap_seconds:
            if self._has_explicit_new_signal(current_intent):
                logger.debug(
                    "Explicit new session signal detected",
                    time_gap=time_gap,
                    signal=current_intent,
                )
                return True

        return False

    def _has_explicit_new_signal(self, message: Optional[str]) -> bool:
        """Check for explicit new session signals in message.

        Looks for keywords indicating user wants to start fresh:
        - "new question"
        - "different topic"
        - "something else"
        - "another thing"
        - "unrelated"

        Args:
            message: User message or intent to check

        Returns:
            True if explicit new session signal found
        """
        if not message:
            return False

        message_lower = message.lower()
        signals = [
            "new question",
            "different topic",
            "something else",
            "another thing",
            "unrelated",
            "start over",
            "new topic",
        ]

        for signal in signals:
            if signal in message_lower:
                logger.debug("Explicit new session signal found", signal=signal)
                return True

        return False

    def _generate_session_summary(self, session_id: int) -> str:
        """Generate a summary of session conversations.

        Builds a concise summary from session's recent messages:
        "Discussed {primary_intent}: {first_query_summary}... ({message_count} messages)"

        Args:
            session_id: ID of session to summarize

        Returns:
            Summary string
        """
        conversations = self.db_ops.get_session_conversations(
            session_id=session_id,
            limit=10,
        )

        if not conversations:
            return "Empty session"

        # Get primary intent from first or most common query
        query_types = [
            c.query_type for c in conversations if c.query_type
        ]
        primary_type = max(set(query_types), key=query_types.count) if query_types else "general"

        # Get first message for summary
        first_message = conversations[0].message[:100] if conversations[0].message else "N/A"

        summary = (
            f"Discussed {primary_type}: {first_message}... "
            f"({len(conversations)} messages)"
        )

        logger.debug(
            "Session summary generated",
            session_id=session_id,
            message_count=len(conversations),
            summary=summary,
        )

        return summary

    def get_session_context(
        self,
        user_id: int,
        current_session_id: int,
    ) -> dict:
        """Get context from previous session(s).

        Returns previous session summary if available, useful for providing
        continuity or understanding recent work.

        Args:
            user_id: The user's ID
            current_session_id: Current session ID

        Returns:
            Dict with keys:
            - "previous_summary": str or None
            - "current_message_count": int
        """
        # Get current session's message count
        current_convs = self.db_ops.get_session_conversations(
            session_id=current_session_id,
        )
        current_count = len(current_convs)

        # Get previous session
        previous_session = self.db_ops.get_previous_session(
            user_id=user_id,
            current_session_id=current_session_id,
        )
        previous_summary = None

        if previous_session:
            previous_summary = previous_session.session_summary
            logger.debug(
                "Retrieved previous session context",
                user_id=user_id,
                previous_session_id=previous_session.id,
                has_summary=bool(previous_summary),
            )

        return {
            "previous_summary": previous_summary,
            "current_message_count": current_count,
        }
