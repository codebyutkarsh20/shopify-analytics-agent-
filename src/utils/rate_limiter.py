"""
In-memory sliding-window rate limiter.

Tracks message timestamps per user and blocks requests when the
configured limit is exceeded within the time window.
"""

import time
from collections import defaultdict
from typing import Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Sliding-window rate limiter for per-user message throttling.

    Stores timestamps in memory (not persistent across restarts, which
    is fine — a restart resets the window).
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed per window.
            window_seconds: Size of the sliding window in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[int, list[float]] = defaultdict(list)
        logger.info(
            "RateLimiter initialized",
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

    def is_allowed(self, user_id: int) -> Tuple[bool, int]:
        """Check if a user is within their rate limit.

        Prunes expired timestamps and checks the count.

        Args:
            user_id: The Telegram user ID.

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: int).
            If allowed is False, retry_after_seconds indicates how long
            the user should wait.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds

        # Prune old timestamps
        timestamps = self._requests[user_id]
        self._requests[user_id] = [t for t in timestamps if t > window_start]

        if len(self._requests[user_id]) >= self.max_requests:
            # Find when the oldest request in the window will expire
            oldest = self._requests[user_id][0]
            retry_after = int(oldest - window_start) + 1
            logger.warning(
                "Rate limit exceeded",
                user_id=user_id,
                requests_in_window=len(self._requests[user_id]),
                retry_after=retry_after,
            )
            return False, max(retry_after, 1)

        # Allow and record
        self._requests[user_id].append(now)
        return True, 0

    def reset(self, user_id: int) -> None:
        """Reset rate limit for a specific user.

        Args:
            user_id: The Telegram user ID.
        """
        self._requests.pop(user_id, None)

    def cleanup(self) -> int:
        """Remove stale entries from users who haven't sent messages recently.

        Returns:
            Number of user entries cleaned up.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds
        stale_users = [
            uid for uid, timestamps in self._requests.items()
            if not timestamps or all(t <= window_start for t in timestamps)
        ]
        for uid in stale_users:
            del self._requests[uid]
        return len(stale_users)
