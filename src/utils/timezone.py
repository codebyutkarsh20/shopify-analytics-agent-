"""Centralized timezone utilities for Indian Standard Time (IST).

All datetime operations in the application should use these helpers
to ensure consistent IST (Asia/Kolkata, UTC+5:30) timestamps.
"""

import pytz
from datetime import datetime

# Indian Standard Time timezone object
IST = pytz.timezone("Asia/Kolkata")


def now_ist() -> datetime:
    """Get current datetime in IST (Asia/Kolkata) as a naive datetime.

    Returns a naive (timezone-unaware) datetime representing the current
    time in IST. This keeps compatibility with SQLite which stores naive
    datetimes, while ensuring all timestamps are in Indian Standard Time.

    Returns:
        Naive datetime representing current IST time.
    """
    return datetime.now(IST).replace(tzinfo=None)


def now_ist_aware() -> datetime:
    """Get current datetime in IST (Asia/Kolkata) as a timezone-aware datetime.

    Use this when you need timezone-aware datetimes (e.g., for date_parser).

    Returns:
        Timezone-aware datetime in IST.
    """
    return datetime.now(IST)


def localize_to_ist(dt: datetime) -> datetime:
    """Localize a naive datetime to IST.

    Args:
        dt: A naive (timezone-unaware) datetime object.

    Returns:
        Timezone-aware datetime in IST.
    """
    if dt.tzinfo is None:
        return IST.localize(dt)
    return dt.astimezone(IST)
