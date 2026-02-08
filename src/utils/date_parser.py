"""
Date parsing utilities for natural language date expressions.

This module provides functionality to parse natural language date expressions
and convert them to specific date ranges with timezone awareness.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytz
from dateutil.parser import parse as dateutil_parse
from dateutil.relativedelta import relativedelta


@dataclass
class DateRange:
    """Represents a date range with start and end dates."""

    start_date: datetime
    end_date: datetime
    label: str

    def __post_init__(self) -> None:
        """Ensure dates are timezone-aware (UTC)."""
        if self.start_date.tzinfo is None:
            self.start_date = pytz.UTC.localize(self.start_date)
        if self.end_date.tzinfo is None:
            self.end_date = pytz.UTC.localize(self.end_date)


def parse_date_range(text: str) -> DateRange:
    """
    Parse natural language date expressions into DateRange objects.

    Supports the following patterns:
    - "yesterday" -> yesterday 00:00 to 23:59
    - "today" -> today 00:00 to now
    - "last 7 days", "last week", "this week", "past week" -> last 7 days
    - "last 30 days", "last month", "this month", "past month" -> last 30 days
    - "last X days" -> last X days (where X is a number)
    - "day before yesterday" -> that specific day
    - Default: last 7 days

    Args:
        text: Natural language date expression (case-insensitive)

    Returns:
        DateRange object with parsed start_date, end_date, and label

    Examples:
        >>> dr = parse_date_range("yesterday")
        >>> # DateRange with yesterday's start and end times

        >>> dr = parse_date_range("last 7 days")
        >>> # DateRange for the last 7 days

        >>> dr = parse_date_range("last 30 days")
        >>> # DateRange for the last 30 days
    """
    text_lower = text.lower().strip()
    now = datetime.now(pytz.UTC)

    # Yesterday
    if text_lower == "yesterday":
        yesterday = now - timedelta(days=1)
        start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return DateRange(start_date, end_date, "Yesterday")

    # Today
    if text_lower == "today":
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return DateRange(start_date, now, "Today")

    # Day before yesterday
    if text_lower == "day before yesterday":
        day_before = now - timedelta(days=2)
        start_date = day_before.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = day_before.replace(hour=23, minute=59, second=59, microsecond=999999)
        return DateRange(start_date, end_date, "Day Before Yesterday")

    # Last 7 days / week patterns (check BEFORE generic "last X days" regex)
    if text_lower in ["last 7 days", "last week", "this week", "past week", "past 7 days"]:
        end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_date = (now - timedelta(days=7)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return DateRange(start_date, end_date, "Last 7 Days")

    # Last 30 days / month patterns (check BEFORE generic "last X days" regex)
    if text_lower in ["last 30 days", "last month", "this month", "past month", "past 30 days"]:
        end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_date = (now - timedelta(days=30)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return DateRange(start_date, end_date, "Last 30 Days")

    # Generic "last X days" pattern (e.g., "last 14 days") — after specific patterns
    match = re.match(r"(?:last|past)\s+(\d+)\s+days?", text_lower)
    if match:
        num_days = int(match.group(1))
        end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_date = (now - timedelta(days=num_days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        label = f"Last {num_days} Days"
        return DateRange(start_date, end_date, label)

    # Default: last 7 days
    end_date = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    start_date = (now - timedelta(days=7)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return DateRange(start_date, end_date, "Last 7 Days")


def get_comparison_range(date_range: DateRange) -> DateRange:
    """
    Get the previous equivalent period for comparison.

    For example, if the input is the last 7 days, returns the 7 days before that.

    Args:
        date_range: The DateRange to find a comparison period for

    Returns:
        DateRange for the previous equivalent period

    Examples:
        >>> current = parse_date_range("last 7 days")
        >>> previous = get_comparison_range(current)
        >>> # previous is 14 days to 7 days ago
    """
    period_length = (date_range.end_date - date_range.start_date).days

    # Calculate new end_date (the day before the current period starts)
    new_end_date = date_range.start_date - timedelta(seconds=1)

    # Calculate new start_date
    new_start_date = new_end_date - timedelta(days=period_length)

    # Ensure we start at midnight
    new_start_date = new_start_date.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    comparison_label = f"Previous Period ({period_length} days)"

    return DateRange(new_start_date, new_end_date, comparison_label)
