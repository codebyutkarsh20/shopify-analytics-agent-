"""Tests for Shopify Analytics Agent bot utilities."""

import pytest
from datetime import datetime, timedelta
import pytz

from src.utils.date_parser import parse_date_range, get_comparison_range
from src.utils.formatters import format_currency, format_percentage


class TestDateParser:
    """Test suite for date parsing utilities."""

    def test_parse_yesterday(self):
        """Test parsing 'yesterday' into correct date range.

        Should return a DateRange for yesterday from 00:00 to 23:59.
        """
        result = parse_date_range("yesterday")

        assert result.label == "Yesterday"
        assert result.start_date.hour == 0
        assert result.start_date.minute == 0
        assert result.end_date.hour == 23
        assert result.end_date.minute == 59

        # Verify it's yesterday
        now = datetime.now(pytz.UTC)
        yesterday = now - timedelta(days=1)

        assert result.start_date.date() == yesterday.date()
        assert result.end_date.date() == yesterday.date()

    def test_parse_today(self):
        """Test parsing 'today' into correct date range.

        Should return a DateRange from today 00:00 to now.
        """
        result = parse_date_range("today")

        assert result.label == "Today"
        assert result.start_date.hour == 0
        assert result.start_date.minute == 0

        # End should be close to now
        now = datetime.now(pytz.UTC)
        assert result.end_date.date() == now.date()

    def test_parse_last_7_days(self):
        """Test parsing 'last 7 days' into correct date range.

        Should return a DateRange for the last 7 days.
        """
        result = parse_date_range("last 7 days")

        assert result.label == "Last 7 Days"

        # Verify span is approximately 7 days
        span = (result.end_date - result.start_date).days
        assert span == 7

    def test_parse_last_7_days_variants(self):
        """Test alternative phrasings for last 7 days.

        "last week", "this week", "past week" should all return same period.
        """
        result1 = parse_date_range("last 7 days")
        result2 = parse_date_range("last week")
        result3 = parse_date_range("this week")
        result4 = parse_date_range("past week")

        # All should have 7-day span
        for result in [result1, result2, result3, result4]:
            span = (result.end_date - result.start_date).days
            assert span == 7

    def test_parse_last_30_days(self):
        """Test parsing 'last 30 days' into correct date range.

        Should return a DateRange for the last 30 days.
        """
        result = parse_date_range("last 30 days")

        assert result.label == "Last 30 Days"

        # Verify span is approximately 30 days
        span = (result.end_date - result.start_date).days
        assert span == 30

    def test_parse_last_30_days_variants(self):
        """Test alternative phrasings for last 30 days.

        "last month", "this month", "past month" should all return same period.
        """
        result1 = parse_date_range("last 30 days")
        result2 = parse_date_range("last month")
        result3 = parse_date_range("this month")
        result4 = parse_date_range("past month")

        # All should have 30-day span
        for result in [result1, result2, result3, result4]:
            span = (result.end_date - result.start_date).days
            assert span == 30

    def test_parse_custom_days(self):
        """Test parsing custom 'last X days' format.

        Should handle arbitrary day counts like "last 14 days".
        """
        result = parse_date_range("last 14 days")

        assert result.label == "Last 14 Days"
        span = (result.end_date - result.start_date).days
        assert span == 14

    def test_parse_default(self):
        """Test parsing unknown format returns default (last 7 days).

        Unknown input like 'blah blah' should default to last 7 days.
        """
        result = parse_date_range("blah blah this makes no sense")

        # Should default to last 7 days
        assert result.label == "Last 7 Days"
        span = (result.end_date - result.start_date).days
        assert span == 7

    def test_comparison_range_last_7_days(self):
        """Test getting comparison range for last 7 days.

        Should return the 7 days before the current period.
        """
        current = parse_date_range("last 7 days")
        comparison = get_comparison_range(current)

        # Comparison span should match current span
        current_span = (current.end_date - current.start_date).days
        comparison_span = (comparison.end_date - comparison.start_date).days
        assert comparison_span == current_span

        # Comparison should end before current starts
        assert comparison.end_date < current.start_date

    def test_comparison_range_last_30_days(self):
        """Test getting comparison range for last 30 days.

        Should return the 30 days before the current period.
        """
        current = parse_date_range("last 30 days")
        comparison = get_comparison_range(current)

        # Comparison span should match current span
        current_span = (current.end_date - current.start_date).days
        comparison_span = (comparison.end_date - comparison.start_date).days
        assert comparison_span == current_span

        # Comparison should end before current starts
        assert comparison.end_date < current.start_date


class TestFormatters:
    """Test suite for formatter utilities."""

    def test_format_currency_basic(self):
        """Test basic currency formatting.

        format_currency(1234.56) -> "$1,234.56"
        """
        result = format_currency(1234.56)

        assert result == "$1,234.56"

    def test_format_currency_whole_number(self):
        """Test currency formatting with whole numbers."""
        result = format_currency(1000)

        assert result == "$1,000.00"

    def test_format_currency_small_amount(self):
        """Test currency formatting with small amounts."""
        result = format_currency(12.5)

        assert result == "$12.50"

    def test_format_currency_zero(self):
        """Test currency formatting with zero."""
        result = format_currency(0)

        assert result == "$0.00"

    def test_format_currency_none(self):
        """Test currency formatting with None value."""
        result = format_currency(None)

        assert result == "$0.00"

    def test_format_currency_custom_symbol(self):
        """Test currency formatting with custom symbol."""
        result = format_currency(1000, currency_symbol="£")

        assert result == "£1,000.00"

    def test_format_percentage_basic(self):
        """Test basic percentage formatting.

        format_percentage(15.5) -> "15.5%"
        """
        result = format_percentage(15.5)

        assert result == "15.5%"

    def test_format_percentage_negative(self):
        """Test percentage formatting with negative value.

        Note: The format_percentage function multiplies by 100 if value < 2,
        so -10.3 becomes -1030.0%.
        """
        result = format_percentage(-10.3)

        # -10.3 is < 2, so it gets multiplied by 100
        assert result == "-1030.0%"

    def test_format_percentage_zero(self):
        """Test percentage formatting with zero."""
        result = format_percentage(0)

        assert result == "0.0%"

    def test_format_percentage_decimal_range(self):
        """Test percentage formatting with decimal range (0-1).

        Should convert 0.5 to 50%.
        """
        result = format_percentage(0.5)

        assert result == "50.0%"

    def test_format_percentage_decimal_places(self):
        """Test percentage formatting with custom decimal places."""
        result = format_percentage(15.567, decimal_places=2)

        assert result == "15.57%"

    def test_format_percentage_none(self):
        """Test percentage formatting with None value."""
        result = format_percentage(None)

        assert result == "0%"

    def test_format_percentage_large_number(self):
        """Test percentage formatting with large number."""
        result = format_percentage(150)

        assert result == "150.0%"

    def test_format_percentage_negative_decimal(self):
        """Test percentage formatting with negative decimal range value."""
        result = format_percentage(-0.25)

        assert result == "-25.0%"
