"""Inline Keyboard generators for Telegram bot.

Provides helper functions to create various inline keyboard markup for user interactions
in the Shopify Analytics bot.
"""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def get_resetstore_confirmation_keyboard() -> InlineKeyboardMarkup:
    """
    Create an inline keyboard for resetstore command confirmation.

    Returns a keyboard with options to confirm or cancel resetting store learning data.

    Returns:
        InlineKeyboardMarkup with Yes/Cancel buttons
    """
    buttons = [
        [
            InlineKeyboardButton(
                "✅ Yes, reset store data",
                callback_data="resetstore_confirm"
            ),
            InlineKeyboardButton(
                "❌ Cancel",
                callback_data="resetstore_cancel"
            ),
        ]
    ]
    return InlineKeyboardMarkup(buttons)


def get_forget_confirmation_keyboard() -> InlineKeyboardMarkup:
    """
    Create an inline keyboard for forget command confirmation.

    Returns a keyboard with options to confirm or cancel clearing all learned data.

    Returns:
        InlineKeyboardMarkup with Yes/Cancel buttons
    """
    buttons = [
        [
            InlineKeyboardButton(
                "✅ Yes, clear my data",
                callback_data="forget_confirm"
            ),
            InlineKeyboardButton(
                "❌ Cancel",
                callback_data="forget_cancel"
            ),
        ]
    ]
    return InlineKeyboardMarkup(buttons)


def get_time_range_keyboard() -> InlineKeyboardMarkup:
    """
    Create an inline keyboard for time range selection.

    Returns a keyboard with common time range options for analytics queries.

    Returns:
        InlineKeyboardMarkup with time range buttons
    """
    buttons = [
        [
            InlineKeyboardButton("📅 Today", callback_data="range_today"),
            InlineKeyboardButton("📅 Yesterday", callback_data="range_yesterday"),
        ],
        [
            InlineKeyboardButton("📊 Last 7 Days", callback_data="range_7days"),
            InlineKeyboardButton("📊 Last 30 Days", callback_data="range_30days"),
        ],
        [
            InlineKeyboardButton("📈 This Month", callback_data="range_month"),
        ]
    ]
    return InlineKeyboardMarkup(buttons)


def get_quick_actions_keyboard() -> InlineKeyboardMarkup:
    """
    Create an inline keyboard for quick analysis actions.

    Returns a keyboard with quick action buttons for common analytics queries.

    Returns:
        InlineKeyboardMarkup with quick action buttons
    """
    buttons = [
        [
            InlineKeyboardButton("📊 Today's Summary", callback_data="quick_today"),
        ],
        [
            InlineKeyboardButton("📈 Weekly Report", callback_data="quick_weekly"),
        ],
        [
            InlineKeyboardButton("🏆 Top Products", callback_data="quick_products"),
        ]
    ]
    return InlineKeyboardMarkup(buttons)
