"""
Message formatting utilities for Telegram output.

This module provides functions to format analytics data and messages
for Telegram bot responses using HTML formatting (much more robust
than MarkdownV2 for dynamic content).
"""

import html
import re
from typing import Any, Dict, Optional


def escape(text: str) -> str:
    """Escape text for Telegram HTML parse mode."""
    return html.escape(str(text))


def markdown_to_telegram_html(text: str) -> str:
    """
    Convert Claude's Markdown response to Telegram-compatible HTML.

    Telegram HTML supports: <b>, <i>, <code>, <pre>, <a href="">, <s>.
    This function converts common Markdown patterns into clean Telegram
    HTML while preserving readability. Includes a final sanitiser pass
    to strip any malformed/unclosed tags so Telegram never sees broken
    HTML (which causes it to show raw tags).
    """
    if not text:
        return text

    # ── Step 1: Protect code blocks & inline code from processing ──
    protected = {}
    counter = [0]

    def _protect(replacement_html: str) -> str:
        key = f"\x00P{counter[0]}\x00"
        protected[key] = replacement_html
        counter[0] += 1
        return key

    # Fenced code blocks: ```lang\n...\n```
    def _code_block(m):
        return _protect(f"<pre>{html.escape(m.group(1).strip())}</pre>")
    text = re.sub(r"```(?:\w*\n)?(.*?)```", _code_block, text, flags=re.DOTALL)

    # Inline code: `...`
    def _inline_code(m):
        return _protect(f"<code>{html.escape(m.group(1))}</code>")
    text = re.sub(r"`([^`]+)`", _inline_code, text)

    # ── Step 2: Escape HTML entities in remaining plain text ──
    text = html.escape(text)

    # ── Step 3: Convert Markdown → Telegram HTML ──

    # Headings: ## text  or  ## **text** → bold on its own line
    def _heading(m):
        heading_text = m.group(1).strip().strip("*").strip()
        return f"\n<b>{heading_text}</b>\n"
    text = re.sub(r"^#{1,6}\s+(.+?)\s*$", _heading, text, flags=re.MULTILINE)

    # Horizontal rules → blank line
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Bold: **text**  (non-greedy, allows multiword)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # Italic: *text*  (not at line-start bullets, not inside words)
    text = re.sub(r"(?<![*\w])\*([^*\n]+?)\*(?![*\w])", r"<i>\1</i>", text)

    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    # Links: [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    # Numbered lists: "1. item" → "1. item"  (keep number, just clean up)
    # Already looks fine in Telegram — just ensure consistent spacing
    text = re.sub(r"^(\d+)\.\s+", r"\1. ", text, flags=re.MULTILINE)

    # Bullet points: - item or * item → • item  (top-level)
    text = re.sub(r"^[-*]\s+", "• ", text, flags=re.MULTILINE)

    # Sub-bullets: "  - item" or "  * item" → "  ◦ item"
    text = re.sub(r"^(\s{2,})[-*]\s+", r"\1◦ ", text, flags=re.MULTILINE)

    # ── Step 4: Restore protected code blocks / inline code ──
    for key, value in protected.items():
        text = text.replace(key, value)

    # ── Step 5: Sanitise — ensure all HTML tags are properly balanced ──
    text = _sanitise_telegram_html(text)

    # ── Step 6: Clean up excessive blank lines ──
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# Tags that Telegram's HTML parser supports
_TELEGRAM_TAGS = {"b", "i", "u", "s", "code", "pre", "a"}


def _sanitise_telegram_html(text: str) -> str:
    """
    Ensure the HTML string only contains balanced, Telegram-supported tags.

    If a tag is opened but never closed (or vice-versa), it's stripped out
    so Telegram doesn't reject the whole message and show raw tags.
    """
    # Find all HTML-like tags in the text
    tag_pattern = re.compile(r"<(/?)(\w+)(\s[^>]*)?>")

    # First pass: collect every tag and its position
    tag_stack = []   # stack of (tag_name, start_pos) for open tags
    bad_spans = []   # (start, end) spans of tags to strip

    for m in tag_pattern.finditer(text):
        is_close = bool(m.group(1))
        tag_name = m.group(2).lower()

        # Ignore tags Telegram doesn't support → strip them
        if tag_name not in _TELEGRAM_TAGS:
            bad_spans.append((m.start(), m.end()))
            continue

        if not is_close:
            tag_stack.append((tag_name, m.start(), m.end()))
        else:
            # Try to match with the most recent open tag of the same name
            matched = False
            for idx in range(len(tag_stack) - 1, -1, -1):
                if tag_stack[idx][0] == tag_name:
                    tag_stack.pop(idx)
                    matched = True
                    break
            if not matched:
                # Closing tag with no opener → strip it
                bad_spans.append((m.start(), m.end()))

    # Any remaining open tags in the stack are unmatched → strip them
    for tag_name, start, end in tag_stack:
        bad_spans.append((start, end))

    if not bad_spans:
        return text

    # Remove bad spans from right to left so positions stay valid
    bad_spans.sort(reverse=True)
    chars = list(text)
    for start, end in bad_spans:
        del chars[start:end]

    return "".join(chars)


def format_currency(amount: float, currency_symbol: str = "$") -> str:
    """
    Format a number as currency.

    Args:
        amount: The amount to format
        currency_symbol: Currency symbol (default: $)

    Returns:
        Formatted currency string (e.g., "$1,234.56")
    """
    if amount is None:
        return f"{currency_symbol}0.00"

    return f"{currency_symbol}{amount:,.2f}"


def format_percentage(value: float, decimal_places: int = 1, is_ratio: bool = False) -> str:
    """
    Format a number as a percentage.

    Args:
        value: The value to format as percentage. Treated as already in
               percentage form (e.g., 15.5 means 15.5%) unless is_ratio=True.
        decimal_places: Number of decimal places to show
        is_ratio: If True, treat value as a 0-1 ratio and multiply by 100

    Returns:
        Formatted percentage string (e.g., "+15.5%" or "-10.3%")
    """
    if value is None:
        return "0%"

    if is_ratio:
        value = value * 100

    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimal_places}f}%"


def format_change(current: float, previous: float) -> tuple[str, str]:
    """
    Format the change between current and previous values.

    Args:
        current: Current value
        previous: Previous value

    Returns:
        Tuple of (percent_change, absolute_change) formatted strings
    """
    if previous == 0 or previous is None:
        if current is None:
            return ("0%", "0")
        return ("N/A", str(current))

    percent_change = ((current - previous) / previous) * 100
    absolute_change = current - previous

    pct_str = f"{percent_change:+.1f}%"
    abs_str = f"{absolute_change:+.0f}"

    return (pct_str, abs_str)


def format_analytics_response(
    metrics: Dict[str, Any], comparison_metrics: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format analytics metrics into a Telegram message using HTML.
    """
    lines = []

    date_range = escape(metrics.get("date_range", "Last 7 Days"))
    lines.append(f"<b>📊 Analytics Report: {date_range}</b>\n")

    revenue = metrics.get("revenue", 0)
    lines.append(f"💰 <b>Revenue:</b> {format_currency(revenue)}")
    if comparison_metrics:
        prev_revenue = comparison_metrics.get("revenue", 0)
        pct_change, abs_change = format_change(revenue, prev_revenue)
        lines.append(f"   └─ {pct_change} ({abs_change})")

    orders = metrics.get("orders", 0)
    lines.append(f"📦 <b>Orders:</b> {orders:,.0f}")
    if comparison_metrics:
        prev_orders = comparison_metrics.get("orders", 0)
        pct_change, abs_change = format_change(orders, prev_orders)
        lines.append(f"   └─ {pct_change} ({abs_change})")

    aov = metrics.get("aov", 0)
    lines.append(f"💵 <b>AOV:</b> {format_currency(aov)}")
    if comparison_metrics:
        prev_aov = comparison_metrics.get("aov", 0)
        pct_change, abs_change = format_change(aov, prev_aov)
        lines.append(f"   └─ {pct_change} ({abs_change})")

    conversion_rate = metrics.get("conversion_rate")
    if conversion_rate is not None:
        lines.append(f"📈 <b>Conversion Rate:</b> {format_percentage(conversion_rate)}")

    visitors = metrics.get("visitors")
    if visitors is not None:
        lines.append(f"👥 <b>Visitors:</b> {visitors:,.0f}")

    lines.append("")
    insight = metrics.get("insight")
    if insight:
        lines.append(f"💡 <b>Insight:</b> {escape(insight)}")

    return "\n".join(lines)


def format_product_performance(
    products: list[Dict[str, Any]], limit: int = 5
) -> str:
    """Format top products into a Telegram HTML message."""
    lines = ["<b>🏆 Top Products</b>\n"]

    for idx, product in enumerate(products[:limit], 1):
        name = escape(product.get("name", "Unknown"))
        revenue = product.get("revenue", 0)
        units = product.get("units_sold", 0)
        lines.append(f"{idx}. <b>{name}</b>")
        lines.append(f"   💰 {format_currency(revenue)} | 📦 {units:,.0f} units")

    if not products:
        lines.append("No product data available.")

    return "\n".join(lines)


def format_error_message(error: Any) -> str:
    """Format an error message in a user-friendly way (HTML)."""
    if isinstance(error, Exception):
        error_msg = escape(str(error))
        error_type = escape(type(error).__name__)
    else:
        error_msg = escape(str(error))
        error_type = "Error"

    return (
        f"❌ <b>{error_type}</b>\n"
        f"Sorry, I encountered an issue: <code>{error_msg}</code>\n\n"
        f"Please try again or contact support if the problem persists."
    )


def format_welcome_message(user_name: str = "there") -> str:
    """Format a welcome message for new users (HTML)."""
    safe_name = escape(user_name)
    return (
        f"👋 Welcome, {safe_name}!\n\n"
        f"I'm your Shopify Analytics Assistant. I can help you:\n"
        f"• 📊 View sales metrics and trends\n"
        f"• 🏆 See your top products\n"
        f"• 📈 Track performance over time\n"
        f"• 💹 Compare periods\n\n"
        f'Try saying "<b>show me sales for last 7 days</b>" or '
        f'"<b>what were yesterday\'s orders</b>" to get started!'
    )


def format_help_message() -> str:
    """Format a help message listing bot capabilities (HTML)."""
    return (
        "<b>🤖 Shopify Analytics Bot - Commands &amp; Capabilities</b>\n\n"
        "<b>Sales Metrics:</b>\n"
        '• "Show me sales for last 7 days"\n'
        '• "What were yesterday\'s orders?"\n'
        '• "Revenue this month"\n\n'
        "<b>Product Performance:</b>\n"
        '• "Top 5 products last month"\n'
        '• "What are my bestsellers?"\n\n'
        "<b>Comparisons:</b>\n"
        '• "Compare this week to last week"\n'
        '• "How much did sales change last month?"\n\n'
        "<b>Supported Time Ranges:</b>\n"
        "today, yesterday, last 7 days, last 30 days, last month, this month\n\n"
        "<i>Use natural language - I'll understand what you need!</i>"
    )
