"""
Message formatting utilities for Telegram output.

This module provides functions to format analytics data and messages
for Telegram bot responses using HTML formatting (much more robust
than MarkdownV2 for dynamic content).
"""

import html
import re
from typing import Any, Dict, List, Optional


def escape(text: str) -> str:
    """Escape text for Telegram HTML parse mode."""
    return html.escape(str(text))


def _convert_markdown_tables(text: str) -> str:
    """Convert markdown tables to clean Telegram-friendly bulleted lists.

    Markdown tables render as garbage in Telegram, so we convert them
    to a structured list format:
        | Name | Revenue | Units |     →   • **Name:** Revenue (Units)
        |------|---------|-------|
        | Foo  | $100    | 5     |     →   • **Foo** — $100 | 5
    """
    lines = text.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Detect a table row: starts and ends with |
        if line.startswith("|") and line.endswith("|"):
            # Collect all consecutive table rows
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                row = lines[i].strip()
                # Skip separator rows like |---|---|---|
                if not re.match(r"^\|[\s\-:]+\|$", row.replace("|", "|")):
                    cells = [c.strip() for c in row.strip("|").split("|")]
                    if any(c for c in cells):  # Skip empty rows
                        table_rows.append(cells)
                i += 1

            if len(table_rows) >= 2:
                headers = table_rows[0]
                data_rows = table_rows[1:]

                for row in data_rows:
                    parts = []
                    for col_idx, cell in enumerate(row):
                        if col_idx == 0:
                            parts.append(f"**{cell}**")
                        elif cell:
                            parts.append(cell)
                    result.append("• " + " — ".join(parts))
                result.append("")  # blank line after table
            elif table_rows:
                # Single row table, just list the cells
                for row in table_rows:
                    result.append("• " + " | ".join(c for c in row if c))
                result.append("")
        else:
            result.append(lines[i])
            i += 1

    return "\n".join(result)


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

    # Markdown tables → clean bulleted lines (before other conversions)
    text = _convert_markdown_tables(text)

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
        value: The value to format as percentage. Values below 2 are
               auto-detected as ratios and multiplied by 100 (e.g., 0.5 -> 50%).
               Values >= 2 are treated as already in percentage form.
        decimal_places: Number of decimal places to show
        is_ratio: If True, force treating value as a 0-1 ratio and multiply by 100

    Returns:
        Formatted percentage string (e.g., "15.5%" or "-10.3%")
    """
    if value is None:
        return "0%"

    if is_ratio or value < 2:
        value = value * 100

    return f"{value:.{decimal_places}f}%"


def format_change(current: float, previous: float) -> tuple[str, str]:
    """
    Format the change between current and previous values.

    Args:
        current: Current value
        previous: Previous value

    Returns:
        Tuple of (percent_change, direction_indicator) formatted strings
    """
    if previous == 0 or previous is None:
        if current is None:
            return ("0%", "0")
        return ("N/A", str(current))

    percent_change = ((current - previous) / previous) * 100
    absolute_change = current - previous

    # Use arrow indicators for visual clarity
    if percent_change > 0:
        arrow = "▲"
    elif percent_change < 0:
        arrow = "▼"
    else:
        arrow = "─"

    pct_str = f"{arrow} {abs(percent_change):.1f}%"
    abs_str = f"{absolute_change:+,.0f}"

    return (pct_str, abs_str)


def format_analytics_response(
    metrics: Dict[str, Any], comparison_metrics: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format analytics metrics into a Telegram message using HTML.
    """
    lines = []

    date_range = escape(metrics.get("date_range", "Last 7 Days"))
    lines.append(f"<b>📊 Analytics Report — {date_range}</b>")
    lines.append("")

    revenue = metrics.get("revenue", 0)
    rev_line = f"💰 <b>Revenue:</b> <code>{format_currency(revenue)}</code>"
    if comparison_metrics:
        prev_revenue = comparison_metrics.get("revenue", 0)
        pct_change, abs_change = format_change(revenue, prev_revenue)
        rev_line += f"  ({pct_change})"
    lines.append(rev_line)

    orders = metrics.get("orders", 0)
    ord_line = f"📦 <b>Orders:</b> <code>{orders:,.0f}</code>"
    if comparison_metrics:
        prev_orders = comparison_metrics.get("orders", 0)
        pct_change, abs_change = format_change(orders, prev_orders)
        ord_line += f"  ({pct_change})"
    lines.append(ord_line)

    aov = metrics.get("aov", 0)
    aov_line = f"💵 <b>AOV:</b> <code>{format_currency(aov)}</code>"
    if comparison_metrics:
        prev_aov = comparison_metrics.get("aov", 0)
        pct_change, abs_change = format_change(aov, prev_aov)
        aov_line += f"  ({pct_change})"
    lines.append(aov_line)

    conversion_rate = metrics.get("conversion_rate")
    if conversion_rate is not None:
        lines.append(f"📈 <b>Conversion:</b> <code>{format_percentage(conversion_rate)}</code>")

    visitors = metrics.get("visitors")
    if visitors is not None:
        lines.append(f"👥 <b>Visitors:</b> <code>{visitors:,.0f}</code>")

    insight = metrics.get("insight")
    if insight:
        lines.append("")
        lines.append(f"💡 <b>Insight:</b> {escape(insight)}")

    return "\n".join(lines)


def format_product_performance(
    products: list[Dict[str, Any]], limit: int = 5
) -> str:
    """Format top products into a Telegram HTML message."""
    lines = ["<b>🏆 Top Products</b>", ""]

    for idx, product in enumerate(products[:limit], 1):
        name = escape(product.get("name", "Unknown"))
        revenue = product.get("revenue", 0)
        units = product.get("units_sold", 0)
        lines.append(
            f"{idx}. <b>{name}</b>\n"
            f"    💰 <code>{format_currency(revenue)}</code>  •  📦 {units:,.0f} units"
        )

    if not products:
        lines.append("<i>No product data available.</i>")

    return "\n".join(lines)


def format_error_message(error: Any) -> str:
    """Format an error message in a user-friendly way (HTML).

    Hides internal details (GraphQL errors, stack traces, API schema)
    and shows a generic message. Full error details stay in server logs only.
    """
    # Generate a short error ID for log correlation
    import hashlib
    import time
    error_id = hashlib.md5(f"{time.time()}{str(error)}".encode()).hexdigest()[:8]

    return (
        "⚠️ <b>Something went wrong</b>\n\n"
        f"Your request couldn't be completed. Please try rephrasing\n"
        f"your question or use /help to see examples.\n\n"
        f"<i>Error ref: {error_id}</i>"
    )


def format_welcome_message(user_name: str = "there") -> str:
    """Format a welcome message for new users (HTML)."""
    safe_name = escape(user_name)
    return (
        f"👋 <b>Welcome, {safe_name}!</b>\n\n"
        "I'm your <b>Shopify Analytics Assistant</b>. "
        "Ask me anything about your store — sales, products, customers, and more.\n\n"
        "<b>Try asking:</b>\n"
        '• "Show me sales for last 7 days"\n'
        '• "What are my top 5 products?"\n'
        '• "Compare this week to last week"\n\n'
        "Type /help for all commands and examples."
    )


def format_help_message() -> str:
    """Format a help message listing bot capabilities (HTML)."""
    return (
        "<b>📖 Commands &amp; Examples</b>\n\n"

        "💰 <b>Sales &amp; Revenue</b>\n"
        '• "Revenue last 7 days"\n'
        '• "How many orders yesterday?"\n'
        '• "Average order value this month"\n\n'

        "🏆 <b>Products</b>\n"
        '• "Top 5 products by revenue"\n'
        '• "What are my bestsellers?"\n'
        '• "Show products low on inventory"\n\n'

        "👥 <b>Customers</b>\n"
        '• "Who are my top spending customers?"\n'
        '• "New customers this week"\n\n'

        "📈 <b>Comparisons</b>\n"
        '• "Compare this week to last week"\n'
        '• "How did sales change last month?"\n\n'

        "⚙️ <b>Bot Commands</b>\n"
        "• /start — Initialize the bot\n"
        "• /connect — Connect your Shopify store\n"
        "• /status — Check connection status\n"
        "• /settings — View preferences\n"
        "• /forget — Clear learned data\n"
        "• /help — Show this message\n\n"

        "<i>Just type in natural language — I'll figure out the rest.</i>"
    )
