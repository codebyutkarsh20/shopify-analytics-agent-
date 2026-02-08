"""Chart generation service for Shopify Analytics Telegram Bot.

Generates professional chart images (bar, line, pie, horizontal bar)
using matplotlib. Charts are saved as temporary PNG files and sent
as images via Telegram's send_photo() API.
"""

import os
import tempfile
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Shopify-themed color palette ──────────────────────────────────────
COLORS = [
    "#5C6AC4",  # Shopify indigo (primary)
    "#006FBB",  # Blue
    "#47C1BF",  # Teal
    "#50B83C",  # Green
    "#F49342",  # Orange
    "#DE3618",  # Red
    "#9C6ADE",  # Purple
    "#EEC200",  # Yellow
    "#00848E",  # Dark teal
    "#BF0711",  # Dark red
    "#4E8098",  # Slate blue
    "#8DB38B",  # Sage green
]

BACKGROUND_COLOR = "#FAFBFC"
GRID_COLOR = "#DFE3E8"
TEXT_COLOR = "#212B36"
SUBTITLE_COLOR = "#637381"


class ChartGenerator:
    """Generates professional chart images for Telegram delivery."""

    def __init__(self, dpi: int = 100, figsize: tuple = (10, 6)):
        self.dpi = dpi
        self.figsize = figsize
        logger.info("ChartGenerator initialized", dpi=dpi, figsize=figsize)

    def _setup_style(self, fig, ax):
        """Apply consistent Shopify-themed styling to a chart."""
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID_COLOR)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=10)
        ax.title.set_color(TEXT_COLOR)
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.7)

    def _save_chart(self, fig) -> str:
        """Save chart to a secure temporary PNG file and return the file path.

        Uses tempfile.NamedTemporaryFile for secure file creation with
        restricted permissions (owner-only read/write).
        """
        tmp = tempfile.NamedTemporaryFile(
            prefix="shopify_chart_",
            suffix=".png",
            delete=False,
        )
        filepath = tmp.name
        tmp.close()

        # Set restrictive permissions (owner read/write only)
        os.chmod(filepath, 0o600)

        fig.savefig(
            filepath,
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            pad_inches=0.3,
        )
        plt.close(fig)

        logger.debug("Chart saved", filepath=filepath)
        return filepath

    def generate_bar_chart(
        self,
        labels: List[str],
        values: List[float],
        title: str,
        y_axis_label: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a vertical bar chart.

        Args:
            labels: Category names (x-axis)
            values: Numeric values (y-axis)
            title: Chart title
            y_axis_label: Optional y-axis label

        Returns:
            File path to the generated PNG, or None on failure
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            self._setup_style(fig, ax)

            # Truncate long labels
            display_labels = [
                (l[:20] + "...") if len(l) > 23 else l for l in labels
            ]

            colors = [COLORS[i % len(COLORS)] for i in range(len(values))]
            bars = ax.bar(display_labels, values, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

            # Add value labels on top of bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    self._format_number(val),
                    ha="center", va="bottom",
                    fontsize=9, color=SUBTITLE_COLOR, fontweight="bold",
                )

            ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
            if y_axis_label:
                ax.set_ylabel(y_axis_label, fontsize=11, color=SUBTITLE_COLOR)

            # Rotate labels if many items
            if len(labels) > 5:
                plt.xticks(rotation=45, ha="right")

            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: self._format_number(x)))
            fig.tight_layout()

            return self._save_chart(fig)

        except Exception as e:
            logger.error("Failed to generate bar chart", error=str(e), exc_info=True)
            return None

    def generate_line_chart(
        self,
        labels: List[str],
        values: List[float],
        title: str,
        y_axis_label: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a line chart (ideal for trends over time).

        Args:
            labels: Time points or categories (x-axis)
            values: Numeric values (y-axis)
            title: Chart title
            y_axis_label: Optional y-axis label

        Returns:
            File path to the generated PNG, or None on failure
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            self._setup_style(fig, ax)

            ax.plot(
                labels, values,
                color=COLORS[0],
                linewidth=2.5,
                marker="o",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor=COLORS[0],
                markeredgewidth=2,
            )

            # Shade area under line
            ax.fill_between(labels, values, alpha=0.1, color=COLORS[0])

            ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
            if y_axis_label:
                ax.set_ylabel(y_axis_label, fontsize=11, color=SUBTITLE_COLOR)

            # Rotate labels if many
            if len(labels) > 6:
                plt.xticks(rotation=45, ha="right")

            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: self._format_number(x)))
            fig.tight_layout()

            return self._save_chart(fig)

        except Exception as e:
            logger.error("Failed to generate line chart", error=str(e), exc_info=True)
            return None

    def generate_pie_chart(
        self,
        labels: List[str],
        values: List[float],
        title: str,
    ) -> Optional[str]:
        """Generate a pie chart (ideal for distributions/shares).

        Args:
            labels: Category names
            values: Numeric values (proportions)
            title: Chart title

        Returns:
            File path to the generated PNG, or None on failure
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor(BACKGROUND_COLOR)

            colors = [COLORS[i % len(COLORS)] for i in range(len(values))]

            # Truncate labels
            display_labels = [
                (l[:18] + "...") if len(l) > 21 else l for l in labels
            ]

            wedges, texts, autotexts = ax.pie(
                values,
                labels=display_labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=140,
                pctdistance=0.75,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )

            # Style percentage text
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight("bold")
                autotext.set_color("white")

            for text in texts:
                text.set_fontsize(10)
                text.set_color(TEXT_COLOR)

            ax.set_title(title, fontsize=14, fontweight="bold", pad=20, color=TEXT_COLOR)
            fig.tight_layout()

            return self._save_chart(fig)

        except Exception as e:
            logger.error("Failed to generate pie chart", error=str(e), exc_info=True)
            return None

    def generate_horizontal_bar_chart(
        self,
        labels: List[str],
        values: List[float],
        title: str,
        x_axis_label: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a horizontal bar chart (ideal for ranked comparisons).

        Args:
            labels: Category names (y-axis)
            values: Numeric values (x-axis)
            title: Chart title
            x_axis_label: Optional x-axis label

        Returns:
            File path to the generated PNG, or None on failure
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            self._setup_style(fig, ax)
            ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.7)
            ax.grid(False, axis="y")

            # Truncate labels
            display_labels = [
                (l[:25] + "...") if len(l) > 28 else l for l in labels
            ]

            colors = [COLORS[i % len(COLORS)] for i in range(len(values))]

            # Reverse so highest is at top
            bars = ax.barh(
                display_labels[::-1], values[::-1],
                color=colors[::-1], height=0.6,
                edgecolor="white", linewidth=0.5,
            )

            # Add value labels at end of bars
            for bar, val in zip(bars, values[::-1]):
                ax.text(
                    bar.get_width() + max(values) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    self._format_number(val),
                    ha="left", va="center",
                    fontsize=9, color=SUBTITLE_COLOR, fontweight="bold",
                )

            ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
            if x_axis_label:
                ax.set_xlabel(x_axis_label, fontsize=11, color=SUBTITLE_COLOR)

            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: self._format_number(x)))
            fig.tight_layout()

            return self._save_chart(fig)

        except Exception as e:
            logger.error("Failed to generate horizontal bar chart", error=str(e), exc_info=True)
            return None

    @staticmethod
    def _format_number(val: float) -> str:
        """Format a number for display on charts (e.g., 1500 → 1.5K)."""
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:.1f}M"
        elif abs(val) >= 1_000:
            return f"{val / 1_000:.1f}K"
        elif val == int(val):
            return str(int(val))
        else:
            return f"{val:.1f}"

    @staticmethod
    def cleanup_file(filepath: str) -> bool:
        """Safely delete a temporary chart file."""
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                logger.debug("Chart file cleaned up", filepath=filepath)
                return True
        except Exception as e:
            logger.warning("Failed to clean up chart file", filepath=filepath, error=str(e))
        return False
