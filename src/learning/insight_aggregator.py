"""Cross-user intelligence aggregation.

Aggregates patterns from all users to identify best practices,
common mistakes, and tool preferences at scale.
"""

import json
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InsightAggregator:
    """Aggregates cross-user insights for system-wide learning.

    Runs periodic aggregation to:
    - Identify which tools work best for each intent
    - Find common mistakes and their fixes
    - Prune outdated insights
    """

    # Query types to analyze for tool preferences
    QUERY_TYPES = [
        "revenue",
        "orders",
        "products",
        "customers",
        "comparison",
        "general",
    ]

    def __init__(self, db_ops):
        """Initialize InsightAggregator.

        Args:
            db_ops: DatabaseOperations instance
        """
        self.db_ops = db_ops
        logger.info("InsightAggregator initialized")

    def run_aggregation(self) -> None:
        """Run full aggregation across all insight types.

        Runs sub-methods in sequence, logging progress.
        """
        logger.info("Starting insight aggregation")

        try:
            self._aggregate_tool_preferences()
            logger.info("Completed tool preference aggregation")
        except Exception as e:
            logger.error("Error in tool preference aggregation", error=str(e))

        try:
            self._aggregate_common_mistakes()
            logger.info("Completed common mistake aggregation")
        except Exception as e:
            logger.error("Error in common mistake aggregation", error=str(e))

        try:
            self._prune_stale_insights()
            logger.info("Completed stale insight pruning")
        except Exception as e:
            logger.error("Error in stale insight pruning", error=str(e))

        logger.info("Insight aggregation completed")

    def _aggregate_tool_preferences(self) -> None:
        """Aggregate tool usage statistics by intent.

        For each query type, identifies the highest-performing tool
        and stores as a global insight.
        """
        logger.info("Aggregating tool preferences by intent")

        for query_type in self.QUERY_TYPES:
            stats = self.db_ops.get_tool_stats_by_intent(query_type)

            if not stats:
                logger.debug(
                    "No tool stats for query type",
                    query_type=query_type,
                )
                continue

            # Find best performing tool
            best = max(
                stats,
                key=lambda s: s["successes"] / max(s["successes"] + s["failures"], 1),
            )

            success_rate = best["successes"] / max(
                best["successes"] + best["failures"], 1
            )

            logger.debug(
                "Best tool identified for intent",
                query_type=query_type,
                tool=best["tool"],
                success_rate=success_rate,
            )

            # Upsert global insight
            self.db_ops.upsert_global_insight(
                insight_type="tool_preference",
                insight_key=query_type,
                insight_value=json.dumps({
                    "preferred_tool": best["tool"],
                    "success_rate": round(success_rate, 3),
                    "avg_time_ms": best.get("avg_time", 0),
                    "all_tools": stats,
                }),
                sample_size=sum(
                    s["successes"] + s["failures"] for s in stats
                ),
                confidence=success_rate,
            )

    def _aggregate_common_mistakes(self) -> None:
        """Aggregate common error patterns.

        Identifies error types that occur frequently (>= 3 times
        in last 30 days) and stores insights.
        """
        logger.info("Aggregating common mistakes")

        error_groups = self.db_ops.get_error_groups(days=30)

        for group in error_groups:
            # Only store if pattern appears 3+ times
            if group.get("count", 0) < 3:
                continue

            error_type = group.get("error_type", "unknown")
            tool = group.get("tool", "unknown")
            count = group.get("count", 0)

            logger.debug(
                "Common mistake identified",
                error_type=error_type,
                tool=tool,
                count=count,
            )

            # Calculate confidence (capped at 1.0)
            confidence = min(count / 20.0, 1.0)

            self.db_ops.upsert_global_insight(
                insight_type="common_mistake",
                insight_key=f"{error_type}_{tool}",
                insight_value=json.dumps({
                    "description": f"{error_type} with {tool}",
                    "count": count,
                    "lessons": group.get("lessons", []),
                }),
                sample_size=count,
                confidence=confidence,
            )

    def _prune_stale_insights(self, max_age_days: int = 90) -> None:
        """Remove outdated insights older than max_age_days.

        Args:
            max_age_days: Delete insights older than this many days
        """
        logger.info(
            "Pruning stale insights",
            max_age_days=max_age_days,
        )

        deleted = self.db_ops.delete_stale_insights(max_age_days)

        if deleted:
            logger.info(
                "Pruned stale insights",
                count=deleted,
            )
