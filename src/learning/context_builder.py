"""Context Building Module for Shopify Analytics Agent.

Builds rich context from user history to enhance Claude's understanding
of user preferences and patterns.
"""

from datetime import datetime
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContextBuilder:
    """Builds context from user history and patterns.

    Compiles user preferences, query patterns, business context, and recent
    activity into a comprehensive context dictionary that can be included
    in Claude's system prompt for personalized responses.
    """

    def __init__(self, db_ops):
        """Initialize ContextBuilder.

        Args:
            db_ops: DatabaseOperations instance for retrieving user data.
        """
        self.db_ops = db_ops
        logger.info("ContextBuilder initialized")

    def build_context(self, user_id: int) -> dict:
        """Build a complete context dictionary for Claude's system prompt.

        Args:
            user_id: The user ID

        Returns:
            Dictionary containing:
                - favorite_metrics: Top 3 most frequently mentioned metrics
                - preferred_time_ranges: Top 2 time range preferences
                - recent_queries: Last 5 conversation summaries
                - business_context: Any stored business context items
                - query_type_distribution: Distribution of query types
                - user_preferences: All stored user preferences
        """
        logger.info("Building context", user_id=user_id)

        # Get favorite metrics (top 3)
        metric_patterns = self.db_ops.get_top_patterns(
            user_id, pattern_type="metric", limit=3
        )
        favorite_metrics = [item.pattern_value for item in metric_patterns]

        # Get preferred time ranges (top 2)
        time_range_patterns = self.db_ops.get_top_patterns(
            user_id, pattern_type="time_range", limit=2
        )
        preferred_time_ranges = [item.pattern_value for item in time_range_patterns]

        # Get recent queries/conversations (last 5)
        recent_conversations = self.db_ops.get_recent_conversations(user_id, limit=5)
        recent_queries = [
            {
                "summary": conv.message[:100] if conv.message else "",
                "timestamp": conv.created_at.isoformat() if conv.created_at else "",
                "query_type": conv.query_type or "",
            }
            for conv in recent_conversations
        ]

        # Get business context items
        all_preferences = self.db_ops.get_preferences_dict(user_id)
        business_context = [
            {
                "key": key,
                "value": value,
            }
            for key, value in all_preferences.items()
            if key.startswith("business_context_")
        ]

        # Get query type distribution
        query_type_patterns = self.db_ops.get_top_patterns(
            user_id, pattern_type="query_type", limit=10
        )
        total_query_patterns = sum(
            item.frequency for item in query_type_patterns
        )
        query_type_distribution = {
            item.pattern_value: (
                item.frequency / total_query_patterns * 100
                if total_query_patterns > 0
                else 0
            )
            for item in query_type_patterns
        }

        # Build recent patterns list for claude_service compatibility
        recent_patterns = []
        all_patterns = self.db_ops.get_top_patterns(user_id, limit=5)
        for p in all_patterns:
            recent_patterns.append({
                "type": p.pattern_type,
                "value": p.pattern_value,
                "frequency": p.frequency,
            })

        context = {
            "favorite_metrics": favorite_metrics,
            "preferred_time_ranges": preferred_time_ranges,
            "recent_queries": recent_queries,
            "business_context": business_context,
            "query_type_distribution": query_type_distribution,
            "user_preferences": all_preferences,
            # Aliases used by claude_service._build_system_prompt
            "preferences": all_preferences,
            "recent_patterns": recent_patterns,
        }

        logger.info(
            "Context built successfully",
            user_id=user_id,
            favorite_metrics=favorite_metrics,
            preferred_time_ranges=preferred_time_ranges,
            recent_queries_count=len(recent_queries),
        )

        return context

    def format_context_for_prompt(self, context: dict) -> str:
        """Format the context dictionary into a readable string for system prompt.

        Args:
            context: Context dictionary from build_context()

        Returns:
            Formatted string suitable for inclusion in Claude's system prompt
        """
        lines = []

        # User Preferences section
        lines.append("User Preferences:")
        if context.get("favorite_metrics"):
            metrics_str = ", ".join(context["favorite_metrics"])
            lines.append(f"  - Favorite metrics: {metrics_str}")
        else:
            lines.append("  - Favorite metrics: Not yet determined")

        if context.get("preferred_time_ranges"):
            ranges_str = ", ".join(context["preferred_time_ranges"])
            lines.append(f"  - Preferred time ranges: {ranges_str}")
        else:
            lines.append("  - Preferred time ranges: Not yet determined")

        # Query type distribution
        if context.get("query_type_distribution"):
            distribution = context["query_type_distribution"]
            sorted_types = sorted(
                distribution.items(), key=lambda x: x[1], reverse=True
            )
            top_types = sorted_types[:3]
            if top_types:
                types_str = ", ".join(
                    f"{qtype} ({freq:.0f}%)" for qtype, freq in top_types
                )
                lines.append(f"  - Most common query types: {types_str}")

        # Recent Activity section
        lines.append("\nRecent Activity:")
        if context.get("recent_queries"):
            for i, query in enumerate(context["recent_queries"][:3], 1):
                if query["summary"]:
                    lines.append(f"  - {query['summary']}")
        else:
            lines.append("  - No recent activity")

        # Business Context section
        if context.get("business_context"):
            lines.append("\nBusiness Context:")
            for item in context["business_context"]:
                lines.append(f"  - {item['value']}")
        else:
            lines.append("\nBusiness Context:")
            lines.append("  - No business context recorded")

        formatted = "\n".join(lines)
        logger.debug("Context formatted for prompt", length=len(formatted))

        return formatted

    def get_quick_context(self, user_id: int) -> str:
        """Build and format context in one call.

        Convenience method that combines build_context() and
        format_context_for_prompt().

        Args:
            user_id: The user ID

        Returns:
            Formatted context string ready for system prompt
        """
        logger.info("Getting quick context", user_id=user_id)
        context = self.build_context(user_id)
        formatted = self.format_context_for_prompt(context)
        return formatted
