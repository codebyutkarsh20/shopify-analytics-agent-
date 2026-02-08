"""Context Building Module for Shopify Analytics Agent.

Builds rich context from user history to enhance Claude's understanding
of user preferences and patterns.
"""

import json
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

        # Get recent query errors for learning context
        past_errors = self._build_error_context()

        # Get query templates relevant to user's patterns
        recommended_templates = self._get_relevant_templates(user_id)

        # Get error recovery patterns
        recovery_patterns = self._get_recovery_patterns()

        # Get global insights
        global_insights = self._get_global_insights(user_id)

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
            "past_errors": past_errors,
            "recommended_templates": recommended_templates,
            "recovery_patterns": recovery_patterns,
            "global_insights": global_insights,
        }

        logger.info(
            "Context built successfully",
            user_id=user_id,
            favorite_metrics=favorite_metrics,
            preferred_time_ranges=preferred_time_ranges,
            recent_queries_count=len(recent_queries),
            past_errors_count=len(past_errors),
        )

        return context

    def _get_relevant_templates(self, user_id: int, limit: int = 5) -> list:
        """Get query templates relevant to this user's common intents.

        Args:
            user_id: The user ID
            limit: Maximum number of templates to return

        Returns:
            List of template objects matching user's common intents
        """
        try:
            top_intents = self.db_ops.get_top_patterns(
                user_id, pattern_type="intent", limit=3
            )
            templates = []
            for intent_pattern in top_intents:
                matching = self.db_ops.get_templates_by_intent(
                    intent_pattern.pattern_value, min_confidence=0.7, limit=2
                )
                templates.extend(matching)
            return templates[:limit]
        except Exception as e:
            logger.warning("Failed to get relevant templates", error=str(e))
            return []

    def _get_recovery_patterns(self, limit: int = 5) -> list:
        """Get most useful error recovery patterns.

        Args:
            limit: Maximum number of patterns to return

        Returns:
            List of recovery pattern objects
        """
        try:
            return self.db_ops.get_top_recovery_patterns(
                min_confidence=0.7, min_times_applied=1, limit=limit
            )
        except Exception as e:
            logger.warning("Failed to get recovery patterns", error=str(e))
            return []

    def _get_global_insights(self, user_id: int) -> list:
        """Get global insights relevant to this user.

        Args:
            user_id: The user ID

        Returns:
            List of insight objects relevant to the user
        """
        try:
            insights = []
            top_types = self.db_ops.get_top_patterns(
                user_id, pattern_type="query_type", limit=3
            )
            for tp in top_types:
                insight = self.db_ops.get_global_insight("tool_preference", tp.pattern_value)
                if insight:
                    insights.append(insight)

            mistakes = self.db_ops.get_global_insights_by_type("common_mistake", limit=3)
            insights.extend(mistakes)
            return insights
        except Exception as e:
            logger.warning("Failed to get global insights", error=str(e))
            return []

    def _build_error_context(self) -> list:
        """Build error learning context from past failed queries.

        Retrieves recent unresolved query errors and formats them for
        inclusion in Claude's system prompt so it can learn from past
        mistakes and avoid repeating the same failing patterns.

        Returns:
            List of error dictionaries with tool_name, query_text,
            error_message, error_type, and lesson fields.
        """
        try:
            recent_errors = self.db_ops.get_recent_query_errors(
                user_id=0,  # system-level — get all errors
                limit=15,
                days=30,
                include_resolved=True,  # include resolved ones too — they have lessons
            )

            errors = []
            for err in recent_errors:
                error_entry = {
                    "tool_name": err.tool_name,
                    "query_text": err.query_text[:500],  # Truncate to avoid huge prompts
                    "error_message": err.error_message[:300],
                    "error_type": err.error_type,
                }
                if err.lesson:
                    error_entry["lesson"] = err.lesson
                if err.resolved:
                    error_entry["resolved"] = True
                errors.append(error_entry)

            return errors

        except Exception as e:
            logger.warning(
                "Failed to build error context",
                error=str(e),
            )
            return []

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

        # Past Errors section (learning from mistakes)
        if context.get("past_errors"):
            lines.append("\nPast Query Errors (AVOID repeating these mistakes):")
            for i, err in enumerate(context["past_errors"], 1):
                tool = err.get("tool_name", "unknown")
                error_msg = err.get("error_message", "")
                error_type = err.get("error_type", "unknown")
                lesson = err.get("lesson", "")
                query_snippet = err.get("query_text", "")[:200]

                lines.append(f"  Error {i} [{tool}] ({error_type}):")
                lines.append(f"    Query: {query_snippet}")
                lines.append(f"    Error: {error_msg}")
                if lesson:
                    lines.append(f"    Lesson: {lesson}")
                if err.get("resolved"):
                    lines.append(f"    Status: RESOLVED")

        # Recommended Templates section
        if context.get("recommended_templates"):
            lines.append("\nPROVEN QUERY TEMPLATES (use these when the intent matches):")
            for t in context["recommended_templates"][:3]:
                lines.append(f"  Intent: {t.intent_description[:100]}")
                lines.append(f"  Tool: {t.tool_name}, Confidence: {t.confidence:.0%}")
                params_str = t.tool_parameters[:200] if t.tool_parameters else ""
                lines.append(f"  Parameters: {params_str}")

        # Recovery Patterns section
        if context.get("recovery_patterns"):
            lines.append("\nERROR RECOVERY PATTERNS (apply if these errors occur):")
            for rp in context["recovery_patterns"][:3]:
                lines.append(f"  If {rp.error_type} with {rp.failed_tool_name}:")
                lines.append(f"    → {rp.recovery_description}")
                lines.append(f"    Success rate: {rp.confidence:.0%}")

        # Global Insights section
        if context.get("global_insights"):
            lines.append("\nGLOBAL INSIGHTS:")
            for gi in context["global_insights"][:3]:
                try:
                    val = json.loads(gi.insight_value)
                    if gi.insight_type == "tool_preference":
                        lines.append(
                            f"  For {gi.insight_key} queries: prefer {val.get('preferred_tool')} ({val.get('success_rate', 0):.0%} success)"
                        )
                    elif gi.insight_type == "common_mistake":
                        lines.append(
                            f"  Common mistake: {val.get('description', gi.insight_key)}"
                        )
                except (json.JSONDecodeError, AttributeError):
                    pass

        formatted = "\n".join(lines)
        logger.debug("Context formatted for prompt", length=len(formatted))

        # Token budget management: ~4 chars per token, max 2000 tokens
        max_chars = 8000
        if len(formatted) > max_chars:
            formatted = (
                formatted[:max_chars]
                + "\n  [Context truncated to fit token budget]"
            )

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
