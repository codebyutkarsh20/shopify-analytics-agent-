"""Preference Management Module for Shopify Analytics Agent.

Manages user preferences, learns from patterns, and provides default parameters
for analysis.
"""

from datetime import datetime
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreferenceManager:
    """Manages user preferences and learning.

    Automatically learns user preferences from query patterns and provides
    default parameters for analysis. Manages both automatic learning and
    manual preference settings.
    """

    def __init__(self, db_ops, pattern_threshold: int = 5):
        """Initialize PreferenceManager.

        Args:
            db_ops: DatabaseOperations instance for storing/retrieving data.
            pattern_threshold: Minimum frequency before auto-setting a preference
                             (default: 5).
        """
        self.db_ops = db_ops
        self.pattern_threshold = pattern_threshold
        logger.info("PreferenceManager initialized", threshold=pattern_threshold)

    def update_preferences_from_patterns(self, user_id: int) -> None:
        """Analyze patterns and automatically update preferences.

        Updates preferences when patterns are strong enough (frequency > threshold):
        - Top metrics become "favorite_metric"
        - Top time ranges become "preferred_time_range"
        - Most common query type becomes "primary_query_type"

        Args:
            user_id: The user ID
        """
        logger.info("Updating preferences from patterns", user_id=user_id)

        # 1. Update favorite metrics if any exceed threshold
        metric_patterns = self.db_ops.get_top_patterns(
            user_id, pattern_type="metric", limit=5
        )

        if metric_patterns:
            # Calculate total frequency for confidence scoring
            total_freq = sum(item.frequency for item in metric_patterns)

            for item in metric_patterns:
                if item.frequency > self.pattern_threshold:
                    metric_name = item.pattern_value
                    confidence = item.frequency / total_freq if total_freq > 0 else 0

                    self.db_ops.set_preference(
                        user_id=user_id,
                        preference_key="favorite_metric",
                        preference_value=metric_name,
                        confidence_score=confidence,
                    )
                    logger.debug(
                        "Favorite metric updated",
                        user_id=user_id,
                        metric=metric_name,
                        confidence=confidence,
                    )
                    break  # Only set the top metric as favorite

        # 2. Update preferred time ranges if any exceed threshold
        time_range_patterns = self.db_ops.get_top_patterns(
            user_id, pattern_type="time_range", limit=5
        )

        if time_range_patterns:
            total_freq = sum(item.frequency for item in time_range_patterns)

            for item in time_range_patterns:
                if item.frequency > self.pattern_threshold:
                    time_range = item.pattern_value
                    confidence = item.frequency / total_freq if total_freq > 0 else 0

                    self.db_ops.set_preference(
                        user_id=user_id,
                        preference_key="preferred_time_range",
                        preference_value=time_range,
                        confidence_score=confidence,
                    )
                    logger.debug(
                        "Preferred time range updated",
                        user_id=user_id,
                        time_range=time_range,
                        confidence=confidence,
                    )
                    break  # Only set the top time range as preferred

        # 3. Update primary query type
        query_type_patterns = self.db_ops.get_top_patterns(
            user_id, pattern_type="query_type", limit=5
        )

        if query_type_patterns:
            primary_type = query_type_patterns[0].pattern_value
            total_freq = sum(item.frequency for item in query_type_patterns)
            confidence = query_type_patterns[0].frequency / total_freq if total_freq > 0 else 0

            self.db_ops.set_preference(
                user_id=user_id,
                preference_key="primary_query_type",
                preference_value=primary_type,
                confidence_score=confidence,
            )
            logger.debug(
                "Primary query type updated",
                user_id=user_id,
                query_type=primary_type,
                confidence=confidence,
            )

        logger.info("Preference update completed", user_id=user_id)

    def get_user_profile(self, user_id: int) -> dict:
        """Get a comprehensive user profile.

        Args:
            user_id: The user ID

        Returns:
            Dictionary containing:
                - favorite_metric: User's most common metric
                - preferred_time_range: User's preferred time period
                - primary_query_type: User's most common query type
                - query_count: Total queries made by user
                - member_since: When user was first seen
                - preferences: All stored preferences dict
        """
        logger.info("Getting user profile", user_id=user_id)

        # Get all preferences
        preferences = self.db_ops.get_preferences_dict(user_id)

        # Get user's first query timestamp (member_since)
        # Note: This assumes a method exists to get first query time
        # If not, we'd need to add it to db_ops
        try:
            member_info = self.db_ops.get_user_first_query_time(user_id)
            member_since = (
                member_info.get("first_query_time", datetime.utcnow().isoformat())
                if member_info
                else datetime.utcnow().isoformat()
            )
        except (AttributeError, TypeError):
            logger.debug("Member since not available, using current time")
            member_since = datetime.utcnow().isoformat()

        # Get total query count
        try:
            query_count = self.db_ops.get_user_query_count(user_id)
        except (AttributeError, TypeError):
            logger.debug("Query count not available")
            query_count = 0

        profile = {
            "favorite_metric": preferences.get("favorite_metric", "revenue"),
            "preferred_time_range": preferences.get("preferred_time_range", "last_7_days"),
            "primary_query_type": preferences.get("primary_query_type", "general"),
            "query_count": query_count,
            "member_since": member_since,
            "preferences": preferences,
        }

        logger.info(
            "User profile retrieved",
            user_id=user_id,
            favorite_metric=profile["favorite_metric"],
            preferred_time_range=profile["preferred_time_range"],
        )

        return profile

    def clear_learned_data(self, user_id: int) -> None:
        """Clear all learned data for privacy (/forget command).

        Deletes all patterns, preferences, and history for the user.

        Args:
            user_id: The user ID
        """
        logger.info("Clearing learned data", user_id=user_id)

        try:
            self.db_ops.clear_user_data(user_id)
            logger.info("Learned data cleared successfully", user_id=user_id)
        except Exception as e:
            logger.error(
                "Failed to clear learned data",
                user_id=user_id,
                error=str(e),
            )
            raise

    def set_manual_preference(self, user_id: int, key: str, value: str) -> None:
        """Set a preference manually (from /settings command).

        Manual preferences have full confidence (1.0) and override learned
        preferences.

        Args:
            user_id: The user ID
            key: Preference key
            value: Preference value
        """
        logger.info(
            "Setting manual preference",
            user_id=user_id,
            key=key,
            value=value,
        )

        try:
            self.db_ops.set_preference(
                user_id=user_id,
                preference_key=key,
                preference_value=value,
                confidence_score=1.0,  # Manual preferences have full confidence
            )
            logger.info(
                "Manual preference set",
                user_id=user_id,
                key=key,
                value=value,
            )
        except Exception as e:
            logger.error(
                "Failed to set manual preference",
                user_id=user_id,
                key=key,
                error=str(e),
            )
            raise

    def get_default_analysis_params(self, user_id: int) -> dict:
        """Get default parameters for vague analysis requests.

        Based on user preferences, returns sensible defaults for:
        - Time range
        - Metrics to include
        - Whether to include comparisons
        - Whether to include product analysis

        Args:
            user_id: The user ID

        Returns:
            Dictionary with default analysis parameters:
                - time_range: Preferred or default time range
                - metrics: List of favorite or default metrics
                - include_comparison: Whether to include comparison data
                - include_products: Whether to include product analysis
        """
        logger.info("Getting default analysis parameters", user_id=user_id)

        profile = self.get_user_profile(user_id)
        preferences = profile["preferences"]

        # Get time range - use preferred or default to last 7 days
        time_range = profile["preferred_time_range"]

        # Get metrics - use favorites or default set
        metrics = []
        if "favorite_metric" in preferences:
            metrics.append(preferences["favorite_metric"])

        # Add common metrics if not already included
        default_metrics = ["revenue", "orders", "aov"]
        for metric in default_metrics:
            if metric not in metrics:
                metrics.append(metric)

        # Decide whether to include comparisons (if comparison queries are common)
        include_comparison = True  # Default to true for most users

        # Decide whether to include product analysis
        # Check if "products" is in favorite metrics or common patterns
        include_products = "products" in metrics or any(
            pref.startswith("favorite_metric_products")
            for pref in preferences.keys()
        )

        params = {
            "time_range": time_range,
            "metrics": metrics[:3],  # Limit to top 3 metrics
            "include_comparison": include_comparison,
            "include_products": include_products,
        }

        logger.info(
            "Default analysis parameters generated",
            user_id=user_id,
            time_range=time_range,
            metrics=params["metrics"],
        )

        return params
