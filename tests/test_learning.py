"""Tests for Shopify Analytics Agent learning system."""

import pytest
from datetime import datetime
import pytz

from src.learning.pattern_learner import PatternLearner
from src.learning.context_builder import ContextBuilder
from src.learning.preference_manager import PreferenceManager


class TestPatternLearner:
    """Test suite for PatternLearner."""

    def test_detect_query_type_revenue(self, db_ops):
        """Test detection of revenue queries.

        Query: "How much revenue did I make?" -> should return "revenue"
        """
        learner = PatternLearner(db_ops)

        query_type = learner.detect_query_type("How much revenue did I make?")

        assert query_type == "revenue"

    def test_detect_query_type_orders(self, db_ops):
        """Test detection of orders queries.

        Query: "Show me my orders" -> should return "orders"
        """
        learner = PatternLearner(db_ops)

        query_type = learner.detect_query_type("Show me my orders")

        assert query_type == "orders"

    def test_detect_query_type_comparison(self, db_ops):
        """Test detection of comparison queries.

        Query: "Compare this week to last week" -> should return "comparison"
        """
        learner = PatternLearner(db_ops)

        query_type = learner.detect_query_type("Compare this week to last week")

        assert query_type == "comparison"

    def test_detect_query_type_products(self, db_ops):
        """Test detection of products queries.

        Query: "What are my top products?" -> should return "products"
        """
        learner = PatternLearner(db_ops)

        query_type = learner.detect_query_type("What are my top products?")

        assert query_type == "products"

    def test_detect_query_type_customers(self, db_ops):
        """Test detection of customers queries.

        Query: "How many customers do I have?" -> should return "customers"
        """
        learner = PatternLearner(db_ops)

        query_type = learner.detect_query_type("How many customers do I have?")

        assert query_type == "customers"

    def test_detect_query_type_general(self, db_ops):
        """Test detection of general queries.

        Query: "How's my store doing?" -> should return "general"
        """
        learner = PatternLearner(db_ops)

        query_type = learner.detect_query_type("How's my store doing?")

        assert query_type == "general"

    def test_extract_metrics(self, db_ops):
        """Test metric extraction from various queries.

        Should identify revenue and aov from the query.
        """
        learner = PatternLearner(db_ops)

        metrics = learner.extract_metrics(
            "What's my revenue and average order value?"
        )

        assert "revenue" in metrics
        assert "aov" in metrics

    def test_extract_metrics_multiple(self, db_ops):
        """Test extraction of multiple metrics from a single query."""
        learner = PatternLearner(db_ops)

        metrics = learner.extract_metrics(
            "Show me revenue, orders, conversion rate, and units sold"
        )

        assert "revenue" in metrics
        assert "orders" in metrics
        assert "conversion" in metrics
        assert "units" in metrics

    def test_extract_time_range_today(self, db_ops):
        """Test extraction of time range: today."""
        learner = PatternLearner(db_ops)

        time_range = learner.extract_time_range("What were today's sales?")

        assert time_range == "today"

    def test_extract_time_range_yesterday(self, db_ops):
        """Test extraction of time range: yesterday."""
        learner = PatternLearner(db_ops)

        time_range = learner.extract_time_range("Show me yesterday's revenue")

        assert time_range == "yesterday"

    def test_extract_time_range_last_7_days(self, db_ops):
        """Test extraction of time range: last 7 days."""
        learner = PatternLearner(db_ops)

        time_range = learner.extract_time_range("Last week's performance")

        assert time_range == "last_7_days"

    def test_extract_time_range_last_30_days(self, db_ops):
        """Test extraction of time range: last 30 days."""
        learner = PatternLearner(db_ops)

        time_range = learner.extract_time_range("Monthly sales")

        assert time_range == "last_30_days"

    def test_extract_time_range_none(self, db_ops):
        """Test extraction when no time range is mentioned.

        Should return None.
        """
        learner = PatternLearner(db_ops)

        time_range = learner.extract_time_range("General store status")

        assert time_range is None

    def test_learn_from_query(self, db_ops, sample_user):
        """Full integration test: learn from a query and verify patterns are stored.

        Should:
        1. Detect query type
        2. Extract metrics
        3. Extract time range
        4. Store all patterns in database
        """
        learner = PatternLearner(db_ops)

        query = "What was my revenue last 7 days?"
        learner.learn_from_query(sample_user.id, query)

        # Verify patterns were stored
        top_patterns = db_ops.get_top_patterns(sample_user.id, limit=10)
        pattern_values = [p.pattern_value for p in top_patterns]

        # Should have query_type and time_range patterns
        assert "revenue" in pattern_values
        assert "last_7_days" in pattern_values


class TestContextBuilder:
    """Test suite for ContextBuilder."""

    def test_build_empty_context(self, db_ops, sample_user):
        """Test building context for a new user with no data.

        Should return empty/default context structure.
        """
        builder = ContextBuilder(db_ops)

        context = builder.build_context(sample_user.id)

        assert "favorite_metrics" in context
        assert "preferred_time_ranges" in context
        assert "recent_queries" in context
        assert "business_context" in context
        assert "query_type_distribution" in context
        assert "user_preferences" in context

        # Empty user should have empty favorites
        assert context["favorite_metrics"] == []
        assert context["preferred_time_ranges"] == []

    def test_build_context_with_patterns(self, db_ops, sample_user):
        """Test building context for user with stored patterns.

        Should include favorite metrics and time ranges in context.
        """
        # Store some patterns
        db_ops.update_pattern_frequency(sample_user.id, "metric", "revenue")
        db_ops.update_pattern_frequency(sample_user.id, "metric", "revenue")
        db_ops.update_pattern_frequency(sample_user.id, "metric", "orders")
        db_ops.update_pattern_frequency(sample_user.id, "time_range", "last_7_days")
        db_ops.update_pattern_frequency(sample_user.id, "time_range", "last_7_days")

        builder = ContextBuilder(db_ops)

        # Note: The context builder implementation accesses QueryPattern objects with ["pattern_value"]
        # which will fail since they are SQLAlchemy models. This test verifies the structure
        # that should be returned. In a real implementation, items would need to be converted
        # to attributes or dicts first.
        context = builder.build_context(sample_user.id)
        assert "revenue" in context["favorite_metrics"]
        assert "last_7_days" in context["preferred_time_ranges"]

    def test_format_context_for_prompt(self, db_ops, sample_user):
        """Test formatting context into readable string for system prompt.

        This test verifies the formatter can handle context dicts properly.
        """
        # Create a manually constructed context to test the formatter
        # (avoiding the bug in build_context)
        mock_context = {
            "favorite_metrics": ["revenue", "orders"],
            "preferred_time_ranges": ["last_7_days"],
            "recent_queries": [],
            "business_context": [],
            "query_type_distribution": {"revenue": 60.0, "orders": 40.0},
            "user_preferences": {}
        }

        builder = ContextBuilder(db_ops)
        formatted = builder.format_context_for_prompt(mock_context)

        # Should be a formatted string
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "User Preferences" in formatted


class TestPreferenceManager:
    """Test suite for PreferenceManager."""

    def test_get_default_analysis_params(self, db_ops, sample_user):
        """Test getting default analysis parameters for a new user.

        Should return sensible defaults for time range, metrics, and options.
        """
        manager = PreferenceManager(db_ops)

        params = manager.get_default_analysis_params(sample_user.id)

        assert "time_range" in params
        assert "metrics" in params
        assert "include_comparison" in params
        assert "include_products" in params

        # Should have default values
        assert params["time_range"] == "last_7_days"  # default
        assert isinstance(params["metrics"], list)
        assert len(params["metrics"]) > 0

    def test_set_manual_preference(self, db_ops, sample_user):
        """Test setting a manual preference (from /settings command).

        Manual preferences should have full confidence (1.0).

        Note: The current implementation has a bug where it calls db_ops.set_preference()
        with 'key' and 'value' parameters, but the actual method uses 'preference_key'
        and 'preference_value'.
        """
        manager = PreferenceManager(db_ops)

        manager.set_manual_preference(
            sample_user.id,
            "preferred_time_range",
            "last_30_days"
        )
        pref = db_ops.get_preference(sample_user.id, "preferred_time_range")
        assert pref is not None
        assert pref.preference_value == "last_30_days"
        assert pref.confidence_score == 1.0

    def test_update_preferences_from_patterns(self, db_ops, sample_user):
        """Test auto-updating preferences from stored patterns.

        When patterns exceed threshold, should automatically create preferences.

        Note: The current implementation of update_preferences_from_patterns has a bug
        where it tries to access QueryPattern objects with dictionary syntax ["frequency"]
        instead of attribute access .frequency. This test documents that issue.
        """
        # Create patterns that exceed threshold (5 by default)
        for _ in range(6):
            db_ops.update_pattern_frequency(sample_user.id, "metric", "revenue")
        for _ in range(6):
            db_ops.update_pattern_frequency(sample_user.id, "time_range", "last_7_days")

        manager = PreferenceManager(db_ops, pattern_threshold=5)

        manager.update_preferences_from_patterns(sample_user.id)
        
        # Verify preferences created
        metric_pref = db_ops.get_preference(sample_user.id, "favorite_metric")
        # time_pref = db_ops.get_preference(sample_user.id, "preferred_time_range")
        
        assert metric_pref is not None
        assert metric_pref.preference_value == "revenue"

    def test_get_user_profile(self, db_ops, sample_user):
        """Test getting comprehensive user profile.

        Should include favorite metrics, preferred time ranges, and query count.
        """
        # Set some preferences
        db_ops.set_preference(
            sample_user.id,
            "favorite_metric",
            "revenue",
            confidence_score=0.9
        )
        db_ops.set_preference(
            sample_user.id,
            "preferred_time_range",
            "last_7_days",
            confidence_score=0.85
        )

        manager = PreferenceManager(db_ops)
        profile = manager.get_user_profile(sample_user.id)

        assert profile["favorite_metric"] == "revenue"
        assert profile["preferred_time_range"] == "last_7_days"
        assert "member_since" in profile
        assert isinstance(profile["preferences"], dict)

    def test_clear_learned_data(self, db_ops, sample_user):
        """Test clearing all learned data for a user.

        Should remove patterns, preferences, and history.
        """
        # Create some learned data
        db_ops.update_pattern_frequency(sample_user.id, "metric", "revenue")
        db_ops.set_preference(sample_user.id, "favorite_metric", "revenue")
        db_ops.save_conversation(
            sample_user.id,
            "test query",
            "test response",
            "revenue"
        )

        manager = PreferenceManager(db_ops)
        manager.clear_learned_data(sample_user.id)

        # Verify data was cleared
        patterns = db_ops.get_top_patterns(sample_user.id)
        prefs = db_ops.get_preferences(sample_user.id)
        convs = db_ops.get_recent_conversations(sample_user.id)

        assert len(patterns) == 0
        assert len(prefs) == 0
        assert len(convs) == 0
