"""Pattern Learning Module for Shopify Analytics Agent.

Learns from user queries to understand their analysis patterns and preferences.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Intent:
    """Two-level intent classification."""
    coarse: str   # revenue, orders, products, customers, comparison, general
    fine: str     # products_ranking_revenue_last_30_days, etc.


class PatternLearner:
    """Learns query patterns from user interactions.

    Analyzes user queries to extract patterns including query types, metrics,
    time ranges, and business context. Stores these patterns in the database
    to build a user profile over time.
    """

    # Query type detection keywords
    QUERY_TYPE_KEYWORDS = {
        "revenue": ["revenue", "sales", "money", "income", "earned", "gross"],
        "orders": ["orders", "order count", "order volume", "order number"],
        "products": ["products", "top products", "best selling", "popular items", "product performance"],
        "customers": ["customers", "customer", "users", "user base", "client"],
        "comparison": ["compare", "vs", "versus", "compared to", "difference", "against"],
    }

    # Metric extraction keywords
    METRIC_KEYWORDS = {
        "revenue": ["revenue", "sales", "income", "earnings", "gross"],
        "aov": ["aov", "average order", "average order value", "avg order"],
        "orders": ["orders", "order count", "total orders", "order volume"],
        "conversion": ["conversion", "conversion rate", "conv rate"],
        "units": ["units", "items sold", "units sold", "quantity"],
        "customer_count": ["customers", "new customers", "customer count", "total customers"],
        "repeat_rate": ["repeat", "repeat customer", "repeat rate", "returning"],
    }

    # Time range detection keywords
    TIME_RANGE_KEYWORDS = {
        "today": ["today"],
        "yesterday": ["yesterday"],
        "last_7_days": ["last week", "7 days", "past 7", "last 7"],
        "last_30_days": ["last month", "30 days", "past 30", "last 30", "monthly"],
        "last_90_days": ["last 90", "past 90", "quarterly", "3 months"],
        "year_to_date": ["ytd", "year to date", "since january"],
    }

    # Business context keywords
    BUSINESS_CONTEXT_KEYWORDS = [
        "launched", "new product", "sale started", "promotion", "campaign",
        "discount", "flash sale", "seasonal", "holiday", "event", "update",
        "release", "deal", "special offer", "limited time", "ended", "started",
    ]

    def __init__(self, db_ops):
        """Initialize PatternLearner.

        Args:
            db_ops: DatabaseOperations instance for storing patterns.
        """
        self.db_ops = db_ops
        logger.info("PatternLearner initialized")

    def learn_from_query(self, user_id: int, query: str, query_type: str = None) -> None:
        """Analyze a user query and update patterns in the database.

        Extracts and stores:
        - Query type (revenue, orders, products, customers, comparison, general)
        - Metrics mentioned (revenue, aov, orders, conversion, etc.)
        - Time range preferences
        - Business context indicators

        Args:
            user_id: The user ID
            query: The user's query string
            query_type: Optional pre-determined query type
        """
        logger.info(
            "Learning from query",
            user_id=user_id,
            query_length=len(query),
            preset_query_type=query_type,
        )

        # 1. Detect and record query type
        detected_type = query_type or self.detect_query_type(query)
        self.db_ops.update_pattern_frequency(user_id, "query_type", detected_type)
        logger.debug("Query type detected", user_id=user_id, type=detected_type)

        # 2. Extract and record metrics
        metrics = self.extract_metrics(query)
        for metric in metrics:
            self.db_ops.update_pattern_frequency(user_id, "metric", metric)
        logger.debug("Metrics extracted", user_id=user_id, metrics=metrics)

        # 3. Extract and record time range preference
        time_range = self.extract_time_range(query)
        if time_range:
            self.db_ops.update_pattern_frequency(user_id, "time_range", time_range)
            logger.debug("Time range detected", user_id=user_id, time_range=time_range)

        # 4. Extract and record business context
        business_context = self.extract_business_context(query)
        if business_context:
            timestamp = datetime.utcnow().isoformat()
            preference_key = f"business_context_{timestamp}"
            self.db_ops.set_preference(
                user_id=user_id,
                preference_key=preference_key,
                preference_value=business_context,
                confidence_score=0.7,
            )
            logger.debug(
                "Business context extracted",
                user_id=user_id,
                context=business_context,
            )

        # 5. Store fine-grained intent
        intent = self.classify_intent(query)
        self.db_ops.update_pattern_frequency(user_id, "intent", intent.fine)

        # 6. Store complexity
        complexity = self.assess_query_complexity(query)
        self.db_ops.update_pattern_frequency(user_id, "complexity", complexity)

        logger.info("Query learning completed", user_id=user_id)

    def detect_query_type(self, query: str) -> str:
        """Classify the query type based on keywords.

        Args:
            query: The user query string

        Returns:
            Query type: "revenue", "orders", "products", "customers",
                       "comparison", or "general"
        """
        query_lower = query.lower()

        # Check each query type in order of specificity
        for query_type, keywords in self.QUERY_TYPE_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type

        return "general"

    def extract_metrics(self, query: str) -> list[str]:
        """Extract metric keywords from the query.

        Args:
            query: The user query string

        Returns:
            List of detected metric names
        """
        query_lower = query.lower()
        detected_metrics = []

        for metric, keywords in self.METRIC_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_metrics.append(metric)

        return detected_metrics

    def extract_time_range(self, query: str) -> Optional[str]:
        """Extract time range preference from the query.

        Args:
            query: The user query string

        Returns:
            Time range identifier or None if not detected
        """
        query_lower = query.lower()

        for time_range, keywords in self.TIME_RANGE_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                return time_range

        return None

    def extract_business_context(self, query: str) -> Optional[str]:
        """Extract business context indicators from the query.

        Detects mentions of product launches, promotions, campaigns, etc.
        that help understand the business situation.

        Args:
            query: The user query string

        Returns:
            Business context description or None if not detected
        """
        query_lower = query.lower()

        # Check for business context keywords
        matched_contexts = [
            keyword
            for keyword in self.BUSINESS_CONTEXT_KEYWORDS
            if keyword in query_lower
        ]

        if matched_contexts:
            # Build a descriptive context string
            context = f"Business event detected: {', '.join(matched_contexts)}"
            return context

        return None

    def classify_intent(self, query: str) -> Intent:
        """Two-level intent classification.

        Coarse: the existing query type (revenue, orders, etc.)
        Fine: a more specific intent string combining type + ranking + metric + time

        Args:
            query: The user query string

        Returns:
            Intent object with coarse and fine-grained classifications
        """
        coarse = self.detect_query_type(query)
        fine = self._build_fine_intent(coarse, query)
        return Intent(coarse=coarse, fine=fine)

    def _build_fine_intent(self, coarse: str, query: str) -> str:
        """Build fine-grained intent string from coarse intent and query.

        Args:
            coarse: The coarse intent (query type)
            query: The user query string

        Returns:
            Fine-grained intent string combining type, ranking, metric, and time
        """
        query_lower = query.lower()
        metrics = self.extract_metrics(query)
        time_range = self.extract_time_range(query)
        has_ranking = any(
            w in query_lower
            for w in ["top", "best", "worst", "highest", "lowest", "most", "least"]
        )
        has_comparison = coarse == "comparison"

        parts = [coarse]
        if has_ranking:
            parts.append("ranking")
        if metrics:
            parts.append(metrics[0])
        if time_range:
            parts.append(time_range)
        if has_comparison:
            parts.append("comparison")
        return "_".join(parts)

    def assess_query_complexity(self, query: str) -> str:
        """Assess the complexity of a query.

        Evaluates signals including multiple metrics, comparisons, conjunctions,
        and query length to classify complexity as simple, moderate, or complex.

        Args:
            query: The user query string

        Returns:
            Complexity level: "simple", "moderate", or "complex"
        """
        signals = 0
        if len(self.extract_metrics(query)) > 1:
            signals += 1
        if self.detect_query_type(query) == "comparison":
            signals += 1
        if any(w in query.lower() for w in ["and", "also", "plus", "as well as"]):
            signals += 1
        if len(query.split()) > 20:
            signals += 1

        if signals == 0:
            return "simple"
        elif signals <= 2:
            return "moderate"
        else:
            return "complex"
