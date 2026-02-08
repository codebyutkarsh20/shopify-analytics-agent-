"""Analytics service for processing and caching Shopify analytics data."""

import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.database.operations import DatabaseOperations
from src.services.shopify_service import ShopifyService
from src.utils.date_parser import DateRange, get_comparison_range
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnalyticsService:
    """Service for analytics calculations and caching."""

    def __init__(
        self,
        shopify_service: ShopifyService,
        db_ops: DatabaseOperations,
    ):
        """
        Initialize the Analytics service.

        Args:
            shopify_service: ShopifyService instance for fetching Shopify data
            db_ops: DatabaseOperations instance for caching
        """
        self.shopify_service = shopify_service
        self.db_ops = db_ops

    async def get_period_analytics(
        self,
        user_id: int,
        start_date: datetime,
        end_date: datetime,
        store_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get analytics metrics for a specific date period.

        Checks cache first. If not cached, fetches orders and calculates metrics.

        Args:
            user_id: User ID for context
            start_date: Start date for the period
            end_date: End date for the period
            store_id: Optional store ID (defaults to user's primary store)

        Returns:
            Dictionary containing:
                - total_revenue: Total revenue for period
                - order_count: Number of orders
                - aov: Average order value
                - top_products: List of top products by revenue
                - currency: Currency code
                - date_range: Period label
        """
        try:
            logger.info(
                "Getting period analytics",
                user_id=user_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )

            # Generate cache key
            cache_key = self._generate_cache_key(start_date, end_date)

            # Get store ID if not provided
            if store_id is None:
                store = self.db_ops.get_store_by_user(user_id)
                if store is None:
                    logger.warning(
                        "No store found for user",
                        user_id=user_id,
                    )
                    return {
                        "total_revenue": 0.0,
                        "order_count": 0,
                        "aov": 0.0,
                        "top_products": [],
                        "currency": None,
                        "date_range": f"{start_date.date()} to {end_date.date()}",
                    }
                store_id = store.id

            # Check cache first
            cached_data = self.db_ops.get_cache(store_id, cache_key)
            if cached_data:
                logger.info(
                    "Analytics retrieved from cache",
                    cache_key=cache_key,
                )
                result = json.loads(cached_data)
                result["from_cache"] = True
                return result

            # Fetch orders from Shopify
            orders = await self.shopify_service.get_orders(
                start_date=start_date,
                end_date=end_date,
            )

            # Process orders to get metrics
            metrics = self.shopify_service.process_orders_data(orders)

            # Add date range label
            metrics["date_range"] = f"{start_date.date()} to {end_date.date()}"
            metrics["from_cache"] = False

            # Cache the results
            self.db_ops.set_cache(
                store_id=store_id,
                cache_key=cache_key,
                cache_data=json.dumps(metrics),
                ttl_hours=1,
            )

            logger.info(
                "Analytics calculated and cached",
                cache_key=cache_key,
                total_revenue=metrics.get("total_revenue"),
                order_count=metrics.get("order_count"),
            )

            return metrics

        except Exception as e:
            logger.error(
                "Failed to get period analytics",
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            raise

    async def get_comparison_analytics(
        self,
        user_id: int,
        date_range: DateRange,
        store_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get analytics with period-over-period comparison.

        Fetches metrics for current period and equivalent previous period,
        then calculates comparison data.

        Args:
            user_id: User ID for context
            date_range: Current DateRange to analyze
            store_id: Optional store ID

        Returns:
            Dictionary containing:
                - current: Current period metrics
                - previous: Previous period metrics
                - comparison: Comparison data with changes
                - date_range: Current period label
                - comparison_range: Previous period label
        """
        try:
            logger.info(
                "Getting comparison analytics",
                user_id=user_id,
                date_range=date_range.label,
            )

            # Get current period metrics
            current_metrics = await self.get_period_analytics(
                user_id=user_id,
                start_date=date_range.start_date,
                end_date=date_range.end_date,
                store_id=store_id,
            )

            # Get comparison period
            comparison_range = get_comparison_range(date_range)

            # Get previous period metrics
            previous_metrics = await self.get_period_analytics(
                user_id=user_id,
                start_date=comparison_range.start_date,
                end_date=comparison_range.end_date,
                store_id=store_id,
            )

            # Calculate comparison
            comparison = self.shopify_service.calculate_comparison(
                current_metrics,
                previous_metrics,
            )

            result = {
                "current": current_metrics,
                "previous": previous_metrics,
                "comparison": comparison,
                "date_range": date_range.label,
                "comparison_range": comparison_range.label,
            }

            logger.info(
                "Comparison analytics calculated",
                revenue_change_pct=comparison["revenue_change"]["percentage"],
                orders_change_abs=comparison["orders_change"]["absolute"],
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to get comparison analytics",
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            raise

    async def get_top_products(
        self,
        user_id: int,
        start_date: datetime,
        end_date: datetime,
        limit: int = 5,
        store_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get top products by revenue for a date period.

        Args:
            user_id: User ID for context
            start_date: Start date for the period
            end_date: End date for the period
            limit: Number of top products to return (default: 5)
            store_id: Optional store ID

        Returns:
            List of top products with revenue and units sold
        """
        try:
            logger.info(
                "Getting top products",
                user_id=user_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                limit=limit,
            )

            # Get analytics for period (which includes top products)
            metrics = await self.get_period_analytics(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                store_id=store_id,
            )

            # Extract and limit top products
            top_products = metrics.get("top_products", [])[:limit]

            logger.info(
                "Top products retrieved",
                count=len(top_products),
            )

            return top_products

        except Exception as e:
            logger.error(
                "Failed to get top products",
                user_id=user_id,
                error=str(e),
                exc_info=True,
            )
            raise

    async def generate_insights(
        self,
        current_metrics: Dict[str, Any],
        comparison: Dict[str, Any],
    ) -> List[str]:
        """
        Generate text insights based on metrics and comparison data.

        Args:
            current_metrics: Current period metrics
            comparison: Comparison data from calculate_comparison

        Returns:
            List of insight strings

        Raises:
            ValueError: If metrics are malformed
        """
        try:
            logger.info("Generating insights from metrics")

            insights = []

            # Revenue trends
            revenue_change = comparison.get("revenue_change", {})
            revenue_pct = revenue_change.get("percentage", 0)
            revenue_abs = revenue_change.get("absolute", 0)

            if revenue_pct > 5:
                insights.append(
                    f"Revenue is trending upward with a {revenue_pct:.1f}% increase "
                    f"(${revenue_abs:.2f}) compared to the previous period."
                )
            elif revenue_pct < -5:
                insights.append(
                    f"Revenue has declined by {abs(revenue_pct):.1f}% "
                    f"(${revenue_abs:.2f}) compared to the previous period."
                )
            else:
                insights.append(
                    "Revenue remains relatively stable compared to the previous period."
                )

            # Order volume analysis
            orders_change = comparison.get("orders_change", {})
            orders_abs = orders_change.get("absolute", 0)
            orders_pct = orders_change.get("percentage", 0)
            current_orders = current_metrics.get("order_count", 0)

            if current_orders > 0:
                if orders_abs > 0:
                    insights.append(
                        f"You received {orders_abs} more orders "
                        f"({orders_pct:.1f}% increase) this period, "
                        f"with a total of {current_orders} orders."
                    )
                elif orders_abs < 0:
                    insights.append(
                        f"Order volume decreased by {abs(orders_abs)} orders "
                        f"({abs(orders_pct):.1f}% decrease) compared to the previous period."
                    )
            else:
                insights.append("No orders recorded in this period.")

            # AOV analysis
            aov_change = comparison.get("aov_change", {})
            aov_abs = aov_change.get("absolute", 0)
            aov_pct = aov_change.get("percentage", 0)
            current_aov = current_metrics.get("aov", 0)

            if current_aov > 0:
                if abs(aov_pct) > 5:
                    direction = "increased" if aov_abs > 0 else "decreased"
                    insights.append(
                        f"Average order value {direction} by ${abs(aov_abs):.2f} "
                        f"({abs(aov_pct):.1f}%) to ${current_aov:.2f}."
                    )

            # Top products insight
            top_products = current_metrics.get("top_products", [])
            if top_products:
                top_product = top_products[0]
                top_name = top_product.get("name", "Unknown")
                top_revenue = top_product.get("revenue", 0)
                insights.append(
                    f"Your top-performing product is '{top_name}' "
                    f"with ${top_revenue:.2f} in revenue."
                )

            logger.info(
                "Insights generated",
                insight_count=len(insights),
            )

            return insights

        except Exception as e:
            logger.error(
                "Failed to generate insights",
                error=str(e),
                exc_info=True,
            )
            raise ValueError(f"Failed to generate insights: {str(e)}")

    def _generate_cache_key(self, start_date: datetime, end_date: datetime) -> str:
        """
        Generate a cache key from date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Cache key string
        """
        date_str = f"{start_date.isoformat()}_{end_date.isoformat()}"
        cache_hash = hashlib.sha256(date_str.encode()).hexdigest()[:16]  # Use first 16 chars
        return f"analytics_{cache_hash}"
