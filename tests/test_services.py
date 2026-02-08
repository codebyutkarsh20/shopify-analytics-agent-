"""Tests for Shopify Analytics Agent services."""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch

from src.services.shopify_service import ShopifyService
from src.services.analytics_service import AnalyticsService
from src.utils.date_parser import DateRange
from datetime import datetime
import pytz


class TestShopifyService:
    """Test suite for ShopifyService."""

    def test_process_orders_data(self, mock_mcp_service, sample_orders):
        """Test processing sample orders into analytics metrics.

        Verifies:
        - Total revenue calculated correctly: $425.50
        - Order count: 3
        - Average order value calculated correctly
        - Top products identified and ranked by revenue
        """
        service = ShopifyService(mock_mcp_service)

        result = service.process_orders_data(sample_orders)

        # Verify revenue calculation
        assert result["total_revenue"] == 425.50
        assert result["order_count"] == 3

        # Verify AOV calculation
        expected_aov = 425.50 / 3
        assert abs(result["aov"] - expected_aov) < 0.01

        # Verify top products
        assert len(result["top_products"]) > 0
        assert result["top_products"][0]["name"] == "Premium Widget"
        assert result["top_products"][0]["revenue"] == 200.00

        # Verify currency
        assert result["currency"] == "USD"

    def test_process_empty_orders(self, mock_mcp_service):
        """Test processing with empty order list.

        Should return zero metrics gracefully.
        """
        service = ShopifyService(mock_mcp_service)

        result = service.process_orders_data([])

        assert result["total_revenue"] == 0.0
        assert result["order_count"] == 0
        assert result["aov"] == 0.0
        assert result["top_products"] == []
        assert result["currency"] is None

    def test_calculate_comparison(self, mock_mcp_service):
        """Test comparison calculation between two metric sets.

        Verifies percentage changes and absolute changes are calculated correctly.
        """
        service = ShopifyService(mock_mcp_service)

        current_metrics = {
            "total_revenue": 1000.0,
            "order_count": 50,
            "aov": 20.0
        }

        previous_metrics = {
            "total_revenue": 800.0,
            "order_count": 40,
            "aov": 20.0
        }

        comparison = service.calculate_comparison(current_metrics, previous_metrics)

        # Revenue change: (1000 - 800) / 800 * 100 = 25%
        assert comparison["revenue_change"]["absolute"] == 200.0
        assert comparison["revenue_change"]["percentage"] == 25.0

        # Orders change: (50 - 40) / 40 * 100 = 25%
        assert comparison["orders_change"]["absolute"] == 10
        assert comparison["orders_change"]["percentage"] == 25.0

        # AOV change: no change, so 0%
        assert comparison["aov_change"]["absolute"] == 0.0
        assert comparison["aov_change"]["percentage"] == 0.0

    def test_calculate_comparison_zero_previous(self, mock_mcp_service):
        """Test comparison with zero previous metrics (division by zero case).

        Verifies graceful handling of edge case where previous period is zero.
        """
        service = ShopifyService(mock_mcp_service)

        current_metrics = {
            "total_revenue": 1000.0,
            "order_count": 50,
            "aov": 20.0
        }

        previous_metrics = {
            "total_revenue": 0.0,
            "order_count": 0,
            "aov": 0.0
        }

        comparison = service.calculate_comparison(current_metrics, previous_metrics)

        # When previous is 0 and current > 0, should return 100%
        assert comparison["revenue_change"]["percentage"] == 100.0
        assert comparison["orders_change"]["percentage"] == 100.0
        assert comparison["aov_change"]["percentage"] == 100.0

    def test_calculate_comparison_zero_both(self, mock_mcp_service):
        """Test comparison when both current and previous are zero.

        Should return 0% change.
        """
        service = ShopifyService(mock_mcp_service)

        current_metrics = {
            "total_revenue": 0.0,
            "order_count": 0,
            "aov": 0.0
        }

        previous_metrics = {
            "total_revenue": 0.0,
            "order_count": 0,
            "aov": 0.0
        }

        comparison = service.calculate_comparison(current_metrics, previous_metrics)

        assert comparison["revenue_change"]["percentage"] == 0.0
        assert comparison["orders_change"]["percentage"] == 0.0
        assert comparison["aov_change"]["percentage"] == 0.0


class TestAnalyticsService:
    """Test suite for AnalyticsService."""

    @pytest.mark.asyncio
    async def test_generate_insights_growth(self, mock_mcp_service, db_ops):
        """Test insight generation with positive growth metrics.

        Verifies that growth insights are generated when metrics show improvement.
        """
        analytics_service = AnalyticsService(
            ShopifyService(mock_mcp_service),
            db_ops
        )

        current_metrics = {
            "total_revenue": 1500.0,
            "order_count": 60,
            "aov": 25.0,
            "top_products": [
                {
                    "name": "Widget A",
                    "revenue": 600.0,
                    "units_sold": 30
                }
            ]
        }

        comparison = {
            "revenue_change": {
                "absolute": 500.0,
                "percentage": 50.0
            },
            "orders_change": {
                "absolute": 20,
                "percentage": 50.0
            },
            "aov_change": {
                "absolute": 5.0,
                "percentage": 25.0
            }
        }

        insights = await analytics_service.generate_insights(current_metrics, comparison)

        assert len(insights) > 0
        assert any("upward" in insight.lower() for insight in insights)
        assert any("50" in insight for insight in insights)

    @pytest.mark.asyncio
    async def test_generate_insights_decline(self, mock_mcp_service, db_ops):
        """Test insight generation with negative growth metrics.

        Verifies that decline insights are generated when metrics show decline.
        """
        analytics_service = AnalyticsService(
            ShopifyService(mock_mcp_service),
            db_ops
        )

        current_metrics = {
            "total_revenue": 500.0,
            "order_count": 20,
            "aov": 25.0,
            "top_products": [
                {
                    "name": "Widget A",
                    "revenue": 200.0,
                    "units_sold": 10
                }
            ]
        }

        comparison = {
            "revenue_change": {
                "absolute": -500.0,
                "percentage": -50.0
            },
            "orders_change": {
                "absolute": -20,
                "percentage": -50.0
            },
            "aov_change": {
                "absolute": -5.0,
                "percentage": -25.0
            }
        }

        insights = await analytics_service.generate_insights(current_metrics, comparison)

        assert len(insights) > 0
        assert any("declined" in insight.lower() or "decreased" in insight.lower() for insight in insights)

    @pytest.mark.asyncio
    async def test_generate_insights_stable(self, mock_mcp_service, db_ops):
        """Test insight generation with stable (minimal change) metrics."""
        analytics_service = AnalyticsService(
            ShopifyService(mock_mcp_service),
            db_ops
        )

        current_metrics = {
            "total_revenue": 1000.0,
            "order_count": 40,
            "aov": 25.0,
            "top_products": []
        }

        comparison = {
            "revenue_change": {
                "absolute": 0.0,
                "percentage": 0.0
            },
            "orders_change": {
                "absolute": 0,
                "percentage": 0.0
            },
            "aov_change": {
                "absolute": 0.0,
                "percentage": 0.0
            }
        }

        insights = await analytics_service.generate_insights(current_metrics, comparison)

        assert len(insights) > 0
        assert any("stable" in insight.lower() for insight in insights)
