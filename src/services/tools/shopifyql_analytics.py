
from typing import Dict, Any
from .base import BaseTool


class ShopifyQLAnalyticsTool(BaseTool):
    """Tool for running ShopifyQL analytics queries — server-side aggregation across ALL data."""

    @property
    def name(self) -> str:
        return "shopifyql_analytics"

    @property
    def description(self) -> str:
        return (
            "Run a ShopifyQL analytics query on Shopify's server-side analytics engine. "
            "This is the PRIMARY tool for any analytics, reporting, or aggregate question. "
            "Shopify computes results across ALL data (no pagination limits, no 250-item cap). "
            "Use for: total revenue, order counts, growth trends, product performance, "
            "customer segmentation, discount impact, refund analysis, regional breakdowns, "
            "time-series comparisons, and any question requiring accurate aggregate numbers. "
            "Returns structured tabular data (columns + rows). "
            "Requires the ShopifyQL query string as input."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "ShopifyQL query string. Syntax reference:\n"
                        "\n"
                        "DATASETS (FROM):\n"
                        "  - sales: Revenue, discounts, returns, tax, shipping, product/variant info, billing/shipping location\n"
                        "  - orders: Order-level data (order_id, status, fulfillment, payment)\n"
                        "  - products: Product performance combining sales + sessions data\n"
                        "  - customers: Customer-level analytics\n"
                        "  - sessions: Traffic/session data\n"
                        "\n"
                        "KEY METRICS (sales dataset):\n"
                        "  DO NOT USE `sum()` or `count()`. Just use the metric name directly:\n"
                        "  - total_sales, net_sales, gross_sales\n"
                        "  - returns, discounts, taxes, shipping\n"
                        "  - orders (number of orders), ordered_quantity, net_quantity\n"
                        "\n"
                        "KEY DIMENSIONS (sales dataset):\n"
                        "  - product_title, product_type, product_vendor, variant_title, sku\n"
                        "  - billing_country, billing_city, billing_region\n"
                        "  - shipping_country, shipping_city\n"
                        "  - discount_code, sale_kind, sale_line_type\n"
                        "\n"
                        "TIME FILTERING (SINCE/UNTIL):\n"
                        "  Named: today, yesterday, this_week, last_week, this_month, last_month, "
                        "this_quarter, last_quarter, this_year, last_year\n"
                        "  Relative: -7d, -30d, -3m, -1y\n"
                        "  Absolute: 2026-01-01\n"
                        "\n"
                        "TIMESERIES / GROUPING: GROUP BY day, week, month, quarter, year\n"
                        "\n"
                        "COMPARE TO: previous_period, previous_year\n"
                        "\n"
                        "Examples:\n"
                        "  SHOW total_sales, net_sales, orders FROM sales SINCE last_month UNTIL this_month\n"
                        "  SHOW net_sales FROM sales GROUP BY product_title ORDER BY net_sales DESC LIMIT 10\n"
                        "  SHOW net_sales FROM sales GROUP BY month SINCE -6m\n"
                        "  SHOW returns, discounts FROM sales GROUP BY product_title ORDER BY returns DESC LIMIT 10\n"
                        "  SHOW net_sales FROM sales GROUP BY billing_country ORDER BY net_sales DESC\n"
                        "  SHOW total_sales FROM sales SINCE -30d UNTIL today COMPARE TO previous_period\n"
                        "  SHOW net_sales, view_sessions FROM products GROUP BY product_title SINCE -7d ORDER BY net_sales DESC LIMIT 20\n"
                    ),
                },
            },
            "required": ["query"],
        }

    def validate(self, input_data: Dict[str, Any]) -> bool:
        super().validate(input_data)

        query = input_data.get("query", "").strip()
        if not query:
            raise ValueError("ShopifyQL query cannot be empty")

        # Basic syntax check — must have FROM and SHOW
        query_upper = query.upper()
        if "FROM" not in query_upper:
            raise ValueError("ShopifyQL query must include FROM clause")
        if "SHOW" not in query_upper:
            raise ValueError("ShopifyQL query must include SHOW clause")

        return True
