from datetime import datetime
from typing import Dict, Any, List, Optional
from decimal import Decimal

from src.services.mcp_service import MCPService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShopifyService:
    """Service for processing and analyzing Shopify data via MCP."""

    def __init__(self, mcp_service: MCPService):
        """
        Initialize the Shopify service.

        Args:
            mcp_service: MCPService instance for communicating with Shopify MCP server
        """
        self.mcp_service = mcp_service

    async def get_orders(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 250,
    ) -> List[Dict[str, Any]]:
        """
        Fetch orders from Shopify within the specified date range.

        Args:
            start_date: Start date for order filtering (inclusive)
            end_date: End date for order filtering (inclusive)
            limit: Maximum number of orders to fetch (default: 250)

        Returns:
            List of order dictionaries

        Raises:
            ValueError: If MCP returns an error
            RuntimeError: If MCP service is not running
        """
        try:
            logger.info(
                "Fetching orders from Shopify",
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                limit=limit,
            )

            # Format dates as ISO 8601 strings
            created_at_min = start_date.isoformat()
            created_at_max = end_date.isoformat()

            # Call the MCP tool
            result = await self.mcp_service.call_tool(
                "get_orders",
                {
                    "created_at_min": created_at_min,
                    "created_at_max": created_at_max,
                    "limit": limit,
                },
            )

            orders = result.get("orders", [])
            logger.info("Orders fetched successfully", count=len(orders))

            return orders

        except Exception as e:
            logger.error(
                "Failed to fetch orders",
                error=str(e),
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                exc_info=True,
            )
            raise

    async def get_products(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch products from Shopify.

        Args:
            limit: Maximum number of products to fetch (default: 50)

        Returns:
            List of product dictionaries

        Raises:
            ValueError: If MCP returns an error
            RuntimeError: If MCP service is not running
        """
        try:
            logger.info("Fetching products from Shopify", limit=limit)

            result = await self.mcp_service.call_tool(
                "get_products",
                {"limit": limit},
            )

            products = result.get("products", [])
            logger.info("Products fetched successfully", count=len(products))

            return products

        except Exception as e:
            logger.error(
                "Failed to fetch products",
                error=str(e),
                exc_info=True,
            )
            raise

    async def search_products(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for products in Shopify.

        Args:
            query: Search query string

        Returns:
            List of matching product dictionaries

        Raises:
            ValueError: If MCP returns an error
            RuntimeError: If MCP service is not running
        """
        try:
            logger.info("Searching products", query=query)

            result = await self.mcp_service.call_tool(
                "search_products",
                {"query": query},
            )

            products = result.get("products", [])
            logger.info("Product search completed", query=query, count=len(products))

            return products

        except Exception as e:
            logger.error(
                "Product search failed",
                error=str(e),
                query=query,
                exc_info=True,
            )
            raise

    async def get_customers(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch customers from Shopify.

        Args:
            limit: Maximum number of customers to fetch (default: 50)

        Returns:
            List of customer dictionaries

        Raises:
            ValueError: If MCP returns an error
            RuntimeError: If MCP service is not running
        """
        try:
            logger.info("Fetching customers from Shopify", limit=limit)

            result = await self.mcp_service.call_tool(
                "get_customers",
                {"limit": limit},
            )

            customers = result.get("customers", [])
            logger.info("Customers fetched successfully", count=len(customers))

            return customers

        except Exception as e:
            logger.error(
                "Failed to fetch customers",
                error=str(e),
                exc_info=True,
            )
            raise

    def process_orders_data(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process raw Shopify orders into analytics metrics.

        Args:
            orders: List of raw order dictionaries from Shopify

        Returns:
            Dictionary containing:
                - total_revenue: Total revenue across all orders
                - order_count: Number of orders
                - aov: Average order value
                - top_products: List of top products by revenue
                - currency: Currency code (if available)

        Raises:
            ValueError: If orders list is empty or malformed
        """
        try:
            if not orders:
                logger.warning("No orders provided for processing")
                return {
                    "total_revenue": 0.0,
                    "order_count": 0,
                    "aov": 0.0,
                    "top_products": [],
                    "currency": None,
                }

            logger.info("Processing order data", order_count=len(orders))

            # Initialize metrics
            total_revenue = Decimal("0")
            product_metrics: Dict[str, Dict[str, Any]] = {}
            currency = None

            # Process each order
            for order in orders:
                try:
                    # Get order total
                    order_total = Decimal(str(order.get("total_price", 0)))
                    total_revenue += order_total

                    # Extract currency from first order
                    if currency is None:
                        currency = order.get("currency", "USD")

                    # Process line items (products in the order)
                    line_items = order.get("line_items", [])
                    for item in line_items:
                        product_name = item.get("title", "Unknown Product")
                        item_price = Decimal(str(item.get("price", 0)))
                        quantity = int(item.get("quantity", 0))
                        item_revenue = item_price * quantity

                        if product_name not in product_metrics:
                            product_metrics[product_name] = {
                                "name": product_name,
                                "revenue": Decimal("0"),
                                "units_sold": 0,
                            }

                        product_metrics[product_name]["revenue"] += item_revenue
                        product_metrics[product_name]["units_sold"] += quantity

                except (TypeError, ValueError, KeyError) as e:
                    logger.warning(
                        "Error processing order",
                        error=str(e),
                        order_id=order.get("id"),
                        exc_info=True,
                    )
                    continue

            # Calculate metrics
            order_count = len(orders)
            aov = total_revenue / order_count if order_count > 0 else Decimal("0")

            # Get top 10 products by revenue
            top_products = sorted(
                product_metrics.values(),
                key=lambda x: x["revenue"],
                reverse=True,
            )[:10]

            # Convert Decimal to float for JSON serialization
            result = {
                "total_revenue": float(total_revenue),
                "order_count": order_count,
                "aov": float(aov),
                "top_products": [
                    {
                        "name": p["name"],
                        "revenue": float(p["revenue"]),
                        "units_sold": p["units_sold"],
                    }
                    for p in top_products
                ],
                "currency": currency,
            }

            logger.info(
                "Order data processed successfully",
                total_revenue=result["total_revenue"],
                order_count=order_count,
                aov=result["aov"],
                top_products_count=len(result["top_products"]),
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to process order data",
                error=str(e),
                order_count=len(orders),
                exc_info=True,
            )
            raise

    def calculate_comparison(
        self,
        current_metrics: Dict[str, Any],
        previous_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate comparison between two metric periods.

        Args:
            current_metrics: Current period metrics (from process_orders_data)
            previous_metrics: Previous period metrics for comparison

        Returns:
            Dictionary containing:
                - revenue_change: {'absolute': float, 'percentage': float}
                - orders_change: {'absolute': int, 'percentage': float}
                - aov_change: {'absolute': float, 'percentage': float}

        Raises:
            ValueError: If metrics dictionaries are malformed
            ZeroDivisionError: Handled gracefully with appropriate values
        """
        try:
            logger.info("Calculating metric comparison")

            comparison = {}

            # Revenue comparison
            current_revenue = Decimal(str(current_metrics.get("total_revenue", 0)))
            previous_revenue = Decimal(str(previous_metrics.get("total_revenue", 0)))
            revenue_absolute = float(current_revenue - previous_revenue)

            if previous_revenue > 0:
                revenue_percentage = float(
                    ((current_revenue - previous_revenue) / previous_revenue) * 100
                )
            else:
                revenue_percentage = 0.0 if current_revenue == 0 else 100.0

            comparison["revenue_change"] = {
                "absolute": revenue_absolute,
                "percentage": revenue_percentage,
            }

            # Order count comparison
            current_orders = current_metrics.get("order_count", 0)
            previous_orders = previous_metrics.get("order_count", 0)
            orders_absolute = current_orders - previous_orders

            if previous_orders > 0:
                orders_percentage = (
                    (current_orders - previous_orders) / previous_orders * 100
                )
            else:
                orders_percentage = 0.0 if current_orders == 0 else 100.0

            comparison["orders_change"] = {
                "absolute": orders_absolute,
                "percentage": orders_percentage,
            }

            # AOV comparison
            current_aov = Decimal(str(current_metrics.get("aov", 0)))
            previous_aov = Decimal(str(previous_metrics.get("aov", 0)))
            aov_absolute = float(current_aov - previous_aov)

            if previous_aov > 0:
                aov_percentage = float(
                    ((current_aov - previous_aov) / previous_aov) * 100
                )
            else:
                aov_percentage = 0.0 if current_aov == 0 else 100.0

            comparison["aov_change"] = {
                "absolute": aov_absolute,
                "percentage": aov_percentage,
            }

            logger.info(
                "Comparison calculated successfully",
                revenue_change_percentage=comparison["revenue_change"]["percentage"],
                orders_change_absolute=comparison["orders_change"]["absolute"],
                aov_change_percentage=comparison["aov_change"]["percentage"],
            )

            return comparison

        except (TypeError, ValueError, KeyError) as e:
            logger.error(
                "Failed to calculate comparison",
                error=str(e),
                exc_info=True,
            )
            raise ValueError(f"Invalid metrics format for comparison: {e}")
