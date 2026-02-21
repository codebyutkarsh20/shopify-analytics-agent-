
from typing import Dict, Any
from .base import BaseTool

class ShopifyAnalyticsTool(BaseTool):
    """Tool for retrieving structured analytics data from Shopify."""
    
    @property
    def name(self) -> str:
        return "shopify_analytics"

    @property
    def description(self) -> str:
        return (
            "Query Shopify products, orders, or customers with sorting, filtering, and pagination. "
            "Use for: rankings (top products, biggest orders, best customers), "
            "date-filtered queries, sorting by any field (price, revenue, date), "
            "and paginating through large result sets. "
            "Queries Shopify's GraphQL Admin API directly. "
            "All timestamps in the response are already in IST (UTC+5:30) — display as-is, do NOT convert."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "resource": {
                    "type": "string",
                    "enum": ["products", "orders", "customers"],
                    "description": "The type of Shopify resource to query",
                },
                "sort_key": {
                    "type": "string",
                    "description": (
                        "Field to sort by. "
                        "Products: TITLE, PRICE, BEST_SELLING, CREATED_AT, UPDATED_AT, INVENTORY_TOTAL. "
                        "Orders: CREATED_AT, TOTAL_PRICE, ORDER_NUMBER, PROCESSED_AT. "
                        "Customers: NAME, TOTAL_SPENT, ORDERS_COUNT, CREATED_AT, LAST_ORDER_DATE."
                    ),
                },
                "reverse": {
                    "type": "boolean",
                    "description": "Sort descending (highest/newest first) when true. Default: true",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (1-250). Default: 10",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Shopify search/filter string. IMPORTANT: Date filters MUST use UTC "
                        "timestamps (NOT IST dates). Use the pre-computed UTC boundaries from "
                        "the system prompt for 'today', 'yesterday', etc. "
                        "Examples: 'financial_status:paid', 'status:active', "
                        "'created_at:>=2026-02-08T18:30:00Z' (today in IST as UTC), "
                        "'tag:sale'. Multiple filters can be combined with spaces."
                    ),
                },
                "after": {
                    "type": "string",
                    "description": "Pagination cursor from previous response's end_cursor for next page",
                },
            },
            "required": ["resource"],
        }
    
    def validate(self, input_data: Dict[str, Any]) -> bool:
        super().validate(input_data)
        
        # Validate resource enum
        resource = input_data.get("resource")
        if resource not in ["products", "orders", "customers"]:
            raise ValueError(f"Invalid resource: {resource}")
            
        # Validate limit
        if "limit" in input_data:
            limit = input_data["limit"]
            if not isinstance(limit, int) or limit < 1 or limit > 250:
                raise ValueError(f"Limit must be between 1 and 250, got {limit}")
                
        return True
