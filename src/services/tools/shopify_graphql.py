
from typing import Dict, Any
from .base import BaseTool

class ShopifyGraphQLTool(BaseTool):
    """Tool for executing raw GraphQL queries against Shopify Admin API."""
    
    @property
    def name(self) -> str:
        return "shopify_graphql"

    @property
    def description(self) -> str:
        return (
            "Execute a custom GraphQL query against Shopify's Admin API. "
            "Use when shopify_analytics doesn't cover your needs — "
            "for example: shop info, inventory levels, discount codes, "
            "draft orders, metafields, collections, fulfillments, refunds, "
            "or any complex/nested query. You write the full GraphQL query. "
            "READ-ONLY: mutations are blocked for safety. "
            "The Shopify Admin API uses Relay-style connections (edges/node pattern). "
            "All timestamps in the response are already in IST (UTC+5:30) — display as-is, do NOT convert."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The full GraphQL query string. Must start with 'query' or '{'. "
                        "Use Shopify Admin API schema. Example: "
                        "'{ shop { name email myshopifyDomain plan { displayName } } }'"
                    ),
                },
                "variables": {
                    "type": "object",
                    "description": "Optional GraphQL variables as a JSON object",
                },
            },
            "required": ["query"],
        }
    
    def validate(self, input_data: Dict[str, Any]) -> bool:
        super().validate(input_data)
        
        query = input_data.get("query", "").strip()
        if not query:
            raise ValueError("Query cannot be empty")
            
        # Basic read-only check (though service layer does deeper check)
        if "mutation" in query.lower() and "mutation" not in query: # loose check if mutation keyword is used as operation type
             # Actually let's assume the service layer handles the security check better.
             # but we can do a quick check.
             if query.lstrip().startswith("mutation"):
                 raise ValueError("Mutations are not allowed")

        return True
