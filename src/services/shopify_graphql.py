"""
Direct Shopify GraphQL Admin API client for advanced analytics queries.

This module bypasses the MCP server to query Shopify's GraphQL API directly,
enabling features the MCP tools don't support:
- Sorting by any field (price, total_spent, order_value, etc.)
- Cursor-based pagination through large result sets
- Complex query filters (date ranges, status, tags, etc.)
- Client-side sorting for fields Shopify doesn't natively sort by (e.g., product price)
"""

import json
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

import httpx
import pytz

from src.config.settings import Settings
from src.utils.logger import get_logger
from src.utils.timezone import IST

logger = get_logger(__name__)

# ISO-8601 pattern that matches Shopify UTC timestamps like:
# "2026-02-08T18:45:00Z" or "2026-02-08T18:45:00+00:00"
_ISO_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)

# Valid Shopify Admin API GraphQL sort keys per resource
PRODUCT_SORT_KEYS = [
    "TITLE", "PRODUCT_TYPE", "VENDOR", "INVENTORY_TOTAL",
    "UPDATED_AT", "CREATED_AT", "BEST_SELLING", "ID",
]
ORDER_SORT_KEYS = [
    "CREATED_AT", "UPDATED_AT", "TOTAL_PRICE", "ORDER_NUMBER",
    "CUSTOMER_NAME", "FINANCIAL_STATUS", "FULFILLMENT_STATUS",
    "PROCESSED_AT", "ID",
]
CUSTOMER_SORT_KEYS = [
    "NAME", "ORDERS_COUNT", "TOTAL_SPENT", "UPDATED_AT",
    "CREATED_AT", "LAST_ORDER_DATE", "LOCATION", "ID",
]


class ShopifyGraphQLClient:
    """Direct Shopify GraphQL Admin API client for analytics queries."""

    DEFAULT_API_VERSION = "2024-10"

    def __init__(self, settings: Settings):
        # Normalize shop domain: strip protocol, trailing slashes
        domain = settings.shopify.shop_domain.strip()
        domain = domain.replace("https://", "").replace("http://", "").rstrip("/")

        self.shop_domain = domain
        self.access_token = settings.shopify.access_token
        # API version: configurable via env, falls back to default
        import os
        self.api_version = os.getenv("SHOPIFY_API_VERSION", self.DEFAULT_API_VERSION)
        self.endpoint = (
            f"https://{self.shop_domain}/admin/api/{self.api_version}/graphql.json"
        )
        self.headers = {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json",
        }
        logger.info(
            "ShopifyGraphQLClient initialized",
            shop_domain=self.shop_domain,
            api_version=self.api_version,
        )

    async def execute_query(
        self, query: str, variables: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute a raw GraphQL query against the Shopify Admin API."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        logger.debug("Executing GraphQL query", variables=variables)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                error_messages = [
                    e.get("message", str(e)) for e in data["errors"]
                ]
                logger.error("GraphQL errors", errors=error_messages)
                raise ValueError(f"GraphQL errors: {'; '.join(error_messages)}")

            result = data.get("data", {})

            # ── Convert ALL Shopify UTC timestamps to IST at the gateway ──
            self._convert_all_timestamps_to_ist(result)

            return result

    # ── Centralized UTC → IST conversion ────────────────────────────

    def _convert_all_timestamps_to_ist(self, obj: Any) -> None:
        """Recursively walk a GraphQL response and convert UTC timestamps to IST in-place.

        Detects Shopify timestamp fields by two heuristics:
        1. Key ends with "At" (camelCase, e.g. createdAt, updatedAt, processedAt)
        2. Key ends with "_at" (snake_case, e.g. created_at, updated_at)

        Only converts values that look like ISO-8601 strings with a timezone
        component (contains 'Z' or '+00:00'), to avoid touching non-timestamp fields.
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if (
                    isinstance(value, str)
                    and (key.endswith("At") or key.lower().endswith("_at"))
                    and _ISO_TIMESTAMP_RE.match(value)
                ):
                    try:
                        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        obj[key] = dt.astimezone(IST).isoformat()
                    except (ValueError, TypeError):
                        logger.warning(
                            "Failed to convert timestamp to IST",
                            field=key, value=value,
                        )
                else:
                    self._convert_all_timestamps_to_ist(value)
        elif isinstance(obj, list):
            for item in obj:
                self._convert_all_timestamps_to_ist(item)

    async def execute_raw_query(
        self, query: str, variables: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute a Claude-generated GraphQL query with safety checks.

        Blocks mutations to prevent data modification — only read queries allowed.
        This gives Claude full flexibility to query any Shopify data while
        preventing accidental writes/deletes.

        Args:
            query: GraphQL query string (must be a query, not mutation)
            variables: Optional query variables

        Returns:
            Raw GraphQL response data

        Raises:
            ValueError: If query contains a mutation
        """
        # Safety: block mutations and subscriptions — only allow read queries
        # Strip comments (both # line comments and inline) before analysis
        import re as _re
        lines = query.splitlines()
        stripped_lines = []
        for line in lines:
            # Remove # comments (but not inside strings)
            line_clean = _re.sub(r'#.*$', '', line).strip()
            if line_clean:
                stripped_lines.append(line_clean)
        query_body = " ".join(stripped_lines).strip().lower()

        # Detect mutation or subscription operations anywhere in the query
        # Matches: "mutation", "mutation {", "mutation MyMutation", etc.
        if _re.search(r'\bmutation\b', query_body):
            raise ValueError(
                "Mutations are blocked for safety. This bot only supports read queries."
            )
        if _re.search(r'\bsubscription\b', query_body):
            raise ValueError(
                "Subscriptions are blocked for safety. This bot only supports read queries."
            )

        logger.info("Executing raw GraphQL query from Claude")
        return await self.execute_query(query, variables)

    # ── Product queries ─────────────────────────────────────────────

    async def query_products(
        self,
        sort_key: str = "CREATED_AT",
        reverse: bool = True,
        limit: int = 10,
        query: Optional[str] = None,
        after: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query products with sorting, filtering, and pagination.

        Args:
            sort_key: Field to sort by (TITLE, PRICE, BEST_SELLING, CREATED_AT, etc.)
                      PRICE is handled via client-side sort since Shopify doesn't support it natively.
            reverse: If True, sort descending (highest first)
            limit: Max products to return (1-250)
            query: Shopify filter string (e.g., "status:active", "tag:sale")
            after: Pagination cursor from previous response

        Returns:
            Dict with products list, pagination info, and metadata
        """
        # PRICE sort isn't supported natively — handle client-side
        if sort_key.upper() == "PRICE":
            return await self._products_sorted_by_price(
                reverse=reverse, limit=limit, query=query
            )

        sort_key_upper = sort_key.upper()
        if sort_key_upper not in PRODUCT_SORT_KEYS:
            sort_key_upper = "CREATED_AT"

        gql = """
        query GetProducts(
            $first: Int!, $sortKey: ProductSortKeys!, $reverse: Boolean!,
            $query: String, $after: String
        ) {
            products(first: $first, sortKey: $sortKey, reverse: $reverse,
                     query: $query, after: $after) {
                edges {
                    node {
                        id
                        title
                        handle
                        status
                        totalInventory
                        createdAt
                        updatedAt
                        priceRangeV2 {
                            minVariantPrice { amount currencyCode }
                            maxVariantPrice { amount currencyCode }
                        }
                        variants(first: 10) {
                            edges {
                                node {
                                    id title price inventoryQuantity sku
                                }
                            }
                        }
                    }
                    cursor
                }
                pageInfo { hasNextPage endCursor }
            }
        }
        """

        variables = {
            "first": min(limit, 250),
            "sortKey": sort_key_upper,
            "reverse": reverse,
            "query": query,
            "after": after,
        }

        data = await self.execute_query(gql, variables)
        result = self._format_products_response(data)
        result["sorted_by"] = sort_key_upper
        result["sort_order"] = "descending" if reverse else "ascending"
        return result

    async def _products_sorted_by_price(
        self,
        reverse: bool = True,
        limit: int = 10,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch ALL products and sort by price client-side.

        Shopify's GraphQL API doesn't support PRICE as a sort key,
        so we paginate through all products and sort ourselves.
        """
        logger.info("Fetching all products for price sorting")
        all_products = []
        cursor = None
        pages_fetched = 0

        while True:
            gql = """
            query GetProducts($first: Int!, $query: String, $after: String) {
                products(first: $first, query: $query, after: $after) {
                    edges {
                        node {
                            id title handle status totalInventory createdAt updatedAt
                            priceRangeV2 {
                                minVariantPrice { amount currencyCode }
                                maxVariantPrice { amount currencyCode }
                            }
                            variants(first: 10) {
                                edges {
                                    node { id title price inventoryQuantity sku }
                                }
                            }
                        }
                        cursor
                    }
                    pageInfo { hasNextPage endCursor }
                }
            }
            """

            variables = {"first": 250, "query": query, "after": cursor}
            data = await self.execute_query(gql, variables)
            pages_fetched += 1

            products_data = data.get("products", {})
            edges = products_data.get("edges", [])

            for edge in edges:
                node = edge["node"]
                max_price = float(
                    node.get("priceRangeV2", {})
                    .get("maxVariantPrice", {})
                    .get("amount", 0)
                )
                all_products.append({"node": node, "max_price": max_price})

            page_info = products_data.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")

            # Safety limit: don't fetch more than 20 pages (5000 products)
            if pages_fetched >= 20:
                logger.warning("Hit pagination safety limit for price sort")
                break

        logger.info(
            "Products fetched for price sort",
            total_products=len(all_products),
            pages_fetched=pages_fetched,
        )

        # Sort by price client-side
        all_products.sort(key=lambda x: x["max_price"], reverse=reverse)

        # Take requested limit
        selected = all_products[:limit]

        products = [self._format_product_node(item["node"]) for item in selected]

        return {
            "products": products,
            "total_count": len(all_products),
            "showing": len(products),
            "sorted_by": "PRICE",
            "sort_order": "descending" if reverse else "ascending",
        }

    # ── Order queries ───────────────────────────────────────────────

    async def query_orders(
        self,
        sort_key: str = "CREATED_AT",
        reverse: bool = True,
        limit: int = 10,
        query: Optional[str] = None,
        after: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query orders with sorting, filtering, and pagination.

        Args:
            sort_key: CREATED_AT, TOTAL_PRICE, ORDER_NUMBER, etc.
            reverse: If True, sort descending
            limit: Max orders to return
            query: Shopify filter string (e.g., "financial_status:paid", "created_at:>2024-01-01")
            after: Pagination cursor
        """
        sort_key_upper = sort_key.upper()
        if sort_key_upper not in ORDER_SORT_KEYS:
            sort_key_upper = "CREATED_AT"

        gql = """
        query GetOrders(
            $first: Int!, $sortKey: OrderSortKeys!, $reverse: Boolean!,
            $query: String, $after: String
        ) {
            orders(first: $first, sortKey: $sortKey, reverse: $reverse,
                   query: $query, after: $after) {
                edges {
                    node {
                        id
                        name
                        createdAt
                        totalPriceSet { shopMoney { amount currencyCode } }
                        subtotalPriceSet { shopMoney { amount currencyCode } }
                        displayFinancialStatus
                        displayFulfillmentStatus
                        lineItems(first: 10) {
                            edges {
                                node {
                                    title
                                    quantity
                                    originalTotalSet { shopMoney { amount currencyCode } }
                                }
                            }
                        }
                    }
                    cursor
                }
                pageInfo { hasNextPage endCursor }
            }
        }
        """

        variables = {
            "first": min(limit, 250),
            "sortKey": sort_key_upper,
            "reverse": reverse,
            "query": query,
            "after": after,
        }

        data = await self.execute_query(gql, variables)
        result = self._format_orders_response(data)
        result["sorted_by"] = sort_key_upper
        result["sort_order"] = "descending" if reverse else "ascending"
        return result

    # ── Customer queries ────────────────────────────────────────────

    async def query_customers(
        self,
        sort_key: str = "CREATED_AT",
        reverse: bool = True,
        limit: int = 10,
        query: Optional[str] = None,
        after: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query customers with sorting, filtering, and pagination.

        Args:
            sort_key: NAME, TOTAL_SPENT, ORDERS_COUNT, CREATED_AT, etc.
            reverse: If True, sort descending
            limit: Max customers to return
            query: Shopify filter string
            after: Pagination cursor
        """
        sort_key_upper = sort_key.upper()
        if sort_key_upper not in CUSTOMER_SORT_KEYS:
            sort_key_upper = "CREATED_AT"

        gql = """
        query GetCustomers(
            $first: Int!, $sortKey: CustomerSortKeys!, $reverse: Boolean!,
            $query: String, $after: String
        ) {
            customers(first: $first, sortKey: $sortKey, reverse: $reverse,
                      query: $query, after: $after) {
                edges {
                    node {
                        id
                        firstName
                        lastName
                        email
                        phone
                        numberOfOrders
                        amountSpent { amount currencyCode }
                        createdAt
                        updatedAt
                        tags
                        defaultAddress { city province country }
                    }
                    cursor
                }
                pageInfo { hasNextPage endCursor }
            }
        }
        """

        variables = {
            "first": min(limit, 250),
            "sortKey": sort_key_upper,
            "reverse": reverse,
            "query": query,
            "after": after,
        }

        data = await self.execute_query(gql, variables)
        result = self._format_customers_response(data)
        result["sorted_by"] = sort_key_upper
        result["sort_order"] = "descending" if reverse else "ascending"
        return result

    # ── Response formatters ─────────────────────────────────────────

    def _format_product_node(self, node: Dict) -> Dict[str, Any]:
        """Format a single product node into a clean dict."""
        variants = []
        for ve in node.get("variants", {}).get("edges", []):
            v = ve.get("node", {})
            if not v:
                continue
            variants.append({
                "id": v.get("id"),
                "title": v.get("title"),
                "price": v.get("price"),
                "inventory": v.get("inventoryQuantity"),
                "sku": v.get("sku"),
            })

        price_range = node.get("priceRangeV2", {})
        return {
            "id": node.get("id"),
            "title": node.get("title", "Unknown"),
            "handle": node.get("handle"),
            "status": node.get("status"),
            "total_inventory": node.get("totalInventory"),
            "created_at": node.get("createdAt"),
            "min_price": price_range.get("minVariantPrice", {}).get("amount"),
            "max_price": price_range.get("maxVariantPrice", {}).get("amount"),
            "currency": price_range.get("maxVariantPrice", {}).get("currencyCode"),
            "variants": variants,
        }

    def _format_products_response(self, data: Dict) -> Dict[str, Any]:
        """Format the full products GraphQL response."""
        products_data = data.get("products", {})
        edges = products_data.get("edges", [])
        page_info = products_data.get("pageInfo", {})

        products = [
            self._format_product_node(edge.get("node", {}))
            for edge in edges
            if edge.get("node")
        ]

        return {
            "products": products,
            "has_next_page": page_info.get("hasNextPage", False),
            "end_cursor": page_info.get("endCursor"),
        }

    def _format_orders_response(self, data: Dict) -> Dict[str, Any]:
        """Format the full orders GraphQL response."""
        orders_data = data.get("orders", {})
        edges = orders_data.get("edges", [])
        page_info = orders_data.get("pageInfo", {})

        orders = []
        for edge in edges:
            node = edge.get("node", {})
            if not node:
                continue
            total_price = node.get("totalPriceSet", {}).get("shopMoney", {})
            customer = node.get("customer") or {}

            line_items = []
            for li_edge in node.get("lineItems", {}).get("edges", []):
                li = li_edge.get("node", {})
                li_total = li.get("originalTotalSet", {}).get("shopMoney", {})
                line_items.append({
                    "title": li.get("title", "Unknown"),
                    "quantity": li.get("quantity", 0),
                    "total": li_total.get("amount"),
                    "currency": li_total.get("currencyCode"),
                })

            orders.append({
                "id": node.get("id"),
                "name": node.get("name"),
                "created_at": node.get("createdAt"),
                "total_price": total_price.get("amount"),
                "currency": total_price.get("currencyCode"),
                "financial_status": node.get("displayFinancialStatus"),
                "fulfillment_status": node.get("displayFulfillmentStatus"),
                "customer_name": (
                    f"{customer.get('firstName', '')} {customer.get('lastName', '')}".strip()
                ),
                "customer_email": customer.get("email"),
                "line_items": line_items,
            })

        return {
            "orders": orders,
            "has_next_page": page_info.get("hasNextPage", False),
            "end_cursor": page_info.get("endCursor"),
        }

    def _format_customers_response(self, data: Dict) -> Dict[str, Any]:
        """Format the full customers GraphQL response."""
        customers_data = data.get("customers", {})
        edges = customers_data.get("edges", [])
        page_info = customers_data.get("pageInfo", {})

        customers = []
        for edge in edges:
            node = edge.get("node", {})
            if not node:
                continue
            amount_spent = node.get("amountSpent", {})
            address = node.get("defaultAddress") or {}

            location_parts = [
                address.get("city", ""),
                address.get("province", ""),
                address.get("country", ""),
            ]
            location = ", ".join(p for p in location_parts if p)

            customers.append({
                "id": node["id"],
                "name": (
                    f"{node.get('firstName', '')} {node.get('lastName', '')}".strip()
                ),
                "email": node.get("email"),
                "phone": node.get("phone"),
                "orders_count": node.get("numberOfOrders"),
                "total_spent": amount_spent.get("amount"),
                "currency": amount_spent.get("currencyCode"),
                "created_at": node.get("createdAt"),
                "tags": node.get("tags", []),
                "location": location,
            })

        return {
            "customers": customers,
            "has_next_page": page_info.get("hasNextPage", False),
            "end_cursor": page_info.get("endCursor"),
        }
