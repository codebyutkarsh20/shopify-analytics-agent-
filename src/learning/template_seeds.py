"""Seed templates for common Shopify analytics queries.

Provides initial templates for frequently-asked question patterns.
These seed templates give the system a starting point without relying
solely on user interaction data.
"""

import json

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Seed templates for common Shopify analytics queries
SEED_TEMPLATES = [
    {
        "intent_category": "products_ranking",
        "intent_description": "Top selling products ranked by sales",
        "tool_name": "shopify_analytics",
        "tool_parameters": json.dumps({
            "resource": "products",
            "sort_key": "BEST_SELLING",
            "reverse": True,
            "limit": 10,
        }),
        "confidence": 0.8,
        "example_queries": json.dumps([
            "top products",
            "best sellers",
            "most popular products",
            "top selling items",
        ]),
    },
    {
        "intent_category": "orders_recent",
        "intent_description": "Recent orders sorted by date",
        "tool_name": "shopify_analytics",
        "tool_parameters": json.dumps({
            "resource": "orders",
            "sort_key": "CREATED_AT",
            "reverse": True,
            "limit": 20,
        }),
        "confidence": 0.8,
        "example_queries": json.dumps([
            "recent orders",
            "latest orders",
            "orders today",
            "show me orders",
        ]),
    },
    {
        "intent_category": "orders_by_value",
        "intent_description": "Orders sorted by total price",
        "tool_name": "shopify_analytics",
        "tool_parameters": json.dumps({
            "resource": "orders",
            "sort_key": "TOTAL_PRICE",
            "reverse": True,
            "limit": 10,
        }),
        "confidence": 0.8,
        "example_queries": json.dumps([
            "biggest orders",
            "highest value orders",
            "top orders by revenue",
        ]),
    },
    {
        "intent_category": "customers_top_spenders",
        "intent_description": "Top customers by total spending",
        "tool_name": "shopify_analytics",
        "tool_parameters": json.dumps({
            "resource": "customers",
            "sort_key": "TOTAL_SPENT",
            "reverse": True,
            "limit": 10,
        }),
        "confidence": 0.8,
        "example_queries": json.dumps([
            "top customers",
            "best customers",
            "biggest spenders",
            "VIP customers",
        ]),
    },
    {
        "intent_category": "customers_recent",
        "intent_description": "Recently acquired customers",
        "tool_name": "shopify_analytics",
        "tool_parameters": json.dumps({
            "resource": "customers",
            "sort_key": "CREATED_AT",
            "reverse": True,
            "limit": 20,
        }),
        "confidence": 0.75,
        "example_queries": json.dumps([
            "new customers",
            "recent customers",
            "latest customers",
        ]),
    },
    {
        "intent_category": "shop_info",
        "intent_description": "Basic store information and settings",
        "tool_name": "shopify_graphql",
        "tool_parameters": json.dumps({
            "query": "{ shop { name email myshopifyDomain plan { displayName } "
                    "currencyCode timezoneAbbreviation } }"
        }),
        "confidence": 0.9,
        "example_queries": json.dumps([
            "shop info",
            "store details",
            "my store",
            "store name",
            "what plan am I on",
        ]),
    },
    {
        "intent_category": "products_inventory",
        "intent_description": "Products sorted by inventory level",
        "tool_name": "shopify_analytics",
        "tool_parameters": json.dumps({
            "resource": "products",
            "sort_key": "INVENTORY_TOTAL",
            "reverse": False,
            "limit": 10,
        }),
        "confidence": 0.75,
        "example_queries": json.dumps([
            "low stock",
            "inventory check",
            "products running low",
            "out of stock",
        ]),
    },
    {
        "intent_category": "products_new",
        "intent_description": "Recently added products",
        "tool_name": "shopify_analytics",
        "tool_parameters": json.dumps({
            "resource": "products",
            "sort_key": "CREATED_AT",
            "reverse": True,
            "limit": 10,
        }),
        "confidence": 0.8,
        "example_queries": json.dumps([
            "new products",
            "recently added",
            "latest products",
        ]),
    },
    {
        "intent_category": "customers_by_orders",
        "intent_description": "Customers with most orders (repeat buyers)",
        "tool_name": "shopify_analytics",
        "tool_parameters": json.dumps({
            "resource": "customers",
            "sort_key": "ORDERS_COUNT",
            "reverse": True,
            "limit": 10,
        }),
        "confidence": 0.8,
        "example_queries": json.dumps([
            "repeat customers",
            "most orders",
            "loyal customers",
            "frequent buyers",
        ]),
    },
    {
        "intent_category": "products_by_price",
        "intent_description": "Products sorted by price",
        "tool_name": "shopify_analytics",
        "tool_parameters": json.dumps({
            "resource": "products",
            "sort_key": "PRICE",
            "reverse": True,
            "limit": 10,
        }),
        "confidence": 0.75,
        "example_queries": json.dumps([
            "most expensive products",
            "cheapest products",
            "products by price",
        ]),
    },
]


def seed_templates(db_ops) -> None:
    """Insert seed templates if they don't already exist.

    Iterates through SEED_TEMPLATES and creates database entries for any
    that don't already exist (checked by intent_category + tool_name combo).

    Args:
        db_ops: DatabaseOperations instance
    """
    logger.info("Starting template seeding")

    count = 0
    for tpl in SEED_TEMPLATES:
        existing = db_ops.find_template(
            intent_category=tpl["intent_category"],
            tool_name=tpl["tool_name"],
        )

        if not existing:
            logger.debug(
                "Creating seed template",
                intent_category=tpl["intent_category"],
                tool_name=tpl["tool_name"],
            )
            db_ops.create_template(
                intent_category=tpl["intent_category"],
                intent_description=tpl["intent_description"],
                tool_name=tpl["tool_name"],
                tool_parameters=tpl["tool_parameters"],
                confidence=tpl["confidence"],
                example_queries=tpl["example_queries"],
            )
            count += 1

    if count:
        logger.info(
            "Seeded query templates",
            count=count,
        )
    else:
        logger.info("No new seed templates needed")
