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
    # ── ShopifyQL analytics templates (server-side, accurate across ALL data) ──

    {
        "intent_category": "revenue_total",
        "intent_description": "Total revenue, net sales, and order count via ShopifyQL (server-side, no limits)",
        "tool_name": "shopifyql_analytics",
        "tool_parameters": json.dumps({
            "query": "FROM sales SHOW total_sales, net_sales, returns, discounts, orders SINCE this_month",
        }),
        "confidence": 0.9,
        "example_queries": json.dumps([
            "total revenue",
            "how much revenue",
            "total sales",
            "how many orders",
            "total order count",
            "net revenue",
            "total refunds",
            "monthly revenue",
            "this month sales",
        ]),
    },
    {
        "intent_category": "revenue_trend",
        "intent_description": "Revenue growth trend over time via ShopifyQL",
        "tool_name": "shopifyql_analytics",
        "tool_parameters": json.dumps({
            "query": "FROM sales SHOW net_sales, orders GROUP BY month SINCE -6m",
        }),
        "confidence": 0.9,
        "example_queries": json.dumps([
            "are we growing",
            "revenue trend",
            "sales trend",
            "monthly growth",
            "how are we doing",
            "month over month",
            "growth rate",
        ]),
    },
    {
        "intent_category": "products_ranking",
        "intent_description": "Top products by revenue via ShopifyQL (accurate across all data)",
        "tool_name": "shopifyql_analytics",
        "tool_parameters": json.dumps({
            "query": "FROM sales SHOW net_sales, ordered_quantity, orders GROUP BY product_title ORDER BY net_sales DESC LIMIT 10",
        }),
        "confidence": 0.9,
        "example_queries": json.dumps([
            "top products",
            "best sellers",
            "most popular products",
            "top selling items",
            "best performing products",
            "product ranking",
        ]),
    },
    {
        "intent_category": "revenue_leakage",
        "intent_description": "Money leakage analysis — returns, discounts, and refunds via ShopifyQL",
        "tool_name": "shopifyql_analytics",
        "tool_parameters": json.dumps({
            "query": "FROM sales SHOW returns, discounts, net_sales GROUP BY product_title ORDER BY returns DESC LIMIT 10",
        }),
        "confidence": 0.85,
        "example_queries": json.dumps([
            "where are we leaking money",
            "refund analysis",
            "returns by product",
            "discount impact",
            "money leakage",
            "most returned products",
        ]),
    },
    {
        "intent_category": "revenue_by_region",
        "intent_description": "Revenue breakdown by country/region via ShopifyQL",
        "tool_name": "shopifyql_analytics",
        "tool_parameters": json.dumps({
            "query": "FROM sales SHOW net_sales, orders GROUP BY billing_country ORDER BY net_sales DESC",
        }),
        "confidence": 0.85,
        "example_queries": json.dumps([
            "revenue by country",
            "sales by region",
            "where do our customers come from",
            "geographic breakdown",
            "top countries",
        ]),
    },
    {
        "intent_category": "daily_performance",
        "intent_description": "Daily order and revenue trends via ShopifyQL",
        "tool_name": "shopifyql_analytics",
        "tool_parameters": json.dumps({
            "query": "FROM sales SHOW total_sales, net_sales, orders GROUP BY day SINCE -7d",
        }),
        "confidence": 0.85,
        "example_queries": json.dumps([
            "daily revenue",
            "daily orders",
            "last 7 days",
            "this week performance",
            "daily trend",
            "what should I fix today",
        ]),
    },
    {
        "intent_category": "discount_effectiveness",
        "intent_description": "Discount code performance and effectiveness via ShopifyQL",
        "tool_name": "shopifyql_analytics",
        "tool_parameters": json.dumps({
            "query": "FROM sales SHOW net_sales, discounts, orders GROUP BY discount_code ORDER BY net_sales DESC LIMIT 10",
        }),
        "confidence": 0.8,
        "example_queries": json.dumps([
            "discount code performance",
            "which discounts work",
            "coupon analysis",
            "promo code results",
        ]),
    },
    {
        "intent_category": "period_comparison",
        "intent_description": "Compare current period vs previous period via ShopifyQL",
        "tool_name": "shopifyql_analytics",
        "tool_parameters": json.dumps({
            "query": "FROM sales SHOW total_sales, orders SINCE -30d UNTIL today COMPARE TO previous_period",
        }),
        "confidence": 0.85,
        "example_queries": json.dumps([
            "compare to last month",
            "how are we doing vs last period",
            "period comparison",
            "are we improving",
            "performance vs previous",
        ]),
    },
    {
        "intent_category": "product_performance",
        "intent_description": "Product performance with sessions and conversion via ShopifyQL",
        "tool_name": "shopifyql_analytics",
        "tool_parameters": json.dumps({
            "query": "FROM products SHOW net_sales, view_sessions, view_cart_sessions GROUP BY product_title SINCE -7d ORDER BY net_sales DESC LIMIT 20",
        }),
        "confidence": 0.8,
        "example_queries": json.dumps([
            "product conversion",
            "product views vs sales",
            "which products convert",
            "product funnel",
        ]),
    },
    {
        "intent_category": "customers_top_spenders",
        "intent_description": "Top customers by total spending",
        "tool_name": "shopify_graphql",
        "tool_parameters": json.dumps({
            "query": "{ customers(first: 250, sortKey: CREATED_AT, reverse: true) { edges { node { id firstName lastName email numberOfOrders amountSpent { amount currencyCode } } } } }",
        }),
        "confidence": 0.8,
        "example_queries": json.dumps([
            "top customers",
            "best customers",
            "biggest spenders",
            "VIP customers",
        ]),
    },

    # ── Record-level templates (for browsing individual items) ──

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
        "intent_category": "customers_by_orders",
        "intent_description": "Customers with most orders (repeat buyers)",
        "tool_name": "shopify_graphql",
        "tool_parameters": json.dumps({
            "query": "{ customers(first: 250, sortKey: CREATED_AT, reverse: true) { edges { node { id firstName lastName email numberOfOrders amountSpent { amount currencyCode } } } } }",
        }),
        "confidence": 0.8,
        "example_queries": json.dumps([
            "repeat customers",
            "most orders",
            "loyal customers",
            "frequent buyers",
        ]),
    },

    # ── Utility templates ──

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


def seed_templates(db_ops, vector_store=None) -> None:
    """Insert seed templates if they don't already exist.

    Iterates through SEED_TEMPLATES and creates database entries for any
    that don't already exist (checked by intent_category + tool_name combo).
    If a ``vector_store`` is provided, embeddings are computed for every
    newly created template.

    Args:
        db_ops: DatabaseOperations instance
        vector_store: Optional EmbeddingStore for embedding new seeds.
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

            # Embed the newly created seed template
            if vector_store:
                try:
                    created = db_ops.find_template(
                        intent_category=tpl["intent_category"],
                        tool_name=tpl["tool_name"],
                    )
                    if created:
                        from src.learning.vector_store import EmbeddingStore

                        embed_text = EmbeddingStore._build_template_embed_text(created)
                        vector_store.store_embedding(
                            entity_type="template",
                            entity_id=created.id,
                            text=embed_text,
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to embed seed template %s: %s",
                        tpl["intent_category"],
                        exc,
                    )

    if count:
        logger.info(
            "Seeded query templates",
            count=count,
        )
    else:
        logger.info("No new seed templates needed")

    # ── Migration: deprecate old templates superseded by ShopifyQL ──
    # If a ShopifyQL template exists for an intent, lower confidence of
    # any competing shopify_analytics / shopify_graphql template so the
    # context builder prefers the ShopifyQL version.
    _deprecate_old_analytics_templates(db_ops)


# Intent categories that now have ShopifyQL versions
_SHOPIFYQL_INTENTS = {
    tpl["intent_category"]
    for tpl in SEED_TEMPLATES
    if tpl["tool_name"] == "shopifyql_analytics"
}

# Old tool names that should be deprioritised for analytics intents
_OLD_ANALYTICS_TOOLS = {"shopify_analytics", "shopify_graphql"}


def _deprecate_old_analytics_templates(db_ops) -> None:
    """Lower confidence of old templates superseded by ShopifyQL.

    Two-pronged approach:
      1. For every intent that now has a ``shopifyql_analytics`` seed,
         deprecate old templates with the same intent but old tool names.
      2. Deprecate ALL templates (regardless of intent) whose
         ``tool_parameters`` contain ``orders_aggregate`` — these are
         learned templates from before ShopifyQL was available and they
         force the LLM to use the 10,000-order-capped fallback.
    """
    from src.database.models import QueryTemplate
    from sqlalchemy import update, and_, or_

    deprecated = 0
    session = db_ops.get_session()
    try:
        # Strategy 1: Deprecate seed-level templates by intent
        for intent in _SHOPIFYQL_INTENTS:
            for old_tool in _OLD_ANALYTICS_TOOLS:
                stmt = (
                    update(QueryTemplate)
                    .where(
                        and_(
                            QueryTemplate.intent_category == intent,
                            QueryTemplate.tool_name == old_tool,
                            QueryTemplate.confidence > 0.2,
                        )
                    )
                    .values(confidence=0.1)
                )
                result = session.execute(stmt)
                if result.rowcount > 0:
                    deprecated += result.rowcount
                    logger.info(
                        "Deprecated old seed template",
                        intent_category=intent,
                        old_tool=old_tool,
                        rows=result.rowcount,
                    )

        # Strategy 2: Deprecate ALL learned templates that use orders_aggregate
        # These were recorded before ShopifyQL existed and steer the LLM away
        # from the correct tool.
        stmt = (
            update(QueryTemplate)
            .where(
                and_(
                    QueryTemplate.tool_parameters.contains("orders_aggregate"),
                    QueryTemplate.tool_name != "shopifyql_analytics",
                    QueryTemplate.confidence > 0.2,
                )
            )
            .values(confidence=0.1)
        )
        result = session.execute(stmt)
        if result.rowcount > 0:
            deprecated += result.rowcount
            logger.info(
                "Deprecated learned orders_aggregate templates",
                rows=result.rowcount,
            )

        session.commit()
    except Exception as exc:
        session.rollback()
        logger.warning("Failed to deprecate old templates: %s", exc)
    finally:
        session.close()

    if deprecated:
        logger.info("Deprecated old analytics templates", count=deprecated)
