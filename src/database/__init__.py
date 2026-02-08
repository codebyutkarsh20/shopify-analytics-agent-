"""Database module for Shopify Analytics Agent."""

from src.database.models import (
    Base,
    User,
    ShopifyStore,
    Conversation,
    QueryPattern,
    UserPreference,
    MCPToolUsage,
    AnalyticsCache,
)
from src.database.operations import DatabaseOperations
from src.database.init_db import (
    init_database,
    check_database,
    reset_database,
    get_database_ops,
)

__all__ = [
    "Base",
    "User",
    "ShopifyStore",
    "Conversation",
    "QueryPattern",
    "UserPreference",
    "MCPToolUsage",
    "AnalyticsCache",
    "DatabaseOperations",
    "init_database",
    "check_database",
    "reset_database",
    "get_database_ops",
]
