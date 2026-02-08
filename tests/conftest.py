"""Pytest configuration and shared fixtures for Shopify Analytics Agent tests."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from src.database.operations import DatabaseOperations
from src.database.models import User, ShopifyStore


@pytest.fixture
def db_ops():
    """Create an in-memory SQLite DatabaseOperations instance for testing.

    Initializes all database tables and yields the instance.
    Cleans up after tests complete.
    """
    # Create in-memory SQLite database
    db = DatabaseOperations("sqlite:///:memory:")

    # Initialize database tables
    db.init_database()

    yield db

    # Cleanup happens automatically with in-memory database


@pytest.fixture
def sample_user(db_ops):
    """Create a test user in the database.

    Returns a User object created with test data.
    """
    user = db_ops.get_or_create_user(
        telegram_user_id=12345,
        telegram_username="testuser",
        first_name="Test"
    )
    return user


@pytest.fixture
def sample_store(db_ops, sample_user):
    """Create a test Shopify store for the sample user.

    Returns a ShopifyStore object.
    """
    store = db_ops.add_store(
        user_id=sample_user.id,
        shop_domain="test-shop.myshopify.com",
        access_token="test_token_12345"
    )
    return store


@pytest.fixture
def sample_orders():
    """Return a list of sample Shopify order dictionaries.

    Provides realistic order data for testing analytics processing.
    """
    return [
        {
            "id": 1,
            "total_price": "150.00",
            "created_at": "2026-02-06T10:00:00Z",
            "currency": "USD",
            "line_items": [
                {
                    "title": "Widget A",
                    "quantity": 2,
                    "price": "50.00"
                },
                {
                    "title": "Widget B",
                    "quantity": 1,
                    "price": "50.00"
                }
            ]
        },
        {
            "id": 2,
            "total_price": "75.50",
            "created_at": "2026-02-06T14:00:00Z",
            "currency": "USD",
            "line_items": [
                {
                    "title": "Widget A",
                    "quantity": 1,
                    "price": "50.00"
                },
                {
                    "title": "Gadget X",
                    "quantity": 1,
                    "price": "25.50"
                }
            ]
        },
        {
            "id": 3,
            "total_price": "200.00",
            "created_at": "2026-02-06T18:00:00Z",
            "currency": "USD",
            "line_items": [
                {
                    "title": "Premium Widget",
                    "quantity": 1,
                    "price": "200.00"
                }
            ]
        }
    ]


@pytest.fixture
def mock_graphql_client():
    """Return a mock ShopifyGraphQLClient for testing without external dependencies.

    Uses AsyncMock to support async methods.
    """
    mock = AsyncMock()
    mock.query_products = AsyncMock(return_value={"products": []})
    mock.query_orders = AsyncMock(return_value={"orders": []})
    mock.query_customers = AsyncMock(return_value={"customers": []})
    mock.execute_raw_query = AsyncMock(return_value={})
    return mock
