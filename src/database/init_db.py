"""Database initialization for Shopify Analytics Agent."""

import logging
from typing import Optional

from src.database.models import Base
from src.database.operations import DatabaseOperations

logger = logging.getLogger(__name__)


def init_database(database_url: Optional[str] = None) -> DatabaseOperations:
    """
    Initialize database and create all tables.

    Args:
        database_url: Optional SQLAlchemy database URL. If not provided,
                     will use settings.database.url from config.

    Returns:
        DatabaseOperations instance for database access.

    Raises:
        Exception: If database initialization fails.
    """
    if database_url is None:
        from src.config.settings import settings
        database_url = settings.database.url

    logger.info(f"Initializing database at {database_url}")

    db_ops = DatabaseOperations(database_url)

    try:
        db_ops.init_database()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    return db_ops


def check_database(database_url: Optional[str] = None) -> bool:
    """
    Verify that all required database tables exist.

    Args:
        database_url: Optional SQLAlchemy database URL. If not provided,
                     will use settings.database.url from config.

    Returns:
        True if all tables exist, False otherwise.

    Raises:
        Exception: If database connection fails.
    """
    if database_url is None:
        from src.config.settings import settings
        database_url = settings.database.url

    logger.info(f"Checking database at {database_url}")

    db_ops = DatabaseOperations(database_url)

    try:
        exists = db_ops.check_database()

        if exists:
            logger.info("All required database tables exist")
        else:
            logger.warning("Some required database tables are missing")

        return exists
    except Exception as e:
        logger.error(f"Failed to check database: {e}")
        raise


def reset_database(database_url: Optional[str] = None) -> None:
    """
    Drop all tables and recreate them. WARNING: This will delete all data.

    Args:
        database_url: Optional SQLAlchemy database URL. If not provided,
                     will use settings.database.url from config.

    Raises:
        Exception: If database reset fails.
    """
    if database_url is None:
        from src.config.settings import settings
        database_url = settings.database.url

    logger.warning("Resetting database - all data will be deleted!")

    db_ops = DatabaseOperations(database_url)

    try:
        Base.metadata.drop_all(db_ops.engine)
        logger.info("All database tables dropped")

        db_ops.init_database()
        logger.info("Database tables recreated successfully")
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise


def get_database_ops(database_url: Optional[str] = None) -> DatabaseOperations:
    """
    Get a DatabaseOperations instance without initializing tables.

    Args:
        database_url: Optional SQLAlchemy database URL. If not provided,
                     will use settings.database.url from config.

    Returns:
        DatabaseOperations instance for database access.
    """
    if database_url is None:
        from src.config.settings import settings
        database_url = settings.database.url

    return DatabaseOperations(database_url)


if __name__ == "__main__":
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        reset_database()
    else:
        init_database()
