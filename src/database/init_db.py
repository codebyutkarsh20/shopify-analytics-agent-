"""Database initialization for Shopify Analytics Agent."""

import logging
from typing import Optional

from sqlalchemy import text, inspect
from src.database.models import Base
from src.database.operations import DatabaseOperations

logger = logging.getLogger(__name__)


def _get_column_type_sql(column) -> str:
    """Get SQLite-compatible column type from a SQLAlchemy column."""
    try:
        col_type = str(column.type)
    except Exception:
        col_type = "TEXT"

    type_map = {
        "VARCHAR": "TEXT",
        "STRING": "TEXT",
        "INTEGER": "INTEGER",
        "FLOAT": "REAL",
        "BOOLEAN": "INTEGER",
        "DATETIME": "TIMESTAMP",
        "TEXT": "TEXT",
    }
    upper = col_type.upper().split("(")[0]
    return type_map.get(upper, "TEXT")


def _migrate_missing_columns(engine) -> None:
    """Add missing columns to existing tables.

    SQLAlchemy's create_all() only creates NEW tables — it does NOT add
    columns to existing tables.  This function compares every model's
    columns against the live database and issues ALTER TABLE ADD COLUMN
    for anything that's missing.
    """
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    migrations_applied = 0

    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            continue

        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}

        for col in table.columns:
            if col.name in existing_columns:
                continue

            col_type = _get_column_type_sql(col)
            nullable = col.nullable if col.nullable is not None else True
            default = "NULL" if nullable else "''"

            sql = f'ALTER TABLE "{table_name}" ADD COLUMN "{col.name}" {col_type} DEFAULT {default}'
            try:
                with engine.connect() as conn:
                    conn.execute(text(sql))
                    conn.commit()
                logger.info(f"Migration: added column {table_name}.{col.name} ({col_type})")
                migrations_applied += 1
            except Exception as e:
                logger.warning(f"Failed to add column {table_name}.{col.name}: {e}")

    if migrations_applied:
        logger.info(f"Schema migration complete: {migrations_applied} column(s) added")
    else:
        logger.info("Schema migration check: no changes needed")


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

    # Migration: rename mcp_tool_usage → tool_usage if needed
    try:
        inspector = inspect(db_ops.engine)
        existing_tables = inspector.get_table_names()
        if "mcp_tool_usage" in existing_tables and "tool_usage" not in existing_tables:
            with db_ops.engine.connect() as conn:
                conn.execute(text("ALTER TABLE mcp_tool_usage RENAME TO tool_usage"))
                conn.commit()
            logger.info("Migrated table: mcp_tool_usage → tool_usage")
    except Exception as e:
        logger.warning("Table migration check failed (safe to ignore on fresh DB)")

    # Migration: add any missing columns to existing tables
    try:
        _migrate_missing_columns(db_ops.engine)
    except Exception as e:
        logger.warning(f"Column migration check failed: {e}")

    # Seed query templates on first run
    try:
        from src.learning.template_seeds import seed_templates
        seed_templates(db_ops)
    except Exception as e:
        logger.warning("Template seeding failed")

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
