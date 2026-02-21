#!/usr/bin/env python3
"""One-time script to backfill embeddings for all existing templates.

Usage (from project root):
    python -m scripts.backfill_embeddings

Safe to run repeatedly — skips templates that already have embeddings.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logger import setup_logging, get_logger
from src.database.init_db import init_database
from src.learning.vector_store import EmbeddingStore

setup_logging(level="INFO")
logger = get_logger(__name__)


def main():
    logger.info("Starting embedding backfill...")

    # Initialise database (creates tables if needed, runs migrations)
    db_ops = init_database()

    # Initialise vector store with eager model loading
    vector_store = EmbeddingStore(db_ops, lazy_load=False)

    if not vector_store.is_available:
        logger.error(
            "Embedding model could not be loaded. "
            "Install sentence-transformers: pip install sentence-transformers"
        )
        sys.exit(1)

    count = vector_store.backfill_templates()
    logger.info("Backfill complete: %d templates embedded", count)


if __name__ == "__main__":
    main()
