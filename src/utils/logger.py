"""
Structured logging utilities using structlog.

This module provides structured logging configuration and a factory function
for creating loggers across the application.
"""

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """
    Configure structured logging with console and optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If provided, logs will be written to file.

    Raises:
        ValueError: If level is not a valid logging level.
    """
    level_upper = level.upper()
    if level_upper not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(f"Invalid logging level: {level}")

    # Configure standard library logging (required by structlog)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level_upper,
    )

    # Setup structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure file handler if log_file is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level_upper)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        A structlog logger instance with the given name.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("message", key="value")
    """
    return structlog.get_logger(name)
