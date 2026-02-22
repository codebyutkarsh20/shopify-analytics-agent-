"""
Structured logging utilities using structlog.

This module provides structured logging configuration and a factory function
for creating loggers across the application. Includes automatic redaction
of sensitive values (API keys, tokens, passwords) and request correlation IDs.
"""

import contextvars
import logging
import re
import sys
import uuid
from pathlib import Path

import structlog

# Correlation ID for tracking requests across the processing pipeline
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


# Patterns that indicate sensitive values
_SENSITIVE_PATTERNS = [
    re.compile(r"shpat_[A-Za-z0-9]+"),       # Shopify access tokens
    re.compile(r"sk-[A-Za-z0-9]+"),           # OpenAI / Anthropic API keys
    re.compile(r"xoxb-[A-Za-z0-9\-]+"),       # Slack tokens
    re.compile(r"gAAAAA[A-Za-z0-9_\-=]+"),    # Fernet encrypted tokens
]

_SENSITIVE_KEYS = {
    "access_token", "api_key", "token", "secret", "password",
    "authorization", "credentials", "encryption_key",
}


def _redact_value(value):
    """Redact a single value if it matches sensitive patterns."""
    if not isinstance(value, str):
        return value
    for pattern in _SENSITIVE_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value


def _redact_processor(logger, method_name, event_dict):
    """Structlog processor that redacts sensitive values from log events."""
    for key in list(event_dict.keys()):
        # Redact known sensitive keys entirely
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = "[REDACTED]"
            continue
        # Redact pattern matches in string values
        if isinstance(event_dict[key], str):
            event_dict[key] = _redact_value(event_dict[key])
    return event_dict


def _correlation_id_processor(logger, method_name, event_dict):
    """Structlog processor that injects the current correlation ID into log events."""
    cid = correlation_id_var.get("")
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def new_correlation_id() -> str:
    """Generate a new correlation ID and set it in the current context.

    Call this at the start of each incoming message/request to create a
    unique ID that will be attached to every log line produced while
    processing that request.

    Returns:
        The generated correlation ID (8-char hex).
    """
    cid = uuid.uuid4().hex[:8]
    correlation_id_var.set(cid)
    return cid


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

    # Setup structlog with redaction processor
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
            _correlation_id_processor,  # Inject correlation ID
            _redact_processor,  # Redact sensitive values before rendering
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

        # Set restrictive file permissions (owner read/write only)
        import os
        try:
            os.chmod(log_file, 0o600)
        except OSError:
            pass  # May fail if file doesn't exist yet; handler will create it

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
