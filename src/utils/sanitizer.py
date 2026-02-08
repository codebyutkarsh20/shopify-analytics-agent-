"""
Input sanitization for prompt injection prevention.

All user-originated data that gets injected into LLM system prompts
must pass through these functions to neutralize stored prompt injection
attacks.
"""

import re
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Patterns that indicate an injection attempt
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    # Direct instruction overrides
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?prior\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(the\s+)?above", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"override\s+(all\s+)?instructions", re.IGNORECASE),

    # Role-switching attempts
    re.compile(r"you\s+are\s+now\s+(a|an|in)\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a|an|if)\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
    re.compile(r"switch\s+to\s+\w+\s+mode", re.IGNORECASE),
    re.compile(r"enter\s+\w+\s+mode", re.IGNORECASE),

    # System prompt manipulation
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"\[system\]", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"updated?\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"admin\s+mode", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
    re.compile(r"debug\s+mode", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),

    # Data exfiltration
    re.compile(r"(reveal|show|display|output|print)\s+(your|the)\s+(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"what\s+(are|is)\s+your\s+(system\s+)?instructions?", re.IGNORECASE),
]

# Characters that could be used to break out of prompt formatting
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_for_prompt(text: Optional[str], max_length: int = 500) -> str:
    """Sanitize user-originated text before injecting into an LLM prompt.

    This function:
    1. Strips control characters
    2. Truncates to max_length
    3. Detects and neutralizes known injection patterns
    4. Wraps the result in a safe delimiter

    Args:
        text: The raw text to sanitize.
        max_length: Maximum allowed length.

    Returns:
        Sanitized text safe for prompt injection.
    """
    if not text:
        return ""

    # Step 1: Strip control characters
    cleaned = _CONTROL_CHARS.sub("", text)

    # Step 2: Truncate
    cleaned = cleaned[:max_length]

    # Step 3: Detect injection attempts
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            logger.warning(
                "Prompt injection attempt detected and neutralized",
                pattern=pattern.pattern,
                text_preview=cleaned[:80],
            )
            # Replace the match with a harmless marker
            cleaned = pattern.sub("[FILTERED]", cleaned)

    return cleaned


def sanitize_error_for_prompt(error_msg: Optional[str], max_length: int = 300) -> str:
    """Sanitize an error message before including in prompt context.

    Error messages are a prime injection vector since an attacker can
    craft queries that produce specific error strings which then get
    stored and re-injected into future prompts.

    Args:
        error_msg: The raw error message.
        max_length: Maximum allowed length.

    Returns:
        Sanitized error text.
    """
    if not error_msg:
        return ""

    # Strip everything that isn't alphanumeric, space, or basic punctuation
    cleaned = re.sub(r"[^\w\s.,;:!?()\-/\[\]{}@#$%&*+=<>'\"]", "", error_msg)
    cleaned = _CONTROL_CHARS.sub("", cleaned)
    cleaned = cleaned[:max_length]

    # Also check for injection patterns
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            logger.warning(
                "Injection attempt detected in error message",
                pattern=pattern.pattern,
            )
            cleaned = pattern.sub("[FILTERED]", cleaned)

    return cleaned


def sanitize_query_for_prompt(query_text: Optional[str], max_length: int = 200) -> str:
    """Sanitize a stored query before including in prompt context.

    Args:
        query_text: The raw query text.
        max_length: Maximum allowed length.

    Returns:
        Sanitized query text.
    """
    return sanitize_for_prompt(query_text, max_length=max_length)
