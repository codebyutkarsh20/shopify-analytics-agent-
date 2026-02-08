"""
Encryption utilities for securing sensitive data at rest.

Uses Fernet symmetric encryption (AES-128-CBC with HMAC-SHA256)
from the `cryptography` library. The encryption key is loaded from
the ENCRYPTION_KEY environment variable; if absent, a new key is
generated and the user is warned to persist it.
"""

import os
import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------

def _derive_key(secret: str) -> bytes:
    """Derive a 32-byte Fernet-compatible key from an arbitrary secret string.

    Fernet requires a URL-safe base64-encoded 32-byte key.  We use SHA-256
    to hash any user-supplied secret into exactly 32 bytes, then base64-encode
    the result.

    Args:
        secret: An arbitrary secret string (e.g. from ENCRYPTION_KEY env var).

    Returns:
        A 32-byte base64-encoded key suitable for Fernet.
    """
    digest = hashlib.sha256(secret.encode()).digest()
    return base64.urlsafe_b64encode(digest)


def _get_fernet() -> Fernet:
    """Return a Fernet instance using the configured encryption key.

    Reads ENCRYPTION_KEY from the environment.  If the variable is missing
    a RuntimeError is raised — the caller should ensure the key is set
    before the application encrypts anything.

    Returns:
        A Fernet cipher instance.

    Raises:
        RuntimeError: If ENCRYPTION_KEY is not set.
    """
    key_raw = os.getenv("ENCRYPTION_KEY", "")
    if not key_raw:
        raise RuntimeError(
            "ENCRYPTION_KEY is not set. Add a strong random string to your "
            ".env file.  Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(_derive_key(key_raw))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encrypt_token(plaintext: str) -> str:
    """Encrypt a sensitive token for database storage.

    Args:
        plaintext: The token to encrypt.

    Returns:
        A base64-encoded ciphertext string safe for storage in a VARCHAR column.
    """
    if not plaintext:
        return plaintext
    f = _get_fernet()
    return f.encrypt(plaintext.encode()).decode()


def decrypt_token(ciphertext: str) -> str:
    """Decrypt a stored token.

    If the ciphertext is not valid Fernet, it is assumed to be a
    legacy plaintext token and returned as-is. This provides backward
    compatibility during the migration from plaintext to encrypted tokens.

    Args:
        ciphertext: The encrypted token string (or legacy plaintext).

    Returns:
        The decrypted plaintext token.
    """
    if not ciphertext:
        return ciphertext
    try:
        f = _get_fernet()
        return f.decrypt(ciphertext.encode()).decode()
    except (InvalidToken, Exception):
        # Not encrypted — legacy plaintext token.  Return as-is so the
        # migration path works transparently.
        logger.debug("Token appears to be legacy plaintext; returning as-is")
        return ciphertext


def is_encrypted(value: str) -> bool:
    """Check whether a string looks like Fernet ciphertext.

    Fernet tokens always start with 'gAAAAA' after base64 encoding.

    Args:
        value: The string to check.

    Returns:
        True if the value appears to be Fernet-encrypted.
    """
    if not value:
        return False
    return value.startswith("gAAAAA")
