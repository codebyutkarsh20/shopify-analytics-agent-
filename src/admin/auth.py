"""Authentication utilities: JWT (httpOnly cookies) + bcrypt password hashing."""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from src.config.settings import settings


_ALGORITHM = "HS256"
_TOKEN_LIFETIME_HOURS = 24


# ---------------------------------------------------------------------------
# Password hashing (raw bcrypt)
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Check *plain* against a bcrypt *hashed* value."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _get_secret() -> str:
    secret = settings.admin_dashboard.jwt_secret
    if not secret:
        # Fallback — NOT safe for production; warn at startup instead
        secret = "insecure-default-jwt-secret"
    return secret


def create_access_token(username: str) -> str:
    """Create a signed JWT containing the admin username."""
    payload = {
        "sub": username,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=_TOKEN_LIFETIME_HOURS),
    }
    return jwt.encode(payload, _get_secret(), algorithm=_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    """Decode and validate a JWT. Returns payload dict or ``None``."""
    try:
        return jwt.decode(token, _get_secret(), algorithms=[_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None
