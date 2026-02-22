"""FastAPI dependencies for the admin dashboard."""

from fastapi import Cookie, HTTPException, status

from src.admin.auth import decode_access_token


async def get_current_admin(admin_token: str | None = Cookie(default=None)) -> str:
    """Extract and validate the admin JWT from the httpOnly cookie.

    Returns the admin username on success; raises 401 otherwise.
    """
    if not admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    payload = decode_access_token(admin_token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return payload["sub"]
