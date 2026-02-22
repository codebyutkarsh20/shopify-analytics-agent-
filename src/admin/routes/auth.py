"""Authentication routes: login, logout, password change."""

from fastapi import APIRouter, Depends, HTTPException, Response, status

from src.admin.auth import create_access_token, verify_password, hash_password
from src.admin.dependencies import get_current_admin
from src.admin.schemas import LoginRequest, LoginResponse, ChangePasswordRequest

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Will be set by app.py at startup
_queries = None


def set_queries(queries):
    global _queries
    _queries = queries


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, response: Response):
    admin = _queries.get_admin_by_username(body.username)
    if not admin or not verify_password(body.password, admin.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    if not admin.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    token = create_access_token(admin.username)
    _queries.update_admin_last_login(admin.username)

    # Set httpOnly cookie
    response.set_cookie(
        key="admin_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,  # 24 hours
        path="/",
    )

    return LoginResponse(
        message="Login successful",
        must_change_password=admin.must_change_password,
    )


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(key="admin_token", path="/")
    return {"message": "Logged out"}


@router.get("/me")
async def get_me(username: str = Depends(get_current_admin)):
    must_change = _queries.admin_must_change_password(username)
    return {"username": username, "must_change_password": must_change}


@router.post("/change-password")
async def change_password(
    body: ChangePasswordRequest,
    response: Response,
    username: str = Depends(get_current_admin),
):
    admin = _queries.get_admin_by_username(username)
    if not admin or not verify_password(body.current_password, admin.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    new_hash = hash_password(body.new_password)
    _queries.update_admin_password(username, new_hash)

    # Re-issue token
    token = create_access_token(username)
    response.set_cookie(
        key="admin_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,
        path="/",
    )

    return {"message": "Password changed successfully"}
