"""User management routes."""

from fastapi import APIRouter, Depends, HTTPException

from src.admin.dependencies import get_current_admin
from src.admin.schemas import PaginatedUsers, UserOut, UserDetailOut

router = APIRouter(prefix="/api/users", tags=["users"])

_queries = None


def set_queries(queries):
    global _queries
    _queries = queries


@router.get("", response_model=PaginatedUsers)
async def list_users(
    page: int = 1,
    limit: int = 20,
    sort_by: str = "last_active",
    order: str = "desc",
    _admin: str = Depends(get_current_admin),
):
    page = max(1, page)
    limit = max(1, min(limit, 100))
    users, total = _queries.get_users_paginated(page, limit, sort_by, order)
    return PaginatedUsers(
        total=total,
        page=page,
        limit=limit,
        users=[UserOut.model_validate(u) for u in users],
    )


@router.get("/{user_id}")
async def get_user(user_id: int, _admin: str = Depends(get_current_admin)):
    detail = _queries.get_user_detail(user_id)
    if not detail:
        raise HTTPException(status_code=404, detail="User not found")
    return UserDetailOut(**detail)
