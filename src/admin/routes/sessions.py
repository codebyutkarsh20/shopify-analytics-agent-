"""Session routes: list sessions and get session chat thread."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.admin.dependencies import get_current_admin

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

_queries = None


def set_queries(queries):
    global _queries
    _queries = queries


@router.get("")
async def list_sessions(
    page: int = 1,
    limit: int = 20,
    user_id: Optional[int] = None,
    channel: Optional[str] = None,
    _admin: str = Depends(get_current_admin),
):
    page = max(1, page)
    limit = max(1, min(limit, 100))
    sessions, total = _queries.get_sessions_paginated(page, limit, user_id, channel)
    return {"total": total, "page": page, "limit": limit, "sessions": sessions}


@router.get("/{session_id}")
async def get_session(session_id: int, _admin: str = Depends(get_current_admin)):
    detail = _queries.get_session_conversations(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Session not found")
    return detail
