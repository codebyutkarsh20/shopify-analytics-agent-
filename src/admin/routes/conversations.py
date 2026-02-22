"""Conversation routes: list & detail."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.admin.dependencies import get_current_admin
from src.admin.schemas import PaginatedConversations, ConversationOut, ConversationDetailOut

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

_queries = None


def set_queries(queries):
    global _queries
    _queries = queries


@router.get("", response_model=PaginatedConversations)
async def list_conversations(
    page: int = 1,
    limit: int = 50,
    user_id: Optional[int] = None,
    query_type: Optional[str] = None,
    channel: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    _admin: str = Depends(get_current_admin),
):
    page = max(1, page)
    limit = max(1, min(limit, 100))
    df = datetime.fromisoformat(date_from) if date_from else None
    dt = datetime.fromisoformat(date_to) if date_to else None

    conversations, total = _queries.get_conversations_paginated(
        page=page,
        limit=limit,
        user_id=user_id,
        query_type=query_type,
        channel=channel,
        date_from=df,
        date_to=dt,
    )
    return PaginatedConversations(
        total=total,
        page=page,
        limit=limit,
        conversations=[ConversationOut(**c) for c in conversations],
    )


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: int, _admin: str = Depends(get_current_admin)):
    detail = _queries.get_conversation_detail(conversation_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetailOut(**detail)
