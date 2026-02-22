"""Analytics routes: feedback, channels, tool usage, errors."""

from typing import Optional

from fastapi import APIRouter, Depends

from src.admin.dependencies import get_current_admin
from src.admin.schemas import (
    FeedbackSummary,
    ChannelBreakdown,
    ChannelStats,
    ToolUsageStat,
    PaginatedErrors,
    ErrorStat,
)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_queries = None


def set_queries(queries):
    global _queries
    _queries = queries


@router.get("/feedback", response_model=FeedbackSummary)
async def feedback_summary(days: int = 30, _admin: str = Depends(get_current_admin)):
    return FeedbackSummary(**_queries.get_feedback_summary(days))


@router.get("/channels", response_model=ChannelBreakdown)
async def channel_breakdown(days: int = 30, _admin: str = Depends(get_current_admin)):
    data = _queries.get_channel_stats(days)
    return ChannelBreakdown(
        channels=[ChannelStats(**ch) for ch in data["channels"]],
        total_conversations=data["total_conversations"],
    )


@router.get("/tools", response_model=list[ToolUsageStat])
async def tool_usage(days: int = 30, limit: int = 20, _admin: str = Depends(get_current_admin)):
    stats = _queries.get_tool_usage_stats(days, limit)
    return [ToolUsageStat(**s) for s in stats]


@router.get("/errors", response_model=PaginatedErrors)
async def errors_list(
    days: int = 30,
    resolved: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
    _admin: str = Depends(get_current_admin),
):
    errors, total = _queries.get_errors_paginated(days, resolved, limit, offset)
    return PaginatedErrors(total=total, errors=[ErrorStat(**e) for e in errors])
