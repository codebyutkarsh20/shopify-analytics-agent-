"""Pydantic schemas for admin dashboard API responses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    message: str
    must_change_password: bool = False


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class UserOut(BaseModel):
    id: int
    telegram_user_id: Optional[int] = None
    telegram_username: Optional[str] = None
    whatsapp_phone: Optional[str] = None
    display_name: Optional[str] = None
    first_name: Optional[str] = None
    is_verified: Optional[bool] = False
    interaction_count: int = 0
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserDetailOut(UserOut):
    conversation_count: int = 0
    feedback_count: int = 0
    avg_quality_score: Optional[float] = None
    channels: list[str] = []


class PaginatedUsers(BaseModel):
    total: int
    page: int
    limit: int
    users: list[UserOut]


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

class ConversationOut(BaseModel):
    id: int
    user_id: int
    user_display_name: Optional[str] = None
    message: str
    response: str
    query_type: str
    channel_type: str = "telegram"
    response_quality_score: Optional[float] = None
    created_at: Optional[datetime] = None
    feedback_count: int = 0

    class Config:
        from_attributes = True


class ConversationDetailOut(ConversationOut):
    tool_calls_json: Optional[str] = None
    template_id_used: Optional[int] = None
    session_id: Optional[int] = None
    quality_completeness_score: Optional[float] = None
    quality_sentiment_score: Optional[float] = None
    quality_tool_score: Optional[float] = None
    feedback: list[dict] = []


class PaginatedConversations(BaseModel):
    total: int
    page: int
    limit: int
    conversations: list[ConversationOut]


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class FeedbackSummary(BaseModel):
    total_feedback: int = 0
    avg_quality_score: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    distribution: dict[str, int] = {}
    daily_trend: list[dict] = []


class ChannelStats(BaseModel):
    channel: str
    conversation_count: int = 0
    unique_users: int = 0
    avg_quality: Optional[float] = None


class ChannelBreakdown(BaseModel):
    channels: list[ChannelStats]
    total_conversations: int = 0


class ToolUsageStat(BaseModel):
    tool_name: str
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    avg_execution_time_ms: float = 0.0


class ErrorStat(BaseModel):
    id: int
    tool_name: str
    error_type: str
    error_message: str
    resolved: bool = False
    created_at: Optional[datetime] = None


class PaginatedErrors(BaseModel):
    total: int
    errors: list[ErrorStat]


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

class SystemStats(BaseModel):
    total_users: int = 0
    active_users_24h: int = 0
    total_conversations: int = 0
    total_sessions: int = 0
    avg_quality_score: float = 0.0
    total_feedback: int = 0
    total_errors: int = 0
    unresolved_errors: int = 0
    database_size_mb: float = 0.0
    cache_entries: int = 0


class HealthCheck(BaseModel):
    status: str = "healthy"
    database: str = "ok"
    bot_running: bool = True
