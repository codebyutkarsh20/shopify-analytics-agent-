"""SQLAlchemy ORM models for Shopify Analytics Agent database."""

from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, String, Integer, Text, Boolean, Float, DateTime, LargeBinary, UniqueConstraint, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from src.utils.timezone import now_ist


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


# ---------------------------------------------------------------------------
# Core identity models
# ---------------------------------------------------------------------------

class User(Base):
    """User model storing user information (channel-agnostic).

    A user can arrive via Telegram, WhatsApp, or both.  The
    ``telegram_user_id`` is nullable so that WhatsApp-only users can
    be created without a Telegram ID.  Cross-channel linking is handled
    by the ``ChannelSession`` table.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_user_id: Mapped[Optional[int]] = mapped_column(Integer, unique=True, nullable=True, index=True)
    telegram_username: Mapped[Optional[str]] = mapped_column(String(255))
    whatsapp_phone: Mapped[Optional[str]] = mapped_column(String(50), unique=True, nullable=True, index=True)
    first_name: Mapped[Optional[str]] = mapped_column(String(255))
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=True)
    interaction_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)
    last_active: Mapped[datetime] = mapped_column(DateTime, default=now_ist, onupdate=now_ist, nullable=False)

    # Relationships
    stores: Mapped[list["ShopifyStore"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    query_patterns: Mapped[list["QueryPattern"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    preferences: Mapped[list["UserPreference"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    tool_usage: Mapped[list["ToolUsage"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    query_errors: Mapped[list["QueryError"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    sessions: Mapped[list["Session"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    channel_sessions: Mapped[list["ChannelSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    response_feedback: Mapped[list["ResponseFeedback"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, telegram_user_id={self.telegram_user_id}, username={self.telegram_username})>"


class ChannelSession(Base):
    """Maps channel-specific identifiers to internal users.

    Enables multi-channel support: the same user can interact via
    Telegram, WhatsApp, web, etc. and all data stays linked.
    """

    __tablename__ = "channel_sessions"
    __table_args__ = (
        UniqueConstraint("channel_type", "channel_user_id", name="unique_channel_user"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    channel_type: Mapped[str] = mapped_column(String(50), nullable=False)
    channel_user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    channel_username: Mapped[Optional[str]] = mapped_column(String(255))
    channel_metadata: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)
    last_active: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="channel_sessions")

    def __repr__(self) -> str:
        return f"<ChannelSession(id={self.id}, channel_type={self.channel_type}, channel_user_id={self.channel_user_id})>"


# ---------------------------------------------------------------------------
# Store & cache models
# ---------------------------------------------------------------------------

class ShopifyStore(Base):
    """Shopify store model storing store access information."""

    __tablename__ = "shopify_stores"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    shop_domain: Mapped[str] = mapped_column(String(255), nullable=False)
    access_token: Mapped[str] = mapped_column(String(255), nullable=False)
    installed_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="stores")
    analytics_cache: Mapped[list["AnalyticsCache"]] = relationship(back_populates="store", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ShopifyStore(id={self.id}, shop_domain={self.shop_domain}, user_id={self.user_id})>"


class AnalyticsCache(Base):
    """Analytics cache model storing cached API responses."""

    __tablename__ = "analytics_cache"
    __table_args__ = (UniqueConstraint("store_id", "cache_key", name="unique_store_cache_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("shopify_stores.id"), nullable=False, index=True)
    cache_key: Mapped[str] = mapped_column(String(255), nullable=False)
    cache_data: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False, index=True)

    # Relationships
    store: Mapped["ShopifyStore"] = relationship(back_populates="analytics_cache")

    def __repr__(self) -> str:
        return f"<AnalyticsCache(id={self.id}, cache_key={self.cache_key}, store_id={self.store_id})>"


# ---------------------------------------------------------------------------
# Session & conversation models
# ---------------------------------------------------------------------------

class Session(Base):
    """Conversation session tracking.

    In single-thread chats (Telegram, WhatsApp) there is no explicit
    "new conversation" button.  The agent detects session boundaries
    using time gaps, topic shifts, and explicit user signals.
    """

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    channel_type: Mapped[str] = mapped_column(String(50), default="telegram", nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)
    last_message_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    primary_intent: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    session_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="sessions")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="session")

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, user_id={self.user_id}, messages={self.message_count})>"


class Conversation(Base):
    """Conversation model storing user interactions and responses."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    session_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sessions.id"), nullable=True, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(100), nullable=False)
    channel_type: Mapped[str] = mapped_column(String(50), default="telegram", nullable=False)
    tool_calls_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    response_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    template_id_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False, index=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversations")
    session: Mapped[Optional["Session"]] = relationship(back_populates="conversations")
    feedback: Mapped[list["ResponseFeedback"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, user_id={self.user_id}, query_type={self.query_type})>"


# ---------------------------------------------------------------------------
# Learning & pattern models
# ---------------------------------------------------------------------------

class QueryPattern(Base):
    """Query pattern model tracking user query patterns."""

    __tablename__ = "query_patterns"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    pattern_type: Mapped[str] = mapped_column(String(100), nullable=False)
    pattern_value: Mapped[str] = mapped_column(String(255), nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_used: Mapped[datetime] = mapped_column(DateTime, default=now_ist, onupdate=now_ist, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="query_patterns")

    def __repr__(self) -> str:
        return f"<QueryPattern(id={self.id}, pattern_type={self.pattern_type}, frequency={self.frequency})>"


class UserPreference(Base):
    """User preference model storing user configuration preferences."""

    __tablename__ = "user_preferences"
    __table_args__ = (UniqueConstraint("user_id", "preference_key", name="unique_user_preference"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    preference_key: Mapped[str] = mapped_column(String(100), nullable=False)
    preference_value: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, onupdate=now_ist, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="preferences")

    def __repr__(self) -> str:
        return f"<UserPreference(id={self.id}, preference_key={self.preference_key}, confidence_score={self.confidence_score})>"


class QueryTemplate(Base):
    """Stores successful query patterns for reuse.

    When a tool call succeeds, the query is saved here indexed by
    intent.  Future similar questions can reuse a known-good query
    instead of constructing one from scratch.
    """

    __tablename__ = "query_templates"

    id: Mapped[int] = mapped_column(primary_key=True)
    intent_category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    intent_description: Mapped[str] = mapped_column(Text, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    tool_parameters: Mapped[str] = mapped_column(Text, nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    avg_execution_time_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_by_user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)
    last_used_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)
    example_queries: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<QueryTemplate(id={self.id}, intent={self.intent_category}, confidence={self.confidence:.2f})>"


# ---------------------------------------------------------------------------
# Error learning models
# ---------------------------------------------------------------------------

class ToolUsage(Base):
    """Tool usage model logging tool invocations and performance."""

    __tablename__ = "tool_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    parameters: Mapped[str] = mapped_column(Text, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    execution_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    intent_category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    query_template_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    channel_type: Mapped[str] = mapped_column(String(50), default="telegram", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False, index=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="tool_usage")

    def __repr__(self) -> str:
        return f"<ToolUsage(id={self.id}, tool_name={self.tool_name}, success={self.success})>"


class QueryError(Base):
    """Query error model storing failed tool calls for learning.

    When a tool call fails, the error details are stored here so the
    agent can learn from past mistakes and avoid repeating the same
    failing queries.
    """

    __tablename__ = "query_errors"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    error_type: Mapped[str] = mapped_column(String(100), nullable=False, default="unknown")
    lesson: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    recovery_pattern_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("error_recovery_patterns.id"), nullable=True,
    )
    was_auto_recovered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    original_intent: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False, index=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="query_errors")
    recovery_pattern: Mapped[Optional["ErrorRecoveryPattern"]] = relationship()

    def __repr__(self) -> str:
        return f"<QueryError(id={self.id}, tool_name={self.tool_name}, error_type={self.error_type})>"


class ErrorRecoveryPattern(Base):
    """Maps failed queries to their successful recoveries.

    When tool A with params X fails but tool B with params Y succeeds
    right after, this pattern is stored so the system can suggest the
    fix automatically next time.
    """

    __tablename__ = "error_recovery_patterns"

    id: Mapped[int] = mapped_column(primary_key=True)
    error_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    failed_tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    failed_parameters: Mapped[str] = mapped_column(Text, nullable=False)
    error_fingerprint: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    recovery_tool_name: Mapped[str] = mapped_column(String(100), nullable=False)
    recovery_parameters: Mapped[str] = mapped_column(Text, nullable=False)
    recovery_description: Mapped[str] = mapped_column(Text, nullable=False)
    times_applied: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    times_succeeded: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)
    last_used_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)

    def __repr__(self) -> str:
        return f"<ErrorRecoveryPattern(id={self.id}, error_type={self.error_type}, confidence={self.confidence:.2f})>"


# ---------------------------------------------------------------------------
# Cross-user intelligence models
# ---------------------------------------------------------------------------

class GlobalInsight(Base):
    """Anonymized cross-user intelligence.

    Stores aggregated patterns across ALL users — what tools work best,
    common mistakes, proven query patterns, etc.
    """

    __tablename__ = "global_insights"
    __table_args__ = (
        UniqueConstraint("insight_type", "insight_key", name="unique_insight"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    insight_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    insight_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    insight_value: Mapped[str] = mapped_column(Text, nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)

    def __repr__(self) -> str:
        return f"<GlobalInsight(id={self.id}, type={self.insight_type}, key={self.insight_key})>"


class ResponseFeedback(Base):
    """Tracks response quality signals.

    Uses implicit signals (user corrections, thanks, frustration) to
    score response quality and feed it back into template learning.
    """

    __tablename__ = "response_feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    feedback_type: Mapped[str] = mapped_column(String(50), nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    query_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tool_used: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    signal_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="feedback")
    user: Mapped["User"] = relationship(back_populates="response_feedback")

    def __repr__(self) -> str:
        return f"<ResponseFeedback(id={self.id}, type={self.feedback_type}, score={self.quality_score})>"


# ---------------------------------------------------------------------------
# Vector embeddings for semantic search
# ---------------------------------------------------------------------------

class EmbeddingVector(Base):
    """Stores embedding vectors for semantic similarity search.

    Each row maps an (entity_type, entity_id) pair to a dense float32
    vector stored as a BLOB.  Used to find templates and learnings that
    are semantically similar to a user's query even when keywords differ.
    """

    __tablename__ = "embedding_vectors"
    __table_args__ = (
        UniqueConstraint("entity_type", "entity_id", name="uq_embedding_entity"),
        Index("idx_embedding_type", "entity_type"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    embedding_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    source_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_ist, nullable=False)

    def __repr__(self) -> str:
        return f"<EmbeddingVector(entity_type={self.entity_type}, entity_id={self.entity_id})>"
