"""SQLAlchemy ORM models for Shopify Analytics Agent database."""

from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, String, Integer, Text, Boolean, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class User(Base):
    """User model storing Telegram user information."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_user_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    telegram_username: Mapped[Optional[str]] = mapped_column(String(255))
    first_name: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_active: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    stores: Mapped[list["ShopifyStore"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    query_patterns: Mapped[list["QueryPattern"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    preferences: Mapped[list["UserPreference"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    mcp_tool_usage: Mapped[list["MCPToolUsage"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, telegram_user_id={self.telegram_user_id}, username={self.telegram_username})>"


class ShopifyStore(Base):
    """Shopify store model storing store access information."""

    __tablename__ = "shopify_stores"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    shop_domain: Mapped[str] = mapped_column(String(255), nullable=False)
    access_token: Mapped[str] = mapped_column(String(255), nullable=False)
    installed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="stores")
    analytics_cache: Mapped[list["AnalyticsCache"]] = relationship(back_populates="store", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ShopifyStore(id={self.id}, shop_domain={self.shop_domain}, user_id={self.user_id})>"


class Conversation(Base):
    """Conversation model storing user interactions and responses."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversations")

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, user_id={self.user_id}, query_type={self.query_type})>"


class QueryPattern(Base):
    """Query pattern model tracking user query patterns."""

    __tablename__ = "query_patterns"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    pattern_type: Mapped[str] = mapped_column(String(100), nullable=False)
    pattern_value: Mapped[str] = mapped_column(String(255), nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_used: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

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
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="preferences")

    def __repr__(self) -> str:
        return f"<UserPreference(id={self.id}, preference_key={self.preference_key}, confidence_score={self.confidence_score})>"


class MCPToolUsage(Base):
    """MCP tool usage model logging tool invocations."""

    __tablename__ = "mcp_tool_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    parameters: Mapped[str] = mapped_column(Text, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    execution_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="mcp_tool_usage")

    def __repr__(self) -> str:
        return f"<MCPToolUsage(id={self.id}, tool_name={self.tool_name}, success={self.success}, execution_time_ms={self.execution_time_ms})>"


class AnalyticsCache(Base):
    """Analytics cache model storing cached API responses."""

    __tablename__ = "analytics_cache"
    __table_args__ = (UniqueConstraint("store_id", "cache_key", name="unique_store_cache_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("shopify_stores.id"), nullable=False, index=True)
    cache_key: Mapped[str] = mapped_column(String(255), nullable=False)
    cache_data: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    store: Mapped["ShopifyStore"] = relationship(back_populates="analytics_cache")

    def __repr__(self) -> str:
        return f"<AnalyticsCache(id={self.id}, cache_key={self.cache_key}, store_id={self.store_id})>"
