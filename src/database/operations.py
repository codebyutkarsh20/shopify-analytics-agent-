"""Database CRUD operations for Shopify Analytics Agent."""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, select, func, and_
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.models import (
    Base,
    User,
    ShopifyStore,
    Conversation,
    QueryPattern,
    UserPreference,
    MCPToolUsage,
    AnalyticsCache,
)


class DatabaseOperations:
    """Manages database sessions and CRUD operations."""

    def __init__(self, database_url: str):
        """
        Initialize database operations.

        Args:
            database_url: SQLAlchemy database URL (e.g., sqlite:///shopify_agent.db)
        """
        self.database_url = database_url

        # Configure SQLite-specific options
        if "sqlite" in database_url:
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False,
            )
        else:
            self.engine = create_engine(database_url, echo=False)

        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            SQLAlchemy session object.
        """
        return self.SessionLocal()

    def init_database(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self.engine)

    def check_database(self) -> bool:
        """
        Check if all required tables exist.

        Returns:
            True if all tables exist, False otherwise.
        """
        inspector = self._get_inspector()
        required_tables = {
            "users",
            "shopify_stores",
            "conversations",
            "query_patterns",
            "user_preferences",
            "mcp_tool_usage",
            "analytics_cache",
        }
        existing_tables = set(inspector.get_table_names())
        return required_tables.issubset(existing_tables)

    def _get_inspector(self):
        """Get SQLAlchemy inspector for database introspection."""
        from sqlalchemy import inspect
        return inspect(self.engine)

    # ==================== User Operations ====================

    def get_or_create_user(
        self,
        telegram_user_id: int,
        telegram_username: Optional[str] = None,
        first_name: Optional[str] = None,
    ) -> User:
        """
        Get existing user or create new user.

        Args:
            telegram_user_id: Telegram user ID.
            telegram_username: Optional Telegram username.
            first_name: Optional user's first name.

        Returns:
            User object.
        """
        session = self.get_session()
        try:
            stmt = select(User).where(User.telegram_user_id == telegram_user_id)
            user = session.execute(stmt).scalar_one_or_none()

            if user is None:
                user = User(
                    telegram_user_id=telegram_user_id,
                    telegram_username=telegram_username,
                    first_name=first_name,
                )
                session.add(user)
                session.commit()
                session.refresh(user)

            return user
        finally:
            session.close()

    def update_user_activity(self, user_id: int) -> None:
        """
        Update user's last active timestamp.

        Args:
            user_id: User ID.
        """
        session = self.get_session()
        try:
            stmt = select(User).where(User.id == user_id)
            user = session.execute(stmt).scalar_one_or_none()

            if user:
                user.last_active = datetime.utcnow()
                session.commit()
        finally:
            session.close()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID.

        Returns:
            User object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(User).where(User.id == user_id)
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    def get_user_first_query_time(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the timestamp of user's first query/conversation.

        Args:
            user_id: User ID.

        Returns:
            Dictionary with 'first_query_time' or None if no queries exist.
        """
        session = self.get_session()
        try:
            stmt = (
                select(Conversation.created_at)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.created_at.asc())
                .limit(1)
            )
            result = session.execute(stmt).scalar_one_or_none()
            if result:
                return {"first_query_time": result.isoformat()}
            return None
        finally:
            session.close()

    def get_user_query_count(self, user_id: int) -> int:
        """
        Get total number of queries/conversations for a user.

        Args:
            user_id: User ID.

        Returns:
            Total query count.
        """
        session = self.get_session()
        try:
            stmt = (
                select(func.count())
                .select_from(Conversation)
                .where(Conversation.user_id == user_id)
            )
            result = session.execute(stmt).scalar()
            return result or 0
        finally:
            session.close()

    # ==================== Store Operations ====================

    def add_store(
        self,
        user_id: int,
        shop_domain: str,
        access_token: str,
    ) -> ShopifyStore:
        """
        Add a Shopify store for a user.

        Args:
            user_id: User ID.
            shop_domain: Shopify shop domain.
            access_token: Shopify API access token.

        Returns:
            ShopifyStore object.
        """
        session = self.get_session()
        try:
            store = ShopifyStore(
                user_id=user_id,
                shop_domain=shop_domain,
                access_token=access_token,
            )
            session.add(store)
            session.commit()
            session.refresh(store)
            return store
        finally:
            session.close()

    def get_store_by_user(self, user_id: int) -> Optional[ShopifyStore]:
        """
        Get Shopify store for a user (assumes one store per user).

        Args:
            user_id: User ID.

        Returns:
            ShopifyStore object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(ShopifyStore).where(ShopifyStore.user_id == user_id)
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    def get_all_stores_by_user(self, user_id: int) -> List[ShopifyStore]:
        """
        Get all Shopify stores for a user.

        Args:
            user_id: User ID.

        Returns:
            List of ShopifyStore objects.
        """
        session = self.get_session()
        try:
            stmt = select(ShopifyStore).where(ShopifyStore.user_id == user_id)
            return session.execute(stmt).scalars().all()
        finally:
            session.close()

    def get_store_by_id(self, store_id: int) -> Optional[ShopifyStore]:
        """
        Get Shopify store by ID.

        Args:
            store_id: Store ID.

        Returns:
            ShopifyStore object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(ShopifyStore).where(ShopifyStore.id == store_id)
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    # ==================== Conversation Operations ====================

    def save_conversation(
        self,
        user_id: int,
        message: str,
        response: str,
        query_type: str,
    ) -> Conversation:
        """
        Save a user conversation.

        Args:
            user_id: User ID.
            message: User's message.
            response: Agent's response.
            query_type: Type of query (e.g., "sales", "products", "analytics").

        Returns:
            Conversation object.
        """
        session = self.get_session()
        try:
            conversation = Conversation(
                user_id=user_id,
                message=message,
                response=response,
                query_type=query_type,
            )
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            return conversation
        finally:
            session.close()

    def get_recent_conversations(
        self,
        user_id: int,
        limit: int = 10,
        days: int = 30,
    ) -> List[Conversation]:
        """
        Get recent conversations for a user.

        Args:
            user_id: User ID.
            limit: Maximum number of conversations to return.
            days: Number of days to look back.

        Returns:
            List of Conversation objects.
        """
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            stmt = (
                select(Conversation)
                .where(
                    and_(
                        Conversation.user_id == user_id,
                        Conversation.created_at >= cutoff_date,
                    )
                )
                .order_by(Conversation.created_at.desc())
                .limit(limit)
            )
            return session.execute(stmt).scalars().all()
        finally:
            session.close()

    def get_conversations_by_type(
        self,
        user_id: int,
        query_type: str,
        limit: int = 10,
    ) -> List[Conversation]:
        """
        Get conversations by query type.

        Args:
            user_id: User ID.
            query_type: Type of query.
            limit: Maximum number of conversations to return.

        Returns:
            List of Conversation objects.
        """
        session = self.get_session()
        try:
            stmt = (
                select(Conversation)
                .where(
                    and_(
                        Conversation.user_id == user_id,
                        Conversation.query_type == query_type,
                    )
                )
                .order_by(Conversation.created_at.desc())
                .limit(limit)
            )
            return session.execute(stmt).scalars().all()
        finally:
            session.close()

    # ==================== Query Pattern Operations ====================

    def update_pattern_frequency(
        self,
        user_id: int,
        pattern_type: str,
        pattern_value: str,
    ) -> QueryPattern:
        """
        Update or create a query pattern with incremented frequency.

        Args:
            user_id: User ID.
            pattern_type: Type of pattern (e.g., "metric", "dimension").
            pattern_value: Pattern value.

        Returns:
            QueryPattern object.
        """
        session = self.get_session()
        try:
            stmt = select(QueryPattern).where(
                and_(
                    QueryPattern.user_id == user_id,
                    QueryPattern.pattern_type == pattern_type,
                    QueryPattern.pattern_value == pattern_value,
                )
            )
            pattern = session.execute(stmt).scalar_one_or_none()

            if pattern is None:
                pattern = QueryPattern(
                    user_id=user_id,
                    pattern_type=pattern_type,
                    pattern_value=pattern_value,
                    frequency=1,
                )
            else:
                pattern.frequency += 1
                pattern.last_used = datetime.utcnow()

            session.add(pattern)
            session.commit()
            session.refresh(pattern)
            return pattern
        finally:
            session.close()

    def get_top_patterns(
        self,
        user_id: int,
        pattern_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[QueryPattern]:
        """
        Get top query patterns for a user.

        Args:
            user_id: User ID.
            pattern_type: Optional pattern type filter.
            limit: Maximum number of patterns to return.

        Returns:
            List of QueryPattern objects sorted by frequency.
        """
        session = self.get_session()
        try:
            stmt = select(QueryPattern).where(QueryPattern.user_id == user_id)

            if pattern_type:
                stmt = stmt.where(QueryPattern.pattern_type == pattern_type)

            stmt = stmt.order_by(QueryPattern.frequency.desc()).limit(limit)
            return session.execute(stmt).scalars().all()
        finally:
            session.close()

    def get_pattern_frequency(
        self,
        user_id: int,
        pattern_type: str,
        pattern_value: str,
    ) -> int:
        """
        Get frequency of a specific pattern.

        Args:
            user_id: User ID.
            pattern_type: Type of pattern.
            pattern_value: Pattern value.

        Returns:
            Frequency count (0 if not found).
        """
        session = self.get_session()
        try:
            stmt = select(QueryPattern).where(
                and_(
                    QueryPattern.user_id == user_id,
                    QueryPattern.pattern_type == pattern_type,
                    QueryPattern.pattern_value == pattern_value,
                )
            )
            pattern = session.execute(stmt).scalar_one_or_none()
            return pattern.frequency if pattern else 0
        finally:
            session.close()

    # ==================== Preference Operations ====================

    def set_preference(
        self,
        user_id: int,
        preference_key: str,
        preference_value: str,
        confidence_score: float = 0.5,
    ) -> UserPreference:
        """
        Set or update a user preference.

        Args:
            user_id: User ID.
            preference_key: Preference key.
            preference_value: Preference value.
            confidence_score: Confidence score (0.0-1.0).

        Returns:
            UserPreference object.
        """
        session = self.get_session()
        try:
            stmt = select(UserPreference).where(
                and_(
                    UserPreference.user_id == user_id,
                    UserPreference.preference_key == preference_key,
                )
            )
            preference = session.execute(stmt).scalar_one_or_none()

            if preference is None:
                preference = UserPreference(
                    user_id=user_id,
                    preference_key=preference_key,
                    preference_value=preference_value,
                    confidence_score=confidence_score,
                )
            else:
                preference.preference_value = preference_value
                preference.confidence_score = confidence_score
                preference.updated_at = datetime.utcnow()

            session.add(preference)
            session.commit()
            session.refresh(preference)
            return preference
        finally:
            session.close()

    def get_preference(
        self,
        user_id: int,
        preference_key: str,
    ) -> Optional[UserPreference]:
        """
        Get a specific user preference.

        Args:
            user_id: User ID.
            preference_key: Preference key.

        Returns:
            UserPreference object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(UserPreference).where(
                and_(
                    UserPreference.user_id == user_id,
                    UserPreference.preference_key == preference_key,
                )
            )
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    def get_preferences(self, user_id: int) -> List[UserPreference]:
        """
        Get all preferences for a user.

        Args:
            user_id: User ID.

        Returns:
            List of UserPreference objects.
        """
        session = self.get_session()
        try:
            stmt = select(UserPreference).where(UserPreference.user_id == user_id)
            return session.execute(stmt).scalars().all()
        finally:
            session.close()

    def get_preferences_dict(self, user_id: int) -> Dict[str, str]:
        """
        Get all preferences for a user as a dictionary.

        Args:
            user_id: User ID.

        Returns:
            Dictionary of preference_key -> preference_value.
        """
        preferences = self.get_preferences(user_id)
        return {pref.preference_key: pref.preference_value for pref in preferences}

    # ==================== MCP Tool Usage Operations ====================

    def log_tool_usage(
        self,
        user_id: int,
        tool_name: str,
        parameters: str,
        success: bool,
        execution_time_ms: int,
    ) -> MCPToolUsage:
        """
        Log MCP tool usage.

        Args:
            user_id: User ID.
            tool_name: Name of the tool.
            parameters: Tool parameters (JSON string).
            success: Whether execution was successful.
            execution_time_ms: Execution time in milliseconds.

        Returns:
            MCPToolUsage object.
        """
        session = self.get_session()
        try:
            tool_usage = MCPToolUsage(
                user_id=user_id,
                tool_name=tool_name,
                parameters=parameters,
                success=success,
                execution_time_ms=execution_time_ms,
            )
            session.add(tool_usage)
            session.commit()
            session.refresh(tool_usage)
            return tool_usage
        finally:
            session.close()

    def get_tool_usage_stats(
        self,
        user_id: int,
        tool_name: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get tool usage statistics for a user.

        Args:
            user_id: User ID.
            tool_name: Optional tool name filter.
            days: Number of days to look back.

        Returns:
            Dictionary with usage statistics.
        """
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            stmt = select(MCPToolUsage).where(
                and_(
                    MCPToolUsage.user_id == user_id,
                    MCPToolUsage.created_at >= cutoff_date,
                )
            )

            if tool_name:
                stmt = stmt.where(MCPToolUsage.tool_name == tool_name)

            usage_records = session.execute(stmt).scalars().all()

            total = len(usage_records)
            successful = sum(1 for u in usage_records if u.success)
            failed = total - successful
            avg_time_ms = (
                sum(u.execution_time_ms for u in usage_records) / total
                if total > 0
                else 0
            )

            return {
                "total": total,
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / total * 100) if total > 0 else 0,
                "avg_execution_time_ms": avg_time_ms,
            }
        finally:
            session.close()

    # ==================== Cache Operations ====================

    def get_cache(self, store_id: int, cache_key: str) -> Optional[str]:
        """
        Get cached data if not expired.

        Args:
            store_id: Store ID.
            cache_key: Cache key.

        Returns:
            Cache data string or None if not found or expired.
        """
        session = self.get_session()
        try:
            stmt = select(AnalyticsCache).where(
                and_(
                    AnalyticsCache.store_id == store_id,
                    AnalyticsCache.cache_key == cache_key,
                )
            )
            cache = session.execute(stmt).scalar_one_or_none()

            if cache is None:
                return None

            if cache.expires_at < datetime.utcnow():
                session.delete(cache)
                session.commit()
                return None

            return cache.cache_data
        finally:
            session.close()

    def set_cache(
        self,
        store_id: int,
        cache_key: str,
        cache_data: str,
        ttl_hours: int = 24,
    ) -> AnalyticsCache:
        """
        Set cached data.

        Args:
            store_id: Store ID.
            cache_key: Cache key.
            cache_data: Cache data (JSON string).
            ttl_hours: Time-to-live in hours.

        Returns:
            AnalyticsCache object.
        """
        session = self.get_session()
        try:
            stmt = select(AnalyticsCache).where(
                and_(
                    AnalyticsCache.store_id == store_id,
                    AnalyticsCache.cache_key == cache_key,
                )
            )
            cache = session.execute(stmt).scalar_one_or_none()

            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

            if cache is None:
                cache = AnalyticsCache(
                    store_id=store_id,
                    cache_key=cache_key,
                    cache_data=cache_data,
                    expires_at=expires_at,
                )
            else:
                cache.cache_data = cache_data
                cache.expires_at = expires_at

            session.add(cache)
            session.commit()
            session.refresh(cache)
            return cache
        finally:
            session.close()

    def clear_expired_cache(self) -> int:
        """
        Delete expired cache entries.

        Returns:
            Number of deleted entries.
        """
        session = self.get_session()
        try:
            stmt = select(AnalyticsCache).where(
                AnalyticsCache.expires_at < datetime.utcnow()
            )
            expired = session.execute(stmt).scalars().all()

            for cache in expired:
                session.delete(cache)

            session.commit()
            return len(expired)
        finally:
            session.close()

    def clear_store_cache(self, store_id: int) -> int:
        """
        Clear all cache for a specific store.

        Args:
            store_id: Store ID.

        Returns:
            Number of deleted entries.
        """
        session = self.get_session()
        try:
            stmt = select(AnalyticsCache).where(AnalyticsCache.store_id == store_id)
            caches = session.execute(stmt).scalars().all()

            for cache in caches:
                session.delete(cache)

            session.commit()
            return len(caches)
        finally:
            session.close()

    # ==================== Data Cleanup Operations ====================

    def clear_user_data(self, user_id: int) -> Dict[str, int]:
        """
        Clear all data for a user (conversations, patterns, preferences, etc.).

        Args:
            user_id: User ID.

        Returns:
            Dictionary with counts of deleted items.
        """
        session = self.get_session()
        try:
            counts = {}

            # Delete conversations
            conversations = session.execute(
                select(Conversation).where(Conversation.user_id == user_id)
            ).scalars().all()
            counts["conversations"] = len(conversations)
            for conv in conversations:
                session.delete(conv)

            # Delete query patterns
            patterns = session.execute(
                select(QueryPattern).where(QueryPattern.user_id == user_id)
            ).scalars().all()
            counts["patterns"] = len(patterns)
            for pattern in patterns:
                session.delete(pattern)

            # Delete preferences
            preferences = session.execute(
                select(UserPreference).where(UserPreference.user_id == user_id)
            ).scalars().all()
            counts["preferences"] = len(preferences)
            for pref in preferences:
                session.delete(pref)

            # Delete MCP tool usage
            tool_usage = session.execute(
                select(MCPToolUsage).where(MCPToolUsage.user_id == user_id)
            ).scalars().all()
            counts["tool_usage"] = len(tool_usage)
            for usage in tool_usage:
                session.delete(usage)

            # Delete stores (which cascades to cache)
            stores = session.execute(
                select(ShopifyStore).where(ShopifyStore.user_id == user_id)
            ).scalars().all()
            counts["stores"] = len(stores)
            for store in stores:
                session.delete(store)

            session.commit()
            return counts
        finally:
            session.close()
