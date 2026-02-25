"""Database CRUD operations for Shopify Analytics Agent."""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
import hashlib

from sqlalchemy import create_engine, select, func, and_
from sqlalchemy.orm import Session as SQLSession, sessionmaker
from sqlalchemy.pool import StaticPool

from src.utils.timezone import now_ist
from src.utils.encryption import encrypt_token, decrypt_token

from src.database.models import (
    Base,
    User,
    ShopifyStore,
    Conversation,
    QueryPattern,
    UserPreference,
    ToolUsage,
    AnalyticsCache,
    QueryError,
    Session,
    QueryTemplate,
    ErrorRecoveryPattern,
    GlobalInsight,
    ResponseFeedback,
    ChannelSession,
    EmbeddingVector,
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

    def get_session(self) -> SQLSession:
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
            "tool_usage",
            "analytics_cache",
            "query_errors",
            "sessions",
            "query_templates",
            "error_recovery_patterns",
            "global_insights",
            "response_feedback",
            "channel_sessions",
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
                user.last_active = now_ist()
                session.commit()
        finally:
            session.close()

    def verify_user(self, user_id: int) -> bool:
        """
        Mark a user as verified (passed access-code check).

        Args:
            user_id: User ID.

        Returns:
            True if user was found and updated, False otherwise.
        """
        session = self.get_session()
        try:
            stmt = select(User).where(User.id == user_id)
            user = session.execute(stmt).scalar_one_or_none()
            if user:
                user.is_verified = True
                session.commit()
                return True
            return False
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

    def get_conversations_by_type(self, user_id: int, query_type: str, limit: int = 5) -> List[Any]:
        """Get recent conversations filtered by query type.

        Args:
            user_id: Telegram user ID
            query_type: specific query type to filter by
            limit: max number of conversations to return

        Returns:
            List of Conversation objects
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
        except Exception as e:
            # logger not globally available here unless imported, assuming it's not or using print for now
            # Actually, let's just return empty list on error for safety
            return []
        finally:
            session.close()

    def increment_user_interaction(self, user_id: int) -> None:
        """
        Increment interaction count for a user.

        Args:
            user_id: User ID.
        """
        session = self.get_session()
        try:
            stmt = select(User).where(User.id == user_id)
            user = session.execute(stmt).scalar_one_or_none()

            if user:
                user.interaction_count = (user.interaction_count or 0) + 1
                session.commit()
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
                access_token=encrypt_token(access_token),
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
        Automatically decrypts the access_token.

        Args:
            user_id: User ID.

        Returns:
            ShopifyStore object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(ShopifyStore).where(ShopifyStore.user_id == user_id)
            # Use scalars().first() to return the first store found, avoiding errors
            # if multiple stores were accidentally created for the same user.
            store = session.execute(stmt).scalars().first()
            if store:
                store.access_token = decrypt_token(store.access_token)
            return store
        finally:
            session.close()

    def update_store_credentials(
        self,
        store_id: int,
        shop_domain: str,
        access_token: str,
    ) -> Optional[ShopifyStore]:
        """
        Update an existing store's domain and access token.

        Args:
            store_id: Store ID to update.
            shop_domain: New shop domain.
            access_token: New access token (will be encrypted).

        Returns:
            Updated ShopifyStore object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(ShopifyStore).where(ShopifyStore.id == store_id)
            store = session.execute(stmt).scalar_one_or_none()

            if store:
                store.shop_domain = shop_domain
                store.access_token = encrypt_token(access_token)
                store.installed_at = now_ist()
                session.commit()
                session.refresh(store)

            return store
        finally:
            session.close()

    def get_all_stores_by_user(self, user_id: int) -> List[ShopifyStore]:
        """
        Get all Shopify stores for a user.
        Automatically decrypts access_tokens.

        Args:
            user_id: User ID.

        Returns:
            List of ShopifyStore objects.
        """
        session = self.get_session()
        try:
            stmt = select(ShopifyStore).where(ShopifyStore.user_id == user_id)
            stores = session.execute(stmt).scalars().all()
            for store in stores:
                store.access_token = decrypt_token(store.access_token)
            return stores
        finally:
            session.close()

    def get_store_by_id(self, store_id: int) -> Optional[ShopifyStore]:
        """
        Get Shopify store by ID.
        Automatically decrypts the access_token.

        Args:
            store_id: Store ID.

        Returns:
            ShopifyStore object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(ShopifyStore).where(ShopifyStore.id == store_id)
            store = session.execute(stmt).scalar_one_or_none()
            if store:
                store.access_token = decrypt_token(store.access_token)
            return store
        finally:
            session.close()

    # ==================== Conversation Operations ====================

    def save_conversation(
        self,
        user_id: int,
        message: str,
        response: str,
        query_type: str,
        session_id: Optional[int] = None,
        channel_type: str = "telegram",
        tool_calls_json: Optional[str] = None,
        template_id_used: Optional[int] = None,
    ) -> Conversation:
        """
        Save a user conversation.

        Args:
            user_id: User ID.
            message: User's message.
            response: Agent's response.
            query_type: Type of query (e.g., "sales", "products", "analytics").
            session_id: Optional session ID.
            channel_type: Channel type (default "telegram").
            tool_calls_json: Optional JSON string of tool calls.
            template_id_used: Optional query template ID used.

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
                session_id=session_id,
                channel_type=channel_type,
                tool_calls_json=tool_calls_json,
                template_id_used=template_id_used,
            )
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            return conversation
        finally:
            session.close()

    def get_latest_conversation(self, user_id: int, limit: int = 1) -> List[Conversation]:
        """
        Get most recent conversation(s) for a user.

        Args:
            user_id: User ID.
            limit: Number of conversations to return.

        Returns:
            List of Conversation objects ordered by most recent first.
        """
        session = self.get_session()
        try:
            stmt = (
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.created_at.desc())
                .limit(limit)
            )
            return session.execute(stmt).scalars().all()
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
            cutoff_date = now_ist() - timedelta(days=days)
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
                pattern.last_used = now_ist()

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
                preference.updated_at = now_ist()

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

    # ==================== Tool Usage Operations ====================

    def log_tool_usage(
        self,
        user_id: int,
        tool_name: str,
        parameters: str,
        success: bool,
        execution_time_ms: int,
    ) -> ToolUsage:
        """
        Log tool usage.

        Args:
            user_id: User ID.
            tool_name: Name of the tool.
            parameters: Tool parameters (JSON string).
            success: Whether execution was successful.
            execution_time_ms: Execution time in milliseconds.

        Returns:
            ToolUsage object.
        """
        session = self.get_session()
        try:
            tool_usage = ToolUsage(
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
            cutoff_date = now_ist() - timedelta(days=days)
            stmt = select(ToolUsage).where(
                and_(
                    ToolUsage.user_id == user_id,
                    ToolUsage.created_at >= cutoff_date,
                )
            )

            if tool_name:
                stmt = stmt.where(ToolUsage.tool_name == tool_name)

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

    def get_tool_stats_by_intent(self, intent_category: str) -> List[Dict]:
        """
        Get tool usage statistics grouped by tool name for an intent category.

        Args:
            intent_category: Intent category to filter by.

        Returns:
            List of dictionaries with tool stats: {tool, successes, failures, avg_time}.
        """
        session = self.get_session()
        try:
            # Get all templates for this intent
            stmt = select(QueryTemplate).where(
                QueryTemplate.intent_category == intent_category
            )
            templates = session.execute(stmt).scalars().all()

            tools_used = {}
            for template in templates:
                tool = template.tool_name
                if tool not in tools_used:
                    tools_used[tool] = {
                        "tool": tool,
                        "successes": template.success_count or 0,
                        "failures": template.failure_count or 0,
                        "avg_time": template.avg_execution_time_ms or 0,
                    }
                else:
                    tools_used[tool]["successes"] += template.success_count or 0
                    tools_used[tool]["failures"] += template.failure_count or 0

            return list(tools_used.values())
        finally:
            session.close()

    # ==================== Query Error Learning Operations ====================

    def log_query_error(
        self,
        user_id: int,
        tool_name: str,
        query_text: str,
        error_message: str,
        error_type: str = "unknown",
        lesson: Optional[str] = None,
    ) -> QueryError:
        """
        Log a failed tool call for learning purposes.

        Args:
            user_id: User ID (0 for system-level errors).
            tool_name: Name of the tool that failed.
            query_text: The query/parameters that caused the error.
            error_message: The error message returned.
            error_type: Category of error (e.g., 'graphql_syntax', 'invalid_field',
                        'api_error', 'timeout', 'unknown').
            lesson: Optional human-readable lesson about what went wrong.

        Returns:
            QueryError object.
        """
        session = self.get_session()
        try:
            query_error = QueryError(
                user_id=user_id,
                tool_name=tool_name,
                query_text=query_text,
                error_message=error_message,
                error_type=error_type,
                lesson=lesson,
                resolved=False,
            )
            session.add(query_error)
            session.commit()
            session.refresh(query_error)
            return query_error
        finally:
            session.close()

    def get_recent_query_errors(
        self,
        user_id: int = 0,
        tool_name: Optional[str] = None,
        limit: int = 10,
        days: int = 30,
        include_resolved: bool = False,
    ) -> List[QueryError]:
        """
        Get recent query errors for learning context.

        Args:
            user_id: User ID (0 for system-level errors).
            tool_name: Optional filter by tool name.
            limit: Maximum number of errors to return.
            days: Number of days to look back.
            include_resolved: Whether to include resolved errors.

        Returns:
            List of QueryError objects sorted by most recent first.
        """
        session = self.get_session()
        try:
            cutoff_date = now_ist() - timedelta(days=days)
            conditions = [QueryError.created_at >= cutoff_date]

            # user_id=0 means system-level — include all; otherwise filter
            if user_id > 0:
                conditions.append(QueryError.user_id == user_id)

            if tool_name:
                conditions.append(QueryError.tool_name == tool_name)

            if not include_resolved:
                conditions.append(QueryError.resolved == False)

            stmt = (
                select(QueryError)
                .where(and_(*conditions))
                .order_by(QueryError.created_at.desc())
                .limit(limit)
            )
            return session.execute(stmt).scalars().all()
        finally:
            session.close()

    def get_similar_query_errors(
        self,
        tool_name: str,
        limit: int = 5,
    ) -> List[QueryError]:
        """
        Get past errors for a specific tool to help avoid repeating mistakes.

        Args:
            tool_name: Tool name to find past errors for.
            limit: Maximum number of errors to return.

        Returns:
            List of QueryError objects for the given tool.
        """
        session = self.get_session()
        try:
            stmt = (
                select(QueryError)
                .where(QueryError.tool_name == tool_name)
                .order_by(QueryError.created_at.desc())
                .limit(limit)
            )
            return session.execute(stmt).scalars().all()
        finally:
            session.close()

    def mark_error_resolved(self, error_id: int, lesson: Optional[str] = None) -> bool:
        """
        Mark a query error as resolved with an optional lesson learned.

        Args:
            error_id: QueryError ID.
            lesson: What was learned from this error.

        Returns:
            True if the error was found and updated, False otherwise.
        """
        session = self.get_session()
        try:
            stmt = select(QueryError).where(QueryError.id == error_id)
            error = session.execute(stmt).scalar_one_or_none()

            if error is None:
                return False

            error.resolved = True
            if lesson:
                error.lesson = lesson
            session.commit()
            return True
        finally:
            session.close()

    def get_error_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get a summary of query errors for monitoring.

        Args:
            days: Number of days to look back.

        Returns:
            Dictionary with error statistics.
        """
        session = self.get_session()
        try:
            cutoff_date = now_ist() - timedelta(days=days)
            stmt = select(QueryError).where(QueryError.created_at >= cutoff_date)
            errors = session.execute(stmt).scalars().all()

            total = len(errors)
            resolved = sum(1 for e in errors if e.resolved)

            # Count by tool name
            by_tool: Dict[str, int] = {}
            for e in errors:
                by_tool[e.tool_name] = by_tool.get(e.tool_name, 0) + 1

            # Count by error type
            by_type: Dict[str, int] = {}
            for e in errors:
                by_type[e.error_type] = by_type.get(e.error_type, 0) + 1

            return {
                "total_errors": total,
                "resolved": resolved,
                "unresolved": total - resolved,
                "errors_by_tool": by_tool,
                "errors_by_type": by_type,
            }
        finally:
            session.close()

    def get_error_groups(self, days: int = 30) -> List[Dict]:
        """
        Get error groups by error type and tool name.

        Args:
            days: Number of days to look back.

        Returns:
            List of dictionaries with error groups: {error_type, tool, count, lessons}.
        """
        session = self.get_session()
        try:
            cutoff_date = now_ist() - timedelta(days=days)
            stmt = select(QueryError).where(QueryError.created_at >= cutoff_date)
            errors = session.execute(stmt).scalars().all()

            groups: Dict[tuple, Dict] = {}
            for error in errors:
                key = (error.error_type, error.tool_name)
                if key not in groups:
                    groups[key] = {
                        "error_type": error.error_type,
                        "tool": error.tool_name,
                        "count": 0,
                        "lessons": [],
                    }
                groups[key]["count"] += 1
                if error.lesson and error.lesson not in groups[key]["lessons"]:
                    groups[key]["lessons"].append(error.lesson)

            return list(groups.values())
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

            if cache.expires_at < now_ist():
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

            expires_at = now_ist() + timedelta(hours=ttl_hours)

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
                AnalyticsCache.expires_at < now_ist()
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

    # ==================== Session Operations ====================

    def create_session(
        self,
        user_id: int,
        channel_type: str = "telegram",
        primary_intent: Optional[str] = None,
    ) -> Session:
        """
        Create a new session for a user.

        Args:
            user_id: User ID.
            channel_type: Channel type (default "telegram").
            primary_intent: Optional primary intent for the session.

        Returns:
            Session object.
        """
        session = self.get_session()
        try:
            new_session = Session(
                user_id=user_id,
                channel_type=channel_type,
                primary_intent=primary_intent,
                message_count=0,
            )
            session.add(new_session)
            session.commit()
            session.refresh(new_session)
            return new_session
        finally:
            session.close()

    def get_active_session(self, user_id: int) -> Optional[Session]:
        """
        Get the active (not ended) session for a user.

        Args:
            user_id: User ID.

        Returns:
            Session object or None if no active session exists.
        """
        session = self.get_session()
        try:
            stmt = (
                select(Session)
                .where(
                    and_(
                        Session.user_id == user_id,
                        Session.ended_at.is_(None),
                    )
                )
                .order_by(Session.last_message_at.desc())
                .limit(1)
            )
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    def end_session(
        self,
        session_id: int,
        summary: Optional[str] = None,
    ) -> Optional[Session]:
        """
        End a session.

        Args:
            session_id: Session ID.
            summary: Optional session summary.

        Returns:
            Updated Session object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(Session).where(Session.id == session_id)
            sess = session.execute(stmt).scalar_one_or_none()

            if sess:
                sess.ended_at = now_ist()
                if summary:
                    sess.session_summary = summary
                session.commit()
                session.refresh(sess)

            return sess
        finally:
            session.close()

    def update_session_activity(
        self,
        session_id: int,
        intent: Optional[str] = None,
    ) -> Optional[Session]:
        """
        Update session activity timestamp and optionally update intent.

        Args:
            session_id: Session ID.
            intent: Optional primary intent to update.

        Returns:
            Updated Session object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(Session).where(Session.id == session_id)
            sess = session.execute(stmt).scalar_one_or_none()

            if sess:
                sess.last_message_at = now_ist()
                sess.message_count = (sess.message_count or 0) + 1
                if intent:
                    sess.primary_intent = intent
                session.commit()
                session.refresh(sess)

            return sess
        finally:
            session.close()

    def get_session_conversations(
        self,
        session_id: int,
        limit: int = 50,
    ) -> List[Conversation]:
        """
        Get conversations for a specific session.

        Args:
            session_id: Session ID.
            limit: Maximum number of conversations to return.

        Returns:
            List of Conversation objects.
        """
        session = self.get_session()
        try:
            stmt = (
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .order_by(Conversation.created_at.asc())
                .limit(limit)
            )
            return session.execute(stmt).scalars().all()
        finally:
            session.close()

    def get_previous_session(
        self,
        user_id: int,
        current_session_id: int,
    ) -> Optional[Session]:
        """
        Get the most recent ended session before the current session.

        Args:
            user_id: User ID.
            current_session_id: Current session ID to exclude.

        Returns:
            Session object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = (
                select(Session)
                .where(
                    and_(
                        Session.user_id == user_id,
                        Session.ended_at.isnot(None),
                        Session.id != current_session_id,
                    )
                )
                .order_by(Session.ended_at.desc())
                .limit(1)
            )
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    # ==================== Query Template Operations ====================

    def create_template(
        self,
        intent_category: str,
        intent_description: str,
        tool_name: str,
        tool_parameters: str,
        created_by_user_id: Optional[int] = None,
        example_queries: Optional[List[str]] = None,
        avg_execution_time_ms: int = 0,
        confidence: float = 1.0,
    ) -> QueryTemplate:
        """
        Create a new query template.

        Args:
            intent_category: Intent category.
            intent_description: Human-readable intent description.
            tool_name: Tool name for this template.
            tool_parameters: Tool parameters template (JSON string).
            created_by_user_id: Optional user ID who created this template.
            example_queries: Optional list of example queries.
            avg_execution_time_ms: Average execution time.
            confidence: Confidence score (0.0-1.0).

        Returns:
            QueryTemplate object.
        """
        session = self.get_session()
        try:
            template = QueryTemplate(
                intent_category=intent_category,
                intent_description=intent_description,
                tool_name=tool_name,
                tool_parameters=tool_parameters,
                created_by_user_id=created_by_user_id,
                example_queries=json.dumps(example_queries) if example_queries else None,
                avg_execution_time_ms=avg_execution_time_ms,
                confidence=confidence,
                success_count=0,
                failure_count=0,
            )
            session.add(template)
            session.commit()
            session.refresh(template)
            return template
        finally:
            session.close()

    def find_template(
        self,
        intent_category: str,
        tool_name: str,
    ) -> Optional[QueryTemplate]:
        """
        Find a template by intent category and tool name, ordered by confidence.

        Args:
            intent_category: Intent category.
            tool_name: Tool name.

        Returns:
            QueryTemplate object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = (
                select(QueryTemplate)
                .where(
                    and_(
                        QueryTemplate.intent_category == intent_category,
                        QueryTemplate.tool_name == tool_name,
                    )
                )
                .order_by(QueryTemplate.confidence.desc())
                .limit(1)
            )
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    def get_best_template(
        self,
        intent_category: str,
        min_confidence: float = 0.7,
    ) -> Optional[QueryTemplate]:
        """
        Get the best template for an intent regardless of tool.

        Args:
            intent_category: Intent category.
            min_confidence: Minimum confidence threshold.

        Returns:
            QueryTemplate object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = (
                select(QueryTemplate)
                .where(
                    and_(
                        QueryTemplate.intent_category == intent_category,
                        QueryTemplate.confidence >= min_confidence,
                    )
                )
                .order_by(QueryTemplate.confidence.desc())
                .limit(1)
            )
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    def increment_template_success(
        self,
        template_id: int,
        execution_time_ms: int,
    ) -> Optional[QueryTemplate]:
        """
        Increment success count and update confidence for a template.

        Args:
            template_id: Template ID.
            execution_time_ms: Execution time in milliseconds.

        Returns:
            Updated QueryTemplate object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(QueryTemplate).where(QueryTemplate.id == template_id)
            template = session.execute(stmt).scalar_one_or_none()

            if template:
                template.success_count = (template.success_count or 0) + 1
                total = (template.success_count or 0) + (template.failure_count or 0)
                template.confidence = (
                    (template.success_count or 0) / total if total > 0 else 0.5
                )
                # Update average execution time
                if template.avg_execution_time_ms:
                    template.avg_execution_time_ms = (
                        (template.avg_execution_time_ms + execution_time_ms) / 2
                    )
                else:
                    template.avg_execution_time_ms = execution_time_ms
                template.last_used_at = now_ist()
                session.commit()
                session.refresh(template)

            return template
        finally:
            session.close()

    def increment_template_failure(
        self,
        template_id: int,
    ) -> Optional[QueryTemplate]:
        """
        Increment failure count and update confidence for a template.

        Args:
            template_id: Template ID.

        Returns:
            Updated QueryTemplate object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(QueryTemplate).where(QueryTemplate.id == template_id)
            template = session.execute(stmt).scalar_one_or_none()

            if template:
                template.failure_count = (template.failure_count or 0) + 1
                total = (template.success_count or 0) + (template.failure_count or 0)
                template.confidence = (
                    (template.success_count or 0) / total if total > 0 else 0.5
                )
                session.commit()
                session.refresh(template)

            return template
        finally:
            session.close()

    def get_templates_by_intent(
        self,
        intent_category: str,
        min_confidence: float = 0.7,
        limit: int = 5,
    ) -> List[QueryTemplate]:
        """
        Get templates for an intent category above a confidence threshold.

        Args:
            intent_category: Intent category.
            min_confidence: Minimum confidence threshold.
            limit: Maximum number of templates to return.

        Returns:
            List of QueryTemplate objects sorted by confidence descending.
        """
        session = self.get_session()
        try:
            stmt = (
                select(QueryTemplate)
                .where(
                    and_(
                        QueryTemplate.intent_category == intent_category,
                        QueryTemplate.confidence >= min_confidence,
                    )
                )
                .order_by(QueryTemplate.confidence.desc())
                .limit(limit)
            )
            return session.execute(stmt).scalars().all()
        finally:
            session.close()

    def add_template_example(
        self,
        template_id: int,
        example_query: str,
    ) -> Optional[QueryTemplate]:
        """
        Add an example query to a template's example_queries JSON array.

        Args:
            template_id: Template ID.
            example_query: Example query string to add.

        Returns:
            Updated QueryTemplate object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(QueryTemplate).where(QueryTemplate.id == template_id)
            template = session.execute(stmt).scalar_one_or_none()

            if template:
                examples = []
                if template.example_queries:
                    try:
                        parsed = json.loads(template.example_queries)
                        if isinstance(parsed, list):
                            examples = parsed
                        elif isinstance(parsed, str):
                            # Handle malformed data: single string instead of array
                            examples = [parsed] if parsed else []
                        else:
                            examples = []
                    except (json.JSONDecodeError, TypeError):
                        # Corrupted JSON — start fresh
                        examples = []
                if example_query not in examples:
                    examples.append(example_query)
                template.example_queries = json.dumps(examples)
                session.commit()
                session.refresh(template)

            return template
        finally:
            session.close()

    # ==================== Error Recovery Pattern Operations ====================

    def create_recovery_pattern(
        self,
        error_type: str,
        failed_tool_name: str,
        failed_parameters: str,
        error_fingerprint: str,
        recovery_tool_name: str,
        recovery_parameters: str,
        recovery_description: str,
    ) -> ErrorRecoveryPattern:
        """
        Create a new error recovery pattern.

        Args:
            error_type: Type of error.
            failed_tool_name: Tool name that failed.
            failed_parameters: Parameters of the failed tool call.
            error_fingerprint: Unique fingerprint of the error.
            recovery_tool_name: Tool name to recover with.
            recovery_parameters: Parameters for recovery tool.
            recovery_description: Description of the recovery.

        Returns:
            ErrorRecoveryPattern object.
        """
        session = self.get_session()
        try:
            pattern = ErrorRecoveryPattern(
                error_type=error_type,
                failed_tool_name=failed_tool_name,
                failed_parameters=failed_parameters,
                error_fingerprint=error_fingerprint,
                recovery_tool_name=recovery_tool_name,
                recovery_parameters=recovery_parameters,
                recovery_description=recovery_description,
                times_applied=0,
                times_succeeded=0,
                confidence=0.5,
            )
            session.add(pattern)
            session.commit()
            session.refresh(pattern)
            return pattern
        finally:
            session.close()

    def find_recovery_pattern(
        self,
        error_fingerprint: str,
    ) -> Optional[ErrorRecoveryPattern]:
        """
        Find a recovery pattern by error fingerprint.

        Args:
            error_fingerprint: Error fingerprint.

        Returns:
            ErrorRecoveryPattern object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(ErrorRecoveryPattern).where(
                ErrorRecoveryPattern.error_fingerprint == error_fingerprint
            )
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    def find_recovery_by_type(
        self,
        error_type: str,
        tool_name: str,
    ) -> Optional[ErrorRecoveryPattern]:
        """
        Find the highest confidence recovery pattern for an error type and tool.

        Args:
            error_type: Error type.
            tool_name: Tool name that failed.

        Returns:
            ErrorRecoveryPattern object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = (
                select(ErrorRecoveryPattern)
                .where(
                    and_(
                        ErrorRecoveryPattern.error_type == error_type,
                        ErrorRecoveryPattern.failed_tool_name == tool_name,
                    )
                )
                .order_by(ErrorRecoveryPattern.confidence.desc())
                .limit(1)
            )
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    def increment_recovery_success(
        self,
        pattern_id: int,
    ) -> Optional[ErrorRecoveryPattern]:
        """
        Increment success counts and update confidence for a recovery pattern.

        Args:
            pattern_id: Pattern ID.

        Returns:
            Updated ErrorRecoveryPattern object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(ErrorRecoveryPattern).where(ErrorRecoveryPattern.id == pattern_id)
            pattern = session.execute(stmt).scalar_one_or_none()

            if pattern:
                pattern.times_applied = (pattern.times_applied or 0) + 1
                pattern.times_succeeded = (pattern.times_succeeded or 0) + 1
                total = pattern.times_applied or 1
                pattern.confidence = (pattern.times_succeeded or 0) / total
                pattern.last_used_at = now_ist()
                session.commit()
                session.refresh(pattern)

            return pattern
        finally:
            session.close()

    def increment_recovery_applied(
        self,
        pattern_id: int,
    ) -> Optional[ErrorRecoveryPattern]:
        """
        Increment applied count and update confidence for a recovery pattern (without success).

        Args:
            pattern_id: Pattern ID.

        Returns:
            Updated ErrorRecoveryPattern object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(ErrorRecoveryPattern).where(ErrorRecoveryPattern.id == pattern_id)
            pattern = session.execute(stmt).scalar_one_or_none()

            if pattern:
                pattern.times_applied = (pattern.times_applied or 0) + 1
                total = pattern.times_applied or 1
                pattern.confidence = (pattern.times_succeeded or 0) / total
                session.commit()
                session.refresh(pattern)

            return pattern
        finally:
            session.close()

    def get_top_recovery_patterns(
        self,
        min_confidence: float = 0.8,
        min_times_applied: int = 2,
        limit: int = 5,
    ) -> List[ErrorRecoveryPattern]:
        """
        Get top recovery patterns above confidence threshold and application count.

        Args:
            min_confidence: Minimum confidence threshold.
            min_times_applied: Minimum number of times applied.
            limit: Maximum number of patterns to return.

        Returns:
            List of ErrorRecoveryPattern objects sorted by confidence descending.
        """
        session = self.get_session()
        try:
            stmt = (
                select(ErrorRecoveryPattern)
                .where(
                    and_(
                        ErrorRecoveryPattern.confidence >= min_confidence,
                        ErrorRecoveryPattern.times_applied >= min_times_applied,
                    )
                )
                .order_by(ErrorRecoveryPattern.confidence.desc())
                .limit(limit)
            )
            return session.execute(stmt).scalars().all()
        finally:
            session.close()

    # ==================== Global Insight Operations ====================

    def upsert_global_insight(
        self,
        insight_type: str,
        insight_key: str,
        insight_value: str,
        sample_size: int = 1,
        confidence: float = 0.5,
    ) -> GlobalInsight:
        """
        Create or update a global insight (upsert).

        Args:
            insight_type: Type of insight.
            insight_key: Key for the insight.
            insight_value: Value of the insight.
            sample_size: Number of samples this insight is based on.
            confidence: Confidence score.

        Returns:
            GlobalInsight object.
        """
        session = self.get_session()
        try:
            stmt = select(GlobalInsight).where(
                and_(
                    GlobalInsight.insight_type == insight_type,
                    GlobalInsight.insight_key == insight_key,
                )
            )
            insight = session.execute(stmt).scalar_one_or_none()

            if insight is None:
                insight = GlobalInsight(
                    insight_type=insight_type,
                    insight_key=insight_key,
                    insight_value=insight_value,
                    sample_size=sample_size,
                    confidence=confidence,
                )
            else:
                insight.insight_value = insight_value
                insight.sample_size = (insight.sample_size or 0) + sample_size
                insight.confidence = confidence
                insight.updated_at = now_ist()

            session.add(insight)
            session.commit()
            session.refresh(insight)
            return insight
        finally:
            session.close()

    def get_global_insight(
        self,
        insight_type: str,
        insight_key: str,
    ) -> Optional[GlobalInsight]:
        """
        Get a specific global insight.

        Args:
            insight_type: Type of insight.
            insight_key: Key for the insight.

        Returns:
            GlobalInsight object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(GlobalInsight).where(
                and_(
                    GlobalInsight.insight_type == insight_type,
                    GlobalInsight.insight_key == insight_key,
                )
            )
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    def get_global_insights_by_type(
        self,
        insight_type: str,
        limit: int = 10,
    ) -> List[GlobalInsight]:
        """
        Get all global insights of a specific type.

        Args:
            insight_type: Type of insight.
            limit: Maximum number of insights to return.

        Returns:
            List of GlobalInsight objects.
        """
        session = self.get_session()
        try:
            stmt = (
                select(GlobalInsight)
                .where(GlobalInsight.insight_type == insight_type)
                .order_by(GlobalInsight.updated_at.desc())
                .limit(limit)
            )
            return session.execute(stmt).scalars().all()
        finally:
            session.close()

    def delete_stale_insights(self, max_age_days: int = 90) -> int:
        """
        Delete global insights older than max_age_days.

        Args:
            max_age_days: Maximum age in days.

        Returns:
            Number of deleted insights.
        """
        session = self.get_session()
        try:
            cutoff_date = now_ist() - timedelta(days=max_age_days)
            stmt = select(GlobalInsight).where(GlobalInsight.updated_at < cutoff_date)
            stale = session.execute(stmt).scalars().all()

            for insight in stale:
                session.delete(insight)

            session.commit()
            return len(stale)
        finally:
            session.close()

    # ==================== Response Feedback Operations ====================

    def save_response_feedback(
        self,
        conversation_id: int,
        user_id: int,
        feedback_type: str,
        quality_score: float,
        query_type: Optional[str] = None,
        tool_used: Optional[str] = None,
        signal_text: Optional[str] = None,
    ) -> ResponseFeedback:
        """
        Save response feedback from a user.

        Args:
            conversation_id: Conversation ID.
            user_id: User ID.
            feedback_type: Type of feedback (e.g., "like", "dislike", "corrected").
            quality_score: Quality score (0.0-1.0).
            query_type: Optional query type.
            tool_used: Optional tool that was used.
            signal_text: Optional text signal from user.

        Returns:
            ResponseFeedback object.
        """
        session = self.get_session()
        try:
            feedback = ResponseFeedback(
                conversation_id=conversation_id,
                user_id=user_id,
                feedback_type=feedback_type,
                quality_score=quality_score,
                query_type=query_type,
                tool_used=tool_used,
                signal_text=signal_text,
            )
            session.add(feedback)
            session.commit()
            session.refresh(feedback)
            return feedback
        finally:
            session.close()

    def get_avg_quality_score(self, user_id: int, days: int = 30) -> float:
        """
        Get average quality score for a user's responses.

        Args:
            user_id: User ID.
            days: Number of days to look back.

        Returns:
            Average quality score (0.0-1.0).
        """
        session = self.get_session()
        try:
            cutoff_date = now_ist() - timedelta(days=days)
            stmt = (
                select(func.avg(ResponseFeedback.quality_score))
                .where(
                    and_(
                        ResponseFeedback.user_id == user_id,
                        ResponseFeedback.created_at >= cutoff_date,
                    )
                )
            )
            result = session.execute(stmt).scalar()
            return float(result) if result else 0.0
        finally:
            session.close()

    def update_conversation_quality(
        self,
        conversation_id: int,
        quality_score: float,
    ) -> Optional[Conversation]:
        """
        Update quality score on a conversation.

        Args:
            conversation_id: Conversation ID.
            quality_score: Quality score to set.

        Returns:
            Updated Conversation object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(Conversation).where(Conversation.id == conversation_id)
            conversation = session.execute(stmt).scalar_one_or_none()

            if conversation:
                conversation.response_quality_score = quality_score
                session.commit()
                session.refresh(conversation)

            return conversation
        finally:
            session.close()

    def update_conversation_sub_scores(
        self,
        conversation_id: int,
        sub_scores: dict,
    ) -> None:
        """Persist multi-factor quality sub-scores on a conversation.

        Args:
            conversation_id: Conversation ID.
            sub_scores: Dict with keys: completeness, sentiment, tool_performance, composite, reasoning.
        """
        import json

        session = self.get_session()
        try:
            conv = session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            ).scalar_one_or_none()
            if conv:
                conv.quality_completeness_score = sub_scores.get("completeness")
                conv.quality_sentiment_score = sub_scores.get("sentiment")
                conv.quality_tool_score = sub_scores.get("tool_performance")
                conv.quality_factors_json = json.dumps(sub_scores)
                session.commit()
        finally:
            session.close()

    # ==================== Channel Session Operations ====================

    def get_or_create_channel_session(
        self,
        user_id: int,
        channel_type: str,
        channel_user_id: str,
        channel_username: Optional[str] = None,
        channel_metadata: Optional[str] = None,
    ) -> ChannelSession:
        """
        Get or create a channel session for a user.

        Args:
            user_id: User ID.
            channel_type: Type of channel (e.g., "telegram").
            channel_user_id: User ID on the channel.
            channel_username: Optional username on the channel.
            channel_metadata: Optional metadata JSON string.

        Returns:
            ChannelSession object.
        """
        session = self.get_session()
        try:
            stmt = select(ChannelSession).where(
                and_(
                    ChannelSession.user_id == user_id,
                    ChannelSession.channel_type == channel_type,
                )
            )
            ch_session = session.execute(stmt).scalar_one_or_none()

            if ch_session is None:
                ch_session = ChannelSession(
                    user_id=user_id,
                    channel_type=channel_type,
                    channel_user_id=channel_user_id,
                    channel_username=channel_username,
                    channel_metadata=channel_metadata,
                )
                session.add(ch_session)
                session.commit()
                session.refresh(ch_session)

            return ch_session
        finally:
            session.close()

    def get_user_by_channel(
        self,
        channel_type: str,
        channel_user_id: str,
    ) -> Optional[User]:
        """
        Get a user by their channel ID and channel type.

        Args:
            channel_type: Type of channel.
            channel_user_id: User ID on the channel.

        Returns:
            User object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(ChannelSession).where(
                and_(
                    ChannelSession.channel_type == channel_type,
                    ChannelSession.channel_user_id == channel_user_id,
                )
            )
            ch_session = session.execute(stmt).scalar_one_or_none()

            if ch_session is None:
                return None

            user_stmt = select(User).where(User.id == ch_session.user_id)
            return session.execute(user_stmt).scalar_one_or_none()
        finally:
            session.close()

    def update_channel_activity(self, channel_session_id: int) -> Optional[ChannelSession]:
        """
        Update the last_active timestamp for a channel session.

        Args:
            channel_session_id: Channel session ID.

        Returns:
            Updated ChannelSession object or None if not found.
        """
        session = self.get_session()
        try:
            stmt = select(ChannelSession).where(ChannelSession.id == channel_session_id)
            ch_session = session.execute(stmt).scalar_one_or_none()

            if ch_session:
                ch_session.last_active = now_ist()
                session.commit()
                session.refresh(ch_session)

            return ch_session
        finally:
            session.close()

    # ==================== Store Management Operations ====================

    def reset_store_learning_data(self, user_id: int) -> Dict[str, int]:
        """
        Clear all learning data for a user when switching stores.

        Preserves User record and ShopifyStore record — only wipes
        store-specific learning data.

        Args:
            user_id: User ID whose learning data to reset.

        Returns:
            Dictionary with counts of deleted items per table.
        """
        session = self.get_session()
        try:
            counts = {}

            # Delete response feedback first (FK → conversations)
            feedback = session.execute(
                select(ResponseFeedback).where(ResponseFeedback.user_id == user_id)
            ).scalars().all()
            counts["response_feedback"] = len(feedback)
            for fb in feedback:
                session.delete(fb)

            # Delete conversations
            conversations = session.execute(
                select(Conversation).where(Conversation.user_id == user_id)
            ).scalars().all()
            counts["conversations"] = len(conversations)
            for conv in conversations:
                session.delete(conv)

            # Delete sessions
            sessions_list = session.execute(
                select(Session).where(Session.user_id == user_id)
            ).scalars().all()
            counts["sessions"] = len(sessions_list)
            for sess in sessions_list:
                session.delete(sess)

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

            # Delete tool usage
            tool_usage = session.execute(
                select(ToolUsage).where(ToolUsage.user_id == user_id)
            ).scalars().all()
            counts["tool_usage"] = len(tool_usage)
            for usage in tool_usage:
                session.delete(usage)

            # Delete query errors
            query_errors = session.execute(
                select(QueryError).where(QueryError.user_id == user_id)
            ).scalars().all()
            counts["query_errors"] = len(query_errors)
            for qe in query_errors:
                session.delete(qe)

            # Clear analytics cache
            stores = session.execute(
                select(ShopifyStore).where(ShopifyStore.user_id == user_id)
            ).scalars().all()
            cache_count = 0
            for store in stores:
                caches = session.execute(
                    select(AnalyticsCache).where(AnalyticsCache.store_id == store.id)
                ).scalars().all()
                cache_count += len(caches)
                for cache in caches:
                    session.delete(cache)
            counts["analytics_cache"] = cache_count

            # Clear global templates and recovery patterns
            templates = session.execute(select(QueryTemplate)).scalars().all()
            counts["query_templates"] = len(templates)
            for tmpl in templates:
                session.delete(tmpl)

            recovery_patterns = session.execute(
                select(ErrorRecoveryPattern)
            ).scalars().all()
            counts["error_recovery_patterns"] = len(recovery_patterns)
            for rp in recovery_patterns:
                session.delete(rp)

            global_insights = session.execute(
                select(GlobalInsight)
            ).scalars().all()
            counts["global_insights"] = len(global_insights)
            for gi in global_insights:
                session.delete(gi)

            # Reset user interaction count
            user = session.execute(
                select(User).where(User.id == user_id)
            ).scalar_one_or_none()
            if user:
                user.interaction_count = 0

            session.commit()
            return counts
        finally:
            session.close()

    # ==================== Data Cleanup Operations ====================

    def clear_user_data(self, user_id: int) -> Dict[str, int]:
        """
        Clear all data for a user (conversations, patterns, preferences, sessions, etc.).

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

            # Delete tool usage
            tool_usage = session.execute(
                select(ToolUsage).where(ToolUsage.user_id == user_id)
            ).scalars().all()
            counts["tool_usage"] = len(tool_usage)
            for usage in tool_usage:
                session.delete(usage)

            # Delete query errors
            query_errors = session.execute(
                select(QueryError).where(QueryError.user_id == user_id)
            ).scalars().all()
            counts["query_errors"] = len(query_errors)
            for qe in query_errors:
                session.delete(qe)

            # Delete sessions
            sessions = session.execute(
                select(Session).where(Session.user_id == user_id)
            ).scalars().all()
            counts["sessions"] = len(sessions)
            for sess in sessions:
                session.delete(sess)

            # Delete channel sessions
            channel_sessions = session.execute(
                select(ChannelSession).where(ChannelSession.user_id == user_id)
            ).scalars().all()
            counts["channel_sessions"] = len(channel_sessions)
            for ch_sess in channel_sessions:
                session.delete(ch_sess)

            # Delete response feedback
            feedback = session.execute(
                select(ResponseFeedback).where(ResponseFeedback.user_id == user_id)
            ).scalars().all()
            counts["response_feedback"] = len(feedback)
            for fb in feedback:
                session.delete(fb)

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

    # ── WhatsApp / Multi-channel operations ────────────────────────

    def get_channel_session(
        self,
        channel_type: str,
        channel_user_id: str,
    ) -> Optional[ChannelSession]:
        """Look up an existing ChannelSession by channel type and user ID.

        Args:
            channel_type: e.g. "whatsapp", "telegram"
            channel_user_id: Channel-specific identifier (phone number, telegram ID, etc.)

        Returns:
            ChannelSession or None.
        """
        session = self.get_session()
        try:
            stmt = select(ChannelSession).where(
                and_(
                    ChannelSession.channel_type == channel_type,
                    ChannelSession.channel_user_id == channel_user_id,
                )
            )
            return session.execute(stmt).scalar_one_or_none()
        finally:
            session.close()

    def create_channel_session(
        self,
        user_id: int,
        channel_type: str,
        channel_user_id: str,
        channel_username: Optional[str] = None,
        channel_metadata: Optional[str] = None,
    ) -> ChannelSession:
        """Create a new ChannelSession linking a channel identifier to an internal user.

        Args:
            user_id: Internal User.id
            channel_type: e.g. "whatsapp", "telegram"
            channel_user_id: Channel-specific identifier
            channel_username: Optional display name / username
            channel_metadata: Optional JSON metadata

        Returns:
            Newly created ChannelSession.
        """
        session = self.get_session()
        try:
            cs = ChannelSession(
                user_id=user_id,
                channel_type=channel_type,
                channel_user_id=channel_user_id,
                channel_username=channel_username,
                channel_metadata=channel_metadata,
                is_active=True,
                created_at=now_ist(),
                last_active=now_ist(),
            )
            session.add(cs)
            session.commit()
            session.refresh(cs)
            return cs
        finally:
            session.close()

    def update_channel_session_activity(self, channel_session_id: int) -> None:
        """Touch the last_active timestamp on a ChannelSession."""
        session = self.get_session()
        try:
            cs = session.get(ChannelSession, channel_session_id)
            if cs:
                cs.last_active = now_ist()
                session.commit()
        finally:
            session.close()

    def merge_users(
        self,
        keep_user_id: int,
        merge_user_id: int,
        target_channel: str,
        target_channel_id: str,
    ) -> None:
        """Merge two user accounts into one canonical identity.

        Re-points all data from ``merge_user_id`` to ``keep_user_id``,
        copies over the channel-specific identifier, updates channel
        sessions, and deletes the merged user.

        This is used when a user links their Telegram and WhatsApp
        accounts via the /link command.

        Args:
            keep_user_id: The user ID to keep (canonical).
            merge_user_id: The user ID to merge into keep and then delete.
            target_channel: The channel type of the merged user ("telegram" or "whatsapp").
            target_channel_id: The channel-specific ID of the merged user.
        """
        from src.utils.logger import get_logger
        _logger = get_logger(__name__)

        session = self.get_session()
        try:
            keep_user = session.get(User, keep_user_id)
            merge_user = session.get(User, merge_user_id)

            if not keep_user or not merge_user:
                raise ValueError(f"User not found: keep={keep_user_id}, merge={merge_user_id}")

            # ── Step 1: Copy channel identifiers to the kept user ──
            if target_channel == "whatsapp" and merge_user.whatsapp_phone:
                wp = merge_user.whatsapp_phone
                # 1. Clear it on the merged user via raw SQL to bypass ORM batching
                session.execute(
                    User.__table__.update()
                    .where(User.id == merge_user_id)
                    .values(whatsapp_phone=None)
                )
                # 2. Assign it to keep user via raw SQL
                session.execute(
                    User.__table__.update()
                    .where(User.id == keep_user_id)
                    .values(whatsapp_phone=wp)
                )
            elif target_channel == "telegram" and merge_user.telegram_user_id:
                tg = merge_user.telegram_user_id
                session.execute(
                    User.__table__.update()
                    .where(User.id == merge_user_id)
                    .values(telegram_user_id=None)
                )
                session.execute(
                    User.__table__.update()
                    .where(User.id == keep_user_id)
                    .values(telegram_user_id=tg)
                )

            # Hard commit the raw ID shift immediately
            session.commit()
            
            # Refresh our local objects so they recognize the new IDs
            session.refresh(keep_user)
            session.refresh(merge_user)

            # Copy display info if the kept user is missing it
            if not keep_user.first_name and merge_user.first_name:
                keep_user.first_name = merge_user.first_name
            if not keep_user.display_name and merge_user.display_name:
                keep_user.display_name = merge_user.display_name

            # Merge interaction counts
            keep_user.interaction_count += merge_user.interaction_count

            # Carry over verification status
            if merge_user.is_verified:
                keep_user.is_verified = True

            session.flush()  # Ensure unique constraints are clear

            # ── Step 2: Re-point all foreign-key references ──

            # Conversations
            session.execute(
                Conversation.__table__.update()
                .where(Conversation.user_id == merge_user_id)
                .values(user_id=keep_user_id)
            )

            # Sessions
            session.execute(
                Session.__table__.update()
                .where(Session.user_id == merge_user_id)
                .values(user_id=keep_user_id)
            )

            # Query patterns (handle potential key conflicts)
            merge_patterns = session.execute(
                select(QueryPattern).where(QueryPattern.user_id == merge_user_id)
            ).scalars().all()
            
            for pattern in merge_patterns:
                existing = session.execute(
                    select(QueryPattern).where(
                        and_(
                            QueryPattern.user_id == keep_user_id,
                            QueryPattern.pattern_type == pattern.pattern_type,
                            QueryPattern.pattern_value == pattern.pattern_value,
                        )
                    )
                ).scalar_one_or_none()
                
                if existing:
                    # Merge frequencies and advance last_used
                    existing.frequency += pattern.frequency
                    if pattern.last_used > existing.last_used:
                        existing.last_used = pattern.last_used
                    session.delete(pattern)
                else:
                    pattern.user_id = keep_user_id

            # User preferences (handle potential key conflicts)
            merge_prefs = session.execute(
                select(UserPreference).where(UserPreference.user_id == merge_user_id)
            ).scalars().all()
            for pref in merge_prefs:
                existing = session.execute(
                    select(UserPreference).where(
                        and_(
                            UserPreference.user_id == keep_user_id,
                            UserPreference.preference_key == pref.preference_key,
                        )
                    )
                ).scalar_one_or_none()
                if existing:
                    # Keep the higher-confidence one
                    if pref.confidence_score > existing.confidence_score:
                        existing.preference_value = pref.preference_value
                        existing.confidence_score = pref.confidence_score
                    session.delete(pref)
                else:
                    pref.user_id = keep_user_id

            # Tool usage
            session.execute(
                ToolUsage.__table__.update()
                .where(ToolUsage.user_id == merge_user_id)
                .values(user_id=keep_user_id)
            )

            # Query errors
            session.execute(
                QueryError.__table__.update()
                .where(QueryError.user_id == merge_user_id)
                .values(user_id=keep_user_id)
            )

            # Response feedback
            session.execute(
                ResponseFeedback.__table__.update()
                .where(ResponseFeedback.user_id == merge_user_id)
                .values(user_id=keep_user_id)
            )

            # Channel sessions — re-point to kept user
            session.execute(
                ChannelSession.__table__.update()
                .where(ChannelSession.user_id == merge_user_id)
                .values(user_id=keep_user_id)
            )

            # Shopify stores — move if kept user doesn't have one
            keep_stores = session.execute(
                select(ShopifyStore).where(ShopifyStore.user_id == keep_user_id)
            ).scalars().all()
            if not keep_stores:
                session.execute(
                    ShopifyStore.__table__.update()
                    .where(ShopifyStore.user_id == merge_user_id)
                    .values(user_id=keep_user_id)
                )
            else:
                merge_stores = session.execute(
                    select(ShopifyStore).where(ShopifyStore.user_id == merge_user_id)
                ).scalars().all()
                for store in merge_stores:
                    session.delete(store)

            # ── Step 3: Delete the merged user ──
            session.delete(merge_user)

            session.commit()

            _logger.info(
                "Users merged successfully",
                keep_user_id=keep_user_id,
                merged_user_id=merge_user_id,
                target_channel=target_channel,
            )

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_whatsapp_user(
        self,
        whatsapp_phone: str,
        display_name: str = "",
        first_name: str = "",
    ) -> User:
        """Create a new User originating from WhatsApp (no Telegram ID).

        If a user with this phone already exists, return them instead.

        Args:
            whatsapp_phone: WhatsApp phone number (digits only, no +).
            display_name: Profile display name.
            first_name: First name extracted from profile.

        Returns:
            User object.
        """
        session = self.get_session()
        try:
            # Check if user already exists by phone
            stmt = select(User).where(User.whatsapp_phone == whatsapp_phone)
            user = session.execute(stmt).scalar_one_or_none()

            if user is None:
                user = User(
                    telegram_user_id=None,
                    whatsapp_phone=whatsapp_phone,
                    display_name=display_name or whatsapp_phone,
                    first_name=first_name,
                )
                session.add(user)
                session.commit()
                session.refresh(user)

            return user
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Embedding vector operations (for semantic search)
    # ------------------------------------------------------------------

    def upsert_embedding(
        self,
        entity_type: str,
        entity_id: int,
        embedding_bytes: bytes,
        source_text: Optional[str] = None,
    ) -> None:
        """Insert or update an embedding vector.

        Args:
            entity_type: ``"template"`` | ``"error_lesson"``
            entity_id:   Primary key in the source table.
            embedding_bytes: Raw bytes from ``numpy_array.tobytes()``.
            source_text: Original text that was embedded (for debugging).
        """
        session = self.get_session()
        try:
            stmt = select(EmbeddingVector).where(
                and_(
                    EmbeddingVector.entity_type == entity_type,
                    EmbeddingVector.entity_id == entity_id,
                )
            )
            existing = session.execute(stmt).scalar_one_or_none()

            if existing:
                existing.embedding_blob = embedding_bytes
                existing.source_text = source_text
            else:
                new_row = EmbeddingVector(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    embedding_blob=embedding_bytes,
                    source_text=source_text,
                )
                session.add(new_row)

            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def get_all_embeddings(
        self,
        entity_type: str,
    ) -> list:
        """Retrieve all embeddings for a given entity type.

        Args:
            entity_type: ``"template"`` or ``"error_lesson"``

        Returns:
            List of ``(entity_id, numpy_array, source_text)`` tuples.
            The numpy array is float32 with shape ``(384,)``.
        """
        import numpy as np

        session = self.get_session()
        try:
            stmt = select(EmbeddingVector).where(
                EmbeddingVector.entity_type == entity_type
            )
            rows = session.execute(stmt).scalars().all()

            results = []
            for row in rows:
                try:
                    vec = np.frombuffer(row.embedding_blob, dtype=np.float32).copy()
                except Exception:
                    vec = None
                results.append((row.entity_id, vec, row.source_text))

            return results
        finally:
            session.close()

    def delete_embedding(
        self,
        entity_type: str,
        entity_id: int,
    ) -> bool:
        """Delete an embedding vector.

        Args:
            entity_type: ``"template"`` or ``"error_lesson"``
            entity_id:   Primary key in the source table.

        Returns:
            True if a row was deleted, False if not found.
        """
        session = self.get_session()
        try:
            stmt = select(EmbeddingVector).where(
                and_(
                    EmbeddingVector.entity_type == entity_type,
                    EmbeddingVector.entity_id == entity_id,
                )
            )
            existing = session.execute(stmt).scalar_one_or_none()

            if existing:
                session.delete(existing)
                session.commit()
                return True
            return False
        finally:
            session.close()
