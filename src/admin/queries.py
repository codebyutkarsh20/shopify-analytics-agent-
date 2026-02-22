"""Dashboard-specific read-only database queries.

Kept separate from the main ``DatabaseOperations`` (already 2800+ lines)
to avoid bloating it further.  All queries here are read-only.
"""

import os
from datetime import datetime, timedelta

from sqlalchemy import Integer, func, select, desc, asc, and_, cast
from sqlalchemy.orm import Session as SASession

from src.database.models import (
    AdminUser,
    AnalyticsCache,
    Conversation,
    ChannelSession,
    QueryError,
    ResponseFeedback,
    Session,
    ToolUsage,
    User,
)
from src.database.operations import DatabaseOperations
from src.utils.timezone import now_ist


class DashboardQueries:
    """Read-only queries powering the admin dashboard."""

    def __init__(self, db_ops: DatabaseOperations):
        self._db_ops = db_ops

    # -- helper -----------------------------------------------------------------

    def _session(self) -> SASession:
        return self._db_ops.SessionLocal()

    # -- admin user CRUD -------------------------------------------------------

    def get_admin_by_username(self, username: str) -> AdminUser | None:
        with self._session() as s:
            return s.execute(
                select(AdminUser).where(AdminUser.username == username)
            ).scalar_one_or_none()

    def create_admin_user(self, username: str, password_hash: str, must_change: bool = True) -> AdminUser:
        with self._session() as s:
            admin = AdminUser(
                username=username,
                password_hash=password_hash,
                must_change_password=must_change,
            )
            s.add(admin)
            s.commit()
            s.refresh(admin)
            return admin

    def update_admin_last_login(self, username: str) -> None:
        with self._session() as s:
            admin = s.execute(
                select(AdminUser).where(AdminUser.username == username)
            ).scalar_one_or_none()
            if admin:
                admin.last_login = now_ist()
                s.commit()

    def update_admin_password(self, username: str, password_hash: str) -> bool:
        with self._session() as s:
            admin = s.execute(
                select(AdminUser).where(AdminUser.username == username)
            ).scalar_one_or_none()
            if not admin:
                return False
            admin.password_hash = password_hash
            admin.must_change_password = False
            s.commit()
            return True

    def admin_must_change_password(self, username: str) -> bool:
        with self._session() as s:
            admin = s.execute(
                select(AdminUser).where(AdminUser.username == username)
            ).scalar_one_or_none()
            return admin.must_change_password if admin else False

    # -- users -----------------------------------------------------------------

    _ALLOWED_USER_SORT_COLS = {
        "last_active", "created_at", "interaction_count", "display_name", "first_name",
    }

    def get_users_paginated(
        self,
        page: int = 1,
        limit: int = 20,
        sort_by: str = "last_active",
        order: str = "desc",
    ) -> tuple[list[User], int]:
        if sort_by not in self._ALLOWED_USER_SORT_COLS:
            sort_by = "last_active"
        with self._session() as s:
            total = s.execute(select(func.count(User.id))).scalar() or 0

            col = getattr(User, sort_by, User.last_active)
            order_fn = desc if order == "desc" else asc
            users = list(
                s.execute(
                    select(User)
                    .order_by(order_fn(col))
                    .offset((page - 1) * limit)
                    .limit(limit)
                ).scalars().all()
            )
            # Detach from session
            for u in users:
                s.expunge(u)
            return users, total

    def get_user_detail(self, user_id: int) -> dict | None:
        with self._session() as s:
            user = s.get(User, user_id)
            if not user:
                return None

            conv_count = s.execute(
                select(func.count(Conversation.id)).where(Conversation.user_id == user_id)
            ).scalar() or 0

            fb_count = s.execute(
                select(func.count(ResponseFeedback.id)).where(ResponseFeedback.user_id == user_id)
            ).scalar() or 0

            avg_q = s.execute(
                select(func.avg(Conversation.response_quality_score)).where(
                    and_(Conversation.user_id == user_id, Conversation.response_quality_score.isnot(None))
                )
            ).scalar()

            channels = [
                row[0]
                for row in s.execute(
                    select(ChannelSession.channel_type).where(ChannelSession.user_id == user_id)
                ).all()
            ]

            return {
                "id": user.id,
                "telegram_user_id": user.telegram_user_id,
                "telegram_username": user.telegram_username,
                "whatsapp_phone": user.whatsapp_phone,
                "display_name": user.display_name,
                "first_name": user.first_name,
                "is_verified": user.is_verified,
                "interaction_count": user.interaction_count,
                "created_at": user.created_at,
                "last_active": user.last_active,
                "conversation_count": conv_count,
                "feedback_count": fb_count,
                "avg_quality_score": round(avg_q, 2) if avg_q else None,
                "channels": channels,
            }

    # -- sessions --------------------------------------------------------------

    def get_sessions_paginated(
        self,
        page: int = 1,
        limit: int = 20,
        user_id: int | None = None,
        channel: str | None = None,
    ) -> tuple[list[dict], int]:
        """List sessions with user info and message counts."""
        with self._session() as s:
            filters = []
            if user_id:
                filters.append(Session.user_id == user_id)
            if channel:
                filters.append(Session.channel_type == channel)

            where = and_(*filters) if filters else True

            total = s.execute(
                select(func.count(Session.id)).where(where)
            ).scalar() or 0

            rows = s.execute(
                select(Session, User.display_name)
                .join(User, Session.user_id == User.id)
                .where(where)
                .order_by(desc(Session.started_at))
                .offset((page - 1) * limit)
                .limit(limit)
            ).all()

            # Batch-fetch first message for all sessions (avoids N+1).
            # SQLite-compatible: get min conversation ID per session,
            # then fetch those rows.
            session_ids = [row[0].id for row in rows]
            first_msgs: dict[int, str | None] = {}
            if session_ids:
                min_ids_sq = (
                    select(
                        Conversation.session_id,
                        func.min(Conversation.id).label("min_id"),
                    )
                    .where(Conversation.session_id.in_(session_ids))
                    .group_by(Conversation.session_id)
                    .subquery()
                )
                first_rows = s.execute(
                    select(Conversation.session_id, Conversation.message)
                    .join(
                        min_ids_sq,
                        Conversation.id == min_ids_sq.c.min_id,
                    )
                ).all()
                first_msgs = {r[0]: r[1] for r in first_rows}

            sessions = []
            for sess, display_name in rows:
                fm = first_msgs.get(sess.id)
                sessions.append({
                    "id": sess.id,
                    "user_id": sess.user_id,
                    "user_display_name": display_name,
                    "channel_type": sess.channel_type,
                    "started_at": sess.started_at,
                    "last_message_at": sess.last_message_at,
                    "ended_at": sess.ended_at,
                    "primary_intent": sess.primary_intent,
                    "message_count": sess.message_count,
                    "session_summary": sess.session_summary,
                    "first_message_preview": fm[:120] if fm else None,
                })

            return sessions, total

    def get_session_conversations(self, session_id: int) -> dict | None:
        """Get a session with all its conversations (the full chat thread)."""
        with self._session() as s:
            sess = s.get(Session, session_id)
            if not sess:
                return None

            user = s.get(User, sess.user_id)

            convs = s.execute(
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .order_by(asc(Conversation.created_at))
            ).scalars().all()

            conversations = []
            for conv in convs:
                feedbacks = s.execute(
                    select(ResponseFeedback).where(
                        ResponseFeedback.conversation_id == conv.id
                    )
                ).scalars().all()

                conversations.append({
                    "id": conv.id,
                    "message": conv.message,
                    "response": conv.response,
                    "query_type": conv.query_type,
                    "channel_type": conv.channel_type,
                    "response_quality_score": conv.response_quality_score,
                    "quality_completeness_score": conv.quality_completeness_score,
                    "quality_sentiment_score": conv.quality_sentiment_score,
                    "quality_tool_score": conv.quality_tool_score,
                    "tool_calls_json": conv.tool_calls_json,
                    "created_at": conv.created_at,
                    "feedback": [
                        {
                            "id": fb.id,
                            "feedback_type": fb.feedback_type,
                            "quality_score": fb.quality_score,
                            "tool_used": fb.tool_used,
                            "signal_text": fb.signal_text,
                            "created_at": str(fb.created_at) if fb.created_at else None,
                        }
                        for fb in feedbacks
                    ],
                })

            return {
                "id": sess.id,
                "user_id": sess.user_id,
                "user_display_name": user.display_name if user else None,
                "channel_type": sess.channel_type,
                "started_at": sess.started_at,
                "last_message_at": sess.last_message_at,
                "ended_at": sess.ended_at,
                "primary_intent": sess.primary_intent,
                "message_count": sess.message_count,
                "session_summary": sess.session_summary,
                "conversations": conversations,
            }

    # -- conversations ---------------------------------------------------------

    def get_conversations_paginated(
        self,
        page: int = 1,
        limit: int = 50,
        user_id: int | None = None,
        query_type: str | None = None,
        channel: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> tuple[list[dict], int]:
        with self._session() as s:
            filters = []
            if user_id:
                filters.append(Conversation.user_id == user_id)
            if query_type:
                filters.append(Conversation.query_type == query_type)
            if channel:
                filters.append(Conversation.channel_type == channel)
            if date_from:
                filters.append(Conversation.created_at >= date_from)
            if date_to:
                filters.append(Conversation.created_at <= date_to)

            where = and_(*filters) if filters else True

            total = s.execute(
                select(func.count(Conversation.id)).where(where)
            ).scalar() or 0

            # Subquery for feedback counts to avoid N+1
            fb_sub = (
                select(
                    ResponseFeedback.conversation_id,
                    func.count(ResponseFeedback.id).label("fb_count"),
                )
                .group_by(ResponseFeedback.conversation_id)
                .subquery()
            )

            rows = s.execute(
                select(Conversation, User.display_name, func.coalesce(fb_sub.c.fb_count, 0))
                .join(User, Conversation.user_id == User.id)
                .outerjoin(fb_sub, Conversation.id == fb_sub.c.conversation_id)
                .where(where)
                .order_by(desc(Conversation.created_at))
                .offset((page - 1) * limit)
                .limit(limit)
            ).all()

            conversations = []
            for conv, display_name, fb_count in rows:
                conversations.append({
                    "id": conv.id,
                    "user_id": conv.user_id,
                    "user_display_name": display_name,
                    "message": conv.message,
                    "response": conv.response,
                    "query_type": conv.query_type,
                    "channel_type": conv.channel_type,
                    "response_quality_score": conv.response_quality_score,
                    "created_at": conv.created_at,
                    "feedback_count": fb_count,
                })

            return conversations, total

    def get_conversation_detail(self, conv_id: int) -> dict | None:
        with self._session() as s:
            conv = s.get(Conversation, conv_id)
            if not conv:
                return None

            feedbacks = s.execute(
                select(ResponseFeedback).where(ResponseFeedback.conversation_id == conv_id)
            ).scalars().all()

            return {
                "id": conv.id,
                "user_id": conv.user_id,
                "message": conv.message,
                "response": conv.response,
                "query_type": conv.query_type,
                "channel_type": conv.channel_type,
                "response_quality_score": conv.response_quality_score,
                "quality_completeness_score": conv.quality_completeness_score,
                "quality_sentiment_score": conv.quality_sentiment_score,
                "quality_tool_score": conv.quality_tool_score,
                "tool_calls_json": conv.tool_calls_json,
                "template_id_used": conv.template_id_used,
                "session_id": conv.session_id,
                "created_at": conv.created_at,
                "feedback": [
                    {
                        "id": fb.id,
                        "feedback_type": fb.feedback_type,
                        "quality_score": fb.quality_score,
                        "tool_used": fb.tool_used,
                        "signal_text": fb.signal_text,
                        "created_at": str(fb.created_at) if fb.created_at else None,
                    }
                    for fb in feedbacks
                ],
            }

    # -- analytics: feedback ---------------------------------------------------

    def get_feedback_summary(self, days: int = 30) -> dict:
        cutoff = now_ist() - timedelta(days=days)
        with self._session() as s:
            total = s.execute(
                select(func.count(ResponseFeedback.id)).where(
                    ResponseFeedback.created_at >= cutoff
                )
            ).scalar() or 0

            avg_score = s.execute(
                select(func.avg(ResponseFeedback.quality_score)).where(
                    ResponseFeedback.created_at >= cutoff
                )
            ).scalar() or 0.0

            positive = s.execute(
                select(func.count(ResponseFeedback.id)).where(
                    and_(
                        ResponseFeedback.created_at >= cutoff,
                        ResponseFeedback.quality_score > 0,
                    )
                )
            ).scalar() or 0

            negative = s.execute(
                select(func.count(ResponseFeedback.id)).where(
                    and_(
                        ResponseFeedback.created_at >= cutoff,
                        ResponseFeedback.quality_score < 0,
                    )
                )
            ).scalar() or 0

            # Distribution by feedback_type
            dist_rows = s.execute(
                select(ResponseFeedback.feedback_type, func.count(ResponseFeedback.id))
                .where(ResponseFeedback.created_at >= cutoff)
                .group_by(ResponseFeedback.feedback_type)
            ).all()
            distribution = {row[0]: row[1] for row in dist_rows}

            # Daily trend (last N days)
            trend_rows = s.execute(
                select(
                    func.date(ResponseFeedback.created_at).label("day"),
                    func.avg(ResponseFeedback.quality_score).label("avg_score"),
                    func.count(ResponseFeedback.id).label("count"),
                )
                .where(ResponseFeedback.created_at >= cutoff)
                .group_by(func.date(ResponseFeedback.created_at))
                .order_by(func.date(ResponseFeedback.created_at))
            ).all()
            daily_trend = [
                {"date": str(row[0]), "avg_score": round(row[1], 2) if row[1] else 0, "count": row[2]}
                for row in trend_rows
            ]

            return {
                "total_feedback": total,
                "avg_quality_score": round(avg_score, 2),
                "positive_count": positive,
                "negative_count": negative,
                "distribution": distribution,
                "daily_trend": daily_trend,
            }

    # -- analytics: channels ---------------------------------------------------

    def get_channel_stats(self, days: int = 30) -> dict:
        cutoff = now_ist() - timedelta(days=days)
        with self._session() as s:
            rows = s.execute(
                select(
                    Conversation.channel_type,
                    func.count(Conversation.id),
                    func.count(func.distinct(Conversation.user_id)),
                    func.avg(Conversation.response_quality_score),
                )
                .where(Conversation.created_at >= cutoff)
                .group_by(Conversation.channel_type)
            ).all()

            channels = []
            total = 0
            for channel_type, conv_count, unique_users, avg_q in rows:
                channels.append({
                    "channel": channel_type,
                    "conversation_count": conv_count,
                    "unique_users": unique_users,
                    "avg_quality": round(avg_q, 2) if avg_q else None,
                })
                total += conv_count

            return {"channels": channels, "total_conversations": total}

    # -- analytics: tool usage -------------------------------------------------

    def get_tool_usage_stats(self, days: int = 30, limit: int = 20) -> list[dict]:
        cutoff = now_ist() - timedelta(days=days)
        with self._session() as s:
            rows = s.execute(
                select(
                    ToolUsage.tool_name,
                    func.count(ToolUsage.id),
                    func.sum(cast(ToolUsage.success, Integer)),
                    func.avg(ToolUsage.execution_time_ms),
                )
                .where(ToolUsage.created_at >= cutoff)
                .group_by(ToolUsage.tool_name)
                .order_by(desc(func.count(ToolUsage.id)))
                .limit(limit)
            ).all()

            result = []
            for tool_name, total_calls, success_sum, avg_time in rows:
                success_count = success_sum or 0
                result.append({
                    "tool_name": tool_name,
                    "total_calls": total_calls,
                    "success_count": success_count,
                    "failure_count": total_calls - success_count,
                    "success_rate": round(success_count / total_calls * 100, 1) if total_calls else 0,
                    "avg_execution_time_ms": round(avg_time, 1) if avg_time else 0,
                })
            return result

    # -- analytics: errors -----------------------------------------------------

    def get_errors_paginated(
        self, days: int = 30, resolved: bool | None = None, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        cutoff = now_ist() - timedelta(days=days)
        with self._session() as s:
            filters = [QueryError.created_at >= cutoff]
            if resolved is not None:
                filters.append(QueryError.resolved == resolved)

            where = and_(*filters)

            total = s.execute(
                select(func.count(QueryError.id)).where(where)
            ).scalar() or 0

            rows = s.execute(
                select(QueryError)
                .where(where)
                .order_by(desc(QueryError.created_at))
                .offset(offset)
                .limit(limit)
            ).scalars().all()

            errors = [
                {
                    "id": e.id,
                    "tool_name": e.tool_name,
                    "error_type": e.error_type,
                    "error_message": e.error_message,
                    "resolved": e.resolved,
                    "created_at": e.created_at,
                }
                for e in rows
            ]
            return errors, total

    # -- monitoring: system stats ----------------------------------------------

    def get_system_stats(self) -> dict:
        cutoff_24h = now_ist() - timedelta(hours=24)
        with self._session() as s:
            total_users = s.execute(select(func.count(User.id))).scalar() or 0
            active_24h = s.execute(
                select(func.count(User.id)).where(User.last_active >= cutoff_24h)
            ).scalar() or 0
            total_conv = s.execute(select(func.count(Conversation.id))).scalar() or 0
            total_sessions = s.execute(select(func.count(Session.id))).scalar() or 0
            avg_quality = s.execute(
                select(func.avg(Conversation.response_quality_score)).where(
                    Conversation.response_quality_score.isnot(None)
                )
            ).scalar() or 0.0
            total_feedback = s.execute(select(func.count(ResponseFeedback.id))).scalar() or 0
            total_errors = s.execute(select(func.count(QueryError.id))).scalar() or 0
            unresolved = s.execute(
                select(func.count(QueryError.id)).where(QueryError.resolved == False)
            ).scalar() or 0
            cache_entries = s.execute(select(func.count(AnalyticsCache.id))).scalar() or 0

        # Database file size
        db_size_mb = 0.0
        try:
            db_url = self._db_ops.engine.url
            db_path = str(db_url).replace("sqlite:///", "")
            if os.path.exists(db_path):
                db_size_mb = round(os.path.getsize(db_path) / (1024 * 1024), 2)
        except Exception:
            pass

        return {
            "total_users": total_users,
            "active_users_24h": active_24h,
            "total_conversations": total_conv,
            "total_sessions": total_sessions,
            "avg_quality_score": round(avg_quality, 2),
            "total_feedback": total_feedback,
            "total_errors": total_errors,
            "unresolved_errors": unresolved,
            "database_size_mb": db_size_mb,
            "cache_entries": cache_entries,
        }
