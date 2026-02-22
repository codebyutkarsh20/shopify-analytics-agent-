"""FastAPI application factory for the admin dashboard."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.admin.queries import DashboardQueries
from src.admin.routes import auth, users, conversations, sessions, analytics, monitoring
from src.database.operations import DatabaseOperations

_STATIC_DIR = Path(__file__).parent / "static"


def create_admin_app(db_ops: DatabaseOperations) -> FastAPI:
    """Build and return the FastAPI admin dashboard application."""

    app = FastAPI(
        title="Shopify Analytics Agent — Admin Dashboard",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
    )

    # CORS (allow same-origin; loosen for dev if needed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Wire up shared DashboardQueries instance
    queries = DashboardQueries(db_ops)
    auth.set_queries(queries)
    users.set_queries(queries)
    conversations.set_queries(queries)
    sessions.set_queries(queries)
    analytics.set_queries(queries)
    monitoring.set_queries(queries)

    # Register route modules
    app.include_router(auth.router)
    app.include_router(users.router)
    app.include_router(conversations.router)
    app.include_router(sessions.router)
    app.include_router(analytics.router)
    app.include_router(monitoring.router)

    # Serve the React SPA for all non-API paths
    @app.get("/")
    async def serve_dashboard():
        return FileResponse(_STATIC_DIR / "dashboard.html")

    # Catch-all for client-side routing (e.g. /users, /analytics)
    @app.get("/{full_path:path}")
    async def catch_all(full_path: str):
        return FileResponse(_STATIC_DIR / "dashboard.html")

    return app
