"""Monitoring routes: health check & system stats."""

from fastapi import APIRouter, Depends

from src.admin.dependencies import get_current_admin
from src.admin.schemas import SystemStats, HealthCheck

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

_queries = None


def set_queries(queries):
    global _queries
    _queries = queries


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Public health endpoint (no auth required)."""
    return HealthCheck(status="healthy", database="ok", bot_running=True)


@router.get("/stats", response_model=SystemStats)
async def system_stats(_admin: str = Depends(get_current_admin)):
    return SystemStats(**_queries.get_system_stats())
