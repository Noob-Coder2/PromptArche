"""
Health check endpoints for monitoring application and external services.

Provides:
- /health - Overall system health
- /health/services - Individual service health
- /health/details - Detailed health report

Usage:
    from app.routers.health import router as health_router
    
    app.include_router(health_router, prefix="/api")
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
import logging

from app.db.supabase import get_supabase
from app.services.health_check import HealthCheckService, ServiceStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Dict[str, Any])
async def health_check(supabase = Depends(get_supabase)):
    """
    Quick health check endpoint.
    
    Returns:
    - 200 OK if all critical services are operational
    - 503 Service Unavailable if any critical service is down
    
    Response:
        {
            "status": "healthy" | "degraded" | "unhealthy",
            "timestamp": "2024-01-20T10:30:00"
        }
    """
    health_service = HealthCheckService(supabase, cache_duration_seconds=10)
    report = await health_service.check_all(use_cache=True, parallel=True)
    
    if report.is_healthy():
        return {
            "status": report.overall_status.value,
            "timestamp": report.checked_at.isoformat(),
            "message": "All systems operational"
        }
    
    # Return degraded or unhealthy status
    status_code = 200 if report.overall_status == ServiceStatus.DEGRADED else 503
    return {
        "status": report.overall_status.value,
        "timestamp": report.checked_at.isoformat(),
        "unhealthy_services": report.get_unhealthy_services(),
        "message": f"System status: {report.overall_status.value}"
    }


@router.get("/health/services", response_model=Dict[str, Any])
async def health_services(supabase = Depends(get_supabase)):
    """
    Detailed health status for each service.
    
    Returns individual service statuses:
    - huggingface_api: Embedding generation API
    - groq_api: AI insights API
    - database: Supabase PostgreSQL
    - embedding_cache: Cache statistics
    
    Response:
        {
            "services": {
                "huggingface_api": {
                    "status": "healthy",
                    "response_time_ms": 145.23,
                    "last_checked": "2024-01-20T10:30:00"
                },
                ...
            },
            "summary": {
                "total_services": 4,
                "healthy": 4,
                "degraded": 0,
                "unhealthy": 0
            }
        }
    """
    health_service = HealthCheckService(supabase, cache_duration_seconds=30)
    report = await health_service.check_all(use_cache=True, parallel=True)
    
    return {
        "services": {
            name: service.to_dict()
            for name, service in report.services.items()
        },
        "summary": report.to_dict()["summary"],
        "checked_at": report.checked_at.isoformat()
    }


@router.get("/health/detailed", response_model=Dict[str, Any])
async def health_detailed(supabase = Depends(get_supabase)):
    """
    Complete health check report with all details.
    
    Includes:
    - Overall system status
    - Individual service status, response times, and errors
    - Summary statistics
    - Recommendations for unhealthy services
    
    Warning: This is a heavy endpoint, use sparingly in production.
    Use /health or /health/services for regular monitoring.
    """
    health_service = HealthCheckService(supabase, cache_duration_seconds=0)
    report = await health_service.check_all(use_cache=False, parallel=True)
    
    unhealthy = report.get_unhealthy_services()
    recommendations = _get_recommendations(unhealthy)
    
    return {
        "status": report.overall_status.value,
        "checked_at": report.checked_at.isoformat(),
        "services": {
            name: service.to_dict()
            for name, service in report.services.items()
        },
        "summary": report.to_dict()["summary"],
        "unhealthy_services": unhealthy,
        "recommendations": recommendations,
        "next_check": (report.checked_at.timestamp() + 30)
    }


def _get_recommendations(unhealthy_services: list) -> Dict[str, str]:
    """Get recovery recommendations for unhealthy services."""
    recommendations = {}
    
    service_recommendations = {
        "huggingface_api": (
            "HuggingFace API is down. Check API status at "
            "https://status.huggingface.co or verify your API key is valid."
        ),
        "groq_api": (
            "Groq API is down. Check your API key and verify service availability. "
            "Contact Groq support if issues persist."
        ),
        "database": (
            "Database connection failed. Check Supabase status, network connectivity, "
            "and verify credentials in environment variables."
        ),
        "embedding_cache": (
            "Cache service is unavailable. This may impact performance but functionality "
            "should degrade gracefully. Restart the service if issues persist."
        ),
    }
    
    for service in unhealthy_services:
        recommendations[service] = service_recommendations.get(
            service,
            f"Service {service} is unhealthy. Check logs for details."
        )
    
    return recommendations
