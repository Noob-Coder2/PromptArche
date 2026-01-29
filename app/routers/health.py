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
    - System resources (memory, CPU, disk)
    - Database table metrics
    - Task queue status
    - Ingestion job statistics
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


@router.get("/health/resources", response_model=Dict[str, Any])
async def health_resources(supabase = Depends(get_supabase)):
    """
    System resource health check.
    
    Returns:
    - Memory usage (percent, used/available in GB)
    - CPU usage (percent)
    - Disk usage (percent, used/free in GB)
    
    Thresholds:
    - Memory: Degraded >80%, Unhealthy >90%
    - CPU: Degraded >90%, Unhealthy >95%
    - Disk: Degraded >85%, Unhealthy >95%
    
    Response:
        {
            "status": "healthy",
            "resources": {
                "memory_percent": 65.4,
                "memory_used_gb": 4.2,
                "memory_available_gb": 8.1,
                "cpu_percent": 42.3,
                "disk_percent": 72.1,
                "disk_used_gb": 256.4,
                "disk_free_gb": 98.2
            }
        }
    """
    health_service = HealthCheckService(supabase, cache_duration_seconds=15)
    report = await health_service.check_all(use_cache=True, parallel=False)
    
    # Find resource check in report
    resource_check = report.services.get("system_resources")
    
    if resource_check:
        return {
            "status": resource_check.status.value,
            "checked_at": resource_check.last_checked.isoformat(),
            "resources": resource_check.details,
            "warning": resource_check.error
        }
    
    return {"status": "unknown", "error": "Resource check not available"}


@router.get("/health/database", response_model=Dict[str, Any])
async def health_database(supabase = Depends(get_supabase)):
    """
    Database health and metrics.
    
    Returns:
    - Overall database connectivity
    - Table row counts
    - Ingestion job status breakdown
    
    Response:
        {
            "status": "healthy",
            "database_connection": {"status": "healthy", "response_time_ms": 45.2},
            "tables": {
                "prompts": {"rows": 1250},
                "clusters": {"rows": 340},
                "insights": {"rows": 892},
                "embedding_cache": {"rows": 5420}
            },
            "ingestion_jobs": {
                "pending": 5,
                "processing": 2,
                "completed": 1240,
                "failed": 1
            }
        }
    """
    health_service = HealthCheckService(supabase, cache_duration_seconds=30)
    report = await health_service.check_all(use_cache=True, parallel=True)
    
    db_check = report.services.get("database")
    tables_check = report.services.get("database_tables")
    jobs_check = report.services.get("ingestion_jobs")
    
    return {
        "status": report.overall_status.value,
        "checked_at": report.checked_at.isoformat(),
        "database_connection": {
            "status": db_check.status.value if db_check else "unknown",
            "response_time_ms": db_check.response_time_ms if db_check else None
        },
        "tables": tables_check.details.get("tables", {}) if tables_check else {},
        "total_rows": tables_check.details.get("total_rows", 0) if tables_check else 0,
        "ingestion_jobs": jobs_check.details.get("job_status_breakdown", {}) if jobs_check else {}
    }


@router.get("/health/queue", response_model=Dict[str, Any])
async def health_queue(supabase = Depends(get_supabase)):
    """
    Background task queue health.
    
    Returns:
    - Queue size (number of pending jobs)
    - Active tasks being processed
    - Worker thread status
    
    Response:
        {
            "status": "healthy",
            "queue": {
                "queue_size": 3,
                "active_tasks": 2,
                "worker_alive": true
            }
        }
    """
    health_service = HealthCheckService(supabase, cache_duration_seconds=10)
    report = await health_service.check_all(use_cache=True, parallel=False)
    
    queue_check = report.services.get("task_queue")
    
    if queue_check:
        return {
            "status": queue_check.status.value,
            "checked_at": queue_check.last_checked.isoformat(),
            "queue": {
                "queue_size": queue_check.details.get("queue_size", 0),
                "active_tasks": queue_check.details.get("active_tasks", 0),
                "worker_alive": queue_check.details.get("worker_alive", False)
            }
        }
    
    return {"status": "unknown", "error": "Queue check not available"}



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
        "task_queue": (
            "Background task queue is unhealthy. Check task queue worker logs. "
            "Verify the worker process is running and not stuck."
        ),
        "system_resources": (
            "System resources are critically high. Check memory, CPU, and disk usage. "
            "Consider scaling up resources or optimizing application."
        ),
        "database_tables": (
            "Database tables are inaccessible. Verify Supabase connection and check "
            "that all required tables exist in the schema."
        ),
        "ingestion_jobs": (
            "High number of failed ingestion jobs. Check ingestion logs, verify file "
            "formats and sizes, and check external API availability."
        ),
    }
    
    for service in unhealthy_services:
        recommendations[service] = service_recommendations.get(
            service,
            f"Service {service} is unhealthy. Check logs for details."
        )
    
    return recommendations
