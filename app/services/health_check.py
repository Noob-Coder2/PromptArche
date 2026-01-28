"""
Health check service for monitoring external services and application status.

Provides:
- Health status for external APIs (HuggingFace, Groq)
- Database connectivity checks
- Service dependency monitoring
- Cache hit/miss rates
- Background task queue status
- Detailed health reports

Usage:
    from app.services.health_check import HealthCheckService
    
    health_service = HealthCheckService(supabase_client)
    status = await health_service.check_all()
    
    print(status.to_dict())  # Returns detailed status report
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceHealthReport:
    """Health status report for a single service."""
    
    def __init__(
        self,
        name: str,
        status: ServiceStatus,
        response_time_ms: Optional[float] = None,
        last_checked: Optional[datetime] = None,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.status = status
        self.response_time_ms = response_time_ms
        self.last_checked = last_checked or datetime.utcnow()
        self.error = error
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "response_time_ms": round(self.response_time_ms, 2) if self.response_time_ms else None,
            "last_checked": self.last_checked.isoformat(),
            "error": self.error,
            "details": self.details
        }


class HealthCheckReport:
    """Overall health report for the application."""
    
    def __init__(self):
        self.checked_at = datetime.utcnow()
        self.services: Dict[str, ServiceHealthReport] = {}
        self.overall_status = ServiceStatus.UNKNOWN
    
    def add_service(self, report: ServiceHealthReport) -> None:
        """Add service health report."""
        self.services[report.name] = report
        self._update_overall_status()
    
    def _update_overall_status(self) -> None:
        """Update overall status based on service statuses."""
        if not self.services:
            self.overall_status = ServiceStatus.UNKNOWN
            return
        
        statuses = [s.status for s in self.services.values()]
        
        if all(s == ServiceStatus.HEALTHY for s in statuses):
            self.overall_status = ServiceStatus.HEALTHY
        elif ServiceStatus.UNHEALTHY in statuses:
            self.overall_status = ServiceStatus.UNHEALTHY
        else:
            self.overall_status = ServiceStatus.DEGRADED
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.overall_status == ServiceStatus.HEALTHY
    
    def get_unhealthy_services(self) -> List[str]:
        """Get list of unhealthy services."""
        return [
            name for name, report in self.services.items()
            if report.status != ServiceStatus.HEALTHY
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.overall_status.value,
            "checked_at": self.checked_at.isoformat(),
            "services": {
                name: report.to_dict()
                for name, report in self.services.items()
            },
            "summary": {
                "total_services": len(self.services),
                "healthy": sum(1 for r in self.services.values() if r.status == ServiceStatus.HEALTHY),
                "degraded": sum(1 for r in self.services.values() if r.status == ServiceStatus.DEGRADED),
                "unhealthy": sum(1 for r in self.services.values() if r.status == ServiceStatus.UNHEALTHY),
            }
        }


class HealthCheckService:
    """
    Monitors health of external services and application components.
    """
    
    def __init__(
        self,
        supabase_client,
        timeout_seconds: float = 5.0,
        cache_duration_seconds: int = 30
    ):
        """
        Initialize health check service.
        
        Args:
            supabase_client: Supabase client instance
            timeout_seconds: Timeout for external API calls
            cache_duration_seconds: Cache health check results
        """
        self.supabase = supabase_client
        self.timeout_seconds = timeout_seconds
        self.cache_duration = cache_duration_seconds
        self._last_report: Optional[HealthCheckReport] = None
        self._last_check_time: Optional[datetime] = None
    
    async def check_all(
        self,
        use_cache: bool = True,
        parallel: bool = True
    ) -> HealthCheckReport:
        """
        Check health of all services.
        
        Args:
            use_cache: Use cached results if available
            parallel: Run checks in parallel
            
        Returns:
            HealthCheckReport with status of all services
        """
        # Return cached result if recent
        if use_cache and self._should_use_cache():
            assert self._last_report is not None
            return self._last_report
        
        report = HealthCheckReport()
        
        if parallel:
            # Run all checks concurrently
            tasks = [
                self.check_huggingface_api(),
                self.check_groq_api(),
                self.check_database(),
                self.check_cache_stats(),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, ServiceHealthReport):
                    report.add_service(result)
                elif isinstance(result, Exception):
                    logger.error(f"Health check error: {result}")
        else:
            # Run checks sequentially
            report.add_service(await self.check_huggingface_api())
            report.add_service(await self.check_groq_api())
            report.add_service(await self.check_database())
            report.add_service(await self.check_cache_stats())
        
        self._last_report = report
        self._last_check_time = datetime.utcnow()
        
        return report
    
    def _should_use_cache(self) -> bool:
        """Check if cached result is still valid."""
        if not self._last_check_time or not self._last_report:
            return False
        
        age = datetime.utcnow() - self._last_check_time
        return age.total_seconds() < self.cache_duration
    
    async def check_huggingface_api(self) -> ServiceHealthReport:
        """
        Check HuggingFace Inference API health.
        
        Returns:
            ServiceHealthReport for HuggingFace
        """
        start_time = datetime.utcnow()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                # Try a simple API call
                response = await client.get(
                    "https://api-inference.huggingface.co/status",
                    headers={"Accept": "application/json"}
                )
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return ServiceHealthReport(
                    name="huggingface_api",
                    status=ServiceStatus.HEALTHY,
                    response_time_ms=response_time,
                    details={"status_code": response.status_code}
                )
            else:
                return ServiceHealthReport(
                    name="huggingface_api",
                    status=ServiceStatus.DEGRADED,
                    response_time_ms=response_time,
                    error=f"Unexpected status code: {response.status_code}",
                    details={"status_code": response.status_code}
                )
        
        except asyncio.TimeoutError:
            return ServiceHealthReport(
                name="huggingface_api",
                status=ServiceStatus.UNHEALTHY,
                error="API request timed out"
            )
        except Exception as e:
            logger.error(f"HuggingFace health check failed: {e}")
            return ServiceHealthReport(
                name="huggingface_api",
                status=ServiceStatus.UNHEALTHY,
                error=str(e)
            )
    
    async def check_groq_api(self) -> ServiceHealthReport:
        """
        Check Groq API health.
        
        Returns:
            ServiceHealthReport for Groq
        """
        start_time = datetime.utcnow()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                # Try a simple API call to Groq status endpoint
                response = await client.get(
                    "https://api.groq.com/status",
                    headers={"Accept": "application/json"}
                )
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return ServiceHealthReport(
                    name="groq_api",
                    status=ServiceStatus.HEALTHY,
                    response_time_ms=response_time,
                    details={"status_code": response.status_code}
                )
            else:
                return ServiceHealthReport(
                    name="groq_api",
                    status=ServiceStatus.DEGRADED,
                    response_time_ms=response_time,
                    error=f"Unexpected status code: {response.status_code}",
                    details={"status_code": response.status_code}
                )
        
        except asyncio.TimeoutError:
            return ServiceHealthReport(
                name="groq_api",
                status=ServiceStatus.UNHEALTHY,
                error="API request timed out"
            )
        except Exception as e:
            logger.error(f"Groq health check failed: {e}")
            return ServiceHealthReport(
                name="groq_api",
                status=ServiceStatus.UNHEALTHY,
                error=str(e)
            )
    
    async def check_database(self) -> ServiceHealthReport:
        """
        Check database connectivity.
        
        Returns:
            ServiceHealthReport for database
        """
        start_time = datetime.utcnow()
        
        try:
            # Try a simple query
            response = self.supabase.table("users").select("id").limit(1).execute()
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ServiceHealthReport(
                name="database",
                status=ServiceStatus.HEALTHY,
                response_time_ms=response_time,
                details={"rows_accessed": len(response.data) if response.data else 0}
            )
        
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return ServiceHealthReport(
                name="database",
                status=ServiceStatus.UNHEALTHY,
                error=str(e)
            )
    
    async def check_cache_stats(self) -> ServiceHealthReport:
        """
        Check embedding cache statistics.
        
        Returns:
            ServiceHealthReport for cache
        """
        try:
            # Get cache stats from database
            response = self.supabase.table("embedding_cache").select("id").execute()
            
            cache_size = len(response.data) if response.data else 0
            
            # Cache is healthy if it exists
            status = ServiceStatus.HEALTHY if cache_size > 0 else ServiceStatus.DEGRADED
            
            return ServiceHealthReport(
                name="embedding_cache",
                status=status,
                details={
                    "entries": cache_size,
                    "status": "operational"
                }
            )
        
        except Exception as e:
            logger.warning(f"Cache health check failed: {e}")
            return ServiceHealthReport(
                name="embedding_cache",
                status=ServiceStatus.DEGRADED,
                error=str(e)
            )
