"""
Health check service for monitoring external services and application status.

Provides:
- Health status for external APIs (HuggingFace, Groq)
- Database connectivity checks
- Service dependency monitoring
- Cache hit/miss rates
- Background task queue status
- System resource monitoring (memory, CPU, disk)
- Database table metrics and ingestion job status
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
import psutil
import os

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
                self.check_task_queue(),
                self.check_system_resources(),
                self.check_database_tables(),
                self.check_ingestion_jobs(),
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
            report.add_service(await self.check_task_queue())
            report.add_service(await self.check_system_resources())
            report.add_service(await self.check_database_tables())
            report.add_service(await self.check_ingestion_jobs())
        
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
                    "https://router.huggingface.co/status",
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
    
    async def check_schema_readiness(self) -> Dict[str, Any]:
        """
        Check if all required database tables exist.
        
        Returns:
            Dict with ready status and missing tables
        """
        required_tables = [
            "prompts", 
            "clusters", 
            "insights", 
            "ingestion_jobs",
            "embedding_cache"
        ]
        
        missing_tables = []
        
        try:
            # Get list of all tables in public schema
            # Note: This query assumes Postgres structure
            try:
                response = self.supabase.rpc(
                    "get_tables_info", {}
                ).execute()
            
            # Fallback if RPC doesn't exist (which it won't by default):
            # We'll try to select 1 row from each table. specific error means missing.
            # This is a bit brute-force but works without admin RPCs.
            except Exception as e:
                logger.warning(f"RPC get_tables_info failed, using fallback: {e}")
            
            for table in required_tables:
                try:
                    # Just check if table exists by selecting nothing
                    # We limit to 0 so it's super fast, effectively a HEAD request
                    self.supabase.table(table).select("count", count="exact").limit(0).execute()
                except Exception as e:
                    # If table is missing, the error usually contains "relation ... does not exist"
                    # or similar code 42P01
                    if "does not exist" in str(e) or "42P01" in str(e) or "not found" in str(e).lower():
                        missing_tables.append(table)
                    else:
                        # Some other error (auth, connection), might be critical
                        logger.warning(f"Error checking table {table}: {e}")
                        # We'll assume it exists but is failing for other reasons, 
                        # or treat as missing if we want to be strict.
                        # For now, if we can't verify it, let's treat as potentially problematic.
                        pass
            
            is_ready = len(missing_tables) == 0
            
            return {
                "ready": is_ready,
                "missing_tables": missing_tables,
                "checked_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Schema check failed: {e}")
            return {
                "ready": False,
                "error": str(e),
                "missing_tables": required_tables # Assume all missing on critical failure
            }
    
    async def check_task_queue(self) -> ServiceHealthReport:
        """
        Check background task queue health.
        
        Returns:
            ServiceHealthReport for task queue
        """
        try:
            from app.services.task_queue import TaskQueueService
            
            # Get queue metrics
            queue_size = 0
            active_tasks = 0
            worker_alive = False
            
            if TaskQueueService._task_queue:
                queue_size = TaskQueueService._task_queue.qsize()
            
            active_tasks = len(TaskQueueService._active_tasks)
            worker_alive = TaskQueueService._worker_task is not None and not TaskQueueService._worker_task.done()
            
            # Health status logic
            status = ServiceStatus.HEALTHY
            if not worker_alive:
                status = ServiceStatus.UNHEALTHY
            elif queue_size > 100:  # Queue backing up
                status = ServiceStatus.DEGRADED
            
            return ServiceHealthReport(
                name="task_queue",
                status=status,
                details={
                    "queue_size": queue_size,
                    "active_tasks": active_tasks,
                    "worker_alive": worker_alive,
                    "max_queue_size_before_degraded": 100
                }
            )
        
        except Exception as e:
            logger.warning(f"Task queue health check failed: {e}")
            return ServiceHealthReport(
                name="task_queue",
                status=ServiceStatus.DEGRADED,
                error=str(e),
                details={"error_type": type(e).__name__}
            )
    
    async def check_system_resources(self) -> ServiceHealthReport:
        """
        Check system resource utilization.
        
        Returns:
            ServiceHealthReport for system resources
        """
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # CPU usage (1-second average)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Disk usage for current directory
            disk = psutil.disk_usage(os.path.dirname(os.path.abspath(__file__)))
            disk_percent = disk.percent
            
            # Determine status based on thresholds
            status = ServiceStatus.HEALTHY
            issues = []
            
            if memory_percent > 90:
                status = ServiceStatus.UNHEALTHY
                issues.append(f"Memory critically high ({memory_percent}%)")
            elif memory_percent > 80:
                status = ServiceStatus.DEGRADED
                issues.append(f"Memory high ({memory_percent}%)")
            
            if cpu_percent > 90:
                status = ServiceStatus.UNHEALTHY if status == ServiceStatus.UNHEALTHY else ServiceStatus.DEGRADED
                issues.append(f"CPU critically high ({cpu_percent}%)")
            
            if disk_percent > 95:
                status = ServiceStatus.UNHEALTHY
                issues.append(f"Disk space critical ({disk_percent}%)")
            elif disk_percent > 85:
                status = ServiceStatus.DEGRADED if status == ServiceStatus.HEALTHY else status
                issues.append(f"Disk space high ({disk_percent}%)")
            
            return ServiceHealthReport(
                name="system_resources",
                status=status,
                error="; ".join(issues) if issues else None,
                details={
                    "memory_percent": round(memory_percent, 2),
                    "memory_used_gb": round(memory.used / (1024**3), 2),
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "cpu_percent": round(cpu_percent, 2),
                    "disk_percent": round(disk_percent, 2),
                    "disk_used_gb": round(disk.used / (1024**3), 2),
                    "disk_free_gb": round(disk.free / (1024**3), 2),
                }
            )
        
        except Exception as e:
            logger.warning(f"System resource check failed: {e}")
            return ServiceHealthReport(
                name="system_resources",
                status=ServiceStatus.DEGRADED,
                error=str(e)
            )
    
    async def check_database_tables(self) -> ServiceHealthReport:
        """
        Check database table health and sizes.
        
        Returns:
            ServiceHealthReport for database tables
        """
        try:
            # Tables to monitor
            tables_to_check = {
                "prompts": "Main data storage",
                "clusters": "Clustering results",
                "insights": "ML insights",
                "embedding_cache": "Embedding cache",
            }
            
            table_stats = {}
            total_rows = 0
            
            for table_name, description in tables_to_check.items():
                try:
                    # Get row count efficiently
                    response = self.supabase.table(table_name).select("count", count="exact").limit(0).execute()
                    
                    row_count = response.count if hasattr(response, 'count') else 0
                    table_stats[table_name] = {
                        "rows": row_count,
                        "description": description
                    }
                    total_rows += row_count
                
                except Exception as e:
                    # Table might not exist or be inaccessible
                    table_stats[table_name] = {
                        "rows": -1,
                        "error": str(e),
                        "description": description
                    }
            
            status = ServiceStatus.HEALTHY if all(
                stats.get("rows", -1) >= 0 for stats in table_stats.values()
            ) else ServiceStatus.DEGRADED
            
            return ServiceHealthReport(
                name="database_tables",
                status=status,
                details={
                    "total_rows": total_rows,
                    "tables": table_stats,
                }
            )
        
        except Exception as e:
            logger.warning(f"Database tables health check failed: {e}")
            return ServiceHealthReport(
                name="database_tables",
                status=ServiceStatus.DEGRADED,
                error=str(e)
            )
    
    async def check_ingestion_jobs(self) -> ServiceHealthReport:
        """
        Check status of ingestion jobs in the queue.
        
        Returns:
            ServiceHealthReport for ingestion jobs
        """
        try:
            # Get job statistics
            response = self.supabase.table("ingestion_jobs").select(
                "status, count"
            ).execute()
            
            # Get counts by status
            job_stats = {
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
            }
            
            try:
                # Try to get specific counts by status
                for status in job_stats.keys():
                    status_response = self.supabase.table("ingestion_jobs").select(
                        "count", count="exact"
                    ).eq("status", status).limit(0).execute()
                    
                    job_stats[status] = status_response.count if hasattr(status_response, 'count') else 0
            
            except Exception as e:
                logger.debug(f"Could not get detailed job status: {e}")
                # Fallback: just get total count
                total_response = self.supabase.table("ingestion_jobs").select(
                    "count", count="exact"
                ).limit(0).execute()
                job_stats["total"] = total_response.count if hasattr(total_response, 'count') else 0
            
            # Determine health based on failed jobs
            status = ServiceStatus.HEALTHY
            if job_stats.get("failed", 0) > 10:
                status = ServiceStatus.DEGRADED
            if job_stats.get("failed", 0) > 50:
                status = ServiceStatus.UNHEALTHY
            
            return ServiceHealthReport(
                name="ingestion_jobs",
                status=status,
                details={
                    "job_status_breakdown": job_stats,
                    "total_jobs": sum(job_stats.values()),
                    "health_thresholds": {
                        "degraded_at_failed_count": 10,
                        "unhealthy_at_failed_count": 50,
                    }
                }
            )
        
        except Exception as e:
            logger.warning(f"Ingestion jobs health check failed: {e}")
            return ServiceHealthReport(
                name="ingestion_jobs",
                status=ServiceStatus.DEGRADED,
                error=str(e)
            )

