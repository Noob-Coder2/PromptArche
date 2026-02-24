"""
Asynchronous task queue service for background job processing.
Provides job persistence and recovery on app restart.
"""

import asyncio
import logging
import os
from typing import Optional, Any, Dict
from datetime import datetime

from app.db.supabase import get_supabase
from app.core.config import settings

logger = logging.getLogger(__name__)


class TaskQueueService:
    """
    Manages background tasks with persistence and recovery.
    
    Features:
    - Asynchronous task execution (non-blocking)
    - Job persistence to database (survives app restart)
    - Automatic recovery of in-progress jobs on startup
    - Task queuing and scheduling
    - Error handling and retry support
    """
    
    # In-memory task queue for current session
    _task_queue: Optional[asyncio.Queue] = None
    _active_tasks: Dict[str, asyncio.Task] = {}
    _worker_task: Optional[asyncio.Task] = None
    
    @classmethod
    async def initialize(cls):
        """
        Initialize the task queue and start worker.
        Call this on application startup.
        """
        if cls._task_queue is None:
            cls._task_queue = asyncio.Queue()
            logger.info("Task queue initialized")
            
            # Recover any in-progress jobs from database
            await cls._recover_jobs()
            
            # Start the worker task
            cls._worker_task = asyncio.create_task(cls._worker())
            logger.info("Task queue worker started")
    
    @classmethod
    async def shutdown(cls):
        """
        Gracefully shutdown the task queue.
        Call this on application shutdown.
        """
        if cls._worker_task:
            cls._worker_task.cancel()
            try:
                await cls._worker_task
            except asyncio.CancelledError:
                pass
            logger.info("Task queue worker stopped")
        
        cls._task_queue = None
        cls._active_tasks.clear()
    
    @classmethod
    async def enqueue_job(
        cls,
        job_id: str,
        job_type: str,
        job_data: Dict[str, Any],
        user_id: str
    ) -> None:
        """
        Enqueue a job for processing.
        
        Args:
            job_id: Unique job identifier
            job_type: Type of job (e.g., 'ingest_file')
            job_data: Job-specific data (e.g., file_path, provider)
            user_id: User who submitted the job
        """
        task_info = {
            "job_id": job_id,
            "job_type": job_type,
            "job_data": job_data,
            "user_id": user_id,
            "enqueued_at": datetime.utcnow().isoformat(),
            "retry_count": 0
        }
        
        # Add to in-memory queue
        if cls._task_queue:
            await cls._task_queue.put(task_info)
            logger.info(f"Enqueued job {job_id} of type {job_type}")
        else:
            logger.error("Task queue not initialized")
    
    @classmethod
    async def _worker(cls):
        """
        Main worker loop - processes tasks from the queue.
        Runs continuously as an async task.
        """
        logger.info("Task queue worker started")
        
        while True:
            try:
                if cls._task_queue is None:
                    await asyncio.sleep(1)
                    continue
                
                # Get next task from queue with timeout
                try:
                    task_info = await asyncio.wait_for(
                        cls._task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # No tasks, continue waiting
                    continue
                
                # Process the task
                job_id = task_info["job_id"]
                job_type = task_info["job_type"]
                
                logger.info(f"Processing job {job_id} of type {job_type}")
                
                # Create async task for job processing
                async_task = asyncio.create_task(
                    cls._process_job(task_info)
                )
                cls._active_tasks[job_id] = async_task
                
                # Wait for task completion
                try:
                    await async_task
                    logger.info(f"Job {job_id} completed successfully")
                except Exception as e:
                    logger.error(f"Job {job_id} failed: {e}")
                finally:
                    # Clean up
                    if job_id in cls._active_tasks:
                        del cls._active_tasks[job_id]
                
            except asyncio.CancelledError:
                logger.info("Task queue worker cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in task queue worker: {e}")
                await asyncio.sleep(1)
    
    @classmethod
    async def _process_job(cls, task_info: Dict[str, Any]) -> None:
        """
        Process a single job with retry logic.
        
        Args:
            task_info: Task information
        """
        from app.services.job_service import IngestionJobService
        
        job_id = task_info["job_id"]
        job_type = task_info["job_type"]
        job_data = task_info["job_data"]
        user_id = task_info["user_id"]
        retry_count = task_info.get("retry_count", 0)
        
        try:
            if job_type == "ingest_file":
                await cls._process_ingest_job(job_id, job_data, user_id)
            else:
                logger.error(f"Unknown job type: {job_type}")
                IngestionJobService.fail_job(job_id, f"Unknown job type: {job_type}")
        
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            
            # Retry logic for transient failures
            if retry_count < settings.MAX_RETRIES:
                retry_count += 1
                task_info["retry_count"] = retry_count
                
                # Exponential backoff
                wait_time = min(30, 2 ** retry_count)
                logger.warning(
                    f"Job {job_id} will retry in {wait_time}s "
                    f"(attempt {retry_count}/{settings.MAX_RETRIES})"
                )
                
                # Re-enqueue with delay
                await asyncio.sleep(wait_time)
                if cls._task_queue:
                    await cls._task_queue.put(task_info)
            else:
                # Max retries exceeded
                IngestionJobService.fail_job(
                    job_id,
                    f"Job failed after {retry_count} retries: {str(e)}"
                )
    
    @classmethod
    async def _process_ingest_job(
        cls,
        job_id: str,
        job_data: Dict[str, Any],
        user_id: str
    ) -> None:
        """
        Process an ingestion job using thread pool executor.
        
        Args:
            job_id: Job ID
            job_data: Contains file_path, provider
            user_id: User ID
        """
        from app.services.ingestion import IngestionService
        
        file_path = job_data.get("file_path")
        provider = job_data.get("provider")
        
        if not file_path or not provider:
            raise ValueError("Missing required job data: file_path, provider")
        
        try:
            # Run async ingestion (fully async pipeline)
            with open(file_path, 'rb') as f:
                result = await IngestionService.ingest_async(
                    f,
                    provider,
                    user_id,
                    job_id
                )
            
            logger.info(f"Ingestion job {job_id} completed: {result}")
            
            # --- Post-Ingestion Pipeline ---
            # Only proceed if ingestion was successful and processed items
            if result.get("status") == "success" and result.get("processed", 0) > 0:
                logger.info(f"Starting post-ingestion analysis for user {user_id}")
                
                try:
                    # 1. Clustering (Sync)
                    # Note: This requires embeddings to be present (via background task or backfill RPC)
                    # The ingestion service waits for embedding tasks, so we should be good IF logic handled it.
                    # BUT: If backfill RPC "backfill_embeddings_from_cache" was not run manually, 
                    # prompts.embedding might be NULL. Clustering checks for non-null embeddings.
                    from app.services.clustering import run_clustering_for_user
                    
                    logger.info("Running clustering...")
                    # Run in thread pool since it's CPU bound (UMAP/HDBSCAN)
                    clustering_res = await asyncio.get_event_loop().run_in_executor(
                        None,
                        run_clustering_for_user,
                        user_id
                    )
                    logger.info(f"Clustering result: {clustering_res}")
                    
                    # 2. Insights (Async)
                    # Only run if clusters were found/created
                    if clustering_res.get("clusters_found", 0) > 0:
                        from app.services.insights import generate_insights_for_user
                        logger.info("Generating insights...")
                        insights_res = await generate_insights_for_user(user_id)
                        logger.info(f"Insights result: {insights_res}")
                    else:
                        logger.info("Skipping insights (no clusters found)")
                        
                except Exception as e:
                    logger.error(f"Post-ingestion analysis failed: {e}", exc_info=True)
                    # We don't fail the *ingestion* job, but we log the error.

        
        finally:
            # Always cleanup temp file
            IngestionService.cleanup_temp_file(file_path)
    
    @classmethod
    async def _recover_jobs(cls):
        """
        Recover in-progress jobs on app startup.
        """
        from app.services.job_service import IngestionJobService
        
        try:
            supabase = get_supabase()
            
            # Find in-progress jobs
            response = supabase.table("ingestion_jobs") \
                .select("*") \
                .in_("status", ["PENDING", "PARSING"]) \
                .execute()
            
            jobs = response.data or []
            
            if not jobs:
                logger.info("No in-progress jobs to recover")
                return
            
            logger.warning(f"Recovering {len(jobs)} in-progress jobs")
            
            for job_data in jobs:
                try:
                    job_id = str(job_data.get("id", ""))
                    user_id = str(job_data.get("user_id", ""))
                    provider = str(job_data.get("provider", ""))
                    temp_file_path = job_data.get("temp_file_path")
                    
                    file_path = str(temp_file_path) if temp_file_path else None
                    
                    if file_path and os.path.exists(file_path):
                        logger.info(f"Recovering job {job_id}: file exists")
                        await cls.enqueue_job(
                            job_id=job_id,
                            job_type="ingest_file",
                            job_data={
                                "file_path": file_path,
                                "provider": provider
                            },
                            user_id=user_id
                        )
                    else:
                        logger.warning(f"Job {job_id}: temp file not found")
                        IngestionJobService.fail_job(
                            job_id,
                            "Temp file lost (app crashed)"
                        )
                except Exception as e:
                    logger.error(f"Error recovering job: {e}")
        
        except Exception as e:
            logger.error(f"Error recovering jobs: {e}")
    
    @classmethod
    async def get_queue_stats(cls) -> Dict[str, Any]:
        """
        Get task queue statistics.
        
        Returns:
            Dict with queue size and active task count
        """
        queue_size = 0
        if cls._task_queue:
            queue_size = cls._task_queue.qsize()
        
        return {
            "queue_size": queue_size,
            "active_tasks": len(cls._active_tasks),
            "total_tasks": queue_size + len(cls._active_tasks)
        }

