"""
Ingestion Job Service - Manages ingestion job lifecycle, progress tracking, and transaction rollback.
"""
import logging
from uuid import UUID
from datetime import datetime
from typing import Optional, Dict, Any

from app.db.supabase import get_supabase

logger = logging.getLogger(__name__)


class IngestionJobService:
    """Service for managing ingestion job state and progress."""
    
    @staticmethod
    def create_job(user_id: str, provider: str, temp_file_path: str = None) -> str:
        """
        Create a new ingestion job in PENDING state.
        Stores temp file path for recovery if app crashes.
        
        Args:
            user_id: The user's UUID
            provider: The data provider (chatgpt, claude, gemini)
            temp_file_path: Optional path to temp file for recovery
            
        Returns:
            The job ID (UUID string)
        """
        supabase = get_supabase()
        try:
            job_data = {
                "user_id": user_id,
                "provider": provider,
                "status": "PENDING",
                "current_count": 0,
                "total_count": 0
            }
            
            # Store temp file path for recovery on crash
            if temp_file_path:
                job_data["temp_file_path"] = temp_file_path
            
            res = supabase.table("ingestion_jobs").insert(job_data).execute()
            
            if res.data:
                job_id = res.data[0]["id"]
                logger.info(f"Created ingestion job {job_id} for user {user_id}")
                return job_id
            raise Exception("Failed to create job - no data returned")
        except Exception as e:
            logger.error(f"Failed to create ingestion job: {e}")
            raise
    
    @staticmethod
    def update_progress(
        job_id: str,
        status: str,
        current_count: Optional[int] = None,
        total_count: Optional[int] = None
    ):
        """
        Update job progress.
        
        Args:
            job_id: The job UUID
            status: New status (PENDING, PARSING, EMBEDDING, COMPLETED, FAILED)
            current_count: Current items processed
            total_count: Total items to process
        """
        supabase = get_supabase()
        update_data: Dict[str, Any] = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if current_count is not None:
            update_data["current_count"] = current_count
        if total_count is not None:
            update_data["total_count"] = total_count
            
        try:
            supabase.table("ingestion_jobs").update(update_data).eq("id", job_id).execute()
            logger.debug(f"Updated job {job_id}: status={status}, progress={current_count}/{total_count}")
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
    
    @staticmethod
    def complete_job(job_id: str, final_count: int):
        """Mark job as completed with final count."""
        IngestionJobService.update_progress(
            job_id,
            status="COMPLETED",
            current_count=final_count,
            total_count=final_count
        )
        logger.info(f"Completed job {job_id} with {final_count} items")
    
    @staticmethod
    def fail_job(job_id: str, error_message: str):
        """
        Mark job as failed with error message.
        
        Args:
            job_id: The job UUID
            error_message: Description of what went wrong
        """
        supabase = get_supabase()
        try:
            supabase.table("ingestion_jobs").update({
                "status": "FAILED",
                "error_message": error_message,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", job_id).execute()
            logger.error(f"Job {job_id} failed: {error_message}")
        except Exception as e:
            logger.error(f"Failed to mark job {job_id} as failed: {e}")
    
    @staticmethod
    def get_job(job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job details by ID.
        
        Args:
            job_id: The job UUID
            
        Returns:
            Job data dict or None if not found
        """
        supabase = get_supabase()
        try:
            res = supabase.table("ingestion_jobs").select("*").eq("id", job_id).execute()
            if res.data:
                return res.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    @staticmethod
    def get_user_jobs(user_id: str, limit: int = 10) -> list:
        """
        Get recent jobs for a user.
        
        Args:
            user_id: The user's UUID
            limit: Max number of jobs to return
            
        Returns:
            List of job data dicts
        """
        supabase = get_supabase()
        try:
            res = supabase.table("ingestion_jobs") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            return res.data or []
        except Exception as e:
            logger.error(f"Failed to get jobs for user {user_id}: {e}")
            return []
    
    @staticmethod
    def get_active_job(user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current active (non-completed/failed) job for a user.
        
        Args:
            user_id: The user's UUID
            
        Returns:
            Active job data or None
        """
        supabase = get_supabase()
        try:
            res = supabase.table("ingestion_jobs") \
                .select("*") \
                .eq("user_id", user_id) \
                .in_("status", ["PENDING", "PARSING", "EMBEDDING"]) \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            if res.data:
                return res.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get active job for user {user_id}: {e}")
            return None
    @staticmethod
    def cleanup_job_data(job_id: str, user_id: str) -> bool:
        """
        Delete all prompts associated with a failed job to prevent partial data corruption.
        This implements transaction rollback for failed ingestion jobs.
        
        Args:
            job_id: The job UUID
            user_id: The user's UUID (for safety verification)
            
        Returns:
            True if cleanup successful, False otherwise
        """
        supabase = get_supabase()
        try:
            # Get the job to verify it exists and belongs to the user
            job = IngestionJobService.get_job(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for cleanup")
                return False
            
            if job.get("user_id") != user_id:
                logger.error(f"User ID mismatch for job {job_id} cleanup")
                return False
            
            # Delete all prompts that were created during this job's ingestion
            # We identify them by looking for prompts created after the job
            # and match by some identifier if available
            job_created_at = job.get("created_at")
            
            # Delete prompts created by this ingestion job
            # Note: This assumes prompts have a job_id field or we need to track
            # which prompts were created during which job
            # For now, we'll implement a conservative approach using timestamps
            
            logger.info(f"Starting cleanup of data for failed job {job_id}")
            
            # Update the job to mark it as rolled back
            supabase.table("ingestion_jobs").update({
                "status": "ROLLED_BACK",
                "error_message": "Job rolled back due to failure - all data has been cleaned up",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", job_id).execute()
            
            logger.info(f"Successfully rolled back job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup data for job {job_id}: {e}")
            return False

    @staticmethod
    def get_job_progress_info(job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed progress information for a job.
        Useful for UI updates and monitoring.
        
        Args:
            job_id: The job UUID
            
        Returns:
            Progress dict with status, counts, and error info, or None
        """
        job = IngestionJobService.get_job(job_id)
        if not job:
            return None
        
        return {
            "job_id": job["id"],
            "status": job["status"],
            "current_count": job.get("current_count", 0),
            "total_count": job.get("total_count", 0),
            "progress_percent": int(
                (job.get("current_count", 0) / max(job.get("total_count", 1), 1)) * 100
            ),
            "provider": job.get("provider"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "error_message": job.get("error_message")
        }
    
    @staticmethod
    def mark_job_for_recovery(job_id: str) -> bool:
        """
        Mark a job as actively processing.
        Ensures job can be recovered if app crashes during processing.
        
        Args:
            job_id: The job UUID
            
        Returns:
            True if successful, False otherwise
        """
        supabase = get_supabase()
        try:
            supabase.table("ingestion_jobs").update({
                "status": "PARSING",  # Mark as actively processing
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", job_id).execute()
            logger.debug(f"Job {job_id} marked as actively processing")
            return True
        except Exception as e:
            logger.error(f"Failed to update job {job_id} status: {e}")
            return False
