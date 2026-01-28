"""
Ingestion Job Service - Manages ingestion job lifecycle and progress tracking.
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
    def create_job(user_id: str, provider: str) -> str:
        """
        Create a new ingestion job in PENDING state.
        
        Args:
            user_id: The user's UUID
            provider: The data provider (chatgpt, claude, gemini)
            
        Returns:
            The job ID (UUID string)
        """
        supabase = get_supabase()
        try:
            res = supabase.table("ingestion_jobs").insert({
                "user_id": user_id,
                "provider": provider,
                "status": "PENDING",
                "current_count": 0,
                "total_count": 0
            }).execute()
            
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
