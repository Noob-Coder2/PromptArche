"""
Ingestion router for handling file upload and job management API endpoints.
Uses async task queue for non-blocking background processing.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import asyncio
from app.core.security import get_current_user_id
from app.core.validators import validate_file_size, validate_file_type, validate_json_structure, validate_provider
from app.core.rate_limiter import rate_limit
from app.services.ingestion import IngestionService
from app.services.job_service import IngestionJobService
from app.services.task_queue import TaskQueueService

router = APIRouter()


@router.post("/api/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    provider: str = Form("chatgpt"),
    user_id: str = Depends(get_current_user_id),
    _: None = Depends(rate_limit)
):
    """
    Start file ingestion in the background using async task queue.
    Returns immediately with a job ID for progress tracking.
    
    The file is processed asynchronously without blocking other users.
    Progress can be tracked via GET /api/jobs/{job_id}
    """
    # Input validation
    try:
        validate_file_size(file)
        validate_file_type(file)
        validate_provider(provider)
        validate_json_structure(file, provider)  # This will also reset file pointer
    except HTTPException:
        # Re-raise validation errors
        raise
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": f"Validation failed: {str(e)}"},
            status_code=400
        )
    
    try:
        # Save upload to temp file using service utility
        file_path = IngestionService.save_upload_to_temp(file)
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": "Failed to save upload"},
            status_code=500
        )
    
    # Create job for tracking (with temp file path for recovery)
    try:
        job_id = IngestionJobService.create_job(user_id, provider, temp_file_path=file_path)
    except Exception as e:
        IngestionService.cleanup_temp_file(file_path)
        return JSONResponse(
            {"status": "error", "message": "Failed to create job"},
            status_code=500
        )
    
    # Enqueue async task (non-blocking)
    try:
        await TaskQueueService.enqueue_job(
            job_id=job_id,
            job_type="ingest_file",
            job_data={
                "file_path": file_path,
                "provider": provider
            },
            user_id=user_id
        )
    except Exception as e:
        IngestionJobService.fail_job(job_id, f"Failed to enqueue task: {str(e)}")
        return JSONResponse(
            {"status": "error", "message": "Failed to enqueue job"},
            status_code=500
        )
    
    return {
        "status": "success",
        "message": "Ingestion started (processing in background)",
        "job_id": job_id
    }


@router.get("/api/jobs/active")
async def get_active_job(user_id: str = Depends(get_current_user_id), _: None = Depends(rate_limit)):
    """Get the currently active ingestion job, if any."""
    job = IngestionJobService.get_active_job(user_id)
    if job:
        return {"active": True, "job": job}
    return {"active": False, "job": None}


@router.get("/api/jobs/stats")
async def get_queue_stats(user_id: str = Depends(get_current_user_id), _: None = Depends(rate_limit)):
    """Get task queue statistics (queue size, active tasks)."""
    stats = await TaskQueueService.get_queue_stats()
    return stats


@router.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, user_id: str = Depends(get_current_user_id), _: None = Depends(rate_limit)):
    """
    Get the current status of an ingestion job.
    Used by frontend to poll for progress.
    """
    job = IngestionJobService.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify ownership
    if job.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "id": job["id"],
        "status": job["status"],
        "current_count": job.get("current_count", 0),
        "total_count": job.get("total_count", 0),
        "error_message": job.get("error_message"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at")
    }


@router.get("/api/jobs")


async def process_upload_background(
    file_path: str,
    provider: str,
    user_id: str,
    job_id: str
):
    """
    DEPRECATED: Legacy background task.
    
    This function is kept for backward compatibility but should not be used.
    Use TaskQueueService.enqueue_job() instead for async processing.
    
    Uses IngestionService for parsing and database operations.
    """
    try:
        with open(file_path, 'rb') as f:
            await IngestionService.ingest_async(f, provider, user_id, job_id)
    except Exception as e:
        # Job failure is handled inside ingest_async, but catch any other errors
        import logging
        logging.getLogger(__name__).error(f"Background ingestion error: {e}")
        IngestionJobService.fail_job(job_id, str(e))
    finally:
        # Cleanup temp file
        IngestionService.cleanup_temp_file(file_path)