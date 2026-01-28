"""
Ingestion router for handling file upload and job management API endpoints.
"""

from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, Form, HTTPException
from fastapi.responses import JSONResponse
from app.core.security import get_current_user_id
from app.services.ingestion import IngestionService
from app.services.job_service import IngestionJobService

router = APIRouter()


@router.post("/api/ingest")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    provider: str = Form("chatgpt"),
    user_id: str = Depends(get_current_user_id)
):
    """
    Start file ingestion in the background.
    Returns a job ID for progress tracking.
    """
    try:
        # Save upload to temp file using service utility
        file_path = IngestionService.save_upload_to_temp(file)
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": "Failed to save upload"},
            status_code=500
        )
    
    # Create job for tracking
    try:
        job_id = IngestionJobService.create_job(user_id, provider)
    except Exception as e:
        IngestionService.cleanup_temp_file(file_path)
        return JSONResponse(
            {"status": "error", "message": "Failed to create job"},
            status_code=500
        )
    
    # Add background task
    background_tasks.add_task(
        process_upload_background,
        file_path,
        provider,
        user_id,
        job_id
    )
    
    return {"status": "success", "message": "Ingestion started", "job_id": job_id}


@router.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, user_id: str = Depends(get_current_user_id)):
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
async def get_user_jobs(user_id: str = Depends(get_current_user_id)):
    """Get recent ingestion jobs for the current user."""
    jobs = IngestionJobService.get_user_jobs(user_id, limit=10)
    return {"jobs": jobs}


@router.get("/api/jobs/active")
async def get_active_job(user_id: str = Depends(get_current_user_id)):
    """Get the currently active ingestion job, if any."""
    job = IngestionJobService.get_active_job(user_id)
    if job:
        return {"active": True, "job": job}
    return {"active": False, "job": None}


def process_upload_background(
    file_path: str,
    provider: str,
    user_id: str,
    job_id: str
):
    """
    Background task for processing uploaded files.
    Uses IngestionService for parsing and database operations.
    """
    try:
        with open(file_path, 'rb') as f:
            IngestionService.ingest_sync(f, provider, user_id, job_id)
    except Exception as e:
        # Job failure is handled inside ingest_sync, but catch any other errors
        import logging
        logging.getLogger(__name__).error(f"Background ingestion error: {e}")
        IngestionJobService.fail_job(job_id, str(e))
    finally:
        # Cleanup temp file
        IngestionService.cleanup_temp_file(file_path)