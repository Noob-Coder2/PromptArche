
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from app.core.security import verify_jwt, get_current_user_id
from app.services.ingestion import IngestionService
from app.services.job_service import IngestionJobService
from app.db.supabase import get_supabase
from app.core.config import settings

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


# --- Page Routes ---

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "supabase_url": settings.SUPABASE_URL,
        "supabase_key": settings.SUPABASE_KEY
    })

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "supabase_url": settings.SUPABASE_URL,
        "supabase_key": settings.SUPABASE_KEY
    })

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_view(request: Request, user_id: str = Depends(get_current_user_id)):
    # This dependency now strictly validates the JWT from cookie or header.
    supabase = get_supabase()
    
    # Fetch Stats
    prompts_count = supabase.table("prompts").select("id", count="exact").eq("user_id", user_id).execute().count
    clusters_count = supabase.table("clusters").select("id", count="exact").eq("user_id", user_id).execute().count
    insights_count = supabase.table("insights").select("id", count="exact").eq("user_id", user_id).execute().count
    
    stats = {
        "total_prompts": prompts_count or 0,
        "total_clusters": clusters_count or 0,
        "total_insights": insights_count or 0
    }
    
    # Fetch Clusters
    clusters_res = supabase.table("clusters").select("*").eq("user_id", user_id).execute()
    clusters = clusters_res.data
    
    # Fetch Timeline Data (Chart.js) - Optimized via View
    timeline_res = supabase.table("prompt_stats_daily").select("day, count").eq("user_id", user_id).order("day", desc=False).limit(365).execute()
    
    date_counts = {}
    for row in timeline_res.data:
        d_str = row['day'].split('T')[0] 
        date_counts[d_str] = row['count']

    sorted_dates = sorted(date_counts.keys())
    chart_data = {
        "labels": sorted_dates,
        "values": [date_counts[d] for d in sorted_dates]
    }

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "clusters": clusters,
        "chart_data": chart_data
    })

@router.get("/upload", response_class=HTMLResponse)
async def upload_view(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


# --- Authentication Endpoints ---

@router.post("/api/login")
async def api_login(response: Response, access_token: str = Form(...)):
    """
    Set HTTP-only cookie for authentication.
    Token is validated by Supabase client-side before this call.
    """
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=False,  # Set to True in production with HTTPS
        max_age=3600 * 24 * 7  # 7 days
    )
    return {"status": "success"}

@router.post("/api/logout")
async def api_logout(response: Response):
    """Clear the authentication cookie."""
    response.delete_cookie(key="access_token")
    return {"status": "success"}


# --- Ingestion Endpoints ---

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


# --- Background Task ---

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

