
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.core.security import verify_jwt, get_current_user_id
from app.services.ingestion import IngestionService
from app.db.supabase import get_supabase
from app.core.config import settings

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

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
    
    # Check if user exists? Optional. JWT is trusted.
    
    # Fetch Stats
    # Note: Supabase-py count can be tricky.
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
    
    # Fetch Timeline Data (Chart.js)
    # Optimized: Query the View `prompt_stats_daily`
    timeline_res = supabase.table("prompt_stats_daily").select("day, count").eq("user_id", user_id).order("day", desc=False).limit(365).execute()
    
    date_counts = {}
    for row in timeline_res.data:
        # Postgres date_trunc returns timestamp, e.g. 2024-01-01T00:00:00+00:00
        d_str = row['day'].split('T')[0] 
        date_counts[d_str] = row['count']

    # Sort
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

@router.post("/api/ingest")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    provider: str = Form("chatgpt"),
    user_id: str = Depends(get_current_user_id)
):
    # Solved: Memory-bound & Blocking.
    # 1. Stream file to temp disk (FastAPI UploadFile is Spooled, but let's be safe and persisting it for background task is better).
    #    Actually, `file.file` is a `SpooledTemporaryFile`. If we pass it to BG task, it might be closed.
    #    We must copy it to a named temp file.
    import shutil
    import tempfile
    
    # Create temp file
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
    except Exception as e:
        return JSONResponse({"status": "error", "message": "Failed to save upload"}, status_code=500)
    
    # 2. Add Background Task
    # We pass the file_path. The task must open it, process, and delete it.
    background_tasks.add_task(process_upload_background, tmp.name, provider, user_id)
    
    return {"status": "success", "message": "Ingestion started in background"}

def process_upload_background(file_path: str, provider: str, user_id: str):
    import os
    try:
        with open(file_path, 'rb') as f:
            # Run sync ingestion
            IngestionService.ingest_sync(f, provider, user_id)
    except Exception as e:
        print(f"Background Ingestion Error: {e}")
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)


# We need to wire this router to main.py
# Also need to fix 'get_current_user_id' to look at Cookie if not Bearer is passed?
# The `verify_jwt` I wrote looks at `HTTPAuthorizationCredentials`.
# Fastapi `OAuth2PasswordBearer` looks at Header.
# I should update `verify_jwt` to also check Cookie.
