
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.core.security import verify_jwt, get_current_user_id
from app.services.ingestion import ingest_chatgpt_export
from app.db.supabase import get_supabase

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_view(request: Request, user_id: str = Depends(get_current_user_id)): # Note: Using get_current_user_id logic that parses cookie/token
    supabase = get_supabase()
    
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
    # Group by Date... PostgREST doesn't do complex group-by easy.
    # We'll fetch 'created_at' of all prompts (lightweight-ish if only id+date) and process in python for MVP.
    # Limitation: If user has 10k prompts, this is slow.
    # Better: Use a Postgres Function (RPC).
    # For MVP: Limit to last 1000 prompts.
    
    timeline_res = supabase.table("prompts").select("created_at").eq("user_id", user_id).order("created_at", desc=True).limit(1000).execute()
    
    # Process in Python: Group by Day
    from collections import defaultdict
    from datetime import datetime
    
    date_counts = defaultdict(int)
    for row in timeline_res.data:
        try:
             # ISO format: 2024-01-24T10:00:00...
             dt_str = row['created_at'].split('T')[0]
             date_counts[dt_str] += 1
        except:
             continue
             
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

@router.post("/api/ingest/chatgpt")
async def ingest_chatgpt(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)
):
    content = await file.read()
    
    # We call the service function. 
    # Ideally should be a background task IF the parsing is heavy.
    # Parsing JSON of 50MB is synchronous and might block.
    # Let's run it in an async wrapper or just handle it.
    
    result = await ingest_chatgpt_export(content, user_id)
    
    return result

# We need to wire this router to main.py
# Also need to fix 'get_current_user_id' to look at Cookie if not Bearer is passed?
# The `verify_jwt` I wrote looks at `HTTPAuthorizationCredentials`.
# Fastapi `OAuth2PasswordBearer` looks at Header.
# I should update `verify_jwt` to also check Cookie.
