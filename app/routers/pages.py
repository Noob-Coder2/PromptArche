"""
Page router for handling HTML page rendering.
"""

from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from app.core.security import get_current_user_id, get_optional_user
from app.core.rate_limiter import rate_limit
from app.core.config import settings
from app.routers.dashboard import get_dashboard_stats, get_user_clusters, get_timeline_data

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the login page."""
    return templates.TemplateResponse("login.html", {
        "request": request,
        "supabase_url": settings.SUPABASE_URL,
        "supabase_key": settings.SUPABASE_KEY
    })


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render the login page."""
    return templates.TemplateResponse("login.html", {
        "request": request,
        "supabase_url": settings.SUPABASE_URL,
        "supabase_key": settings.SUPABASE_KEY
    })


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_view(request: Request, user_id: str = Depends(get_current_user_id), _: None = Depends(rate_limit)):
    """Render the dashboard page with user data."""
    # Get dashboard data
    stats = await get_dashboard_stats(user_id)
    clusters = await get_user_clusters(user_id)
    chart_data = await get_timeline_data(user_id)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "stats": stats,
        "clusters": clusters,
        "chart_data": chart_data
    })


@router.get("/upload", response_class=HTMLResponse)
async def upload_view(request: Request, user_id: str = Depends(get_current_user_id), _: None = Depends(rate_limit)):
    """Render the upload page."""
    return templates.TemplateResponse("upload.html", {"request": request})


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}