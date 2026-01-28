"""
Dashboard router for handling dashboard statistics and data API endpoints.
"""

from fastapi import APIRouter, Depends
from app.core.security import get_current_user_id
from app.db.supabase import get_supabase

router = APIRouter()


@router.get("/api/dashboard/stats")
async def get_dashboard_stats(user_id: str = Depends(get_current_user_id)):
    """Get dashboard statistics for the current user."""
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
    
    return stats


@router.get("/api/dashboard/clusters")
async def get_user_clusters(user_id: str = Depends(get_current_user_id)):
    """Get all clusters for the current user."""
    supabase = get_supabase()
    
    clusters_res = supabase.table("clusters").select("*").eq("user_id", user_id).execute()
    return clusters_res.data or []


@router.get("/api/dashboard/timeline")
async def get_timeline_data(user_id: str = Depends(get_current_user_id)):
    """Get timeline data for charting."""
    supabase = get_supabase()
    
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

    return chart_data