"""
Dashboard router for handling dashboard statistics and data API endpoints.
"""

from fastapi import APIRouter, Depends, Request
from app.core.security import get_current_user_id
from app.core.rate_limiter import rate_limit
from app.db.supabase import get_supabase

router = APIRouter()


@router.get("/api/dashboard/stats")
async def get_dashboard_stats(user_id: str = Depends(get_current_user_id), _: None = Depends(rate_limit)):
    """Get dashboard statistics for the current user."""
    supabase = get_supabase()
    
    # Fetch Stats
    prompts_count = supabase.table("prompts").select("id", count="exact").eq("user_id", user_id).execute().count  # pyright: ignore[reportAttributeAccessIssue]
    clusters_count = supabase.table("clusters").select("id", count="exact").eq("user_id", user_id).execute().count # pyright: ignore[reportAttributeAccessIssue]
    insights_count = supabase.table("insights").select("id", count="exact").eq("user_id", user_id).execute().count # pyright: ignore[reportAttributeAccessIssue]
    
    stats = {
        "total_prompts": prompts_count or 0,
        "total_clusters": clusters_count or 0,
        "total_insights": insights_count or 0
    }
    
    return stats


@router.get("/api/dashboard/clusters")
async def get_user_clusters(user_id: str = Depends(get_current_user_id), _: None = Depends(rate_limit)):
    """Get all clusters for the current user."""
    supabase = get_supabase()
    
    clusters_res = supabase.table("clusters").select("*").eq("user_id", user_id).execute()
    return clusters_res.data or []


@router.get("/api/dashboard/timeline")
async def get_timeline_data(user_id: str = Depends(get_current_user_id), _: None = Depends(rate_limit)):
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


@router.get("/api/analytics/prompt-stats")
async def get_prompt_analytics(user_id: str = Depends(get_current_user_id), _: None = Depends(rate_limit)):
    """
    Get comprehensive prompt analytics for the current user.
    
    Returns:
        - summary: Aggregate statistics (total, avg/min/max length, by source)
        - distribution: Length distribution in histogram buckets
        - by_source: Statistics grouped by provider (chatgpt/claude/grok)
        - recent_trends: Daily trends for the last 30 days
        - extremes: Top 10 longest and shortest prompts
    """
    supabase = get_supabase()
    
    # Call the PostgreSQL function that aggregates all analytics views
    result = supabase.rpc('get_user_analytics', {'target_user_id': user_id}).execute()
    
    # If no data, return empty structure
    if not result.data:
        return {
            "summary": None,
            "distribution": [],
            "by_source": [],
            "recent_trends": [],
            "extremes": []
        }
    
    return result.data
