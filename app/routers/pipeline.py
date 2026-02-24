"""
Pipeline router — manual triggers for clustering and insight generation.

These endpoints let the end user explicitly kick off:
  - Clustering  (for prompts that have embeddings but no cluster_id)
  - Insights    (for clusters that still have the default placeholder description)

The existing automatic pipeline (triggered after file upload) is NOT affected.
"""

import asyncio
import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from app.core.security import get_current_user_id
from app.core.rate_limiter import rate_limit
from app.db.supabase import get_supabase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# Sentinel description written by clustering.py for new clusters
_UNCLUSTERED_DESCRIPTION = "Auto-generated cluster"

# Track in-flight manual jobs so we don't double-fire
_running: Dict[str, bool] = {}


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------

@router.get("/status")
async def get_pipeline_status(
    user_id: str = Depends(get_current_user_id),
    _: None = Depends(rate_limit),
) -> Dict[str, Any]:
    """
    Returns counts of work that still needs to be done for the current user:
      - unclustered_prompts: prompts with an embedding but no cluster_id yet
      - clusters_without_insights: clusters whose description is still the
        auto-generated placeholder (i.e. LLM insight not yet generated)
      - clustering_running / insights_running: whether a manual job is active
    """
    supabase = get_supabase()

    try:
        # Prompts that have an embedding but haven't been assigned to a cluster
        unclustered_res = (
            supabase.table("prompts")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .is_("cluster_id", "null")
            .not_.is_("embedding", "null")
            .execute()
        )
        unclustered_count = unclustered_res.count or 0

        # Clusters that still carry the placeholder description
        pending_insights_res = (
            supabase.table("clusters")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .eq("description", _UNCLUSTERED_DESCRIPTION)
            .execute()
        )
        pending_insights_count = pending_insights_res.count or 0

    except Exception as e:
        # Transient network errors (e.g. RemoteProtocolError) — return safe
        # fallback so the frontend just retries on the next poll cycle.
        logger.warning(f"Pipeline status DB query failed (transient): {e}")
        return {
            "unclustered_prompts": 0,
            "clusters_without_insights": 0,
            "clustering_running": _running.get(f"cluster:{user_id}", False),
            "insights_running": _running.get(f"insights:{user_id}", False),
            "error": "temporary_db_error",
        }

    return {
        "unclustered_prompts": unclustered_count,
        "clusters_without_insights": pending_insights_count,
        "clustering_running": _running.get(f"cluster:{user_id}", False),
        "insights_running": _running.get(f"insights:{user_id}", False),
    }


# ---------------------------------------------------------------------------
# Run clustering
# ---------------------------------------------------------------------------

@router.post("/run-clustering")
async def run_clustering(
    user_id: str = Depends(get_current_user_id),
    _: None = Depends(rate_limit),
) -> Dict[str, Any]:
    """
    Manually trigger clustering for the current user.
    Runs in a background task so the request returns immediately.
    The existing automatic post-ingestion clustering is NOT affected.
    """
    key = f"cluster:{user_id}"
    if _running.get(key):
        return {"status": "already_running", "message": "Clustering is already in progress"}

    async def _run():
        _running[key] = True
        try:
            from app.services.clustering import run_clustering_for_user
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_clustering_for_user, user_id)
            logger.info(f"Manual clustering complete for user {user_id}: {result}")
        except Exception as e:
            logger.error(f"Manual clustering failed for user {user_id}: {e}", exc_info=True)
        finally:
            _running.pop(key, None)

    asyncio.create_task(_run())
    return {"status": "started", "message": "Clustering started in the background"}


# ---------------------------------------------------------------------------
# Run insights
# ---------------------------------------------------------------------------

@router.post("/run-insights")
async def run_insights(
    user_id: str = Depends(get_current_user_id),
    _: None = Depends(rate_limit),
) -> Dict[str, Any]:
    """
    Manually trigger insight generation for all clusters that still have the
    placeholder description.
    Runs in a background task so the request returns immediately.
    The existing automatic post-ingestion insight generation is NOT affected.
    """
    key = f"insights:{user_id}"
    if _running.get(key):
        return {"status": "already_running", "message": "Insight generation is already in progress"}

    async def _run():
        _running[key] = True
        try:
            from app.services.insights import generate_insights_for_user
            result = await generate_insights_for_user(user_id)
            logger.info(f"Manual insights complete for user {user_id}: {result}")
        except Exception as e:
            logger.error(f"Manual insights failed for user {user_id}: {e}", exc_info=True)
        finally:
            _running.pop(key, None)

    asyncio.create_task(_run())
    return {"status": "started", "message": "Insight generation started in the background"}
