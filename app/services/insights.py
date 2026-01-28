
"""
Insights service for generating AI-powered cluster analysis.
Uses Groq API with Pydantic schema validation.
"""
import logging
from uuid import UUID
from typing import Optional, Dict, Any
import httpx
import json
from pydantic import ValidationError

from app.core.config import settings
from app.db.supabase import get_supabase
from app.schemas import InsightResponse

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3-32b"


async def generate_cluster_insight(
    cluster_id: str,
    user_id: str,
    client: Optional[httpx.AsyncClient] = None
) -> Optional[InsightResponse]:
    """
    Generates a brutally honest insight for a specific cluster.
    Uses Pydantic schema to validate LLM response structure.
    
    Args:
        cluster_id: UUID of the cluster to analyze
        user_id: UUID of the user
        client: Optional persistent HTTP client
        
    Returns:
        InsightResponse if successful, None otherwise
    """
    supabase = get_supabase()
    
    # Fetch sample prompts from this cluster
    res = supabase.table("prompts") \
        .select("content") \
        .eq("cluster_id", cluster_id) \
        .limit(10) \
        .execute()
        
    if not res.data:
        logger.warning(f"No prompts found for cluster {cluster_id}")
        return None
        
    prompts = [r['content'] for r in res.data]
    prompt_text = "\n---\n".join(prompts[:5])
    
    system_prompt = """
    You are a brutally honest coding coach and prompt engineer. 
    Analyze the following list of user prompts which form a semantic cluster.
    Identify the common pattern, intent, or bad habit.
    Give a short, punchy title (max 5 words) and a "Brutal Insight" (max 3 sentences).
    Tell them if they are over-engineering, being lazy, or hallucinating.
    Format your response as JSON: {"title": "...", "insight": "..."}
    """
    
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze these prompts:\n{prompt_text}"}
        ],
        "response_format": {"type": "json_object"}
    }
    
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
    }
    
    try:
        # Use provided client or create new one
        if client:
            resp = await client.post(GROQ_URL, headers=headers, json=payload, timeout=30.0)
        else:
            async with httpx.AsyncClient() as temp_client:
                resp = await temp_client.post(GROQ_URL, headers=headers, json=payload, timeout=30.0)
        
        if resp.status_code != 200:
            logger.error(f"Groq API Error: {resp.status_code} - {resp.text}")
            return None
            
        data = resp.json()
        content = data['choices'][0]['message']['content']
        
        # Validate response with Pydantic schema
        parsed = _parse_insight_response(content)
        
        # Update cluster with validated insight
        supabase.table("clusters").update({
            "label": parsed.title,
            "description": parsed.insight
        }).eq("id", cluster_id).execute()
        
        # Store dedicated insight entry
        supabase.table("insights").insert({
            "user_id": user_id,
            "cluster_id": cluster_id,
            "content": parsed.insight
        }).execute()
        
        logger.info(f"Generated insight for cluster {cluster_id}: {parsed.title}")
        return parsed
        
    except Exception as e:
        logger.error(f"Insight generation failed for cluster {cluster_id}: {e}")
        return None


def _parse_insight_response(content: str) -> InsightResponse:
    """
    Parse and validate LLM response using Pydantic schema.
    Falls back to defaults if parsing fails.
    
    Args:
        content: Raw LLM response string
        
    Returns:
        Validated InsightResponse
    """
    try:
        # Try to parse as JSON first
        parsed_json = json.loads(content)
        
        # Validate with Pydantic
        return InsightResponse.model_validate(parsed_json)
        
    except json.JSONDecodeError:
        # LLM returned non-JSON text - use as insight directly
        logger.warning("LLM returned non-JSON response, using as raw insight")
        return InsightResponse(
            title="Cluster Insight",
            insight=content[:500] if len(content) > 500 else content
        )
        
    except ValidationError as e:
        # JSON parsed but didn't match schema
        logger.warning(f"LLM response validation failed: {e}")
        
        # Try to extract what we can
        if isinstance(parsed_json, dict):
            title = str(parsed_json.get("title", "Cluster Insight"))[:50]
            insight = str(parsed_json.get("insight", "No insight generated."))[:500]
            return InsightResponse(title=title, insight=insight)
        
        return InsightResponse()


async def generate_insights_for_user(
    user_id: str,
    client: Optional[httpx.AsyncClient] = None
) -> Dict[str, Any]:
    """
    Generate insights for all clusters belonging to a user.
    
    Args:
        user_id: UUID of the user
        client: Optional persistent HTTP client
        
    Returns:
        Dict with success/failure counts
    """
    supabase = get_supabase()
    
    # Get all clusters for user
    res = supabase.table("clusters") \
        .select("id") \
        .eq("user_id", user_id) \
        .execute()
    
    if not res.data:
        return {"status": "skipped", "message": "No clusters found"}
    
    success_count = 0
    failure_count = 0
    
    for cluster in res.data:
        result = await generate_cluster_insight(cluster["id"], user_id, client)
        if result:
            success_count += 1
        else:
            failure_count += 1
    
    return {
        "status": "success",
        "insights_generated": success_count,
        "failures": failure_count
    }

