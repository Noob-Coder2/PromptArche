
import logging
from uuid import UUID
import httpx
import json
from app.core.config import settings
from app.db.supabase import get_supabase

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

async def generate_cluster_insight(cluster_id: str, user_id: str):
    """
    Generates a brutally honest insight for a specific cluster.
    """
    supabase = get_supabase()
    
    # Fetch random 5 prompts from this cluster to represent it
    # Supabase doesn't have native "random()" easily exposed in py client select logic without RPC.
    # We'll just fetch 10 and pick in python.
    res = supabase.table("prompts").select("content").eq("cluster_id", cluster_id).limit(10).execute()
    if not res.data:
        return
        
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
        "model": "qwen/qwen3-32b", # User specified model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze these prompts:\n{prompt_text}"}
        ],
        "response_format": {"type": "json_object"} 
    }
    
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        # "HTTP-Referer": "https://promp-arche.app", # Optional
    }
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(GROQ_URL, headers=headers, json=payload, timeout=20.0)
            if resp.status_code != 200:
                logger.error(f"Groq Error: {resp.text}")
                return
                
            data = resp.json()
            content = data['choices'][0]['message']['content']
            
            try:
                parsed = json.loads(content)
                title = parsed.get("title", "Cluster Insight")
                insight_text = parsed.get("insight", "No insight generated.")
            except:
                # Fallback if model outputs raw text
                title = "Cluster Insight"
                insight_text = content
                
            # Update Cluster Label
            supabase.table("clusters").update({"label": title, "description": insight_text}).eq("id", cluster_id).execute()
            
            # Store dedicated Insight entry
            supabase.table("insights").insert({
                "user_id": user_id,
                "cluster_id": cluster_id,
                "content": insight_text
            }).execute()
            
    except Exception as e:
        logger.error(f"Insight Generation Failed: {e}")
