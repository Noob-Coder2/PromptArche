
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from uuid import UUID

from app.db.supabase import get_supabase
from app.services.embeddings import generate_embeddings_batch

logger = logging.getLogger(__name__)

async def ingest_chatgpt_export(file_content: bytes, user_id: UUID):
    """
    Parses a ChatGPT `conversations.json` file and stores user prompts.
    """
    try:
        data = json.loads(file_content)
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON")
        return {"status": "error", "message": "Invalid JSON"}

    supabase = get_supabase()
    prompts_to_insert = []
    
    # Iterate through conversations
    for conversation in data:
        # Standard ChatGPT export structure:
        # conversation['mapping'] is a dict of nodes.
        # Check for 'message' -> 'author' -> 'role' == 'user'.
        mapping = conversation.get('mapping', {})
        for node_id, node in mapping.items():
            message = node.get('message')
            if message and message.get('author', {}).get('role') == 'user':
                content_parts = message.get('content', {}).get('parts', [])
                if content_parts and isinstance(content_parts[0], str):
                    text = content_parts[0]
                    if not text.strip():
                        continue
                        
                    # Basic deduplication check happens at DB level via unique constraint.
                    # We can prepare the row.
                    
                    # Timestamps from export are unix epoch
                    create_time = message.get('create_time')
                    created_at = datetime.fromtimestamp(create_time).isoformat() if create_time else datetime.now().isoformat()
                    
                    prompts_to_insert.append({
                        "user_id": str(user_id),
                        "content": text,
                        "source": "chatgpt",
                        "created_at": created_at,
                        "metadata": {"original_id": message.get('id')}
                    })

    # Bulk insert raw prompts first (ignoring embeddings for now to save time/API calls if they fail)
    # OR: Process in chunks, embed and insert.
    # Given the goal: "Cache embeddings... never re-embed same prompt"
    # Strategy:
    # 1. Check if content already exists for user.
    # 2. If valid embedding exists, skip.
    # 3. If new, embed and insert.
    
    # For MVP performance with large files:
    # We will try to rely on "ON CONFLICT DO NOTHING" via Supabase/Postgres.
    # But to get embeddings, we need to know WHICH ones are new.
    
    # Let's filter in python for now (naive approach, assume we want to process all that aren't in DB).
    # Fetch existing hashes or content snippets? Too heavy.
    
    # Better: Insert all valid prompts into a temporary list. 
    # Iterate and check DB 'exists' or handle "upsert" logic carefully.
    
    # Revised Strategy for Reliability:
    # 1. Insert Prompt text WITHOUT embedding. (Handle duplicates via constraint).
    # 2. Background Task: Query prompts where embedding IS NULL.
    # 3. Generate embeddings for those and update.
    
    # Lets try to Insert chunk by chunk.
    
    success_count = 0
    
    # Insert raw prompts first
    for chunk in chunk_list(prompts_to_insert, 100):
        try:
             # Supabase upsert: on_conflict='user_id, content' -> ignore
             response = supabase.table("prompts").upsert(
                 chunk, on_conflict="user_id, content", ignore_duplicates=True
             ).execute()
             # Note: response.data might be empty if all ignored.
             success_count += len(chunk)
        except Exception as e:
            logger.error(f"Error inserting chunk: {e}")
            
    return {"status": "success", "processed": success_count, "message": "Prompts queued for embedding"}

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def process_missing_embeddings():
    """
    Background, scheduled, or triggered task to fill missing embeddings.
    """
    supabase = get_supabase()
    
    # Fetch rows with null embedding
    # limit to avoid timeouts
    response = supabase.table("prompts").select("id, content").is_("embedding", "null").limit(50).execute()
    rows = response.data
    
    if not rows:
        return
        
    for row in rows:
        try:
            embedding = await generate_embeddings_batch([row['content']]) # batch of 1
            if embedding:
                supabase.table("prompts").update({"embedding": embedding[0]}).eq("id", row['id']).execute()
        except Exception as e:
            logger.error(f"Failed to embed prompt {row['id']}: {e}")
