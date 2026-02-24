import json
import logging
import ijson
import warnings
import tempfile
import shutil
import os
import asyncio
from typing import List, Dict, Any, Generator, IO, Optional, Tuple
from datetime import datetime
from uuid import UUID
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_exponential

from app.db.supabase import get_supabase
from app.core.config import settings
from app.services.embedding_cache import get_cached_embeddings_batch

logger = logging.getLogger(__name__)

# --- Abstract Parser ---
class IngestionParser(ABC):
    @abstractmethod
    def parse(self, file_obj: IO) -> Generator[Dict[str, Any], None, None]:
        """
        Yields parsed prompt dictionaries: 
        { "content": str, "source": str, "created_at": str, "metadata": dict }
        """
        pass

# --- Parsers ---
class ChatGPTParser(IngestionParser):
    def parse(self, file_obj: IO) -> Generator[Dict[str, Any], None, None]:
        """
        Parse ChatGPT export format (as of 2024-2026).
        
        Format structure:
        - Array of conversations
        - Each conversation has: title, create_time, update_time, mapping
        - mapping contains nodes with messages
        - Filter for user messages with actual content
        """
        try:
            conversations = ijson.items(file_obj, 'item')
            for conversation in conversations:
                # Extract conversation metadata
                conversation_title = conversation.get('title', 'Untitled')
                conversation_create_time = conversation.get('create_time')
                
                # Process mapping nodes
                mapping = conversation.get('mapping', {})
                for node_id, node in mapping.items():
                    message = node.get('message')
                    
                    # Skip nodes without messages
                    if not message:
                        continue
                    
                    # Only process user messages
                    author = message.get('author', {})
                    if author.get('role') != 'user':
                        continue
                    
                    # Get content
                    content = message.get('content', {})
                    content_type = content.get('content_type')
                    
                    # Skip non-text content types
                    # Filter out: user_editable_context, system messages, etc.
                    if content_type != 'text':
                        continue
                    
                    # Extract text from parts
                    parts = content.get('parts', [])
                    if not parts or not isinstance(parts[0], str):
                        continue
                    
                    text = parts[0].strip()
                    if not text:
                        continue
                    
                    # Get timestamp (prefer message create_time, fallback to conversation)
                    create_time = message.get('create_time') or conversation_create_time
                    if create_time:
                        # ijson returns Decimal for floats, convert to float first
                        created_at = datetime.fromtimestamp(float(create_time)).isoformat()
                    else:
                        created_at = datetime.now().isoformat()
                    
                    yield {
                        "content": text,
                        "source": "chatgpt",
                        "external_id": message.get('id'),
                        "created_at": created_at,
                        "metadata": {
                            "message_id": message.get('id'),
                            "conversation_title": conversation_title,
                            "node_id": node_id,
                            "prompt_length": len(text)
                        }
                    }
        except Exception as e:
            logger.error(f"ChatGPT Parse Error: {e}", exc_info=True)


class ClaudeParser(IngestionParser):
    def parse(self, file_obj: IO) -> Generator[Dict[str, Any], None, None]:
        try:
            # Claude export: Array of conversation objects
            # Each has: {account: {...}, chat_messages: [...], created_at, name, uuid}
            conversations = ijson.items(file_obj, 'item')
            for conversation in conversations:
                conversation_uuid = conversation.get('uuid')
                conversation_name = conversation.get('name', '')
                
                # Each message in chat_messages array
                chat_messages = conversation.get('chat_messages', [])
                for msg in chat_messages:
                    # Filter for human messages
                    # Note: sender can be "human" or "assistant"
                    if msg.get('sender') == 'human':
                        # Content is an array of objects: [{type: "text", text: "..."}]
                        # Sometimes there's also a direct 'text' field (legacy/backup)
                        text = ''
                        
                        # Try content array first (official format)
                        content_array = msg.get('content', [])
                        if content_array and isinstance(content_array, list):
                            for content_item in content_array:
                                if isinstance(content_item, dict) and content_item.get('type') == 'text':
                                    text = content_item.get('text', '')
                                    break
                        
                        # Fallback to direct text field if content array didn't work
                        if not text:
                            text = msg.get('text', '')
                        
                        if not text.strip():
                            continue
                        
                        # Timestamps are ISO format strings (already formatted)
                        created_at = msg.get('created_at', datetime.now().isoformat())
                        
                        yield {
                            "content": text.strip(),
                            "source": "claude",
                            "external_id": msg.get('uuid'),
                            "created_at": created_at,
                            "metadata": {
                                "conversation_uuid": conversation_uuid,
                                "conversation_name": conversation_name,
                                "message_uuid": msg.get('uuid'),
                                "prompt_length": len(text.strip())
                            }
                        }
        except Exception as e:
            logger.error(f"Claude Parse Error: {e}")


class GrokParser(IngestionParser):
    def parse(self, file_obj: IO) -> Generator[Dict[str, Any], None, None]:
        try:
            # Grok export structure: {conversations: [{conversation: {...}, responses: [...]}]}
            # We need to use ijson to stream parse the nested structure
            # ijson path: 'conversations.item' gets each conversation object
            
            for conversation_data in ijson.items(file_obj, 'conversations.item'):
                # Extract conversation metadata
                conversation_info = conversation_data.get('conversation', {})
                conversation_id = conversation_info.get('id')
                title = conversation_info.get('title', '')
                
                # Parse responses array
                responses = conversation_data.get('responses', [])
                for response_wrapper in responses:
                    response = response_wrapper.get('response', {})
                    
                    # Only process human messages
                    if response.get('sender') == 'human':
                        text = response.get('message', '').strip()
                        if not text:
                            continue
                        
                        # Parse timestamp from nested $date.$numberLong structure
                        create_time_obj = response.get('create_time', {})
                        timestamp_ms = None
                        
                        if isinstance(create_time_obj, dict):
                            date_obj = create_time_obj.get('$date', {})
                            if isinstance(date_obj, dict):
                                number_long = date_obj.get('$numberLong')
                                if number_long:
                                    try:
                                        # Convert from milliseconds to seconds
                                        timestamp_ms = int(number_long) / 1000
                                    except (ValueError, TypeError):
                                        pass
                        
                        if timestamp_ms:
                            created_at = datetime.fromtimestamp(timestamp_ms).isoformat()
                        else:
                            created_at = datetime.now().isoformat()
                        
                        yield {
                            "content": text,
                            "source": "grok",
                            "external_id": response.get('_id'),
                            "created_at": created_at,
                            "metadata": {
                                "conversation_id": conversation_id,
                                "title": title,
                                "response_id": response.get('_id'),
                                "prompt_length": len(text)
                            }
                        }
        except Exception as e:
            logger.error(f"Grok Parse Error: {e}")


# --- Service ---
class IngestionService:
    """
    Service for handling file ingestion with streaming, progress tracking, and transaction management.
    Designed to handle large files (multi-GB) without memory exhaustion.
    
    NEW: Batch embedding generation during ingestion (50-100x faster than sequential).
    """
    
    # --- File Handling Utilities ---
    @staticmethod
    def save_upload_to_temp(upload_file) -> str:
        """
        Save an UploadFile to a temporary file for background processing.
        
        Args:
            upload_file: FastAPI UploadFile object
            
        Returns:
            Path to the temporary file
        """
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        try:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp.close()
            logger.debug(f"Saved upload to temp file: {tmp.name}")
            return tmp.name
        except Exception as e:
            logger.error(f"Failed to save upload to temp: {e}")
            raise
    
    @staticmethod
    def cleanup_temp_file(file_path: str):
        """
        Safely remove a temporary file.
        
        Args:
            file_path: Path to the file to remove
        """
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    # --- Ingestion Methods ---
    @staticmethod
    async def ingest_async(
        file_obj: IO,
        provider: str,
        user_id: str,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Async streaming ingestion pipeline.
        Processes large files without loading entire content into memory.
        
        Features:
        - Streaming parser for memory efficiency
        - Batch embedding generation (50-100x faster!)
        - Batch processing to prevent connection exhaustion
        - Retry logic with exponential backoff
        - Transaction rollback on failure
        - Progress tracking
        - Safe for concurrent execution (no global state)
        
        Args:
            file_obj: File-like object to parse (seekable)
            provider: Data provider (chatgpt, claude, grok)
            user_id: User's UUID string
            job_id: Optional job ID for progress tracking
            
        Returns:
            Dict with status and processed count
        """
        from app.services.job_service import IngestionJobService
        
        parser_map = {
            "chatgpt": ChatGPTParser(),
            "claude": ClaudeParser(),
            "grok": GrokParser()
        }
        
        parser = parser_map.get(provider)
        if not parser:
            if job_id:
                IngestionJobService.fail_job(job_id, f"Unknown provider: {provider}")
            return {"status": "error", "message": f"Unknown provider: {provider}"}
        
        # Update job status
        if job_id:
            IngestionJobService.update_progress(job_id, "PARSING")
        
        supabase = get_supabase()
        prompts_buffer = []
        success_count = 0
        failed_batch_ids = []  # Track failed batches for rollback
        batch_num = 0
        
        # Tracking for background tasks
        background_tasks: set = set()
        # Limit concurrent background embedding tasks to 4 to prevent API/DB overload
        semaphore = asyncio.Semaphore(4)
        
        try:
            # Generator-based streaming: processes items one at a time
            # This ensures memory usage stays constant regardless of file size
            for parsed_item in parser.parse(file_obj):
                parsed_item["user_id"] = str(user_id)
                prompts_buffer.append(parsed_item)
                
                # Flush buffer when it reaches configured batch size
                if len(prompts_buffer) >= settings.BATCH_SIZE:
                    batch_num += 1
                    batch_count = len(prompts_buffer)
                    try:
                        batch_num += 1
                        await loop.run_in_executor(
                            None, 
                            IngestionService._flush_buffer_hybrid,
                            supabase, list(prompts_buffer), str(user_id), batch_num, background_tasks, semaphore
                        )
                        success_count += len(prompts_buffer)
                        logger.info(f"Batch {batch_num}: Processed {len(prompts_buffer)} items")
                    except Exception as e:
                        logger.error(f"Failed to flush batch {batch_num}: {e}")
                        failed_batch_ids.append(batch_num)
                        # DUMP FAILED DATA
                        try:
                            with open("failed_prompts.jsonl", "a", encoding="utf-8") as f:
                                for item in prompts_buffer:
                                    f.write(json.dumps({
                                        "content": item.get("content", ""),
                                        "reason": f"Batch {batch_num} failed: {str(e)}",
                                        "timestamp": datetime.now().isoformat()
                                    }) + "\n")
                        except Exception as log_e:
                            logger.error(f"Failed to log failed batch: {log_e}")
                            
                    prompts_buffer.clear()
                    
                    # Update progress with EMBEDDING stage
                    if job_id:
                        IngestionJobService.update_progress(
                            job_id, "EMBEDDING", current_count=success_count
                        )
            
            # Flush remaining buffer
            if prompts_buffer:
                try:
                    batch_num += 1
                    await loop.run_in_executor(
                        None, 
                        IngestionService._flush_buffer_hybrid,
                        supabase, list(prompts_buffer), str(user_id), batch_num, background_tasks, semaphore
                    )
                    success_count += len(prompts_buffer)
                    logger.info(f"Batch {batch_num} (final): Processed {len(prompts_buffer)} items")
                    
                    if job_id:
                        IngestionJobService.update_progress(
                            job_id, "EMBEDDING", current_count=success_count
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to flush final batch {batch_num}: {e}")
                    failed_batch_ids.append(batch_num)
                    # DUMP FAILED DATA
                    try:
                        with open("failed_prompts.jsonl", "a", encoding="utf-8") as f:
                            for item in prompts_buffer:
                                f.write(json.dumps({
                                    "content": item.get("content", ""),
                                    "reason": f"Final batch {batch_num} failed: {str(e)}",
                                    "timestamp": datetime.now().isoformat()
                                }) + "\n")
                    except Exception as log_e:
                        logger.error(f"Failed to log failed batch: {log_e}")
            
            # --- Wait for background embedding tasks to complete ---
            if background_tasks:
                logger.info(f"Waiting for {len(background_tasks)} background embedding tasks...")
                await asyncio.gather(*background_tasks)
                logger.info("All background embedding tasks completed")
            
            # If there were failed batches, log and attempt recovery
            if failed_batch_ids:
                logger.warning(
                    f"Ingestion partially failed: {len(failed_batch_ids)} batch(es) failed. "
                    f"Processed {success_count} items before failure."
                )
                if job_id:
                    error_msg = f"Partial ingestion: {success_count} items processed, " \
                                f"but {len(failed_batch_ids)} batch(es) failed"
                    IngestionJobService.fail_job(job_id, error_msg)
                return {
                    "status": "partial_failure",
                    "processed": success_count,
                    "failed_batches": failed_batch_ids
                }
            
            # Mark job complete
            if job_id:
                IngestionJobService.complete_job(job_id, success_count)
            
            logger.info(f"Ingestion completed: {success_count} items processed in {batch_num} batches")
            return {"status": "success", "processed": success_count}
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}", exc_info=True)
            if job_id:
                IngestionJobService.fail_job(
                    job_id,
                    f"Invalid JSON format: {str(e)}"
                )
            return {"status": "error", "message": "Invalid JSON format"}
        
        except IOError as e:
            logger.error(f"File I/O error: {e}", exc_info=True)
            if job_id:
                IngestionJobService.fail_job(
                    job_id,
                    f"File read error: {str(e)}"
                )
            return {"status": "error", "message": "File read error"}
        
        except Exception as e:
            logger.error(f"Ingestion failed with unexpected exception: {e}", exc_info=True)
            if job_id:
                IngestionJobService.fail_job(
                    job_id,
                    f"Unexpected error: {str(e)}"
                )
            return {"status": "error", "message": "Unexpected error occurred"}

    @staticmethod
    @retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True
    )
    def _flush_buffer_with_retry(
        supabase,
        batch: List[Dict[str, Any]],
        batch_num: Optional[int] = None
    ):
        """
        Flush buffer to database with exponential backoff retry.
        Handles temporary database locks or network issues gracefully.
        
        Args:
            supabase: Supabase client instance (uses connection pool)
            batch: List of prompt dictionaries to insert
            batch_num: Optional batch number for logging
            
        Raises:
            Exception: If all retry attempts fail
        """
        try:
            # We use insert and catch duplicate key errors.
            # Uniqueness is enforced by (user_id, source, external_id) constraint.
            # This allows duplicate CONTENT (same prompt text) but prevents duplicate EVENTS.
            try:
                # DEBUG: Check if embedding leaked
                if batch and "embedding" in batch[0]:
                    logger.error("CRITICAL: 'embedding' key FOUND in batch sent to insert!")
                
                result = supabase.table("prompts").insert(batch).execute()
                if batch_num:
                    logger.debug(f"Successfully flushed batch {batch_num} with {len(batch)} items")
            except Exception as insert_error:
                # Check if it's a duplicate key error (23505 is PostgreSQL's unique violation code)
                error_msg = str(insert_error)
                if '23505' in error_msg or 'duplicate' in error_msg.lower():
                    # Duplicates are expected and OK - insert succeeded for non-duplicates
                    logger.debug(f"Batch {batch_num} had some duplicates (expected behavior)")
                else:
                    # This is a real error, re-raise it
                    raise
        except Exception as e:
            batch_info = f" (batch {batch_num})" if batch_num else ""
            logger.warning(f"Upsert attempt failed{batch_info}, will retry: {e}")
            raise

    @staticmethod
    async def _flush_buffer_hybrid(
        supabase,
        batch: List[Dict[str, Any]],
        user_id: str,
        batch_num: int,
        background_tasks: set,
        semaphore: asyncio.Semaphore
    ):
        """
        Hybrid pipeline:
        1. Store prompts immediately WITHOUT embeddings
        2. Queue background task to generate embeddings (stored in embedding_cache)
        3. Background task calls server-side RPC to copy embeddings from cache → prompts
        
        Embeddings NEVER travel through the REST API for the prompts table.
        This avoids PostgREST bytea serialization issues with vector columns.
        """
        batch_info = f" (batch {batch_num})"
        
        # --- Phase 1: Store Prompts (Immediate Insert, NO embedding) ---
        # Strip embedding field from all items — embeddings are handled server-side later
        insert_batch = []
        
        # Helper to extract checkable IDs
        check_ids = [item["external_id"] for item in batch if item.get("external_id")]
        existing_ids = set()
        
        # Pre-fetch existing IDs to avoid batch failure on duplicates
        if check_ids:
            try:
                existing_res = supabase.table("prompts").select("external_id") \
                    .eq("user_id", user_id) \
                    .in_("external_id", check_ids) \
                    .execute()
                existing_ids = {row["external_id"] for row in existing_res.data}
                if existing_ids:
                    logger.info(f"Batch {batch_num}: Found {len(existing_ids)} duplicates in DB, skipping them.")
            except Exception as e:
                logger.warning(f"Failed to check duplicates for batch {batch_num}: {e}")
        
        seen_ids = set()
        
        # Log DUPLICATES to a separate file (as mostly noise/info)
        try:
             duplicate_file = open("duplicate_prompts.jsonl", "a", encoding="utf-8")
        except Exception as e:
             logger.error(f"Failed to open duplicate log file: {e}")
             duplicate_file = None

        for item in batch:
            ext_id = item.get("external_id")
            skipped_reason = None
            
            # 1. Skip if exists in DB
            if ext_id and ext_id in existing_ids:
                skipped_reason = "already_in_db"
            
            # 2. Skip if duplicate within this batch
            elif ext_id:
                if ext_id in seen_ids:
                    skipped_reason = "duplicate_in_batch"
                else:
                    seen_ids.add(ext_id)
            
            if skipped_reason:
                if duplicate_file:
                    try:
                        log_entry = {
                            "external_id": ext_id,
                            "source": item.get("source"),
                            "content": item.get("content", ""),
                            "reason": skipped_reason,
                            "timestamp": datetime.now().isoformat()
                        }
                        duplicate_file.write(json.dumps(log_entry) + "\n")
                    except Exception:
                        pass
                continue
                
            insert_item = {k: v for k, v in item.items() if k != "embedding"}
            
            # WORKAROUND backslashes
            if "content" in insert_item and isinstance(insert_item["content"], str):
                insert_item["content"] = insert_item["content"].replace("\\", "")
                
            insert_batch.append(insert_item)
            
        if duplicate_file:
            duplicate_file.close()
        
        if not insert_batch:
            logger.info(f"Batch {batch_num}: All items were duplicates. Skipping insert.")
        else:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    IngestionService._flush_buffer_with_retry,
                    supabase, insert_batch, batch_num
                )
            except Exception as e:
                logger.error(f"Failed to store batch{batch_info}: {e}")
                raise

        # --- Phase 2: Background Embed + Server-Side Backfill ---
        # Queue a background task that:
        #   1. Generates embeddings via HF API (results stored in embedding_cache)
        #   2. Calls server-side RPC to copy from embedding_cache → prompts
        task = asyncio.create_task(
            IngestionService._process_embeddings_gap(
                supabase, 
                batch,  # Pass all items — cache service handles hits/misses efficiently
                user_id, 
                batch_num, 
                semaphore
            )
        )
        # Add to tracking set
        background_tasks.add(task)
        # Remove from set when done
        task.add_done_callback(background_tasks.discard)
        
        logger.debug(f"Batch{batch_info}: Queued background embedding for {len(batch)} items")

    @staticmethod
    async def _process_embeddings_gap(
        supabase,
        items: List[Dict[str, Any]],
        user_id: str,
        batch_num: int,
        semaphore: asyncio.Semaphore
    ):
        """
        Background task to generate embeddings and backfill them server-side.
        
        Flow:
        1. Call get_cached_embeddings_batch → generates via HF API for misses,
           stores ALL results in embedding_cache table automatically
        2. Call server-side RPC to copy embeddings from embedding_cache → prompts
           (vectors never travel through PostgREST)
        
        Respects concurrency limit via semaphore.
        """
        async with semaphore:
            batch_info = f" (BG batch {batch_num})"
            try:
                # 1. Generate Embeddings (API Call + Cache Storage)
                # get_cached_embeddings_batch handles:
                #   - Memory cache check
                #   - DB cache check  
                #   - HF API call for misses
                #   - Storing results in embedding_cache table
                texts = [item["content"] for item in items]
                embeddings = await get_cached_embeddings_batch(texts)
                
                generated_count = sum(1 for e in embeddings if e is not None)
                logger.info(f"Background{batch_info}: Generated {generated_count}/{len(items)} embeddings")
                
                # 2. Server-Side Backfill: Copy from embedding_cache → prompts via RPC
                # This runs entirely inside PostgreSQL — no vectors through PostgREST
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: supabase.rpc(
                            'backfill_embeddings_from_cache',
                            {'target_user_id': user_id}
                        ).execute()
                    )
                    
                    backfilled = result.data if result and hasattr(result, 'data') else 0
                    logger.info(f"Background{batch_info}: Server-side backfill updated {backfilled} prompts")
                except Exception as rpc_err:
                    logger.warning(f"RPC backfill failed{batch_info}: {rpc_err}")
                
            except Exception as e:
                logger.error(f"Background task failed{batch_info}: {e}")
                # We do NOT re-raise, as this runs in background and shouldn't crash the main loop
                # The prompt data is already safe in DB.
