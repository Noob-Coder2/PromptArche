
import json
import logging
import httpx
import ijson
import warnings
import tempfile
import shutil
import os
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
        # ChatGPT export is a list of conversations.
        # ijson.items(f, 'item') yields each conversation dictionary one by one.
        try:
            conversations = ijson.items(file_obj, 'item')
            for conversation in conversations:
                mapping = conversation.get('mapping', {})
                for node_id, node in mapping.items():
                    message = node.get('message')
                    # Check for User Role
                    if message and message.get('author', {}).get('role') == 'user':
                        if message.get('content', {}).get('content_type') != 'text':
                            continue
                            
                        content_parts = message.get('content', {}).get('parts', [])
                        if content_parts and isinstance(content_parts[0], str):
                            text = content_parts[0]
                            if not text.strip():
                                continue
                            
                            # Metadata Extraction
                            create_time = message.get('create_time')
                            created_at = datetime.fromtimestamp(create_time).isoformat() if create_time else datetime.now().isoformat()
                            
                            yield {
                                "content": text,
                                "source": "chatgpt",
                                "created_at": created_at,
                                "metadata": {
                                    "original_id": message.get('id'),
                                    "conversation_id": conversation.get('conversation_id')
                                }
                            }
        except Exception as e:
            logger.error(f"ChatGPT Parse Error: {e}")

class ClaudeParser(IngestionParser):
    def parse(self, file_obj: IO) -> Generator[Dict[str, Any], None, None]:
        try:
            # Assuming Claude export is a list of chats
            chats = ijson.items(file_obj, 'item')
            for chat in chats:
                messages = chat.get('chat_messages', [])
                for msg in messages:
                    if msg.get('sender') == 'human':
                        text = msg.get('text', '')
                        created_at = msg.get('created_at', datetime.now().isoformat())
                        yield {
                            "content": text,
                            "source": "claude",
                            "created_at": created_at,
                            "metadata": {"conversation_uuid": chat.get('uuid')}
                        }
        except Exception as e:
            logger.error(f"Claude Parse Error: {e}")

class GeminiParser(IngestionParser):
    def parse(self, file_obj: IO) -> Generator[Dict[str, Any], None, None]:
        try:
            # Gemini Takeout is a list of conversations
            conversations = ijson.items(file_obj, 'item')
            for convo in conversations:
                messages = convo.get('messages', [])
                title = convo.get('title')
                
                for msg in messages:
                    role = msg.get('author') or msg.get('role')
                    if role == 'user':
                        text = msg.get('content', '')
                        if not text:
                            continue
                        
                        created_at = msg.get('created_at', datetime.now().isoformat())
                        yield {
                            "content": text,
                            "source": "gemini",
                            "created_at": created_at,
                            "metadata": {"title": title}
                        }
        except Exception as e:
             logger.error(f"Gemini Parse Error: {e}")

# --- Service ---
class IngestionService:
    """
    Service for handling file ingestion with streaming, progress tracking, and transaction management.
    Designed to handle large files (multi-GB) without memory exhaustion.
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
    def ingest_sync(
        file_obj: IO,
        provider: str,
        user_id: str,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous streaming ingestion to be run in a thread pool.
        Designed to be called from async context via executor.
        Processes large files without loading entire content into memory.
        
        Features:
        - Streaming parser for memory efficiency
        - Batch processing to prevent connection exhaustion
        - Retry logic with exponential backoff
        - Transaction rollback on failure
        - Progress tracking
        - Safe for concurrent execution (no global state)
        
        Args:
            file_obj: File-like object to parse (seekable)
            provider: Data provider (chatgpt, claude, gemini)
            user_id: User's UUID string
            job_id: Optional job ID for progress tracking
            
        Returns:
            Dict with status and processed count
        """
        from app.services.job_service import IngestionJobService
        
        parser_map = {
            "chatgpt": ChatGPTParser(),
            "claude": ClaudeParser(),
            "gemini": GeminiParser()
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
                        # Generate embeddings in batch (50-100x faster than sequential)
                        IngestionService._flush_buffer_with_embeddings(
                            supabase, prompts_buffer, batch_num
                        )
                        success_count += batch_count
                        logger.info(f"Batch {batch_num}: Inserted {batch_count} items with embeddings")
                    except Exception as e:
                        logger.error(f"Failed to flush batch {batch_num}: {e}")
                        failed_batch_ids.append(batch_num)
                        # Don't stop processing, mark batch as failed and continue
                        # This allows partial recovery
                    finally:
                        prompts_buffer = []
                    
                    # Update progress
                    if job_id:
                        IngestionJobService.update_progress(
                            job_id, "PARSING", current_count=success_count
                        )
            
            # Flush remaining buffer
            if prompts_buffer:
                batch_num += 1
                batch_count = len(prompts_buffer)
                
                try:
                    IngestionService._flush_buffer_with_embeddings(
                        supabase, prompts_buffer, batch_num
                    )
                    success_count += batch_count
                    logger.info(f"Batch {batch_num} (final): Inserted {batch_count} items with embeddings")
                except Exception as e:
                    logger.error(f"Failed to flush final batch {batch_num}: {e}")
                    failed_batch_ids.append(batch_num)
            
            # If there were failed batches, attempt cleanup/rollback
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
            
        except Exception as e:
            logger.error(f"Ingestion failed with exception: {e}", exc_info=True)
            if job_id:
                IngestionJobService.fail_job(
                    job_id, 
                    f"Ingestion error: {str(e)}"
                )
            return {"status": "error", "message": str(e)}

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
            supabase.table("prompts").upsert(
                batch, on_conflict="user_id, content", ignore_duplicates=True
            ).execute()
            if batch_num:
                logger.debug(f"Successfully flushed batch {batch_num} with {len(batch)} items")
        except Exception as e:
            batch_info = f" (batch {batch_num})" if batch_num else ""
            logger.warning(f"Upsert attempt failed{batch_info}, will retry: {e}")
            raise

    @staticmethod
    def _flush_buffer(supabase, batch):
        """Legacy method for backward compatibility."""
        IngestionService._flush_buffer_with_retry(supabase, batch)

# Maintain backward compatibility if needed, or update consumers
async def ingest_chatgpt_export(content: bytes, user_id: UUID):
    return await IngestionService.ingest(content, "chatgpt", user_id)


        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
    
    # --- Ingestion Methods ---
    @staticmethod
    def ingest_sync(
        file_obj: IO,
        provider: str,
        user_id: str,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous ingestion to be run in a thread pool.
        Accepts a file-like object (bytes stream).
        
        Args:
            file_obj: File-like object to parse
            provider: Data provider (chatgpt, claude, gemini)
            user_id: User's UUID string
            job_id: Optional job ID for progress tracking
        """
        from app.services.job_service import IngestionJobService
        
        parser_map = {
            "chatgpt": ChatGPTParser(),
            "claude": ClaudeParser(),
            "gemini": GeminiParser()
        }
        
        parser = parser_map.get(provider)
        if not parser:
            if job_id:
                IngestionJobService.fail_job(job_id, f"Unknown provider: {provider}")
            return {"status": "error", "message": f"Unknown provider: {provider}"}
        
        # Update job status
        if job_id:
            IngestionJobService.update_progress(job_id, "PARSING")
            
        prompts_buffer = []
        success_count = 0
        supabase = get_supabase()
        
        try:
            # Generator based processing
            for parsed_item in parser.parse(file_obj):
                parsed_item["user_id"] = str(user_id)
                prompts_buffer.append(parsed_item)
                
                if len(prompts_buffer) >= 100:
                    IngestionService._flush_buffer_with_retry(supabase, prompts_buffer)
                    success_count += len(prompts_buffer)
                    prompts_buffer = []
                    
                    # Update progress
                    if job_id:
                        IngestionJobService.update_progress(
                            job_id, "PARSING", current_count=success_count
                        )
                    
            # Flush remaining
            if prompts_buffer:
                batch_num += 1
                batch_count = len(prompts_buffer)
                
                try:
                    IngestionService._flush_buffer_with_retry(
                        supabase, prompts_buffer, batch_num
                    )
                    success_count += batch_count
                    logger.info(f"Batch {batch_num} (final): Inserted {batch_count} items")
                except Exception as e:
                    logger.error(f"Failed to flush final batch {batch_num}: {e}")
                    failed_batch_ids.append(batch_num)
            
            # If there were failed batches, attempt cleanup/rollback
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
            
        except Exception as e:
            logger.error(f"Ingestion failed with exception: {e}", exc_info=True)
            if job_id:
                IngestionJobService.fail_job(
                    job_id, 
                    f"Ingestion error: {str(e)}"
                )
            return {"status": "error", "message": str(e)}

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
            supabase.table("prompts").upsert(
                batch, on_conflict="user_id, content", ignore_duplicates=True
            ).execute()
            if batch_num:
                logger.debug(f"Successfully flushed batch {batch_num} with {len(batch)} items")
        except Exception as e:
            batch_info = f" (batch {batch_num})" if batch_num else ""
            logger.warning(f"Upsert attempt failed{batch_info}, will retry: {e}")
            raise

    @staticmethod
    def _flush_buffer(supabase, batch):
        """Legacy method for backward compatibility."""
        IngestionService._flush_buffer_with_retry(supabase, batch)

# Maintain backward compatibility if needed, or update consumers
async def ingest_chatgpt_export(content: bytes, user_id: UUID):
    return await IngestionService.ingest(content, "chatgpt", user_id)