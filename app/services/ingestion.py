
import json
import logging
import httpx
import ijson
import warnings
from typing import List, Dict, Any, Generator, IO
from datetime import datetime
from uuid import UUID
from abc import ABC, abstractmethod


from app.db.supabase import get_supabase

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
    @staticmethod
    def ingest_sync(file_obj: IO, provider: str, user_id: UUID):
        """
        Synchronous ingestion to be run in a thread pool.
        Accepts a file-like object (bytes stream).
        """
        parser_map = {
            "chatgpt": ChatGPTParser(),
            "claude": ClaudeParser(),
            "gemini": GeminiParser()
        }
        
        parser = parser_map.get(provider)
        if not parser:
            return {"status": "error", "message": f"Unknown provider: {provider}"}
            
        prompts_buffer = []
        success_count = 0
        supabase = get_supabase()
        
        try:
            # Generator based processing
            for parsed_item in parser.parse(file_obj):
                parsed_item["user_id"] = str(user_id)
                prompts_buffer.append(parsed_item)
                
                if len(prompts_buffer) >= 100:
                    IngestionService._flush_buffer(supabase, prompts_buffer)
                    success_count += len(prompts_buffer)
                    prompts_buffer = []
                    
            # Flush remaining
            if prompts_buffer:
                IngestionService._flush_buffer(supabase, prompts_buffer)
                success_count += len(prompts_buffer)
                
            return {"status": "success", "processed": success_count}
        except Exception as e:
            logger.error(f"Ingestion Failed: {e}")
            return {"status": "error", "message": str(e)}


    @staticmethod
    def _flush_buffer(supabase, batch):
        try:
            supabase.table("prompts").upsert(
                 batch, on_conflict="user_id, content", ignore_duplicates=True
            ).execute()
        except Exception as e:
            logger.error(f"Failed to upsert batch: {e}")

# Maintain backward compatibility if needed, or update consumers
async def ingest_chatgpt_export(content: bytes, user_id: UUID):
    return await IngestionService.ingest(content, "chatgpt", user_id)
