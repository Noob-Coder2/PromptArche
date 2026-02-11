"""
Input validation utilities for the application.
"""
import ijson
import json
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException, UploadFile
from app.core.config import settings

logger = logging.getLogger(__name__)

# Supported providers and their expected JSON structure
SUPPORTED_PROVIDERS = {
    "chatgpt": {
        "required_fields": ["mapping"],
        "description": "ChatGPT conversations.json format"
    },
    "claude": {
        "required_fields": ["chat_messages"],
        "description": "Claude conversation export format"
    },
    "grok": {
        "required_fields": ["conversations"],
        "description": "Grok conversation export format"
    }
}


def validate_file_size(file: UploadFile) -> None:
    """
    Validate that file size is within limits.
    
    Args:
        file: UploadFile to validate
        
    Raises:
        HTTPException: If file is too large
    """
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE} bytes"
        )


def validate_file_type(file: UploadFile) -> None:
    """
    Validate that file is a JSON file.
    
    Args:
        file: UploadFile to validate
        
    Raises:
        HTTPException: If file is not JSON
    """
    if not file.filename or not file.filename.lower().endswith('.json'):
        raise HTTPException(
            status_code=400,
            detail="Only JSON files are supported"
        )


def validate_json_structure(file: UploadFile, provider: str) -> None:
    """
    Validate JSON structure based on provider type using streaming (ijson).
    Only reads enough of the file to confirm the top-level structure,
    keeping memory usage constant regardless of file size.
    
    Args:
        file: UploadFile containing JSON data
        provider: Data provider type (chatgpt, claude, grok)
        
    Raises:
        HTTPException: If JSON is invalid or structure is incorrect
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider: {provider}. Supported: {list(SUPPORTED_PROVIDERS.keys())}"
        )
    
    try:
        file_obj = file.file
        
        if provider in ("chatgpt", "claude"):
            # Expect a top-level JSON array with at least one item
            # ijson.items at '' with prefix 'item' iterates array elements
            try:
                parser = ijson.parse(file_obj)
                first_event = next(parser, None)
                
                if first_event is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{provider.capitalize()} export cannot be empty"
                    )
                
                prefix, event, value = first_event
                if event != 'start_array':
                    raise HTTPException(
                        status_code=400,
                        detail=f"{provider.capitalize()} export must be a list of conversations"
                    )
                
                # Check there's at least one item in the array
                second_event = next(parser, None)
                if second_event is None or second_event[1] == 'end_array':
                    raise HTTPException(
                        status_code=400,
                        detail=f"{provider.capitalize()} export cannot be empty"
                    )
                    
            except ijson.JSONError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON format: {str(e)}"
                )
                
        elif provider == "grok":
            # Expect a top-level JSON object containing a "conversations" key
            try:
                parser = ijson.parse(file_obj)
                first_event = next(parser, None)
                
                if first_event is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Grok export cannot be empty"
                    )
                
                prefix, event, value = first_event
                if event != 'start_map':
                    raise HTTPException(
                        status_code=400,
                        detail="Grok export must be a JSON object"
                    )
                
                # Scan for "conversations" key (read only top-level keys)
                found_conversations = False
                for prefix, event, value in parser:
                    if event == 'map_key' and value == 'conversations':
                        found_conversations = True
                        break
                    # Stop if we've entered a nested structure without finding it
                    # at the top level (prefix would be non-empty for nested keys)
                    if prefix and '.' in prefix:
                        continue
                
                if not found_conversations:
                    raise HTTPException(
                        status_code=400,
                        detail="Grok export must contain 'conversations' key"
                    )
                    
            except ijson.JSONError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON format: {str(e)}"
                )
        
        # Reset file pointer for actual processing
        file_obj.seek(0)
        logger.info(f"Validated {provider} export structure (streaming)")
        
    except HTTPException:
        file.file.seek(0)
        raise
    except UnicodeDecodeError:
        file.file.seek(0)
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded"
        )
    except Exception as e:
        file.file.seek(0)
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Validation failed: {str(e)}"
        )


def validate_provider(provider: str) -> None:
    """
    Validate that provider is supported.
    
    Args:
        provider: Provider name to validate
        
    Raises:
        HTTPException: If provider is not supported
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider: {provider}. Supported: {list(SUPPORTED_PROVIDERS.keys())}"
        )