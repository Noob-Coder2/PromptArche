"""
Input validation utilities for the application.
"""

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
    "gemini": {
        "required_fields": ["messages"],
        "description": "Gemini conversation export format"
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


def validate_json_structure(file: UploadFile, provider: str) -> Dict[str, Any]:
    """
    Validate JSON structure based on provider type.
    
    Args:
        file: UploadFile containing JSON data
        provider: Data provider type (chatgpt, claude, gemini)
        
    Returns:
        Parsed JSON data
        
    Raises:
        HTTPException: If JSON is invalid or structure is incorrect
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider: {provider}. Supported: {list(SUPPORTED_PROVIDERS.keys())}"
        )
    
    try:
        # Read and parse JSON
        content = file.file.read()
        file.file.seek(0)  # Reset file pointer for further processing
        
        json_data = json.loads(content.decode('utf-8'))
        
        # Validate structure based on provider
        provider_config = SUPPORTED_PROVIDERS[provider]
        
        if provider == "chatgpt":
            if not isinstance(json_data, list):
                raise HTTPException(
                    status_code=400,
                    detail="ChatGPT export must be a list of conversations"
                )
            if not json_data:  # Empty list
                raise HTTPException(
                    status_code=400,
                    detail="ChatGPT export cannot be empty"
                )
                
        elif provider == "claude":
            if not isinstance(json_data, list):
                raise HTTPException(
                    status_code=400,
                    detail="Claude export must be a list of conversations"
                )
            if not json_data:
                raise HTTPException(
                    status_code=400,
                    detail="Claude export cannot be empty"
                )
                
        elif provider == "gemini":
            if not isinstance(json_data, list):
                raise HTTPException(
                    status_code=400,
                    detail="Gemini export must be a list of conversations"
                )
            if not json_data:
                raise HTTPException(
                    status_code=400,
                    detail="Gemini export cannot be empty"
                )
        
        logger.info(f"Validated {provider} export with {len(json_data)} items")
        return json_data
        
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON format: {str(e)}"
        )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded"
        )
    except Exception as e:
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