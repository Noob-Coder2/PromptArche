"""
Rate limiting utility for API endpoints.
"""

import time
import logging
from typing import Dict, Optional
from collections import defaultdict
from fastapi import HTTPException, Request
from app.core.config import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.
    For production, consider Redis-based implementation.
    """
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed based on rate limit.
        
        Args:
            identifier: User identifier (IP or user_id)
            
        Returns:
            True if request is allowed, False otherwise
        """
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier] 
            if req_time > minute_ago
        ]
        
        # Check if limit exceeded
        if len(self.requests[identifier]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[identifier].append(current_time)
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """
        Get remaining requests for identifier.
        
        Args:
            identifier: User identifier
            
        Returns:
            Number of remaining requests
        """
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier] 
            if req_time > minute_ago
        ]
        
        return max(0, self.requests_per_minute - len(self.requests[identifier]))


# Global rate limiter instance
rate_limiter = RateLimiter(settings.RATE_LIMIT_REQUESTS)


def rate_limit(request: Request, identifier: Optional[str] = None) -> None:
    """
    Rate limiting dependency for FastAPI.
    
    Args:
        request: FastAPI request object
        identifier: Optional identifier (defaults to client IP)
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    if identifier is None:
        # Use client IP as identifier
        identifier = request.client.host
    
    if not rate_limiter.is_allowed(identifier):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )


def get_rate_limit_info(identifier: Optional[str] = None) -> Dict[str, int]:
    """
    Get rate limit information for display.
    
    Args:
        identifier: User identifier
        
    Returns:
        Dict with rate limit information
    """
    if identifier is None:
        return {
            "limit": settings.RATE_LIMIT_REQUESTS,
            "remaining": settings.RATE_LIMIT_REQUESTS
        }
    
    remaining = rate_limiter.get_remaining_requests(identifier)
    
    return {
        "limit": settings.RATE_LIMIT_REQUESTS,
        "remaining": remaining
    }