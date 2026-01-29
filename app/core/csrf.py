"""
CSRF protection middleware for PromptArche.

Implements stateless CSRF token validation using:
- Double Submit Cookie pattern
- Secure token generation and validation
- Configurable security settings
"""

import secrets
import hashlib
import logging
from typing import Optional, Dict, Any
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)


class CSRFTokenManager:
    """Manages CSRF token generation and validation."""
    
    def __init__(self):
        self.token_length = 32
        self.hash_algorithm = "sha256"
    
    def generate_token(self) -> str:
        """Generate a secure CSRF token."""
        random_bytes = secrets.token_bytes(self.token_length)
        token = secrets.token_urlsafe(self.token_length)
        return token
    
    def hash_token(self, token: str) -> str:
        """Hash a token for secure storage."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def create_csrf_cookie(self, response: Response, token: str) -> None:
        """Set CSRF token in secure cookie."""
        response.set_cookie(
            key="csrf_token",
            value=token,
            httponly=False,  # Accessible by JavaScript
            samesite=settings.validated_samesite,  # type: ignore
            secure=settings.COOKIE_SECURE,
            max_age=3600,  # 1 hour
            path="/"
        )
    
    def validate_csrf_token(self, request: Request, token: str) -> bool:
        """Validate CSRF token from header against cookie."""
        if not token:
            return False
        
        # Get token from cookie
        cookie_token = request.cookies.get("csrf_token")
        if not cookie_token:
            return False
        
        # Compare tokens (timing attack safe)
        return secrets.compare_digest(token, cookie_token)


from starlette.middleware.base import BaseHTTPMiddleware


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """CSRF protection middleware."""
    
    def __init__(self, app, exempt_paths: Optional[list] = None):
        super().__init__(app)
        self.token_manager = CSRFTokenManager()
        self.exempt_paths = exempt_paths or [
            "/api/login",
            "/api/logout",
            "/api/health",
            "/api/docs",
            "/openapi.json"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request with CSRF protection."""
        path = request.url.path
        
        # Skip CSRF check for exempt paths
        if path in self.exempt_paths or path.startswith("/static"):
            return await call_next(request)
        
        # Skip CSRF check for safe methods
        if request.method in ["GET", "HEAD", "OPTIONS", "TRACE"]:
            return await call_next(request)
        
        # Check for CSRF token in header
        csrf_token = request.headers.get("X-CSRF-Token")
        if not csrf_token:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "CSRF token missing from header"}
            )
        
        # Validate CSRF token
        if not self.token_manager.validate_csrf_token(request, csrf_token):
            logger.warning(f"CSRF validation failed for path: {path}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "Invalid CSRF token"}
            )
        
        return await call_next(request)


def get_csrf_token(request: Request) -> Dict[str, Any]:
    """Get or generate CSRF token for the current request."""
    token_manager = CSRFTokenManager()
    
    # Try to get existing token from cookie
    csrf_token = request.cookies.get("csrf_token")
    
    if not csrf_token:
        # Generate new token
        csrf_token = token_manager.generate_token()
        
        # Create response with token
        response = JSONResponse({"csrf_token": csrf_token})
        token_manager.create_csrf_cookie(response, csrf_token)
        return {"response": response, "token": csrf_token}
    
    return {"token": csrf_token}


def create_csrf_dependency():
    """Create CSRF token dependency for routes."""
    async def csrf_dependency(request: Request):
        return get_csrf_token(request)
    return csrf_dependency


# Global CSRF middleware instance
csrf_middleware = CSRFProtectionMiddleware