"""
Global error handling middleware for PromptArche.

Provides:
- Consistent error response formatting
- Exception logging with context
- Status code mapping
- User-friendly vs developer messages
- Error recovery recommendations

Usage:
    from app.core.error_handler import setup_error_handlers
    
    app = FastAPI()
    setup_error_handlers(app)
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
from typing import Dict, Any, Optional
import logging
import traceback
from datetime import datetime

from app.core.exceptions import (
    AppException,
    RateLimitError,
    ValidationError,
    InternalError,
    ErrorCode
)

logger = logging.getLogger(__name__)


class ErrorResponse:
    """Format error responses consistently."""
    
    @staticmethod
    def format(
        error: str,
        code: str,
        status_code: int,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format error response.
        
        Args:
            error: User-friendly error message
            code: Machine-readable error code
            status_code: HTTP status code
            context: Additional context data
            timestamp: Request timestamp
            request_id: Request ID for tracking
            
        Returns:
            Formatted error response dict
        """
        response = {
            "success": False,
            "error": error,
            "code": code,
            "status": status_code,
        }
        
        if context:
            response["context"] = context
        
        if timestamp:
            response["timestamp"] = timestamp
        
        if request_id:
            response["request_id"] = request_id
        
        return response


async def app_exception_handler(request: Request, exc: AppException):
    """
    Handle AppException instances.
    
    Converts app exceptions to proper HTTP responses with logging.
    """
    exc.log(logger)
    
    response = ErrorResponse.format(
        error=exc.user_message,
        code=exc.error_code.value,
        status_code=exc.status_code,
        context=exc.context or None,
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Add retry-after header for rate limit errors
    headers = {}
    if isinstance(exc, RateLimitError) and exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response,
        headers=headers
    )


async def validation_error_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors.
    
    Converts validation errors to user-friendly messages.
    """
    errors = []
    for error in exc.errors():
        loc = " -> ".join(str(x) for x in error["loc"])
        msg = error.get("msg", "Invalid value")
        errors.append(f"{loc}: {msg}")
    
    detail = "; ".join(errors[:3])  # Limit to first 3 errors
    if len(errors) > 3:
        detail += f" (and {len(errors) - 3} more)"
    
    logger.warning(f"Validation error: {detail}")
    
    response = ErrorResponse.format(
        error="Invalid request data",
        code=ErrorCode.VALIDATION_ERROR.value,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        context={
            "validation_errors": errors[:10],  # Include first 10 in context
            "error_count": len(errors)
        },
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response
    )


async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions.
    
    Logs full traceback and returns generic error to user.
    """
    logger.error(
        f"Unhandled exception: {exc.__class__.__name__}",
        exc_info=True
    )
    
    error_context: Optional[Dict[str, Any]] = None
    if str(exc):
        error_context = {"error_type": exc.__class__.__name__}
    
    response = ErrorResponse.format(
        error="An unexpected error occurred. Our team has been notified.",
        code=ErrorCode.INTERNAL_ERROR.value,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        context=error_context,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response
    )


def setup_error_handlers(app: FastAPI) -> None:
    """
    Register all error handlers with FastAPI app.
    
    Usage:
        app = FastAPI()
        setup_error_handlers(app)
    
    Handles:
    - AppException: Custom application exceptions
    - RequestValidationError: Pydantic validation errors
    - Exception: All other unexpected exceptions
    """
    # Define wrapped handlers for sync conversion
    async def app_exc_handler(request: Request, exc: AppException) -> JSONResponse:
        return await app_exception_handler(request, exc)
    
    async def validation_exc_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return await validation_error_handler(request, exc)
    
    async def general_exc_handler(request: Request, exc: Exception) -> JSONResponse:
        return await general_exception_handler(request, exc)
    
    # Register handlers
    app.add_exception_handler(AppException, app_exc_handler)  # type: ignore
    app.add_exception_handler(RequestValidationError, validation_exc_handler)  # type: ignore
    app.add_exception_handler(Exception, general_exc_handler)  # type: ignore
