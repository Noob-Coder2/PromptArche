"""
Unified exception system for PromptArche.

Provides consistent error handling across the application with:
- Specific exception types for different error categories
- User-friendly error messages
- HTTP status code mapping
- Structured error responses
- Contextual information for debugging

Usage:
    from app.core.exceptions import ValidationError, RateLimitError
    
    try:
        validate_file_size(file)
    except ValidationError as e:
        logger.warning(f"Validation failed: {e.user_message}")
        return {"error": e.user_message, "code": e.error_code}
"""

from typing import Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Standard error codes for API responses."""
    VALIDATION_ERROR = "validation_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND = "not_found"
    DUPLICATE_ERROR = "duplicate_error"
    DATABASE_ERROR = "database_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    INTERNAL_ERROR = "internal_error"


class AppException(Exception):
    """
    Base exception for all application errors.
    
    Provides:
    - user_message: Safe message to show to end users
    - developer_message: Detailed message for developers (in logs)
    - error_code: Machine-readable error code
    - status_code: HTTP status code
    - context: Additional context data for debugging
    """
    
    status_code = 500
    error_code = ErrorCode.INTERNAL_ERROR
    user_message = "An unexpected error occurred. Please try again."
    
    def __init__(
        self,
        user_message: Optional[str] = None,
        developer_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize exception.
        
        Args:
            user_message: Message to show to end users (safe, non-technical)
            developer_message: Detailed message for debugging (technical)
            context: Additional context data (file names, IDs, etc.)
        """
        self.user_message = user_message or self.__class__.user_message
        self.developer_message = developer_message or str(self)
        self.context = context or {}
        
        super().__init__(self.user_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to API response dictionary."""
        return {
            "error": self.user_message,
            "code": self.error_code.value,
            "status": self.status_code,
            "context": self.context if self.context else None
        }
    
    def log(self, logger: logging.Logger = logger):
        """Log exception with context."""
        logger.error(
            f"{self.__class__.__name__}: {self.developer_message}",
            extra={"context": self.context}
        )


class ValidationError(AppException):
    """
    Raised when input validation fails.
    
    Examples:
    - File size exceeds limit
    - Invalid JSON format
    - Missing required fields
    - Incorrect file type
    """
    
    status_code = 400
    error_code = ErrorCode.VALIDATION_ERROR
    user_message = "The provided data is invalid. Please check your input."


class RateLimitError(AppException):
    """
    Raised when rate limit is exceeded.
    
    Examples:
    - Too many requests per minute
    - API quota exceeded
    - Service throttling
    """
    
    status_code = 429
    error_code = ErrorCode.RATE_LIMIT_ERROR
    user_message = "Too many requests. Please wait a moment and try again."
    
    def __init__(
        self,
        user_message: Optional[str] = None,
        developer_message: Optional[str] = None,
        retry_after: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(user_message, developer_message, context)
        self.retry_after = retry_after or 60


class ServiceUnavailableError(AppException):
    """
    Raised when external service is unavailable.
    
    Examples:
    - HuggingFace API is down
    - Groq API is unreachable
    - Database connection failed
    """
    
    status_code = 503
    error_code = ErrorCode.SERVICE_UNAVAILABLE
    user_message = "A required service is currently unavailable. Please try again later."


class AuthenticationError(AppException):
    """
    Raised when authentication fails.
    
    Examples:
    - Invalid API key
    - Token expired
    - Missing credentials
    """
    
    status_code = 401
    error_code = ErrorCode.AUTHENTICATION_ERROR
    user_message = "Authentication failed. Please log in again."


class AuthorizationError(AppException):
    """
    Raised when user lacks permission.
    
    Examples:
    - Accessing another user's data
    - Insufficient permissions
    - Resource not owned by user
    """
    
    status_code = 403
    error_code = ErrorCode.AUTHORIZATION_ERROR
    user_message = "You don't have permission to access this resource."


class NotFoundError(AppException):
    """
    Raised when resource is not found.
    
    Examples:
    - Job ID doesn't exist
    - Cluster not found
    - User not found
    """
    
    status_code = 404
    error_code = ErrorCode.NOT_FOUND
    user_message = "The requested resource was not found."


class DuplicateError(AppException):
    """
    Raised when creating duplicate resource.
    
    Examples:
    - Email already registered
    - Duplicate prompt content
    - Duplicate job ID
    """
    
    status_code = 409
    error_code = ErrorCode.DUPLICATE_ERROR
    user_message = "This resource already exists."


class DatabaseError(AppException):
    """
    Raised when database operation fails.
    
    Examples:
    - Query failed
    - Connection lost
    - Transaction rolled back
    """
    
    status_code = 500
    error_code = ErrorCode.DATABASE_ERROR
    user_message = "A database error occurred. Please try again."


class ExternalServiceError(AppException):
    """
    Raised when external service call fails.
    
    Examples:
    - HuggingFace API error
    - Groq API error
    - HTTP request failed
    """
    
    status_code = 502
    error_code = ErrorCode.EXTERNAL_SERVICE_ERROR
    user_message = "An external service returned an error. Please try again."


class InternalError(AppException):
    """
    Raised for internal application errors.
    
    This is the catch-all for unexpected errors that shouldn't happen.
    """
    
    status_code = 500
    error_code = ErrorCode.INTERNAL_ERROR
    user_message = "An internal error occurred. Our team has been notified."


# Error message templates for common scenarios

ERROR_MESSAGES = {
    "file_too_large": "File size exceeds maximum limit of {max_size}MB",
    "invalid_file_type": "Invalid file type '{file_type}'. Accepted types: {accepted_types}",
    "invalid_json": "Invalid JSON format: {detail}",
    "missing_field": "Required field missing: {field}",
    "rate_limit": "Rate limit exceeded: {limit} requests per {window}",
    "api_down": "{service} is currently unavailable (status {status_code})",
    "invalid_credentials": "Invalid {credential_type}",
    "access_denied": "You don't have permission to {action}",
    "resource_not_found": "{resource_type} '{resource_id}' not found",
    "duplicate_resource": "{resource_type} already exists with {field}='{value}'",
}


def create_validation_error(
    message_key: str,
    **kwargs
) -> ValidationError:
    """
    Create validation error with template message.
    
    Args:
        message_key: Key in ERROR_MESSAGES dict
        **kwargs: Values to format into message
        
    Returns:
        ValidationError with formatted message
        
    Example:
        raise create_validation_error(
            "file_too_large",
            max_size=100
        )
    """
    template = ERROR_MESSAGES.get(message_key, message_key)
    user_message = template.format(**kwargs)
    return ValidationError(user_message=user_message, context=kwargs)


def create_rate_limit_error(
    limit: int,
    window: str,
    retry_after: Optional[int] = None
) -> RateLimitError:
    """
    Create rate limit error.
    
    Args:
        limit: Number of requests allowed
        window: Time window (e.g., "per minute")
        retry_after: Seconds to wait before retry
        
    Returns:
        RateLimitError
    """
    message = ERROR_MESSAGES["rate_limit"].format(limit=limit, window=window)
    return RateLimitError(
        user_message=message,
        retry_after=retry_after,
        context={"limit": limit, "window": window}
    )


def create_service_unavailable_error(
    service_name: str,
    status_code: Optional[int] = None
) -> ServiceUnavailableError:
    """
    Create service unavailable error.
    
    Args:
        service_name: Name of unavailable service (HuggingFace, Groq, etc.)
        status_code: HTTP status code from service
        
    Returns:
        ServiceUnavailableError
    """
    if status_code:
        message = ERROR_MESSAGES["api_down"].format(
            service=service_name,
            status_code=status_code
        )
    else:
        message = f"{service_name} is currently unavailable"
    
    return ServiceUnavailableError(
        user_message=message,
        context={"service": service_name, "status_code": status_code}
    )


def create_not_found_error(
    resource_type: str,
    resource_id: str
) -> NotFoundError:
    """
    Create not found error.
    
    Args:
        resource_type: Type of resource (Job, Cluster, etc.)
        resource_id: ID of missing resource
        
    Returns:
        NotFoundError
    """
    message = ERROR_MESSAGES["resource_not_found"].format(
        resource_type=resource_type,
        resource_id=resource_id
    )
    return NotFoundError(
        user_message=message,
        context={"resource_type": resource_type, "resource_id": resource_id}
    )
