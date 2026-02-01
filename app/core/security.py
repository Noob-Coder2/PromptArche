
"""
Security module for JWT authentication and authorization.
Uses OAuth2PasswordBearer for standardized auth handling.
"""
import jwt
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status, Request
from fastapi.security import OAuth2PasswordBearer
from app.core.config import settings

# OAuth2 scheme - auto_error=False allows fallback to cookie auth
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme)
) -> Dict[str, Any]:
    """
    Validate and decode JWT token from either:
    1. Authorization header (Bearer token)
    2. HTTP-only cookie (access_token)
    
    Args:
        request: FastAPI request object
        token: Token from OAuth2PasswordBearer (header)
        
    Returns:
        Decoded JWT payload dict
        
    Raises:
        HTTPException: If no valid token found or token is invalid/expired
    """
    # Priority: Header token > Cookie token
    auth_token = token or request.cookies.get("access_token")
    
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(
            auth_token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["ES256"],
            options={"verify_aud": False}  # Supabase doesn't always set aud
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_id(
    payload: Dict[str, Any] = Depends(get_current_user)
) -> str:
    """
    Extract the user ID (sub claim) from a valid JWT payload.
    
    Args:
        payload: Decoded JWT payload from get_current_user
        
    Returns:
        User ID string (UUID)
        
    Raises:
        HTTPException: If user ID not found in token
    """
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user ID (sub claim)",
        )
    return user_id


async def get_optional_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[Dict[str, Any]]:
    """
    Optionally get current user - returns None if not authenticated.
    Useful for routes that work differently for authenticated vs anonymous users.
    
    Args:
        request: FastAPI request object
        token: Token from OAuth2PasswordBearer (header)
        
    Returns:
        Decoded JWT payload or None
    """
    auth_token = token or request.cookies.get("access_token")
    
    if not auth_token:
        return None
    
    try:
        payload = jwt.decode(
            auth_token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["ES256"],
            options={"verify_aud": False}
        )
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# Legacy compatibility - deprecated, use get_current_user instead
def verify_jwt(request: Request = None, credentials = None):
    """
    DEPRECATED: Use get_current_user dependency instead.
    Kept for backward compatibility.
    """
    import warnings
    warnings.warn(
        "verify_jwt is deprecated, use get_current_user instead",
        DeprecationWarning,
        stacklevel=2
    )
    
    token = None
    if credentials:
        token = credentials.credentials
    elif request:
        token = request.cookies.get("access_token")
        
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
        )

    try:
        payload = jwt.decode(
            token, 
            settings.SUPABASE_JWT_SECRET, 
            algorithms=["ES256"], 
            options={"verify_aud": False}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

