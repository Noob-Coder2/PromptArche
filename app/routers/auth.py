"""
Authentication router for handling login/logout API endpoints.
"""

from fastapi import APIRouter, Depends, Response, Request
from fastapi.responses import JSONResponse
from app.core.security import get_current_user_id
from app.core.config import settings
from app.core.rate_limiter import rate_limit
from app.core.csrf import get_csrf_token

router = APIRouter()


@router.post("/api/login")
async def api_login(response: Response, access_token: str, request: Request, rate_limit_dep: bool = Depends(rate_limit)):
    """
    Set HTTP-only cookie for authentication.
    Token is validated by Supabase client-side before this call.
    """
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite=settings.validated_samesite,  # type: ignore
        secure=settings.COOKIE_SECURE,  # Always secure in production
        max_age=3600 * 24  # 1 day
    )
    return {"status": "success"}


@router.post("/api/logout")
async def api_logout(response: Response):
    """Clear the authentication cookie."""
    response.delete_cookie(key="access_token")
    return {"status": "success"}


@router.get("/api/me")
async def get_current_user_info(user_id: str = Depends(get_current_user_id)):
    """Get current user information."""
    return {"user_id": user_id}


@router.get("/api/csrf-token")
async def get_csrf_token_endpoint(request: Request):
    """Get CSRF token for form submissions."""
    return get_csrf_token(request)