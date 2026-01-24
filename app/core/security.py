
import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

security = HTTPBearer()

def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Verifies the Supabase JWT token.
    """
    token = credentials.credentials
    try:
        # Supabase uses HS256 and the JWT secret to sign tokens.
        # We verify the signature and expiration.
        payload = jwt.decode(
            token, 
            settings.SUPABASE_JWT_SECRET, 
            algorithms=["HS256"], 
            options={"verify_aud": False} # Change if you want to strictly check audience
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

def get_current_user_id(payload: dict = Security(verify_jwt)) -> str:
    """
    Extracts the user ID (sub) from the valid JWT payload.
    """
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
             status_code=status.HTTP_401_UNAUTHORIZED,
             detail="Token missing user ID",
        )
    return user_id
