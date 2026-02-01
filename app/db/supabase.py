
"""
Database connection management with connection pooling for handling concurrent requests.
"""

from supabase import create_client, Client
from app.core.config import settings

# Singleton connection instance with pooling
_supabase_client: Client = None # pyright: ignore[reportAssignmentType]


def get_supabase() -> Client:
    """
    Get or create the Supabase client with connection pooling.
    Uses SERVICE ROLE KEY for backend operations to bypass RLS policies.
    
    This is safe because:
    - Backend code is trusted (not exposed to users)
    - User authentication is handled via JWT validation
    - user_id is explicitly set from validated JWT tokens
    
    Returns:
        Supabase client instance with service role privileges
        
    Raises:
        ValueError: If Supabase credentials are not configured
    """
    global _supabase_client
    
    if _supabase_client is None:
        url: str = settings.SUPABASE_URL
        # Use SERVICE ROLE KEY for backend operations (bypasses RLS)
        key: str = settings.SUPABASE_SERVICE_KEY or settings.SUPABASE_KEY
        
        if not url or not key:
            raise ValueError("Supabase URL and Service Key must be set in .env")
        
        # Create client with pooling configuration
        # Note: Supabase-py uses connection pooling automatically via httpx
        _supabase_client = create_client(url, key)
    
    return _supabase_client


def close_supabase_connection():
    """
    Explicitly close the Supabase connection.
    Call this during application shutdown.
    """
    global _supabase_client
    if _supabase_client is not None:
        try:
            # _supabase_client.postgrest.auth.session = None # Caused error, not needed
            _supabase_client = None
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Error closing Supabase connection: {e}")

