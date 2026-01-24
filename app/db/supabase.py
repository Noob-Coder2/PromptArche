
from supabase import create_client, Client
from app.core.config import settings

def get_supabase() -> Client:
    url: str = settings.SUPABASE_URL
    key: str = settings.SUPABASE_KEY
    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in .env")
    return create_client(url, key)
