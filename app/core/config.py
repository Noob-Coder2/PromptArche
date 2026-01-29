
import os
import logging
from dotenv import load_dotenv

load_dotenv()

class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    SUPABASE_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # BGE-Large-EN-v1.5 dimension is 1024
    EMBEDDING_DIM: int = 1024
    
    # Security settings
    COOKIE_SECURE: bool = os.getenv("COOKIE_SECURE", "true").lower() == "true"  # Default to secure
    COOKIE_SAMESITE: str = os.getenv("COOKIE_SAMESITE", "strict")  # strict, lax, or none
    FORCE_HTTPS: bool = os.getenv("FORCE_HTTPS", "true").lower() == "true"  # Force HTTPS in production
    
    # Validate SameSite value
    @property
    def validated_samesite(self) -> str:
        """Validate and return the SameSite attribute value."""
        valid_values = {"strict", "lax", "none"}
        value = self.COOKIE_SAMESITE.lower()
        if value not in valid_values:
            logging.warning(f"Invalid COOKIE_SAMESITE value: {self.COOKIE_SAMESITE}. Using 'strict'.")
            return "strict"
        return value
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "104857600"))  # 100MB default
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # requests per minute
    
    # Validate required secrets on startup
    @classmethod
    def validate_secrets(cls) -> None:
        """Validate that all required secrets are set."""
        missing_secrets = []
        if not cls.SUPABASE_URL:
            missing_secrets.append("SUPABASE_URL")
        if not cls.SUPABASE_KEY:
            missing_secrets.append("SUPABASE_KEY")
        if not cls.SUPABASE_JWT_SECRET:
            missing_secrets.append("SUPABASE_JWT_SECRET")
        if not cls.HF_TOKEN:
            missing_secrets.append("HF_TOKEN")
        if not cls.GROQ_API_KEY:
            missing_secrets.append("GROQ_API_KEY")
            
        if missing_secrets:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_secrets)}")
    
    # Ingestion settings for large file handling
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1048576"))  # 1MB chunks for streaming
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))  # Items per database batch
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "5"))  # Retry attempts for failed operations
    
    # Database connection pooling
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "20"))  # Max concurrent connections
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))  # Connection timeout in seconds

settings = Settings()
