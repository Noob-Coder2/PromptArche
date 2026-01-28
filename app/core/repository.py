"""
Repository abstraction layer for data access.

Provides:
- Abstract interfaces for data operations
- Concrete Supabase implementation
- Dependency injection for testing
- Decouples business logic from database

Usage:
    from app.core.repository import IRepository, get_repository
    
    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str, repo = Depends(get_repository)):
        return await repo.get_job(job_id)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Data class for ingestion job."""
    id: str
    user_id: str
    filename: str
    status: str
    progress: int
    total_records: int
    parsed_records: int
    failed_records: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    temp_file_path: Optional[str] = None


@dataclass
class Cluster:
    """Data class for document cluster."""
    id: str
    job_id: str
    cluster_id: int
    size: int
    created_at: datetime


@dataclass
class EmbeddingCache:
    """Data class for cached embeddings."""
    text_hash: str
    embedding: List[float]
    created_at: datetime


class IRepository(ABC):
    """
    Abstract repository interface.
    
    Defines all data access operations needed by the application.
    Implementations can use Supabase, PostgreSQL, MongoDB, etc.
    """
    
    # ============ Jobs ============
    
    @abstractmethod
    async def create_job(
        self,
        user_id: str,
        filename: str,
        total_records: int
    ) -> Job:
        """Create a new ingestion job."""
        pass
    
    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        pass
    
    @abstractmethod
    async def get_user_jobs(self, user_id: str, limit: int = 50) -> List[Job]:
        """Get all jobs for a user."""
        pass
    
    @abstractmethod
    async def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: Optional[int] = None,
        parsed_records: Optional[int] = None,
        error_message: Optional[str] = None,
        temp_file_path: Optional[str] = None
    ) -> Job:
        """Update job status and progress."""
        pass
    
    @abstractmethod
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        pass
    
    @abstractmethod
    async def get_pending_jobs(self) -> List[Job]:
        """Get all pending/parsing jobs for recovery."""
        pass
    
    # ============ Clusters ============
    
    @abstractmethod
    async def save_clusters(
        self,
        job_id: str,
        clusters: List[Dict[str, Any]]
    ) -> List[Cluster]:
        """Save clusters for a job."""
        pass
    
    @abstractmethod
    async def get_job_clusters(self, job_id: str) -> List[Cluster]:
        """Get all clusters for a job."""
        pass
    
    @abstractmethod
    async def delete_job_clusters(self, job_id: str) -> int:
        """Delete all clusters for a job."""
        pass
    
    # ============ Embeddings ============
    
    @abstractmethod
    async def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embedding by text hash."""
        pass
    
    @abstractmethod
    async def cache_embeddings(
        self,
        embeddings: Dict[str, List[float]]
    ) -> int:
        """Cache embeddings (text_hash -> embedding)."""
        pass
    
    @abstractmethod
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        pass
    
    # ============ Generic ============
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if repository is healthy."""
        pass


class SupabaseRepository(IRepository):
    """
    Supabase implementation of repository pattern.
    
    Handles all data operations using Supabase PostgreSQL.
    """
    
    def __init__(self, supabase_client):
        """Initialize with Supabase client."""
        self.supabase = supabase_client
    
    # ============ Jobs ============
    
    async def create_job(
        self,
        user_id: str,
        filename: str,
        total_records: int
    ) -> Job:
        """Create a new ingestion job."""
        try:
            response = self.supabase.table("ingestion_jobs").insert({
                "user_id": user_id,
                "filename": filename,
                "status": "pending",
                "progress": 0,
                "total_records": total_records,
                "parsed_records": 0,
                "failed_records": 0,
                "error_message": None,
                "temp_file_path": None,
            }).execute()
            
            job_data = response.data[0]
            return self._row_to_job(job_data)
        
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            raise
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        try:
            response = self.supabase.table("ingestion_jobs").select("*").eq(
                "id", job_id
            ).execute()
            
            if not response.data:
                return None
            
            return self._row_to_job(response.data[0])
        
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    async def get_user_jobs(self, user_id: str, limit: int = 50) -> List[Job]:
        """Get all jobs for a user."""
        try:
            response = self.supabase.table("ingestion_jobs").select("*").eq(
                "user_id", user_id
            ).order("created_at", desc=True).limit(limit).execute()
            
            return [self._row_to_job(row) for row in response.data]
        
        except Exception as e:
            logger.error(f"Failed to get jobs for user {user_id}: {e}")
            return []
    
    async def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: Optional[int] = None,
        parsed_records: Optional[int] = None,
        error_message: Optional[str] = None,
        temp_file_path: Optional[str] = None
    ) -> Job:
        """Update job status and progress."""
        try:
            update_data: Dict[str, Any] = {"status": status}
            
            if progress is not None:
                update_data["progress"] = progress
            
            if parsed_records is not None:
                update_data["parsed_records"] = parsed_records
            
            if error_message is not None:
                update_data["error_message"] = error_message
            
            if temp_file_path is not None:
                update_data["temp_file_path"] = temp_file_path
            
            response = self.supabase.table("ingestion_jobs").update(
                update_data
            ).eq("id", job_id).execute()
            
            return self._row_to_job(response.data[0])
        
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            raise
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        try:
            self.supabase.table("ingestion_jobs").delete().eq("id", job_id).execute()
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False
    
    async def get_pending_jobs(self) -> List[Job]:
        """Get all pending/parsing jobs for recovery."""
        try:
            response = self.supabase.table("ingestion_jobs").select("*").in_(
                "status", ["pending", "parsing"]
            ).execute()
            
            return [self._row_to_job(row) for row in response.data]
        
        except Exception as e:
            logger.error(f"Failed to get pending jobs: {e}")
            return []
    
    # ============ Clusters ============
    
    async def save_clusters(
        self,
        job_id: str,
        clusters: List[Dict[str, Any]]
    ) -> List[Cluster]:
        """Save clusters for a job."""
        try:
            cluster_rows = []
            for cluster_id, size in clusters:
                cluster_rows.append({
                    "job_id": job_id,
                    "cluster_id": cluster_id,
                    "size": size
                })
            
            response = self.supabase.table("document_clusters").insert(
                cluster_rows
            ).execute()
            
            return [self._row_to_cluster(row) for row in response.data]
        
        except Exception as e:
            logger.error(f"Failed to save clusters for job {job_id}: {e}")
            raise
    
    async def get_job_clusters(self, job_id: str) -> List[Cluster]:
        """Get all clusters for a job."""
        try:
            response = self.supabase.table("document_clusters").select("*").eq(
                "job_id", job_id
            ).execute()
            
            return [self._row_to_cluster(row) for row in response.data]
        
        except Exception as e:
            logger.error(f"Failed to get clusters for job {job_id}: {e}")
            return []
    
    async def delete_job_clusters(self, job_id: str) -> int:
        """Delete all clusters for a job."""
        try:
            response = self.supabase.table("document_clusters").delete().eq(
                "job_id", job_id
            ).execute()
            
            return len(response.data) if response.data else 0
        
        except Exception as e:
            logger.error(f"Failed to delete clusters for job {job_id}: {e}")
            return 0
    
    # ============ Embeddings ============
    
    async def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embedding by text hash."""
        try:
            response = self.supabase.table("embedding_cache").select(
                "embedding"
            ).eq("text_hash", text_hash).execute()
            
            if not response.data:
                return None
            
            return response.data[0]["embedding"]
        
        except Exception as e:
            logger.error(f"Failed to get cached embedding: {e}")
            return None
    
    async def cache_embeddings(
        self,
        embeddings: Dict[str, List[float]]
    ) -> int:
        """Cache embeddings (text_hash -> embedding)."""
        try:
            rows = [
                {
                    "text_hash": text_hash,
                    "embedding": embedding
                }
                for text_hash, embedding in embeddings.items()
            ]
            
            response = self.supabase.table("embedding_cache").insert(
                rows,
                count="exact"
            ).execute()
            
            return len(response.data) if response.data else 0
        
        except Exception as e:
            logger.error(f"Failed to cache embeddings: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        try:
            response = self.supabase.table("embedding_cache").select(
                "count",
                count="exact"
            ).execute()
            
            return {
                "cached_count": response.count or 0,
                "status": "healthy"
            }
        
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                "cached_count": 0,
                "status": "error",
                "error": str(e)
            }
    
    # ============ Generic ============
    
    async def health_check(self) -> bool:
        """Check if repository is healthy."""
        try:
            response = self.supabase.table("users").select("id").limit(1).execute()
            return True
        
        except Exception as e:
            logger.error(f"Repository health check failed: {e}")
            return False
    
    # ============ Helpers ============
    
    def _row_to_job(self, row: Dict[str, Any]) -> Job:
        """Convert database row to Job dataclass."""
        return Job(
            id=row["id"],
            user_id=row["user_id"],
            filename=row["filename"],
            status=row["status"],
            progress=row.get("progress", 0),
            total_records=row.get("total_records", 0),
            parsed_records=row.get("parsed_records", 0),
            failed_records=row.get("failed_records", 0),
            error_message=row.get("error_message"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            temp_file_path=row.get("temp_file_path")
        )
    
    def _row_to_cluster(self, row: Dict[str, Any]) -> Cluster:
        """Convert database row to Cluster dataclass."""
        return Cluster(
            id=row["id"],
            job_id=row["job_id"],
            cluster_id=row["cluster_id"],
            size=row["size"],
            created_at=datetime.fromisoformat(row["created_at"])
        )


# Dependency injection for repository
def get_repository() -> IRepository:
    """
    Get repository instance for dependency injection.
    
    Usage:
        from app.core.repository import get_repository
        
        @app.get("/jobs")
        async def list_jobs(repo = Depends(get_repository)):
            return await repo.get_user_jobs(user_id)
    
    For testing, replace this with a mock implementation.
    """
    from app.db.supabase import get_supabase
    supabase = get_supabase()
    return SupabaseRepository(supabase)
