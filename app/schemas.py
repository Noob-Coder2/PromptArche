"""
Pydantic schemas for request/response validation.
Ensures API responses conform to expected structure.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# --- Insight Schemas ---

class InsightResponse(BaseModel):
    """Schema for LLM-generated cluster insights."""
    title: str = Field(
        max_length=50,
        default="Cluster Insight",
        description="Short, punchy title for the insight (max 5 words)"
    )
    insight: str = Field(
        max_length=500,
        default="No insight generated.",
        description="The brutally honest insight text (max 3 sentences)"
    )


# --- Job Schemas ---

class JobStatusResponse(BaseModel):
    """Schema for ingestion job status."""
    id: str
    status: str = Field(
        description="Job status: PENDING, PARSING, EMBEDDING, COMPLETED, or FAILED"
    )
    current_count: int = Field(default=0, ge=0)
    total_count: int = Field(default=0, ge=0)
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Schema for list of jobs."""
    jobs: List[JobStatusResponse]


class ActiveJobResponse(BaseModel):
    """Schema for active job check."""
    active: bool
    job: Optional[JobStatusResponse] = None


# --- Cluster Schemas ---

class ClusterInfo(BaseModel):
    """Schema for cluster information."""
    id: str
    label: str
    description: Optional[str] = None
    prompt_count: int = Field(default=0, ge=0)
    
    class Config:
        from_attributes = True


class ClusteringResult(BaseModel):
    """Schema for clustering operation result."""
    status: str
    clusters_found: int = Field(default=0, ge=0)
    clusters_created: int = Field(default=0, ge=0)
    clusters_updated: int = Field(default=0, ge=0)
    prompts_assigned: int = Field(default=0, ge=0)
    noise_points: int = Field(default=0, ge=0)
    orphaned_clusters: int = Field(default=0, ge=0)
    message: Optional[str] = None


# --- Ingestion Schemas ---

class IngestResponse(BaseModel):
    """Schema for ingestion start response."""
    status: str
    message: str
    job_id: Optional[str] = None


class IngestionResult(BaseModel):
    """Schema for ingestion completion result."""
    status: str
    processed: int = Field(default=0, ge=0)
    message: Optional[str] = None


# --- Dashboard Schemas ---

class DashboardStats(BaseModel):
    """Schema for dashboard statistics."""
    total_prompts: int = Field(default=0, ge=0)
    total_clusters: int = Field(default=0, ge=0)
    total_insights: int = Field(default=0, ge=0)


class ChartData(BaseModel):
    """Schema for timeline chart data."""
    labels: List[str]
    values: List[int]


# --- Auth Schemas ---

class LoginRequest(BaseModel):
    """Schema for login request."""
    access_token: str


class AuthResponse(BaseModel):
    """Schema for auth operation response."""
    status: str
    message: Optional[str] = None
