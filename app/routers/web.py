"""
Legacy router - DEPRECATED
This file has been split into separate routers for better organization:
- auth.py: Authentication endpoints
- ingestion.py: File upload and job management  
- dashboard.py: Dashboard statistics and data
- pages.py: HTML page rendering

All functionality has been migrated to the respective routers.
"""

from fastapi import APIRouter

router = APIRouter()

# This router is kept for backward compatibility but all endpoints have been moved
# to their respective routers in the same directory.
