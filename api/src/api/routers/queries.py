from fastapi import APIRouter, HTTPException
from src.core.models.query import QueryRequest, QueryResponse, QueryResult
from typing import List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/search", response_model=QueryResponse)
async def search_videos(query: QueryRequest):
    """Placeholder - search handled by ML worker after processing"""
    
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    return QueryResponse(
        query=query.text,
        results=[],
        total=0,
        metadata={
            "message": "Search will be available after video processing is complete",
            "note": "ML worker handles indexing and search"
        }
    )

@router.get("/status")
async def search_status():
    """Check if search is available"""
    return {
        "search_available": False,
        "message": "Search functionality requires ML worker to process videos first"
    }