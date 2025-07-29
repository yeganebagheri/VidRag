from fastapi import APIRouter, Depends, HTTPException, Query
from src.core.search.vector_search import get_enhanced_vector_search, HierarchicalVectorSearch
from src.core.models.query import QueryRequest, QueryResponse, QueryResult, HierarchicalQueryRequest
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/search", response_model=QueryResponse)
async def search_videos(
    query: QueryRequest,
    vector_search: HierarchicalVectorSearch = Depends(get_enhanced_vector_search)
):
    """Enhanced video search using hierarchical vector search"""
    
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    try:
        # Use hierarchical search by default
        search_results = await vector_search.hierarchical_search(
            query=query.text,
            top_k=query.limit,
            search_level="both"  # Search both scenes and segments
        )
        
        # Convert to response format
        results = []
        for result in search_results:
            results.append(QueryResult(
                video_id=result["video_id"],
                segment_id=result.get("segment_id", f"{result['video_id']}_{result.get('scene_id', 'unknown')}"),
                text=result["text"],
                score=result["score"],
                timestamp=result.get("start", 0),
                content_type=result["type"],
                metadata={
                    "scene_id": result.get("scene_id"),
                    "start": result.get("start"),
                    "end": result.get("end"),
                    "duration": result.get("duration"),
                    "modality_scores": result.get("modality_scores", {}),
                    "context_score": result.get("context_score", 0),
                    "temporal_score": result.get("temporal_score", 0)
                }
            ))
        
        return QueryResponse(
            query=query.text,
            results=results,
            total=len(results)
        )
        
    except Exception as e:
        logger.error(f"Enhanced search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/search/hierarchical")
async def hierarchical_search(
    query: HierarchicalQueryRequest,
    vector_search: HierarchicalVectorSearch = Depends(get_enhanced_vector_search)
):
    """Advanced hierarchical search with customizable parameters"""
    
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    try:
        # Use hierarchical search with custom parameters
        search_results = await vector_search.hierarchical_search(
            query=query.text,
            top_k=query.limit,
            search_level=query.search_level
        )
        
        return {
            "query": query.text,
            "search_level": query.search_level,
            "results": search_results,
            "total": len(search_results),
            "metadata": {
                "search_parameters": {
                    "search_level": query.search_level,
                    "limit": query.limit
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Hierarchical search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hierarchical search failed: {str(e)}")

@router.post("/search/multimodal")
async def multimodal_search(
    query: str,
    modality_weights: Optional[Dict[str, float]] = None,
    top_k: int = Query(10, ge=1, le=100),
    vector_search: HierarchicalVectorSearch = Depends(get_enhanced_vector_search)
):
    """Advanced multimodal search with adaptive weighting"""
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    try:
        # Use multimodal search with adaptive weights
        search_results = await vector_search.multimodal_search(
            query=query,
            modality_weights=modality_weights,
            top_k=top_k
        )
        
        return {
            "query": query,
            "results": search_results,
            "total": len(search_results),
            "modality_weights_used": search_results[0].get("query_weights") if search_results else None
        }
        
    except Exception as e:
        logger.error(f"Multimodal search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multimodal search failed: {str(e)}")

@router.get("/videos/{video_id}/segments")
async def get_video_segments(
    video_id: str,
    vector_search: HierarchicalVectorSearch = Depends(get_enhanced_vector_search)
):
    """Get all hierarchical segments for a specific video"""
    
    if video_id not in vector_search.video_embeddings:
        raise HTTPException(status_code=404, detail="Video not found or not processed")
    
    # Get video structure
    video_structure = await vector_search.get_video_structure(video_id)
    
    return video_structure

@router.get("/videos/{video_id}/scenes")
async def get_video_scenes(
    video_id: str,
    vector_search: HierarchicalVectorSearch = Depends(get_enhanced_vector_search)
):
    """Get all scenes for a specific video"""
    
    if video_id not in vector_search.scene_embeddings:
        raise HTTPException(status_code=404, detail="Video not found or scenes not detected")
    
    scenes = []
    for scene_data in vector_search.scene_embeddings[video_id]:
        scenes.append({
            "scene_id": scene_data["scene_id"],
            "start": scene_data["start"],
            "end": scene_data["end"],
            "text": scene_data.get("text", ""),
            "num_segments": scene_data.get("num_segments", 0)
        })
    
    return {
        "video_id": video_id,
        "scenes": scenes,
        "total_scenes": len(scenes)
    }

@router.post("/search/semantic")
async def semantic_search_by_modality(
    query: str,
    modality: str = Query(..., regex="^(text|visual|ocr)$"),
    top_k: int = Query(10, ge=1, le=100),
    vector_search: HierarchicalVectorSearch = Depends(get_enhanced_vector_search)
):
    """Search by specific modality (text, visual, or OCR)"""
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    try:
        # Use single modality search
        search_results = await vector_search._search_by_modality(
            modality=modality,
            query_embedding=vector_search.embedding_model.encode(query).astype('float32'),
            k=top_k
        )
        
        return {
            "query": query,
            "modality": modality,
            "results": search_results,
            "total": len(search_results)
        }
        
    except Exception as e:
        logger.error(f"Semantic search by modality failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

@router.get("/search/suggestions")
async def get_search_suggestions(
    partial_query: str = Query(..., min_length=2),
    limit: int = Query(5, ge=1, le=10),
    vector_search: HierarchicalVectorSearch = Depends(get_enhanced_vector_search)
):
    """Get search suggestions based on indexed content"""
    
    try:
        # Simple implementation - in production, you'd want more sophisticated suggestion logic
        suggestions = []
        
        # Get some sample texts from indexed segments
        for video_id, segments in vector_search.segment_metadata.items():
            for segment in segments[:limit * 2]:  # Check more segments than needed
                text = segment.get("text", "")
                if partial_query.lower() in text.lower():
                    # Extract a meaningful phrase containing the partial query
                    words = text.split()
                    for i, word in enumerate(words):
                        if partial_query.lower() in word.lower():
                            # Get context around the matching word
                            start_idx = max(0, i - 2)
                            end_idx = min(len(words), i + 3)
                            suggestion = " ".join(words[start_idx:end_idx])
                            if suggestion not in suggestions:
                                suggestions.append(suggestion)
                                break
                
                if len(suggestions) >= limit:
                    break
            
            if len(suggestions) >= limit:
                break
        
        return {
            "partial_query": partial_query,
            "suggestions": suggestions[:limit]
        }
        
    except Exception as e:
        logger.error(f"Search suggestions failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search suggestions failed: {str(e)}")

@router.get("/analytics/search")
async def get_search_analytics():
    """Get search analytics and system performance metrics"""
    
    return {
        "message": "Search analytics endpoint - implement based on your analytics requirements",
        "available_metrics": [
            "query_frequency",
            "response_times", 
            "result_relevance",
            "modality_usage",
            "search_patterns"
        ]
    }