# src/api/routers/enhanced_queries.py

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from src.core.search.internvl_enhanced_vector_search import get_internvl_enhanced_vector_search, InternVLEnhancedVectorSearch
from src.core.models.query import QueryRequest, QueryResponse, QueryResult, HierarchicalQueryRequest
from src.core.models.internvl_encoder import get_internvl_encoder
from typing import List, Optional, Dict, Any
import logging
import time
import uuid

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/search", response_model=QueryResponse)
async def enhanced_search_videos(
    query: QueryRequest,
    use_internvl: bool = Query(True, description="Use InternVL unified encoding for search"),
    use_knowledge_enhancement: bool = Query(True, description="Enable knowledge graph enhancement"),
    vector_search: InternVLEnhancedVectorSearch = Depends(get_internvl_enhanced_vector_search)
):
    """
    Enhanced video search using InternVL 2.0 + Video-RAG
    
    Key Improvements:
    1. InternVL unified multimodal understanding
    2. Knowledge graph enhanced retrieval  
    3. Temporal context processing
    4. Superior cross-modal query handling
    """
    
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    query_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Enhanced search query: '{query.text}' (ID: {query_id})")
        
        # Use enhanced search with InternVL + Video-RAG
        search_results = await vector_search.enhanced_search(
            query=query.text,
            top_k=query.limit,
            search_level="both",
            use_knowledge_enhancement=use_knowledge_enhancement
        )
        
        # Convert to response format with enhanced metadata
        results = []
        for result in search_results:
            query_result = QueryResult(
                video_id=result["video_id"],
                segment_id=result.get("segment_id", f"{result['video_id']}_{result.get('scene_id', 'unknown')}"),
                text=result["text"],
                score=result.get("enhanced_score", result["score"]),
                timestamp=result.get("start", 0),
                content_type=result["type"],
                metadata={
                    "scene_id": result.get("scene_id"),
                    "start": result.get("start"),
                    "end": result.get("end"),
                    "duration": result.get("duration"),
                    # Enhanced metadata
                    "search_method": result.get("search_method", "unknown"),
                    "internvl_enhanced": result.get("internvl_enhanced", False),
                    "knowledge_score": result.get("knowledge_score", 0.0),
                    "temporal_score": result.get("temporal_score", 0.0),
                    "base_confidence": result.get("base_confidence", 1.0),
                    "temporal_bonus": result.get("temporal_bonus", 0.0)
                }
            )
            results.append(query_result)
        
        execution_time = time.time() - start_time
        
        # Enhanced response with processing metadata
        response = QueryResponse(
            query=query.text,
            results=results,
            total=len(results),
            execution_time=execution_time,
            metadata={
                "query_id": query_id,
                "internvl_used": use_internvl and vector_search.use_internvl,
                "knowledge_enhancement_used": use_knowledge_enhancement,
                "processing_method": "internvl_video_rag" if vector_search.use_internvl else "traditional",
                "enhanced_features": {
                    "unified_encoding": vector_search.use_internvl,
                    "knowledge_graphs": len(vector_search.knowledge_graphs),
                    "temporal_context": True,
                    "cross_modal_understanding": vector_search.use_internvl
                }
            }
        )
        
        logger.info(f"✅ Enhanced search completed in {execution_time:.3f}s: {len(results)} results")
        return response
        
    except Exception as e:
        logger.error(f"❌ Enhanced search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced search failed: {str(e)}")

@router.post("/search/internvl")
async def internvl_unified_search(
    query: str,
    top_k: int = Query(10, ge=1, le=100),
    search_level: str = Query("both", regex="^(scenes|segments|both)$"),
    vector_search: InternVLEnhancedVectorSearch = Depends(get_internvl_enhanced_vector_search)
):
    """
    Direct InternVL unified search (bypasses traditional methods)
    
    This endpoint specifically uses InternVL's unified encoding for
    superior cross-modal understanding and retrieval accuracy.
    """
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    if not vector_search.use_internvl or not vector_search.internvl_encoder:
        raise HTTPException(
            status_code=503, 
            detail="InternVL encoder not available. Check system status."
        )
    
    try:
        start_time = time.time()
        
        # Force InternVL-only search
        results = await vector_search.enhanced_search(
            query=query,
            top_k=top_k,
            search_level=search_level,
            use_knowledge_enhancement=True
        )
        
        # Filter to only InternVL-enhanced results
        internvl_results = [
            result for result in results 
            if result.get("search_method") == "internvl_unified"
        ]
        
        execution_time = time.time() - start_time
        
        return {
            "query": query,
            "search_method": "internvl_unified",
            "results": internvl_results,
            "total": len(internvl_results),
            "execution_time": execution_time,
            "internvl_stats": {
                "encoder_model": vector_search.internvl_encoder.model_name,
                "device": vector_search.internvl_encoder.device,
                "confidence_boost": 1.2,  # InternVL results get 20% boost
                "unified_embeddings": True
            }
        }
        
    except Exception as e:
        logger.error(f"❌ InternVL unified search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"InternVL search failed: {str(e)}")

@router.post("/search/knowledge-enhanced")
async def knowledge_enhanced_search(
    query: str,
    top_k: int = Query(10, ge=1, le=100),
    knowledge_weight: float = Query(0.25, ge=0.0, le=1.0, description="Weight for knowledge graph scores"),
    temporal_weight: float = Query(0.15, ge=0.0, le=1.0, description="Weight for temporal context scores"),
    vector_search: InternVLEnhancedVectorSearch = Depends(get_internvl_enhanced_vector_search)
):
    """
    Video-RAG enhanced search with configurable knowledge weighting
    
    This endpoint showcases the Video-RAG enhancement with external
    knowledge integration and temporal context understanding.
    """
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    try:
        start_time = time.time()
        
        # Get initial results
        initial_results = await vector_search._multi_modal_retrieval(
            await vector_search._encode_query_enhanced(query),
            top_k * 2,
            "both"
        )
        
        # Apply custom knowledge enhancement
        enhanced_results = []
        for result in initial_results:
            video_id = result["video_id"]
            knowledge_graph = vector_search.knowledge_graphs.get(video_id, {})
            
            # Custom knowledge scoring
            knowledge_score = vector_search._calculate_knowledge_relevance(
                query, knowledge_graph, result
            )
            
            # Custom temporal scoring
            temporal_score = vector_search._calculate_temporal_context_score(
                result, initial_results
            )
            
            # Custom weighted scoring
            original_score = result["score"]
            enhanced_score = (
                original_score * (1.0 - knowledge_weight - temporal_weight) +
                knowledge_score * knowledge_weight +
                temporal_score * temporal_weight
            )
            
            result["enhanced_score"] = enhanced_score
            result["knowledge_score"] = knowledge_score
            result["temporal_score"] = temporal_score
            result["scoring_weights"] = {
                "original": 1.0 - knowledge_weight - temporal_weight,
                "knowledge": knowledge_weight,
                "temporal": temporal_weight
            }
            
            enhanced_results.append(result)
        
        # Sort and limit
        enhanced_results.sort(key=lambda x: x["enhanced_score"], reverse=True)
        final_results = enhanced_results[:top_k]
        
        execution_time = time.time() - start_time
        
        return {
            "query": query,
            "search_method": "video_rag_enhanced",
            "results": final_results,
            "total": len(final_results),
            "execution_time": execution_time,
            "enhancement_config": {
                "knowledge_weight": knowledge_weight,
                "temporal_weight": temporal_weight,
                "knowledge_graphs_used": len(vector_search.knowledge_graphs),
                "video_rag_features": [
                    "knowledge_graph_integration",
                    "temporal_context_analysis",
                    "configurable_scoring_weights"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Knowledge-enhanced search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Knowledge-enhanced search failed: {str(e)}")

@router.post("/search/cross-modal")
async def cross_modal_search(
    text_query: Optional[str] = None,
    image_description: Optional[str] = None,
    top_k: int = Query(10, ge=1, le=100),
    vector_search: InternVLEnhancedVectorSearch = Depends(get_internvl_enhanced_vector_search)
):
    """
    Cross-modal search showcasing InternVL's unified understanding
    
    This endpoint demonstrates InternVL's ability to understand
    queries that combine text and visual concepts seamlessly.
    """
    
    if not text_query and not image_description:
        raise HTTPException(
            status_code=400, 
            detail="Either text_query or image_description must be provided"
        )
    
    if not vector_search.use_internvl:
        raise HTTPException(
            status_code=503,
            detail="Cross-modal search requires InternVL encoder"
        )
    
    try:
        start_time = time.time()
        
        # Construct cross-modal query
        combined_query = []
        if text_query:
            combined_query.append(f"Text: {text_query}")
        if image_description:
            combined_query.append(f"Visual: {image_description}")
        
        full_query = " | ".join(combined_query)
        
        # Use InternVL's cross-modal understanding
        results = await vector_search.enhanced_search(
            query=full_query,
            top_k=top_k,
            search_level="both",
            use_knowledge_enhancement=True
        )
        
        # Filter to prioritize cross-modal relevant results
        cross_modal_results = []
        for result in results:
            # Boost results that have both text and visual content
            has_text = bool(result.get("text", "").strip())
            has_visual = result.get("metadata", {}).get("has_visual", False)
            
            cross_modal_score = result.get("enhanced_score", result["score"])
            if has_text and has_visual:
                cross_modal_score *= 1.3  # 30% boost for multimodal content
                
            result["cross_modal_score"] = cross_modal_score
            result["content_modalities"] = {
                "has_text": has_text,
                "has_visual": has_visual,
                "multimodal": has_text and has_visual
            }
            
            cross_modal_results.append(result)
        
        # Re-sort by cross-modal score
        cross_modal_results.sort(key=lambda x: x["cross_modal_score"], reverse=True)
        
        execution_time = time.time() - start_time
        
        return {
            "text_query": text_query,
            "image_description": image_description,
            "combined_query": full_query,
            "search_method": "internvl_cross_modal",
            "results": cross_modal_results[:top_k],
            "total": len(cross_modal_results[:top_k]),
            "execution_time": execution_time,
            "cross_modal_analysis": {
                "multimodal_content_ratio": sum(
                    1 for r in cross_modal_results[:top_k] 
                    if r["content_modalities"]["multimodal"]
                ) / max(1, len(cross_modal_results[:top_k])),
                "internvl_advantage": "Unified understanding of text + visual concepts",
                "enhancement_applied": "30% boost for multimodal content"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Cross-modal search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cross-modal search failed: {str(e)}")

# Keep existing endpoints for backward compatibility
@router.post("/search/hierarchical")
async def hierarchical_search(
    query: HierarchicalQueryRequest,
    vector_search: InternVLEnhancedVectorSearch = Depends(get_internvl_enhanced_vector_search)
):
    """Enhanced hierarchical search (backward compatible)"""
    
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    try:
        # Use enhanced search but maintain backward compatibility
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
            "enhanced": True,  # Indicate this uses enhanced backend
            "metadata": {
                "search_parameters": {
                    "search_level": query.search_level,
                    "limit": query.limit
                },
                "backend": "internvl_enhanced"
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced hierarchical search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hierarchical search failed: {str(e)}")

@router.post("/search/multimodal")
async def multimodal_search(
    query: str,
    modality_weights: Optional[Dict[str, float]] = None,
    top_k: int = Query(10, ge=1, le=100),
    vector_search: InternVLEnhancedVectorSearch = Depends(get_internvl_enhanced_vector_search)
):
    """Enhanced multimodal search (backward compatible)"""
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    try:
        # Use enhanced multimodal search
        search_results = await vector_search.multimodal_search(
            query=query,
            modality_weights=modality_weights,
            top_k=top_k
        )
        
        return {
            "query": query,
            "results": search_results,
            "total": len(search_results),
            "modality_weights_used": modality_weights,
            "enhanced": True,
            "backend_info": {
                "internvl_enabled": vector_search.use_internvl,
                "unified_encoding": vector_search.use_internvl,
                "knowledge_enhancement": True
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced multimodal search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multimodal search failed: {str(e)}")

@router.get("/videos/{video_id}/segments")
async def get_video_segments(
    video_id: str,
    vector_search: InternVLEnhancedVectorSearch = Depends(get_internvl_enhanced_vector_search)
):
    """Get enhanced video structure with InternVL metadata"""
    
    if video_id not in vector_search.video_embeddings:
        raise HTTPException(status_code=404, detail="Video not found or not processed")
    
    # Get enhanced video structure
    video_structure = await vector_search.get_video_structure(video_id)
    
    return video_structure

@router.get("/videos/{video_id}/scenes")
async def get_video_scenes(
    video_id: str,
    vector_search: InternVLEnhancedVectorSearch = Depends(get_internvl_enhanced_vector_search)
):
    """Get enhanced video scenes with InternVL metadata"""
    
    if video_id not in vector_search.scene_embeddings:
        raise HTTPException(status_code=404, detail="Video not found or scenes not detected")
    
    scenes = []
    for scene_data in vector_search.scene_embeddings[video_id]:
        scene_info = {
            "scene_id": scene_data["scene_id"],
            "start": scene_data["start"],
            "end": scene_data["end"],
            "text": scene_data.get("text", ""),
            "num_segments": scene_data.get("num_segments", 0),
            "internvl_enhanced": scene_data.get("internvl_enhanced", False)  # NEW
        }
        scenes.append(scene_info)
    
    return {
        "video_id": video_id,
        "scenes": scenes,
        "total_scenes": len(scenes),
        "internvl_enhanced_scenes": sum(1 for s in scenes if s["internvl_enhanced"]),
        "knowledge_graph_available": video_id in vector_search.knowledge_graphs
    }

@router.post("/search/semantic")
async def semantic_search_by_modality(
    query: str,
    modality: str = Query(..., regex="^(text|visual|ocr|unified)$"),  # Added 'unified'
    top_k: int = Query(10, ge=1, le=100),
    vector_search: InternVLEnhancedVectorSearch = Depends(get_internvl_enhanced_vector_search)
):
    """Enhanced semantic search by modality including unified InternVL"""
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    try:
        if modality == "unified":
            # NEW: Direct unified search
            if not vector_search.use_internvl:
                raise HTTPException(
                    status_code=503,
                    detail="Unified search requires InternVL encoder"
                )
            
            # Get unified query embedding
            query_embeddings = await vector_search._encode_query_enhanced(query)
            unified_embedding = query_embeddings.get("unified")
            
            if unified_embedding is None:
                raise HTTPException(
                    status_code=503,
                    detail="Failed to generate unified embedding"
                )
            
            # Search unified index
            search_results = await vector_search._search_unified_index(
                unified_embedding, top_k, "segments"
            )
            
        else:
            # Traditional single modality search
            query_embedding = vector_search.embedding_model.encode(query).astype('float32')
            search_results = await vector_search._search_by_modality(
                modality=modality,
                query_embedding=query_embedding,
                k=top_k
            )
        
        return {
            "query": query,
            "modality": modality,
            "results": search_results,
            "total": len(search_results),
            "search_method": "unified_internvl" if modality == "unified" else "traditional",
            "enhanced_features": {
                "internvl_unified": modality == "unified",
                "cross_modal_understanding": modality == "unified",
                "superior_accuracy": modality == "unified"
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced semantic search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

@router.get("/search/suggestions")
async def get_search_suggestions(
    partial_query: str = Query(..., min_length=2),
    limit: int = Query(5, ge=1, le=10),
    use_knowledge_graphs: bool = Query(True, description="Use knowledge graphs for suggestions"),
    vector_search: InternVLEnhancedVectorSearch = Depends(get_internvl_enhanced_vector_search)
):
    """Enhanced search suggestions using knowledge graphs"""
    
    try:
        suggestions = []
        
        # Enhanced suggestions using knowledge graphs
        if use_knowledge_graphs and vector_search.knowledge_graphs:
            knowledge_suggestions = []
            
            for video_id, kg in vector_search.knowledge_graphs.items():
                # Check entities for matches
                for entity_type, entities in kg.get("entities", {}).items():
                    for entity in entities:
                        entity_text = entity["text"].lower()
                        if partial_query.lower() in entity_text:
                            confidence = entity.get("confidence", 1.0)
                            knowledge_suggestions.append({
                                "suggestion": entity["text"],
                                "type": "entity",
                                "entity_type": entity_type,
                                "confidence": confidence,
                                "source": "knowledge_graph"
                            })
            
            # Sort by confidence and add to suggestions
            knowledge_suggestions.sort(key=lambda x: x["confidence"], reverse=True)
            suggestions.extend(knowledge_suggestions[:limit//2])
        
        # Traditional content-based suggestions
        traditional_suggestions = []
        for video_id, segments in vector_search.segment_metadata.items():
            for segment in segments[:limit * 2]:
                text = segment.get("text", "")
                if partial_query.lower() in text.lower():
                    words = text.split()
                    for i, word in enumerate(words):
                        if partial_query.lower() in word.lower():
                            start_idx = max(0, i - 2)
                            end_idx = min(len(words), i + 3)
                            suggestion = " ".join(words[start_idx:end_idx])
                            traditional_suggestions.append({
                                "suggestion": suggestion,
                                "type": "content",
                                "confidence": segment.get("confidence", 0.8),
                                "source": "video_content"
                            })
                            break
                
                if len(traditional_suggestions) >= limit//2:
                    break
            
            if len(traditional_suggestions) >= limit//2:
                break
        
        suggestions.extend(traditional_suggestions)
        
        # Remove duplicates and limit
        unique_suggestions = []
        seen = set()
        for suggestion in suggestions:
            suggestion_text = suggestion["suggestion"].lower()
            if suggestion_text not in seen:
                seen.add(suggestion_text)
                unique_suggestions.append(suggestion)
        
        return {
            "partial_query": partial_query,
            "suggestions": unique_suggestions[:limit],
            "total": len(unique_suggestions[:limit]),
            "enhancement_info": {
                "knowledge_graph_used": use_knowledge_graphs,
                "knowledge_graphs_available": len(vector_search.knowledge_graphs),
                "suggestion_sources": list(set(s["source"] for s in unique_suggestions[:limit]))
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced search suggestions failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search suggestions failed: {str(e)}")

@router.get("/analytics/search")
async def get_search_analytics(
    vector_search: InternVLEnhancedVectorSearch = Depends(get_internvl_enhanced_vector_search)
):
    """Enhanced search analytics with InternVL metrics"""
    
    try:
        # Basic analytics
        total_videos = len(vector_search.video_embeddings)
        total_segments = sum(len(segments) for segments in vector_search.video_embeddings.values())
        
        # InternVL-specific analytics
        internvl_enhanced_videos = sum(
            1 for video_id in vector_search.video_embeddings.keys()
            if any(seg.get("internvl_enhanced", False) 
                  for seg in vector_search.segment_metadata.get(video_id, []))
        )
        
        internvl_enhanced_segments = sum(
            sum(1 for seg in segments if seg.get("internvl_enhanced", False))
            for segments in vector_search.segment_metadata.values()
        )
        
        # Knowledge graph analytics
        knowledge_graph_coverage = len(vector_search.knowledge_graphs)
        
        # Index analytics
        index_status = {
            "traditional_indices": sum(
                1 for idx in ["text", "visual", "ocr", "multimodal"] 
                if vector_search.segment_indices.get(idx) is not None
            ),
            "unified_indices": sum(
                1 for idx in ["unified"] 
                if vector_search.segment_indices.get(idx) is not None
            )
        }
        
        return {
            "search_system_status": "enhanced" if vector_search.use_internvl else "traditional",
            "video_analytics": {
                "total_videos": total_videos,
                "total_segments": total_segments,
                "internvl_enhanced_videos": internvl_enhanced_videos,
                "internvl_enhanced_segments": internvl_enhanced_segments,
                "enhancement_coverage": {
                    "video_percentage": (internvl_enhanced_videos / max(1, total_videos)) * 100,
                    "segment_percentage": (internvl_enhanced_segments / max(1, total_segments)) * 100
                }
            },
            "knowledge_graph_analytics": {
                "videos_with_knowledge_graphs": knowledge_graph_coverage,
                "coverage_percentage": (knowledge_graph_coverage / max(1, total_videos)) * 100
            },
            "search_capabilities": {
                "unified_multimodal_search": vector_search.use_internvl,
                "knowledge_enhanced_retrieval": True,
                "cross_modal_understanding": vector_search.use_internvl,
                "temporal_context_processing": True,
                "traditional_fallback": True
            },
            "index_status": index_status,
            "performance_metrics": {
                "internvl_encoder_stats": (
                    vector_search.internvl_encoder.get_stats() 
                    if vector_search.internvl_encoder else {}
                )
            },
            "available_search_methods": [
                "enhanced_search",
                "internvl_unified_search", 
                "knowledge_enhanced_search",
                "cross_modal_search",
                "hierarchical_search",
                "multimodal_search"
            ]
        }
        
    except Exception as e:
        logger.error(f"Enhanced search analytics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search analytics failed: {str(e)}")

@router.post("/search/benchmark")
async def run_search_benchmark(
    background_tasks: BackgroundTasks,
    test_queries: List[str] = Query(default=["machine learning", "climate change", "artificial intelligence"]),
    vector_search: InternVLEnhancedVectorSearch = Depends(get_internvl_enhanced_vector_search)
):
    """Run search performance benchmark comparing traditional vs InternVL methods"""
    
    try:
        benchmark_results = {
            "benchmark_id": str(uuid.uuid4()),
            "test_queries": test_queries,
            "results": []
        }
        
        for query in test_queries:
            query_benchmark = {"query": query, "methods": {}}
            
            # Traditional search
            start_time = time.time()
            traditional_results = await vector_search._search_traditional_multimodal(
                vector_search.embedding_model.encode(query).astype('float32'),
                10,
                "both"
            )
            traditional_time = time.time() - start_time
            
            query_benchmark["methods"]["traditional"] = {
                "execution_time": traditional_time,
                "result_count": len(traditional_results),
                "avg_score": sum(r["score"] for r in traditional_results) / max(1, len(traditional_results))
            }
            
            # InternVL enhanced search (if available)
            if vector_search.use_internvl:
                start_time = time.time()
                enhanced_results = await vector_search.enhanced_search(query, 10, "both", True)
                enhanced_time = time.time() - start_time
                
                query_benchmark["methods"]["internvl_enhanced"] = {
                    "execution_time": enhanced_time,
                    "result_count": len(enhanced_results),
                    "avg_score": sum(
                        r.get("enhanced_score", r["score"]) for r in enhanced_results
                    ) / max(1, len(enhanced_results)),
                    "internvl_results": sum(
                        1 for r in enhanced_results 
                        if r.get("search_method") == "internvl_unified"
                    )
                }
            
            benchmark_results["results"].append(query_benchmark)
        
        # Calculate summary statistics
        if vector_search.use_internvl:
            traditional_avg_time = sum(
                r["methods"]["traditional"]["execution_time"] 
                for r in benchmark_results["results"]
            ) / len(test_queries)
            
            enhanced_avg_time = sum(
                r["methods"]["internvl_enhanced"]["execution_time"] 
                for r in benchmark_results["results"]
            ) / len(test_queries)
            
            benchmark_results["summary"] = {
                "traditional_avg_time": traditional_avg_time,
                "enhanced_avg_time": enhanced_avg_time,
                "speed_improvement": ((traditional_avg_time - enhanced_avg_time) / traditional_avg_time) * 100,
                "internvl_advantage": enhanced_avg_time < traditional_avg_time
            }
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Search benchmark failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search benchmark failed: {str(e)}")