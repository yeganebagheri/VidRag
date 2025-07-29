from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

class QueryRequest(BaseModel):
    """Basic query request model"""
    text: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return")

class HierarchicalQueryRequest(BaseModel):
    """Enhanced query request for hierarchical search"""
    text: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    search_level: Literal["scenes", "segments", "both"] = Field(
        "both", 
        description="Level of search: scenes, segments, or both"
    )
    modality_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Custom weights for text, visual, and OCR modalities"
    )
    context_window: int = Field(
        2, 
        ge=0, 
        le=10, 
        description="Number of adjacent segments to consider for context"
    )

    @validator('modality_weights')
    def validate_modality_weights(cls, v):
        if v is not None:
            required_keys = {'text', 'visual', 'ocr'}
            if not all(key in v for key in required_keys):
                raise ValueError(f"modality_weights must contain all keys: {required_keys}")
            if not all(0 <= weight <= 1 for weight in v.values()):
                raise ValueError("All modality weights must be between 0 and 1")
            total = sum(v.values())
            if not (0.8 <= total <= 1.2):  # Allow some tolerance
                raise ValueError("Modality weights should sum to approximately 1.0")
        return v

class QueryResult(BaseModel):
    """Enhanced query result model with hierarchical information"""
    video_id: str
    segment_id: str
    text: str
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    timestamp: float = Field(..., ge=0, description="Start timestamp in seconds")
    content_type: str = Field(..., description="Type of content (segment, scene, etc.)")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata including scene info, modality scores, etc."
    )

class HierarchicalQueryResult(BaseModel):
    """Enhanced result model for hierarchical search"""
    video_id: str
    segment_id: Optional[str] = None
    scene_id: Optional[str] = None
    text: str
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., ge=0)
    duration: float = Field(..., ge=0)
    
    # Scoring information
    final_score: float = Field(..., ge=0, le=1)
    modality_scores: Dict[str, float] = Field(default_factory=dict)
    context_score: float = Field(0.0, ge=0, le=1)
    temporal_score: float = Field(0.0, ge=0, le=1)
    adaptive_score: float = Field(0.0, ge=0, le=1)
    
    # Content type and hierarchy info
    content_type: Literal["scene", "segment"] = "segment"
    hierarchy_level: int = Field(0, ge=0, description="0 for segments, 1 for scenes")
    
    # Additional metadata
    has_visual_content: bool = False
    has_ocr_content: bool = False
    num_segments: Optional[int] = None  # For scene results
    
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class QueryResponse(BaseModel):
    """Enhanced query response model"""
    query: str
    results: List[QueryResult]
    total: int = Field(..., ge=0)
    execution_time: Optional[float] = Field(None, description="Query execution time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class HierarchicalQueryResponse(BaseModel):
    """Enhanced response for hierarchical queries"""
    query: str
    search_level: str
    results: List[HierarchicalQueryResult]
    total: int = Field(..., ge=0)
    
    # Execution metadata
    execution_time: Optional[float] = None
    modality_weights_used: Optional[Dict[str, float]] = None
    
    # Result statistics
    scene_results_count: int = Field(0, ge=0)
    segment_results_count: int = Field(0, ge=0)
    
    # Search metadata
    search_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional search execution metadata"
    )

class MultimodalSearchRequest(BaseModel):
    """Request model for multimodal search"""
    query: str = Field(..., min_length=1, max_length=1000)
    modality_weights: Optional[Dict[str, float]] = None
    top_k: int = Field(10, ge=1, le=100)
    adaptive_weighting: bool = Field(True, description="Enable adaptive modality weighting")
    
    @validator('modality_weights')
    def validate_modality_weights(cls, v):
        if v is not None:
            required_keys = {'text', 'visual', 'ocr'}
            if not all(key in v for key in required_keys):
                raise ValueError(f"modality_weights must contain all keys: {required_keys}")
        return v

class SearchAnalytics(BaseModel):
    """Model for search analytics data"""
    query: str
    execution_time: float
    result_count: int
    modalities_used: List[str]
    search_level: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class VideoStructure(BaseModel):
    """Model representing hierarchical video structure"""
    video_id: str
    total_duration: float
    total_segments: int
    total_scenes: int
    
    scenes: List[Dict[str, Any]] = Field(default_factory=list)
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    
    processing_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class SearchSuggestion(BaseModel):
    """Model for search suggestions"""
    suggestion: str
    relevance_score: float = Field(..., ge=0, le=1)
    category: Optional[str] = None
    context: Optional[str] = None

class SearchSuggestionsResponse(BaseModel):
    """Response model for search suggestions"""
    partial_query: str
    suggestions: List[SearchSuggestion]
    total: int = Field(..., ge=0)