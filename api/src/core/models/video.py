from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class ProcessingStatus(str, Enum):
    """Enhanced processing status enumeration"""
    uploaded = "uploaded"
    queued_for_processing = "queued_for_processing"
    processing = "processing"
    transcribing = "transcribing"
    analyzing_scenes = "analyzing_scenes"
    extracting_features = "extracting_features"
    building_index = "building_index"
    completed = "completed"
    failed = "failed"

class VideoCreate(BaseModel):
    filename: str
    title: Optional[str] = None
    description: Optional[str] = None
    user_id: str
    file_size: Optional[int] = None  # ← Make it optional
    duration: Optional[float] = None

class Video(BaseModel):
    """Enhanced video model with hierarchical processing info"""
    video_id: str
    filename: str
    file_size: int
    status: ProcessingStatus = ProcessingStatus.uploaded
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    # Processing results
    processing_results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Video metadata
    duration: Optional[float] = None
    fps: Optional[float] = None
    resolution: Optional[Dict[str, int]] = None
    
    # Hierarchical processing info
    total_scenes: Optional[int] = None
    total_segments: Optional[int] = None
    
    # Processing statistics
    processing_stats: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Processing performance and statistics"
    )

class ProcessingProgress(BaseModel):
    """Model for tracking processing progress"""
    video_id: str
    current_stage: ProcessingStatus
    progress_percentage: float = Field(..., ge=0, le=100)
    current_operation: Optional[str] = None
    estimated_time_remaining: Optional[float] = None  # seconds
    stages_completed: List[ProcessingStatus] = Field(default_factory=list)
    
    # Detailed progress for each stage
    stage_progress: Optional[Dict[str, float]] = Field(default_factory=dict)
    
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class SceneInfo(BaseModel):
    """Model for scene information"""
    scene_id: str
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., ge=0)
    duration: float = Field(..., ge=0)
    
    # Scene content
    description: Optional[str] = None
    key_frames: List[float] = Field(default_factory=list)  # Timestamps of key frames
    
    # Scene statistics
    num_segments: int = Field(0, ge=0)
    has_speech: bool = False
    has_text: bool = False
    visual_complexity: Optional[float] = Field(None, ge=0, le=1)
    
    # Scene embeddings info (don't store actual embeddings in model)
    embedding_dimensions: Optional[Dict[str, int]] = None

class SegmentInfo(BaseModel):
    """Model for segment information"""
    segment_id: str
    scene_id: str
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., ge=0)
    duration: float = Field(..., ge=0)
    
    # Content
    transcript: Optional[str] = None
    ocr_text: Optional[str] = None
    
    # Content flags
    has_speech: bool = False
    has_visual_content: bool = False
    has_ocr_content: bool = False
    
    # Quality metrics
    audio_quality: Optional[float] = Field(None, ge=0, le=1)
    visual_quality: Optional[float] = Field(None, ge=0, le=1)
    transcription_confidence: Optional[float] = Field(None, ge=0, le=1)

class KnowledgeGraph(BaseModel):
    """Model for extracted knowledge graph"""
    video_id: str
    
    # Entities extracted from video
    entities: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    
    # Relationships between entities
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Topics and themes
    topics: List[str] = Field(default_factory=list)
    
    # Temporal information
    entity_timeline: Optional[Dict[str, List[Dict[str, Any]]]] = None
    
    extraction_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ProcessingResults(BaseModel):
    """Comprehensive processing results model"""
    video_id: str
    processing_version: str = "2.0"
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Basic video information
    video_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Hierarchical structure
    scenes: List[SceneInfo] = Field(default_factory=list)
    segments: List[SegmentInfo] = Field(default_factory=list)
    
    # Knowledge extraction
    knowledge_graph: Optional[KnowledgeGraph] = None
    
    # Processing statistics
    processing_time: Optional[float] = None  # Total processing time in seconds
    stage_times: Optional[Dict[str, float]] = Field(default_factory=dict)
    
    # Quality metrics
    overall_quality: Optional[float] = Field(None, ge=0, le=1)
    modality_quality: Optional[Dict[str, float]] = Field(default_factory=dict)
    
    # Error information
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

class VideoUploadResponse(BaseModel):
    """Response model for video upload"""
    video_id: str
    filename: str
    status: ProcessingStatus
    message: str
    
    # Upload metadata
    file_size: int
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    
    # Processing information
    estimated_processing_time: Optional[float] = None
    processing_stages: List[ProcessingStatus] = Field(default_factory=list)

class VideoListResponse(BaseModel):
    """Response model for video listing"""
    videos: List[Video]
    total: int = Field(..., ge=0)
    page: int = Field(1, ge=1)
    page_size: int = Field(10, ge=1, le=100)
    
    # Summary statistics
    summary: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "total_videos": 0,
            "processing_videos": 0,
            "completed_videos": 0,
            "failed_videos": 0,
            "total_duration": 0,
            "total_scenes": 0,
            "total_segments": 0
        }
    )

class VideoAnalytics(BaseModel):
    """Model for video analytics data"""
    video_id: str
    
    # Usage statistics
    search_count: int = Field(0, ge=0)
    view_count: int = Field(0, ge=0)
    last_accessed: Optional[datetime] = None
    
    # Content statistics
    most_queried_segments: List[Dict[str, Any]] = Field(default_factory=list)
    popular_topics: List[str] = Field(default_factory=list)
    
    # Performance metrics
    average_search_time: Optional[float] = None
    search_success_rate: Optional[float] = Field(None, ge=0, le=1)

class BatchProcessingRequest(BaseModel):
    """Model for batch processing requests"""
    video_ids: List[str] = Field(..., min_items=1, max_items=50)
    processing_options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    priority: Literal["low", "normal", "high"] = "normal"
    
    # Callback configuration
    callback_url: Optional[str] = None
    callback_token: Optional[str] = None

class VideoSearchFilters(BaseModel):
    """Model for video search filtering"""
    status: Optional[List[ProcessingStatus]] = None
    min_duration: Optional[float] = Field(None, ge=0)
    max_duration: Optional[float] = Field(None, ge=0)
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    has_scenes: Optional[bool] = None
    has_knowledge_graph: Optional[bool] = None
    tags: Optional[List[str]] = None

    @validator('max_duration')
    def validate_duration_range(cls, v, values):
        """Validate that max_duration is greater than min_duration"""
        if v is not None and 'min_duration' in values and values['min_duration'] is not None:
            if v <= values['min_duration']:
                raise ValueError('max_duration must be greater than min_duration')
        return v