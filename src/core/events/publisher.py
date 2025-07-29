import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

class EventType(str, Enum):
    # Video processing events
    VIDEO_UPLOADED = "video.uploaded"
    VIDEO_PROCESSING_STARTED = "video.processing.started"
    VIDEO_TRANSCRIPTION_COMPLETED = "video.transcription.completed"
    VIDEO_SCENE_DETECTION_COMPLETED = "video.scene_detection.completed"
    VIDEO_HIERARCHICAL_SEGMENTATION_COMPLETED = "video.hierarchical_segmentation.completed"
    VIDEO_EMBEDDINGS_GENERATED = "video.embeddings.generated"
    VIDEO_INDEXING_COMPLETED = "video.indexing.completed"
    VIDEO_PROCESSING_COMPLETED = "video.processing.completed"
    VIDEO_PROCESSING_FAILED = "video.processing.failed"
    
    # Query events
    QUERY_RECEIVED = "query.received"
    RETRIEVAL_COMPLETED = "retrieval.completed"
    RESPONSE_GENERATED = "response.generated"
    
    # System events
    SYSTEM_HEALTH_CHECK = "system.health_check"
    INDEX_REBUILT = "index.rebuilt"

@dataclass
class VideoProcessingEvent:
    video_id: str
    stage: str
    progress: float
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class QueryEvent:
    query_id: str
    query_text: str
    response_type: str
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EnhancedEventPublisher:
    def __init__(self):
        self.initialized = False
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Dict[str, Any]] = []
        self.metrics = {
            "events_published": 0,
            "events_by_type": {},
            "subscribers_count": 0
        }
    
    async def initialize(self):
        """Initialize enhanced event publisher"""
        logger.info("Initializing Enhanced Event Publisher")
        self.initialized = True
    
    async def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to specific event types"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        self.metrics["subscribers_count"] = sum(len(subs) for subs in self.subscribers.values())
        
        logger.info(f"New subscriber for {event_type}. Total subscribers: {self.metrics['subscribers_count']}")
    
    async def publish_event(self, event_type: EventType, payload: Dict[str, Any]):
        """Publish event with enhanced tracking and metrics"""
        if not self.initialized:
            await self.initialize()
        
        event = {
            "id": f"{event_type}_{datetime.utcnow().timestamp()}",
            "type": event_type,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "videorag_api"
        }
        
        # Store in history (keep last 1000 events)
        self.event_history.append(event)
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]
        
        # Update metrics
        self.metrics["events_published"] += 1
        if event_type not in self.metrics["events_by_type"]:
            self.metrics["events_by_type"][event_type] = 0
        self.metrics["events_by_type"][event_type] += 1
        
        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error notifying subscriber for {event_type}: {e}")
        
        logger.info(f"Published event: {event_type} with payload keys: {list(payload.keys())}")
        return True
    
    async def publish_video_processing_event(self, video_id: str, stage: str, 
                                           progress: float, metadata: Optional[Dict] = None,
                                           error: Optional[str] = None):
        """Convenience method for video processing events"""
        event_data = VideoProcessingEvent(
            video_id=video_id,
            stage=stage,
            progress=progress,
            metadata=metadata or {},
            error=error
        )
        
        # Determine event type based on stage
        event_type_map = {
            "uploaded": EventType.VIDEO_UPLOADED,
            "transcription": EventType.VIDEO_TRANSCRIPTION_COMPLETED,
            "scene_detection": EventType.VIDEO_SCENE_DETECTION_COMPLETED,
            "hierarchical_segmentation": EventType.VIDEO_HIERARCHICAL_SEGMENTATION_COMPLETED,
            "embeddings": EventType.VIDEO_EMBEDDINGS_GENERATED,
            "indexing": EventType.VIDEO_INDEXING_COMPLETED,
            "completed": EventType.VIDEO_PROCESSING_COMPLETED,
            "failed": EventType.VIDEO_PROCESSING_FAILED
        }
        
        event_type = event_type_map.get(stage, EventType.VIDEO_PROCESSING_STARTED)
        
        await self.publish_event(event_type, asdict(event_data))
    
    async def publish_query_event(self, query_id: str, query_text: str, 
                                 response_type: str, user_id: Optional[str] = None,
                                 metadata: Optional[Dict] = None):
        """Convenience method for query events"""
        event_data = QueryEvent(
            query_id=query_id,
            query_text=query_text,
            response_type=response_type,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        await self.publish_event(EventType.QUERY_RECEIVED, asdict(event_data))
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get event publisher metrics"""
        return {
            **self.metrics,
            "recent_events": self.event_history[-10:],  # Last 10 events
            "event_types_available": [e.value for e in EventType]
        }
    
    async def get_events_by_type(self, event_type: EventType, limit: int = 50) -> List[Dict]:
        """Get recent events of specific type"""
        return [
            event for event in self.event_history[-limit:]
            if event["type"] == event_type
        ]
    
    async def close(self):
        """Close event publisher and cleanup"""
        logger.info("Closing Enhanced Event Publisher")
        self.subscribers.clear()
        self.initialized = False

# Global instance
enhanced_event_publisher = EnhancedEventPublisher()

async def get_enhanced_event_publisher() -> EnhancedEventPublisher:
    if not enhanced_event_publisher.initialized:
        await enhanced_event_publisher.initialize()
    return enhanced_event_publisher