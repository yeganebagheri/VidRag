# src/core/events/sqs_publisher.py

import boto3
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
from botocore.exceptions import ClientError

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
class SQSEventMessage:
    """Standard SQS event message format"""
    event_id: str
    event_type: EventType
    payload: Dict[str, Any]
    timestamp: str
    source: str = "videorag_api"
    retry_count: int = 0
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None

class SQSEventPublisher:
    """Enhanced Event Publisher with AWS SQS integration"""
    
    def __init__(self, aws_region: str = "us-east-1"):
        self.aws_region = aws_region
        self.sqs_client = None
        self.queue_urls = {}
        self.initialized = False
        
        # Queue configuration
        self.queue_config = {
            # Processing queues
            EventType.VIDEO_UPLOADED: "videorag-video-uploaded",
            EventType.VIDEO_PROCESSING_STARTED: "videorag-processing-started", 
            EventType.VIDEO_TRANSCRIPTION_COMPLETED: "videorag-transcription-completed",
            EventType.VIDEO_SCENE_DETECTION_COMPLETED: "videorag-scene-detection-completed",
            EventType.VIDEO_HIERARCHICAL_SEGMENTATION_COMPLETED: "videorag-segmentation-completed",
            EventType.VIDEO_EMBEDDINGS_GENERATED: "videorag-embeddings-generated",
            EventType.VIDEO_INDEXING_COMPLETED: "videorag-indexing-completed",
            EventType.VIDEO_PROCESSING_COMPLETED: "videorag-processing-completed",
            EventType.VIDEO_PROCESSING_FAILED: "videorag-processing-failed-dlq",
            
            # Query queues
            EventType.QUERY_RECEIVED: "videorag-query-received",
            EventType.RETRIEVAL_COMPLETED: "videorag-retrieval-completed",
            EventType.RESPONSE_GENERATED: "videorag-response-generated",
            
            # System queues
            EventType.SYSTEM_HEALTH_CHECK: "videorag-health-check",
            EventType.INDEX_REBUILT: "videorag-index-rebuilt"
        }
        
        # Queue attributes for different event types
        self.queue_attributes = {
            # High priority queues (transcription, embeddings)
            "videorag-video-uploaded": {
                "VisibilityTimeoutSeconds": "300",  # 5 minutes
                "MessageRetentionPeriod": "1209600",  # 14 days
                "DelaySeconds": "0",
                "ReceiveMessageWaitTimeSeconds": "20"  # Long polling
            },
            "videorag-transcription-completed": {
                "VisibilityTimeoutSeconds": "600",  # 10 minutes
                "MessageRetentionPeriod": "1209600",
                "DelaySeconds": "0",
                "ReceiveMessageWaitTimeSeconds": "20"
            },
            "videorag-embeddings-generated": {
                "VisibilityTimeoutSeconds": "300",
                "MessageRetentionPeriod": "1209600", 
                "DelaySeconds": "0",
                "ReceiveMessageWaitTimeSeconds": "20"
            },
            
            # Standard processing queues
            "videorag-processing-started": {
                "VisibilityTimeoutSeconds": "1800",  # 30 minutes
                "MessageRetentionPeriod": "1209600",
                "DelaySeconds": "0",
                "ReceiveMessageWaitTimeSeconds": "20"
            },
            
            # Dead letter queue for failed processing
            "videorag-processing-failed-dlq": {
                "VisibilityTimeoutSeconds": "300",
                "MessageRetentionPeriod": "1209600",
                "DelaySeconds": "0",
                "maxReceiveCount": "3",  # DLQ attribute
                "ReceiveMessageWaitTimeSeconds": "20"
            }
        }
        
        self.metrics = {
            "messages_sent": 0,
            "messages_failed": 0,
            "messages_by_queue": {},
        }
    
    async def initialize(self):
        """Initialize SQS client and ensure queues exist"""
        try:
            # Initialize SQS client
            self.sqs_client = boto3.client('sqs', region_name=self.aws_region)
            
            # Ensure all queues exist
            await self._ensure_queues_exist()
            
            self.initialized = True
            logger.info("✅ SQS Event Publisher initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQS Event Publisher: {e}")
            raise
    
    async def _ensure_queues_exist(self):
        """Ensure all required SQS queues exist"""
        for event_type, queue_name in self.queue_config.items():
            try:
                # Get queue URL (creates if doesn't exist)
                loop = asyncio.get_event_loop()
                
                # Check if queue exists
                try:
                    response = await loop.run_in_executor(
                        None,
                        self.sqs_client.get_queue_url,
                        queue_name
                    )
                    queue_url = response['QueueUrl']
                    
                except ClientError as e:
                    if e.response['Error']['Code'] == 'AWS.SimpleQueueService.NonExistentQueue':
                        # Create queue
                        attributes = self.queue_attributes.get(queue_name, {})
                        
                        response = await loop.run_in_executor(
                            None,
                            self.sqs_client.create_queue,
                            queue_name,
                            attributes
                        )
                        queue_url = response['QueueUrl']
                        logger.info(f"✅ Created SQS queue: {queue_name}")
                    else:
                        raise
                
                self.queue_urls[event_type] = queue_url
                
            except Exception as e:
                logger.error(f"❌ Failed to ensure queue {queue_name}: {e}")
                raise
    
    async def publish_event(self, event_type: EventType, payload: Dict[str, Any],
                           correlation_id: Optional[str] = None,
                           delay_seconds: int = 0) -> bool:
        """Publish event to appropriate SQS queue"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Create event message
            event_message = SQSEventMessage(
                event_id=f"{event_type}_{datetime.utcnow().timestamp()}",
                event_type=event_type,
                payload=payload,
                timestamp=datetime.utcnow().isoformat(),
                correlation_id=correlation_id
            )
            
            # Get queue URL
            queue_url = self.queue_urls.get(event_type)
            if not queue_url:
                logger.error(f"No queue configured for event type: {event_type}")
                return False
            
            # Prepare message
            message_body = json.dumps(asdict(event_message), default=str)
            
            # Message attributes for filtering and routing
            message_attributes = {
                'EventType': {
                    'StringValue': event_type.value,
                    'DataType': 'String'
                },
                'Source': {
                    'StringValue': 'videorag_api',
                    'DataType': 'String'
                }
            }
            
            if correlation_id:
                message_attributes['CorrelationId'] = {
                    'StringValue': correlation_id,
                    'DataType': 'String'
                }
            
            # Send message
            loop = asyncio.get_event_loop()
            send_params = {
                'QueueUrl': queue_url,
                'MessageBody': message_body,
                'MessageAttributes': message_attributes
            }
            
            if delay_seconds > 0:
                send_params['DelaySeconds'] = delay_seconds
            
            response = await loop.run_in_executor(
                None,
                self.sqs_client.send_message,
                **send_params
            )
            
            # Update metrics
            self.metrics["messages_sent"] += 1
            queue_name = self.queue_config[event_type]
            if queue_name not in self.metrics["messages_by_queue"]:
                self.metrics["messages_by_queue"][queue_name] = 0
            self.metrics["messages_by_queue"][queue_name] += 1
            
            logger.info(f"✅ Published event {event_type} to queue {queue_name}")
            logger.debug(f"Message ID: {response.get('MessageId')}")
            
            return True
            
        except Exception as e:
            self.metrics["messages_failed"] += 1
            logger.error(f"❌ Failed to publish event {event_type}: {e}")
            return False
    
    async def publish_video_processing_event(self, video_id: str, stage: str,
                                           progress: float, metadata: Optional[Dict] = None,
                                           error: Optional[str] = None,
                                           correlation_id: Optional[str] = None):
        """Convenience method for video processing events"""
        payload = {
            "video_id": video_id,
            "stage": stage,
            "progress": progress,
            "metadata": metadata or {},
            "error": error
        }
        
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
        
        return await self.publish_event(event_type, payload, correlation_id)
    
    async def publish_batch_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Publish multiple events in batch (up to 10 per SQS limitation)"""
        if not events:
            return {"success": 0, "failed": 0, "errors": []}
        
        # Group events by queue
        events_by_queue = {}
        for event in events:
            event_type = EventType(event["event_type"])
            queue_url = self.queue_urls.get(event_type)
            
            if queue_url:
                if queue_url not in events_by_queue:
                    events_by_queue[queue_url] = []
                events_by_queue[queue_url].append(event)
        
        results = {"success": 0, "failed": 0, "errors": []}
        
        # Send batches to each queue
        for queue_url, queue_events in events_by_queue.items():
            # SQS batch limit is 10 messages
            for i in range(0, len(queue_events), 10):
                batch = queue_events[i:i+10]
                batch_result = await self._send_batch_to_queue(queue_url, batch)
                
                results["success"] += batch_result["success"]
                results["failed"] += batch_result["failed"] 
                results["errors"].extend(batch_result["errors"])
        
        return results
    
    async def _send_batch_to_queue(self, queue_url: str, events: List[Dict]) -> Dict[str, Any]:
        """Send batch of events to specific queue"""
        try:
            entries = []
            for i, event in enumerate(events):
                event_message = SQSEventMessage(**event)
                entries.append({
                    'Id': str(i),
                    'MessageBody': json.dumps(asdict(event_message), default=str),
                    'MessageAttributes': {
                        'EventType': {
                            'StringValue': event_message.event_type.value,
                            'DataType': 'String'
                        }
                    }
                })
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.sqs_client.send_message_batch,
                queue_url,
                entries
            )
            
            successful = len(response.get('Successful', []))
            failed = len(response.get('Failed', []))
            errors = [f"ID {f['Id']}: {f['Message']}" for f in response.get('Failed', [])]
            
            return {"success": successful, "failed": failed, "errors": errors}
            
        except Exception as e:
            return {"success": 0, "failed": len(events), "errors": [str(e)]}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get publisher metrics"""
        return {
            **self.metrics,
            "initialized": self.initialized,
            "queues_configured": len(self.queue_urls),
            "queue_names": list(self.queue_config.values())
        }
    
    async def close(self):
        """Close SQS client"""
        if self.sqs_client:
            self.sqs_client.close()
        self.initialized = False
        logger.info("SQS Event Publisher closed")

# Global instance
sqs_event_publisher = SQSEventPublisher()

async def get_sqs_event_publisher() -> SQSEventPublisher:
    """Get singleton SQS event publisher instance"""
    if not sqs_event_publisher.initialized:
        await sqs_event_publisher.initialize()
    return sqs_event_publisher