# src/workers/sqs_base_worker.py

import asyncio
import json
import logging
import boto3
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class WorkerStatus(str, Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class WorkerMetrics:
    messages_processed: int = 0
    messages_failed: int = 0
    messages_in_progress: int = 0
    average_processing_time: float = 0.0
    last_activity: Optional[datetime] = None
    uptime: float = 0.0
    messages_deleted: int = 0
    messages_returned_to_queue: int = 0

class SQSBaseWorker(ABC):
    """Enhanced base worker that consumes from SQS queues"""
    
    def __init__(self, worker_type: str, queue_names: List[str], 
                 aws_region: str = "us-east-1", worker_id: Optional[str] = None):
        self.worker_type = worker_type
        self.worker_id = worker_id or f"{worker_type}_{datetime.utcnow().timestamp()}"
        self.queue_names = queue_names
        self.aws_region = aws_region
        
        self.status = WorkerStatus.INITIALIZING
        self.metrics = WorkerMetrics()
        self.start_time: Optional[datetime] = None
        
        # SQS configuration
        self.sqs_client = None
        self.queue_urls = {}
        self.current_messages: Dict[str, Dict[str, Any]] = {}
        
        # Worker configuration
        self.max_concurrent_messages = 5
        self.visibility_timeout = 300  # 5 minutes
        self.wait_time_seconds = 20  # Long polling
        self.max_receive_count = 3
        self.batch_size = 10  # Max messages to receive at once
        
        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        self.consumer_tasks: List[asyncio.Task] = []
    
    async def initialize(self):
        """Initialize the SQS worker"""
        logger.info(f"Initializing {self.worker_type} SQS worker {self.worker_id}")
        
        try:
            # Initialize SQS client
            self.sqs_client = boto3.client('sqs', region_name=self.aws_region)
            
            # Get queue URLs
            await self._get_queue_urls()
            
            self.start_time = datetime.utcnow()
            self.status = WorkerStatus.RUNNING
            
            logger.info(f"✅ {self.worker_type} SQS worker {self.worker_id} initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQS worker: {e}")
            self.status = WorkerStatus.ERROR
            raise
    
    async def _get_queue_urls(self):
        """Get URLs for all configured queues"""
        loop = asyncio.get_event_loop()
        
        for queue_name in self.queue_names:
            try:
                response = await loop.run_in_executor(
                    None,
                    self.sqs_client.get_queue_url,
                    queue_name
                )
                self.queue_urls[queue_name] = response['QueueUrl']
                logger.info(f"✅ Found queue: {queue_name}")
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'AWS.SimpleQueueService.NonExistentQueue':
                    logger.error(f"❌ Queue {queue_name} does not exist")
                    raise Exception(f"Required queue {queue_name} not found")
                else:
                    raise
    
    @abstractmethod
    async def process_message(self, message_body: Dict[str, Any], 
                            message_attributes: Dict[str, Any]) -> bool:
        """Process a single SQS message - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_queue_names(self) -> List[str]:
        """Return list of queue names this worker should consume from"""
        pass
    
    async def start(self):
        """Start the SQS worker"""
        await self.initialize()
        
        try:
            logger.info(f"🚀 Starting {self.worker_type} SQS worker {self.worker_id}")
            
            # Start consumer tasks for each queue
            for queue_name, queue_url in self.queue_urls.items():
                task = asyncio.create_task(
                    self._consume_queue(queue_name, queue_url)
                )
                self.consumer_tasks.append(task)
            
            # Monitor and update metrics
            metrics_task = asyncio.create_task(self._metrics_monitor())
            self.consumer_tasks.append(metrics_task)
            
            # Wait for shutdown signal or task completion
            await self.shutdown_event.wait()
            
        except KeyboardInterrupt:
            logger.info(f"🛑 Received shutdown signal for worker {self.worker_id}")
        except Exception as e:
            logger.error(f"❌ Worker {self.worker_id} encountered error: {e}")
            self.status = WorkerStatus.ERROR
        finally:
            await self.stop()
    
    async def _consume_queue(self, queue_name: str, queue_url: str):
        """Consume messages from a specific SQS queue"""
        logger.info(f"Starting consumer for queue: {queue_name}")
        
        while not self.shutdown_event.is_set() and self.status == WorkerStatus.RUNNING:
            try:
                # Check if we can handle more messages
                if len(self.current_messages) >= self.max_concurrent_messages:
                    await asyncio.sleep(1)
                    continue
                
                # Receive messages
                messages = await self._receive_messages(queue_url)
                
                if not messages:
                    continue
                
                # Process messages concurrently
                tasks = []
                for message in messages:
                    if len(self.current_messages) < self.max_concurrent_messages:
                        task = asyncio.create_task(
                            self._handle_message(queue_name, queue_url, message)
                        )
                        tasks.append(task)
                    else:
                        # Return message to queue by not deleting it
                        logger.warning(f"At capacity, message will be retried: {message.get('MessageId')}")
                
                # Wait for some tasks to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                logger.error(f"Error consuming from queue {queue_name}: {e}")
                await asyncio.sleep(5)  # Backoff on error
    
    async def _receive_messages(self, queue_url: str) -> List[Dict[str, Any]]:
        """Receive messages from SQS queue"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.sqs_client.receive_message,
                {
                    'QueueUrl': queue_url,
                    'MaxNumberOfMessages': min(self.batch_size, 
                                             self.max_concurrent_messages - len(self.current_messages)),
                    'WaitTimeSeconds': self.wait_time_seconds,
                    'MessageAttributeNames': ['All'],
                    'AttributeNames': ['All']
                }
            )
            
            return response.get('Messages', [])
            
        except Exception as e:
            logger.error(f"Failed to receive messages: {e}")
            return []
    
    async def _handle_message(self, queue_name: str, queue_url: str, message: Dict[str, Any]):
        """Handle a single SQS message"""
        message_id = message.get('MessageId')
        receipt_handle = message.get('ReceiptHandle')
        
        # Track message processing
        processing_start = datetime.utcnow()
        self.current_messages[message_id] = {
            "queue_name": queue_name,
            "start_time": processing_start,
            "receipt_handle": receipt_handle
        }
        
        self.metrics.messages_in_progress += 1
        self.metrics.last_activity = processing_start
        
        try:
            # Parse message body
            message_body = json.loads(message.get('Body', '{}'))
            message_attributes = message.get('MessageAttributes', {})
            
            # Extract string values from SQS message attributes
            parsed_attributes = {}
            for key, value in message_attributes.items():
                parsed_attributes[key] = value.get('StringValue', value.get('BinaryValue'))
            
            logger.info(f"Processing message {message_id} from {queue_name}")
            
            # Process the message
            success = await asyncio.wait_for(
                self.process_message(message_body, parsed_attributes),
                timeout=self.visibility_timeout - 30  # Leave buffer for cleanup
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - processing_start).total_seconds()
            self._update_average_processing_time(processing_time)
            
            if success:
                # Delete message from queue
                await self._delete_message(queue_url, receipt_handle)
                self.metrics.messages_processed += 1
                self.metrics.messages_deleted += 1
                
                logger.info(f"✅ Successfully processed message {message_id} in {processing_time:.2f}s")
            else:
                # Leave message in queue for retry
                self.metrics.messages_failed += 1
                self.metrics.messages_returned_to_queue += 1
                logger.warning(f"⚠️ Failed to process message {message_id}, will retry")
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Message {message_id} timed out")
            self.metrics.messages_failed += 1
            self.metrics.messages_returned_to_queue += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing message {message_id}: {e}")
            self.metrics.messages_failed += 1
            # Decide whether to delete or retry based on error type
            if self._should_retry_message(message, e):
                self.metrics.messages_returned_to_queue += 1
            else:
                await self._delete_message(queue_url, receipt_handle)
                self.metrics.messages_deleted += 1
            
        finally:
            # Clean up tracking
            self.current_messages.pop(message_id, None)
            self.metrics.messages_in_progress -= 1
    
    def _should_retry_message(self, message: Dict[str, Any], error: Exception) -> bool:
        """Determine if message should be retried based on error type"""
        # Implement retry logic based on your needs
        # For example, retry on temporary failures, not on validation errors
        
        # Check approximate receive count
        attributes = message.get('Attributes', {})
        receive_count = int(attributes.get('ApproximateReceiveCount', 0))
        
        if receive_count >= self.max_receive_count:
            logger.warning(f"Message has been retried {receive_count} times, giving up")
            return False
        
        # Add more sophisticated retry logic here
        return True
    
    async def _delete_message(self, queue_url: str, receipt_handle: str):
        """Delete message from SQS queue"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.sqs_client.delete_message,
                queue_url,
                receipt_handle
            )
        except Exception as e:
            logger.error(f"Failed to delete message: {e}")
    
    def _update_average_processing_time(self, new_duration: float):
        """Update rolling average processing time"""
        if self.metrics.messages_processed == 0:
            self.metrics.average_processing_time = new_duration
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_processing_time = (
                alpha * new_duration + 
                (1 - alpha) * self.metrics.average_processing_time
            )
    
    async def _metrics_monitor(self):
        """Monitor and update worker metrics"""
        while not self.shutdown_event.is_set():
            if self.start_time:
                self.metrics.uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            await asyncio.sleep(10)  # Update every 10 seconds
    
    async def stop(self):
        """Gracefully stop the worker"""
        logger.info(f"🔄 Stopping {self.worker_type} SQS worker {self.worker_id}")
        self.status = WorkerStatus.STOPPING
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for current messages to complete (with timeout)
        timeout = 60  # 60 seconds to complete current messages
        start_time = datetime.utcnow()
        
        while self.current_messages and (datetime.utcnow() - start_time).total_seconds() < timeout:
            logger.info(f"Waiting for {len(self.current_messages)} messages to complete...")
            await asyncio.sleep(2)
        
        # Cancel remaining tasks
        for task in self.consumer_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish
        if self.consumer_tasks:
            await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
        
        if self.current_messages:
            logger.warning(f"Force stopping worker with {len(self.current_messages)} incomplete messages")
        
        # Close SQS client
        if self.sqs_client:
            self.sqs_client.close()
        
        self.status = WorkerStatus.STOPPED
        logger.info(f"✅ Worker {self.worker_id} stopped")
    
    async def pause(self):
        """Pause the worker"""
        if self.status == WorkerStatus.RUNNING:
            self.status = WorkerStatus.PAUSED
            logger.info(f"⏸️ Worker {self.worker_id} paused")
    
    async def resume(self):
        """Resume the worker"""
        if self.status == WorkerStatus.PAUSED:
            self.status = WorkerStatus.RUNNING
            logger.info(f"▶️ Worker {self.worker_id} resumed")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get worker status and metrics"""
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type,
            "status": self.status.value,
            "queue_names": self.queue_names,
            "metrics": {
                "messages_processed": self.metrics.messages_processed,
                "messages_failed": self.metrics.messages_failed,
                "messages_in_progress": self.metrics.messages_in_progress,
                "messages_deleted": self.metrics.messages_deleted,
                "messages_returned_to_queue": self.metrics.messages_returned_to_queue,
                "average_processing_time": round(self.metrics.average_processing_time, 2),
                "success_rate": (
                    self.metrics.messages_processed / 
                    max(1, self.metrics.messages_processed + self.metrics.messages_failed)
                ) if (self.metrics.messages_processed + self.metrics.messages_failed) > 0 else 0,
                "uptime": round(self.metrics.uptime, 2),
                "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
            },
            "current_messages": list(self.current_messages.keys()),
            "queue_urls": self.queue_urls
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        is_healthy = (
            self.status == WorkerStatus.RUNNING and
            self.metrics.messages_in_progress < self.max_concurrent_messages and
            len(self.current_messages) < self.max_concurrent_messages and
            self.sqs_client is not None
        )
        
        return {
            "healthy": is_healthy,
            "status": self.status.value,
            "worker_id": self.worker_id,
            "uptime": self.metrics.uptime,
            "load": len(self.current_messages) / self.max_concurrent_messages,
            "queues_connected": len(self.queue_urls) == len(self.queue_names)
        }