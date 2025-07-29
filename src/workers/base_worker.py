import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from publisher import EnhancedEventPublisher, EventType, get_enhanced_event_publisher
from dataclasses import dataclass

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
    tasks_processed: int = 0
    tasks_failed: int = 0
    tasks_in_progress: int = 0
    average_processing_time: float = 0.0
    last_activity: Optional[datetime] = None
    uptime: float = 0.0

class EnhancedBaseWorker(ABC):
    def __init__(self, worker_type: str, worker_id: Optional[str] = None):
        self.worker_type = worker_type
        self.worker_id = worker_id or f"{worker_type}_{datetime.utcnow().timestamp()}"
        self.status = WorkerStatus.INITIALIZING
        self.event_publisher: Optional[EnhancedEventPublisher] = None
        self.metrics = WorkerMetrics()
        self.start_time: Optional[datetime] = None
        self.current_tasks: Dict[str, Dict[str, Any]] = {}
        self.max_concurrent_tasks = 5
        self.task_timeout = 300  # 5 minutes default
        
    async def initialize(self):
        """Initialize the worker"""
        logger.info(f"Initializing {self.worker_type} worker {self.worker_id}")
        
        self.event_publisher = await get_enhanced_event_publisher()
        
        # Subscribe to relevant events
        for event_type in self.get_event_types():
            await self.event_publisher.subscribe(event_type, self.handle_event)
        
        self.start_time = datetime.utcnow()
        self.status = WorkerStatus.RUNNING
        
        logger.info(f"✅ {self.worker_type} worker {self.worker_id} initialized and ready")
    
    @abstractmethod
    async def process_event(self, event_type: EventType, payload: Dict[str, Any]) -> bool:
        """Process a specific event - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_event_types(self) -> List[EventType]:
        """Return list of event types this worker handles"""
        pass
    
    async def handle_event(self, event: Dict[str, Any]):
        """Handle incoming events with error handling and metrics"""
        if self.status != WorkerStatus.RUNNING:
            logger.warning(f"Worker {self.worker_id} not running, ignoring event {event['type']}")
            return
        
        event_type = EventType(event["type"])
        payload = event["payload"]
        task_id = f"{event['id']}_{self.worker_id}"
        
        # Check concurrent task limit
        if len(self.current_tasks) >= self.max_concurrent_tasks:
            logger.warning(f"Worker {self.worker_id} at max capacity, dropping event {event_type}")
            return
        
        # Track task start
        task_start = datetime.utcnow()
        self.current_tasks[task_id] = {
            "event_type": event_type,
            "start_time": task_start,
            "payload": payload
        }
        
        self.metrics.tasks_in_progress += 1
        self.metrics.last_activity = task_start
        
        try:
            logger.info(f"Worker {self.worker_id} processing {event_type}")
            
            # Set task timeout
            success = await asyncio.wait_for(
                self.process_event(event_type, payload),
                timeout=self.task_timeout
            )
            
            # Update metrics on success
            task_duration = (datetime.utcnow() - task_start).total_seconds()
            self.metrics.tasks_processed += 1
            self._update_average_processing_time(task_duration)
            
            if success:
                logger.info(f"✅ Worker {self.worker_id} completed {event_type} in {task_duration:.2f}s")
            else:
                logger.warning(f"⚠️ Worker {self.worker_id} failed to process {event_type}")
                self.metrics.tasks_failed += 1
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Worker {self.worker_id} timed out processing {event_type}")
            self.metrics.tasks_failed += 1
            
        except Exception as e:
            logger.error(f"❌ Worker {self.worker_id} error processing {event_type}: {e}")
            self.metrics.tasks_failed += 1
            
        finally:
            # Clean up task tracking
            self.current_tasks.pop(task_id, None)
            self.metrics.tasks_in_progress -= 1
    
    def _update_average_processing_time(self, new_duration: float):
        """Update rolling average processing time"""
        if self.metrics.tasks_processed == 1:
            self.metrics.average_processing_time = new_duration
        else:
            # Exponential moving average with alpha = 0.1
            alpha = 0.1
            self.metrics.average_processing_time = (
                alpha * new_duration + 
                (1 - alpha) * self.metrics.average_processing_time
            )
    
    async def start(self):
        """Start the worker"""
        await self.initialize()
        
        try:
            logger.info(f"🚀 Starting {self.worker_type} worker {self.worker_id}")
            
            # Keep worker running
            while self.status == WorkerStatus.RUNNING:
                await asyncio.sleep(1)
                
                # Update uptime
                if self.start_time:
                    self.metrics.uptime = (datetime.utcnow() - self.start_time).total_seconds()
                
        except KeyboardInterrupt:
            logger.info(f"🛑 Received shutdown signal for worker {self.worker_id}")
            await self.stop()
            
        except Exception as e:
            logger.error(f"❌ Worker {self.worker_id} encountered error: {e}")
            self.status = WorkerStatus.ERROR
            await self.stop()
    
    async def stop(self):
        """Gracefully stop the worker"""
        logger.info(f"🔄 Stopping {self.worker_type} worker {self.worker_id}")
        self.status = WorkerStatus.STOPPING
        
        # Wait for current tasks to complete (with timeout)
        timeout = 30  # 30 seconds to complete current tasks
        start_time = datetime.utcnow()
        
        while self.current_tasks and (datetime.utcnow() - start_time).total_seconds() < timeout:
            logger.info(f"Waiting for {len(self.current_tasks)} tasks to complete...")
            await asyncio.sleep(1)
        
        if self.current_tasks:
            logger.warning(f"Force stopping worker with {len(self.current_tasks)} incomplete tasks")
        
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
            "metrics": {
                "tasks_processed": self.metrics.tasks_processed,
                "tasks_failed": self.metrics.tasks_failed,
                "tasks_in_progress": self.metrics.tasks_in_progress,
                "average_processing_time": round(self.metrics.average_processing_time, 2),
                "success_rate": (
                    self.metrics.tasks_processed / 
                    max(1, self.metrics.tasks_processed + self.metrics.tasks_failed)
                ) if (self.metrics.tasks_processed + self.metrics.tasks_failed) > 0 else 0,
                "uptime": round(self.metrics.uptime, 2),
                "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
            },
            "current_tasks": list(self.current_tasks.keys()),
            "event_types": [et.value for et in self.get_event_types()]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        is_healthy = (
            self.status == WorkerStatus.RUNNING and
            self.metrics.tasks_in_progress < self.max_concurrent_tasks and
            len(self.current_tasks) < self.max_concurrent_tasks
        )
        
        return {
            "healthy": is_healthy,
            "status": self.status.value,
            "worker_id": self.worker_id,
            "uptime": self.metrics.uptime,
            "load": len(self.current_tasks) / self.max_concurrent_tasks
        }