# src/workers/sqs_transcription_worker.py

import asyncio
import os
import tempfile
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
from faster_whisper import WhisperModel
from sqs_base_worker import SQSBaseWorker, WorkerStatus
from sqs_publisher import get_sqs_event_publisher, EventType
from src.core.database.repositories.video_repository import get_video_repository
import logging

logger = logging.getLogger(__name__)

class SQSTranscriptionWorker(SQSBaseWorker):
    """Enhanced transcription worker that consumes from SQS queues"""
    
    def __init__(self, model_size: str = "base", aws_region: str = "us-east-1", 
                 worker_id: Optional[str] = None):
        # Define queues this worker consumes from
        queue_names = [
            "videorag-video-uploaded",
            "videorag-processing-started"
        ]
        
        super().__init__("transcription", queue_names, aws_region, worker_id)
        
        self.model_size = model_size
        self.whisper_model: Optional[WhisperModel] = None
        self.video_repo = None
        self.event_publisher = None
        self.temp_dir = Path(tempfile.gettempdir()) / "videorag_transcription"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Worker-specific configuration
        self.max_concurrent_messages = 3  # Transcription is resource-intensive
        self.visibility_timeout = 1800  # 30 minutes for transcription
    
    async def initialize(self):
        """Initialize transcription worker with models and dependencies"""
        await super().initialize()
        
        # Initialize video repository
        self.video_repo = await get_video_repository()
        
        # Initialize event publisher
        self.event_publisher = await get_sqs_event_publisher()
        
        # Initialize Whisper model
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            device = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            self.whisper_model = WhisperModel(
                self.model_size, 
                device=device, 
                compute_type=compute_type
            )
            logger.info(f"✅ Whisper model loaded on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.status = WorkerStatus.ERROR
            raise
    
    def get_queue_names(self) -> List[str]:
        """Return queue names this worker consumes from"""
        return [
            "videorag-video-uploaded",
            "videorag-processing-started"
        ]
    
    async def process_message(self, message_body: Dict[str, Any], 
                            message_attributes: Dict[str, Any]) -> bool:
        """Process SQS message for transcription"""
        
        # Extract event information
        event_type = message_attributes.get('EventType')
        payload = message_body.get('payload', {})
        correlation_id = message_body.get('correlation_id')
        
        logger.info(f"Processing event: {event_type} for video: {payload.get('video_id')}")
        
        # Route based on event type
        if event_type in ['video.uploaded', 'video.processing.started']:
            return await self._process_video_transcription(payload, correlation_id)
        
        logger.warning(f"Unknown event type: {event_type}")
        return False
    
    async def _process_video_transcription(self, payload: Dict[str, Any], 
                                         correlation_id: Optional[str] = None) -> bool:
        """Process video transcription with enhanced error handling"""
        video_id = payload.get("video_id")
        
        if not video_id:
            logger.error("No video_id in payload")
            return False
        
        audio_path = None
        
        try:
            # Update video status
            await self.video_repo.update_status(video_id, "transcribing")
            
            # Publish progress event
            await self.event_publisher.publish_video_processing_event(
                video_id=video_id,
                stage="transcription",
                progress=0.1,
                metadata={"status": "started", "model": self.model_size, "worker_id": self.worker_id},
                correlation_id=correlation_id
            )
            
            # Get video file path
            video_path = await self._get_video_path(video_id, payload)
            
            if not video_path or not os.path.exists(video_path):
                logger.error(f"Video file not found for {video_id}: {video_path}")
                return False
            
            # Extract audio
            audio_path = await self._extract_audio(video_id, video_path)
            
            # Publish progress
            await self.event_publisher.publish_video_processing_event(
                video_id=video_id,
                stage="transcription",
                progress=0.3,
                metadata={"status": "audio_extracted", "audio_duration": await self._get_audio_duration(audio_path)},
                correlation_id=correlation_id
            )
            
            # Transcribe with enhanced options
            segments = await self._transcribe_audio(video_id, audio_path, correlation_id)
            
            # Publish progress
            await self.event_publisher.publish_video_processing_event(
                video_id=video_id,
                stage="transcription", 
                progress=0.8,
                metadata={"status": "transcription_completed", "segments_count": len(segments)},
                correlation_id=correlation_id
            )
            
            # Save segments to database
            await self.video_repo.save_transcription_segments(video_id, segments)
            
            # Update video status
            await self.video_repo.update_status(video_id, "transcribed")
            
            # Publish completion event
            await self.event_publisher.publish_video_processing_event(
                video_id=video_id,
                stage="transcription",
                progress=1.0,
                metadata={
                    "status": "completed",
                    "segments_count": len(segments),
                    "total_duration": segments[-1]["end"] if segments else 0,
                    "language": segments[0].get("language") if segments else "unknown",
                    "worker_id": self.worker_id
                },
                correlation_id=correlation_id
            )
            
            # Trigger next stage - publish to scene detection queue
            await self.event_publisher.publish_event(
                EventType.VIDEO_TRANSCRIPTION_COMPLETED,
                {
                    "video_id": video_id,
                    "segments_count": len(segments),
                    "next_stage": "scene_detection",
                    "video_path": video_path
                },
                correlation_id=correlation_id
            )
            
            logger.info(f"✅ Transcription completed for video {video_id}: {len(segments)} segments")
            return True
            
        except Exception as e:
            logger.error(f"❌ Transcription failed for video {video_id}: {e}")
            
            # Update status and publish error
            await self.video_repo.update_status(video_id, "failed", str(e))
            await self.event_publisher.publish_video_processing_event(
                video_id=video_id,
                stage="transcription",
                progress=0.0,
                error=str(e),
                metadata={"worker_id": self.worker_id},
                correlation_id=correlation_id
            )
            return False
        
        finally:
            # Cleanup temporary files
            try:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
                    logger.debug(f"Cleaned up temporary audio file: {audio_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup audio file: {e}")
    
    async def _get_video_path(self, video_id: str, payload: Dict[str, Any]) -> Optional[str]:
        """Get video file path from storage (Supabase or local)"""
        
        # Check payload first
        if "file_path" in payload:
            return payload["file_path"]
        
        # Try to get from video repository
        video = await self.video_repo.get_by_id(video_id)
        if video:
            if hasattr(video, 'storage_path') and video.storage_path:
                # If using Supabase Storage, download to temp location
                return await self._download_from_storage(video_id, video.storage_path)
            elif hasattr(video, 'file_path'):
                return video.file_path
        
        # Fallback path construction
        return f"/tmp/videorag_uploads/{video_id}"
    
    async def _download_from_storage(self, video_id: str, storage_path: str) -> str:
        """Download video from Supabase Storage to local temp file"""
        # This would integrate with your Supabase Storage manager
        # For now, return the storage path (assuming it's accessible)
        return storage_path
    
    async def _extract_audio(self, video_id: str, video_path: str) -> str:
        """Extract audio from video using ffmpeg"""
        audio_path = self.temp_dir / f"{video_id}_audio.wav"
        
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(audio_path)
        ]
        
        logger.debug(f"Extracting audio: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg failed: {stderr.decode()}")
        
        if not os.path.exists(audio_path):
            raise Exception(f"Audio extraction failed - file not created: {audio_path}")
        
        logger.debug(f"Audio extracted successfully: {audio_path}")
        return str(audio_path)
    
    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using ffprobe"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json", 
                "-show_format", audio_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                import json
                info = json.loads(stdout.decode())
                return float(info['format']['duration'])
            
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}")
        
        return 0.0
    
    async def _transcribe_audio(self, video_id: str, audio_path: str, 
                              correlation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Transcribe audio using Whisper with progress updates"""
        if not self.whisper_model:
            raise Exception("Whisper model not initialized")
        
        logger.info(f"Starting transcription for {video_id}")
        
        # Run transcription in thread to avoid blocking
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            self._run_whisper_transcription,
            audio_path
        )
        
        # Format segments with enhanced metadata
        formatted_segments = []
        for i, segment in enumerate(segments):
            segment_data = {
                "segment_index": i,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "confidence": float(getattr(segment, 'avg_logprob', 0.0)),
                "language": info.language,
                "language_probability": float(info.language_probability),
                "no_speech_prob": float(getattr(segment, 'no_speech_prob', 0.0)),
                "worker_id": self.worker_id
            }
            
            # Add word-level timing if available
            if hasattr(segment, 'words') and segment.words:
                segment_data["words"] = [
                    {
                        "word": word.word,
                        "start": float(word.start),
                        "end": float(word.end),
                        "probability": float(word.probability)
                    }
                    for word in segment.words
                ]
            
            formatted_segments.append(segment_data)
            
            # Publish progress updates every 50 segments
            if i > 0 and i % 50 == 0:
                progress = 0.3 + (i / len(list(segments))) * 0.5  # 30% to 80% range
                await self.event_publisher.publish_video_processing_event(
                    video_id=video_id,
                    stage="transcription",
                    progress=progress,
                    metadata={
                        "status": "transcribing", 
                        "segments_processed": i,
                        "worker_id": self.worker_id
                    },
                    correlation_id=correlation_id
                )
        
        logger.info(f"Transcription completed: {len(formatted_segments)} segments, language: {info.language}")
        return formatted_segments
    
    def _run_whisper_transcription(self, audio_path: str):
        """Run Whisper transcription (blocking operation)"""
        return self.whisper_model.transcribe(
            audio_path,
            language=None,  # Auto-detect
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            initial_prompt=None,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6
        )
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get transcription-specific processing stats"""
        base_stats = await self.get_status()
        
        # Add transcription-specific metrics
        transcription_stats = {
            "model_size": self.model_size,
            "model_loaded": self.whisper_model is not None,
            "temp_dir": str(self.temp_dir),
            "temp_files": len(list(self.temp_dir.glob("*"))) if self.temp_dir.exists() else 0,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        base_stats["transcription_specific"] = transcription_stats
        return base_stats

# Factory function for dependency injection
def create_sqs_transcription_worker(model_size: str = "base", 
                                   aws_region: str = "us-east-1") -> SQSTranscriptionWorker:
    return SQSTranscriptionWorker(model_size, aws_region)

if __name__ == "__main__":
    async def main():
        worker = SQSTranscriptionWorker()
        await worker.start()
    
    asyncio.run(main())