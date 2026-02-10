import asyncio
import os
import tempfile
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
from faster_whisper import WhisperModel
from base_worker import EnhancedBaseWorker, WorkerStatus
from publisher import EventType, get_enhanced_event_publisher
from video_repository import get_video_repository
import logging

logger = logging.getLogger(__name__)

class EnhancedTranscriptionWorker(EnhancedBaseWorker):
    def __init__(self, model_size: str = "base", worker_id: Optional[str] = None):
        super().__init__("transcription", worker_id)
        self.model_size = model_size
        self.whisper_model: Optional[WhisperModel] = None
        self.video_repo = None
        self.temp_dir = Path(tempfile.gettempdir()) / "videorag_transcription"
        self.temp_dir.mkdir(exist_ok=True)
        
    async def initialize(self):
        """Initialize transcription worker with models"""
        await super().initialize()
        
        # Initialize video repository
        self.video_repo = get_video_repository()
        
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
    
    def get_event_types(self) -> List[EventType]:
        return [
            EventType.VIDEO_UPLOADED,
            EventType.VIDEO_PROCESSING_STARTED
        ]
    
    async def process_event(self, event_type: EventType, payload: Dict[str, Any]) -> bool:
        """Process transcription events"""
        
        if event_type in [EventType.VIDEO_UPLOADED, EventType.VIDEO_PROCESSING_STARTED]:
            return await self._process_video_transcription(payload)
        
        return False
    
    async def _process_video_transcription(self, payload: Dict[str, Any]) -> bool:
        """Process video transcription"""
        video_id = payload.get("video_id")
        
        if not video_id:
            logger.error("No video_id in payload")
            return False
        
        try:
            # Update video status
            await self.video_repo.update_status(video_id, "transcribing")
            
            # Publish progress event
            await self.event_publisher.publish_video_processing_event(
                video_id=video_id,
                stage="transcription",
                progress=0.1,
                metadata={"status": "started", "model": self.model_size}
            )
            
            # Get video file path (this would come from your storage system)
            video_path = await self._get_video_path(video_id, payload)
            
            if not video_path or not os.path.exists(video_path):
                logger.error(f"Video file not found for {video_id}")
                return False
            
            # Extract audio
            audio_path = await self._extract_audio(video_id, video_path)
            
            # Publish progress
            await self.event_publisher.publish_video_processing_event(
                video_id=video_id,
                stage="transcription",
                progress=0.3,
                metadata={"status": "audio_extracted"}
            )
            
            # Transcribe with enhanced options
            segments = await self._transcribe_audio(video_id, audio_path)
            
            # Publish progress
            await self.event_publisher.publish_video_processing_event(
                video_id=video_id,
                stage="transcription",
                progress=0.8,
                metadata={"status": "transcription_completed", "segments_count": len(segments)}
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
                    "language": segments[0].get("language") if segments else "unknown"
                }
            )
            
            # Trigger next stage
            await self.event_publisher.publish_event(
                EventType.VIDEO_TRANSCRIPTION_COMPLETED,
                {
                    "video_id": video_id,
                    "segments_count": len(segments),
                    "next_stage": "scene_detection"
                }
            )
            
            logger.info(f"✅ Transcription completed for video {video_id}: {len(segments)} segments")
            return True
            
        except Exception as e:
            logger.error(f"❌ Transcription failed for video {video_id}: {e}")
            
            # Update status and publish error
            await self.video_repo.update_status(video_id, "failed")
            await self.event_publisher.publish_video_processing_event(
                video_id=video_id,
                stage="transcription",
                progress=0.0,
                error=str(e)
            )
            return False
        
        finally:
            # Cleanup temporary files
            try:
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.unlink(audio_path)
            except:
                pass
    
    async def _get_video_path(self, video_id: str, payload: Dict[str, Any]) -> Optional[str]:
        """Get video file path from storage"""
        # This should be implemented based on your storage system
        # For now, assuming it's in the payload or retrievable from video_repo
        
        if "file_path" in payload:
            return payload["file_path"]
        
        # Try to get from video repository
        video = await self.video_repo.get_by_id(video_id)
        if video and hasattr(video, 'file_path'):
            return video.file_path
        
        # Default path construction (adjust based on your setup)
        return f"/tmp/videorag_uploads/{video_id}"
    
    async def _extract_audio(self, video_id: str, video_path: str) -> str:
        """Extract audio from video using ffmpeg"""
        audio_path = self.temp_dir / f"{video_id}_audio.wav"
        
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(audio_path)
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg failed: {stderr.decode()}")
        
        return str(audio_path)
    
    async def _transcribe_audio(self, video_id: str, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe audio using Whisper with enhanced options"""
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
        for segment in segments:
            segment_data = {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "confidence": float(getattr(segment, 'avg_logprob', 0.0)),
                "language": info.language,
                "language_probability": float(info.language_probability),
                "no_speech_prob": float(getattr(segment, 'no_speech_prob', 0.0))
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
            "temp_files": len(list(self.temp_dir.glob("*"))) if self.temp_dir.exists() else 0
        }
        
        base_stats["transcription_specific"] = transcription_stats
        return base_stats

# Factory function for dependency injection
def create_transcription_worker(model_size: str = "base") -> EnhancedTranscriptionWorker:
    return EnhancedTranscriptionWorker(model_size)

if __name__ == "__main__":
    async def main():
        worker = EnhancedTranscriptionWorker()
        await worker.start()
    
    asyncio.run(main())