# src/core/services/enhanced_video_processor.py
# Minimal version to get your server running

import os
import cv2
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from PIL import Image
import asyncio

logger = logging.getLogger(__name__)

class InternVLEnhancedVideoProcessor:
    """
    Minimal Enhanced Video Processor - Core functionality only
    Heavy ML dependencies are loaded lazily to avoid startup crashes
    """
    
    def __init__(self):
        # Core attributes
        self.whisper_model = None
        self.ocr_reader = None
        self.clip_model = None
        self.clip_preprocess = None
        self.embedding_model = None
        self.nlp = None
        self.nlp_method = "basic"
        self.internvl_encoder = None
        
        self.temp_dir = Path(tempfile.gettempdir()) / "videorag"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Parameters
        self.scene_threshold = 0.3
        self.max_segment_duration = 30.0
        self.frame_sample_rate = 1.0
        self.use_internvl = False  # Disabled for minimal version
        self.internvl_frame_limit = 8
        
        # Track initialization status
        self.components_loaded = {
            "whisper": False,
            "ocr": False,
            "clip": False,
            "embeddings": False,
            "nlp": False,
            "internvl": False
        }
        
    async def initialize(self):
        """Initialize with lazy loading and error handling"""
        logger.info("🚀 Initializing Minimal Video Processor...")
        
        # Initialize NLP first (most reliable)
        await self._initialize_nlp_minimal()
        
        # Try to initialize other components without failing
        await self._initialize_optional_components()
        
        logger.info("✅ Minimal Video Processor initialized!")
        logger.info(f"Loaded components: {[k for k, v in self.components_loaded.items() if v]}")
    
    async def _initialize_nlp_minimal(self):
        """Initialize minimal NLP capabilities"""
        try:
            import nltk
            import ssl
            
            # Handle SSL certificate issues
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # Download essential NLTK data quietly
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            self.nlp_method = "nltk_basic"
            self.components_loaded["nlp"] = True
            logger.info("✅ Minimal NLP (NLTK) loaded")
            
        except Exception as e:
            logger.warning(f"NLTK failed to load: {e}")
            self.nlp_method = "basic"
            logger.info("Using basic text processing only")
    
    async def _initialize_optional_components(self):
        """Try to initialize optional components without failing startup"""
        
        # Try Whisper
        try:
            from faster_whisper import WhisperModel
            self.whisper_model = WhisperModel("base", device="cpu")
            self.components_loaded["whisper"] = True
            logger.info("✅ Whisper loaded")
        except Exception as e:
            logger.warning(f"Whisper failed to load: {e}")
        
        # Try EasyOCR
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'])
            self.components_loaded["ocr"] = True
            logger.info("✅ EasyOCR loaded")
        except Exception as e:
            logger.warning(f"EasyOCR failed to load: {e}")
        
        # Try SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.components_loaded["embeddings"] = True
            logger.info("✅ Embeddings loaded")
        except Exception as e:
            logger.warning(f"SentenceTransformers failed to load: {e}")
    
    async def process_video(self, video_path: str, video_id: str) -> Dict[str, Any]:
        """Minimal video processing with graceful degradation"""
        logger.info(f"🎬 Processing video (minimal mode): {video_path}")
        
        results = {
            "video_id": video_id,
            "transcription": [],
            "visual_segments": [],
            "scene_boundaries": [],
            "hierarchical_segments": [],
            "knowledge_graph": {},
            "embeddings": [],
            "processing_mode": "minimal",
            "components_used": [k for k, v in self.components_loaded.items() if v]
        }
        
        try:
            # Step 1: Basic video info (always works)
            video_info = await self._extract_video_info(video_path)
            results["video_info"] = video_info
            
            # Step 2: Basic scene detection
            scene_boundaries = await self._detect_scenes_minimal(video_path)
            results["scene_boundaries"] = scene_boundaries
            
            # Step 3: Transcription (if Whisper available)
            if self.components_loaded["whisper"]:
                transcription = await self._enhanced_transcription(video_path)
                results["transcription"] = transcription
            else:
                logger.warning("Whisper not available - skipping transcription")
            
            # Step 4: Basic visual processing
            visual_segments = await self._process_visual_content_minimal(video_path, scene_boundaries)
            results["visual_segments"] = visual_segments
            
            # Step 5: Create segments
            hierarchical_segments = await self._create_hierarchical_segments_minimal(
                results["transcription"], visual_segments, scene_boundaries
            )
            results["hierarchical_segments"] = hierarchical_segments
            
            # Step 6: Basic knowledge extraction
            knowledge_graph = await self._extract_knowledge_graph_minimal(hierarchical_segments)
            results["knowledge_graph"] = knowledge_graph
            
            # Step 7: Generate embeddings (if available)
            if self.components_loaded["embeddings"]:
                embeddings = await self._generate_embeddings_minimal(hierarchical_segments)
                results["embeddings"] = embeddings
            
            logger.info(f"✅ Minimal video processing completed for {video_id}")
            
        except Exception as e:
            logger.error(f"❌ Minimal video processing failed for {video_id}: {e}")
            raise
        
        return self._convert_numpy_types(results)
    
    async def _detect_scenes_minimal(self, video_path: str) -> List[Dict]:
        """Basic scene detection without CLIP"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 60
        cap.release()
        
        # Simple time-based scenes (every 30 seconds)
        scenes = []
        scene_duration = 30.0
        scene_id = 0
        
        start_time = 0
        while start_time < duration:
            end_time = min(start_time + scene_duration, duration)
            scenes.append({
                "start": start_time,
                "end": end_time,
                "scene_id": scene_id
            })
            start_time = end_time
            scene_id += 1
        
        return scenes
    
    async def _process_visual_content_minimal(self, video_path: str, scene_boundaries: List[Dict]) -> List[Dict]:
        """Minimal visual processing"""
        visual_segments = []
        
        for scene in scene_boundaries:
            scene_data = {
                "scene_id": scene["scene_id"],
                "start": scene["start"],
                "end": scene["end"],
                "visual_features": [0.0] * 512,  # Placeholder
                "ocr_results": [],
                "num_frames": 0,
                "internvl_features": [0.0] * 768,
                "internvl_enhanced": False
            }
            
            # Try OCR if available
            if self.components_loaded["ocr"]:
                try:
                    # Extract one frame for OCR
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_MSEC, scene["start"] * 1000)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        ocr_result = self.ocr_reader.readtext(frame)
                        for bbox, text, confidence in ocr_result:
                            if confidence > 0.5:
                                scene_data["ocr_results"].append({
                                    "text": text,
                                    "confidence": float(confidence),
                                    "bbox": [[float(x), float(y)] for x, y in bbox],
                                    "timestamp": scene["start"]
                                })
                except Exception as e:
                    logger.warning(f"OCR failed for scene {scene['scene_id']}: {e}")
            
            visual_segments.append(scene_data)
        
        return visual_segments
    
    async def _create_hierarchical_segments_minimal(self, transcription, visual_segments, scene_boundaries):
        """Create minimal hierarchical segments"""
        hierarchical_segments = []
        
        for scene in scene_boundaries:
            scene_start = scene["start"]
            scene_end = scene["end"]
            scene_id = scene["scene_id"]
            
            # Find transcription for this scene
            scene_transcripts = [
                t for t in transcription 
                if t["start"] >= scene_start and t["end"] <= scene_end
            ]
            
            # Find visual data
            scene_visual = next(
                (v for v in visual_segments if v["scene_id"] == scene_id), 
                None
            )
            
            # Create segment
            combined_text = " ".join(t["text"] for t in scene_transcripts) if scene_transcripts else ""
            
            segment = {
                "segment_id": f"{scene_id}_0",
                "scene_id": scene_id,
                "start": scene_start,
                "end": scene_end,
                "text": combined_text,
                "transcripts": scene_transcripts,
                "visual_data": scene_visual,
                "duration": scene_end - scene_start,
                "internvl_enhanced": False
            }
            
            hierarchical_segments.append(segment)
        
        return hierarchical_segments
    
    async def _extract_knowledge_graph_minimal(self, hierarchical_segments):
        """Minimal knowledge graph extraction"""
        knowledge_graph = {
            "entities": {},
            "relationships": [],
            "topics": [],
            "sentiment_analysis": {"polarity": 0.0, "subjectivity": 0.0},
            "nlp_method_used": self.nlp_method
        }
        
        all_text = " ".join(seg["text"] for seg in hierarchical_segments if seg["text"])
        
        if all_text.strip():
            # Basic entity extraction using regex
            import re
            
            # Simple patterns
            persons = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', all_text)
            orgs = re.findall(r'\b[A-Z][a-z]+ (?:Inc|Corp|Company|LLC)\b', all_text)
            
            if persons:
                knowledge_graph["entities"]["PERSON"] = [
                    {"text": person, "confidence": 0.6} for person in set(persons)
                ]
            
            if orgs:
                knowledge_graph["entities"]["ORG"] = [
                    {"text": org, "confidence": 0.6} for org in set(orgs)
                ]
            
            # Basic topics
            words = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
            from collections import Counter
            common_words = Counter(words).most_common(5)
            knowledge_graph["topics"] = [word for word, count in common_words if count > 1]
        
        return knowledge_graph
    
    async def _generate_embeddings_minimal(self, hierarchical_segments):
        """Generate minimal embeddings"""
        embeddings = []
        
        for segment in hierarchical_segments:
            embedding_data = {
                "segment_id": segment["segment_id"],
                "scene_id": segment["scene_id"],
                "timestamp": segment["start"],
                "embeddings": {},
                "processing_method": "minimal"
            }
            
            # Text embedding if available
            if self.components_loaded["embeddings"] and segment["text"]:
                try:
                    text_embedding = self.embedding_model.encode(segment["text"])
                    embedding_data["embeddings"]["text"] = text_embedding.tolist()
                    embedding_data["embeddings"]["unified"] = text_embedding.tolist()
                except:
                    embedding_data["embeddings"]["text"] = [0.0] * 384
                    embedding_data["embeddings"]["unified"] = [0.0] * 384
            else:
                embedding_data["embeddings"]["text"] = [0.0] * 384
                embedding_data["embeddings"]["unified"] = [0.0] * 384
            
            # Placeholder embeddings
            embedding_data["embeddings"]["visual"] = [0.0] * 512
            embedding_data["embeddings"]["ocr"] = [0.0] * 384
            
            embedding_data["metadata"] = {
                "duration": segment["duration"],
                "has_text": bool(segment["text"]),
                "components_available": self.components_loaded
            }
            
            embeddings.append(embedding_data)
        
        return embeddings
    
    # Keep essential methods
    async def _enhanced_transcription(self, video_path: str) -> List[Dict]:
        """Transcription if Whisper is available"""
        if not self.whisper_model:
            return []
        
        # Extract audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_path = temp_audio.name

        try:
            import subprocess
            result = subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode != 0:
                return []

            segments, info = self.whisper_model.transcribe(audio_path)
            
            transcription = []
            for segment in segments:
                transcription.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip(),
                    "confidence": float(getattr(segment, 'avg_logprob', 0.0)),
                    "language": info.language,
                    "words": []
                })
            
            return transcription

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return []
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    async def _extract_video_info(self, video_path: str) -> Dict:
        """Extract basic video info"""
        cap = cv2.VideoCapture(video_path)
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        return info
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj
    
    def get_nlp_capabilities(self):
        """Get NLP capabilities"""
        return {
            "nlp_method": self.nlp_method,
            "components_loaded": self.components_loaded,
            "entity_extraction": self.nlp_method != "none",
            "sentiment_analysis": False,
            "minimal_mode": True
        }

# Global instances for backward compatibility
internvl_enhanced_video_processor = InternVLEnhancedVideoProcessor()
enhanced_video_processor = InternVLEnhancedVideoProcessor()

async def get_internvl_enhanced_video_processor() -> InternVLEnhancedVideoProcessor:
    return internvl_enhanced_video_processor

async def get_enhanced_video_processor() -> InternVLEnhancedVideoProcessor:
    return enhanced_video_processor

# Compatibility
EnhancedVideoProcessor = InternVLEnhancedVideoProcessor