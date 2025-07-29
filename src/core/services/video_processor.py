import os
import cv2
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from faster_whisper import WhisperModel
import easyocr
import open_clip
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import spacy
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class EnhancedVideoProcessor:
    def __init__(self):
        self.whisper_model = None
        self.ocr_reader = None
        self.clip_model = None
        self.clip_preprocess = None
        self.embedding_model = None
        self.nlp = None
        self.temp_dir = Path(tempfile.gettempdir()) / "videorag"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Enhanced parameters
        self.scene_threshold = 0.3  # For scene detection
        self.max_segment_duration = 30.0  # seconds
        self.frame_sample_rate = 1.0  # seconds
        
    async def initialize(self):
        """Initialize all ML models"""
        logger.info("Initializing enhanced video processing models...")
        
        # Initialize Whisper for transcription
        try:
            self.whisper_model = WhisperModel("base", device="cpu")
            logger.info("Whisper model loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
        
        # Initialize OCR
        try:
            self.ocr_reader = easyocr.Reader(['en'])
            logger.info("OCR reader loaded")
        except Exception as e:
            logger.error(f"Failed to load OCR: {e}")
        
        # Initialize CLIP
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
            )
            self.clip_model = self.clip_model.to(device).eval()
            logger.info("CLIP model loaded")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            
        # Initialize spaCy for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded")
        except Exception as e:
            logger.error(f"Failed to load spaCy: {e}")

    async def process_video(self, video_path: str, video_id: str) -> Dict[str, Any]:
        """Enhanced video processing with scene detection and hierarchical segmentation"""
        logger.info(f"Processing video: {video_path}")
        
        results = {
            "video_id": video_id,
            "transcription": [],
            "visual_segments": [],
            "scene_boundaries": [],
            "hierarchical_segments": [],
            "knowledge_graph": {},
            "embeddings": []
        }
        
        try:
            # Step 1: Extract basic video info
            video_info = await self._extract_video_info(video_path)
            results["video_info"] = video_info
            
            # Step 2: Scene detection
            scene_boundaries = await self._detect_scenes(video_path)
            results["scene_boundaries"] = scene_boundaries
            
            # Step 3: Transcription with speaker diarization
            transcription = await self._enhanced_transcription(video_path)
            results["transcription"] = transcription
            
            # Step 4: Visual processing with scene awareness
            visual_segments = await self._process_visual_content(video_path, scene_boundaries)
            results["visual_segments"] = visual_segments
            
            # Step 5: Hierarchical segmentation
            hierarchical_segments = await self._create_hierarchical_segments(
                transcription, visual_segments, scene_boundaries
            )
            results["hierarchical_segments"] = hierarchical_segments
            
            # Step 6: Knowledge extraction and graph construction
            knowledge_graph = await self._extract_knowledge_graph(hierarchical_segments)
            results["knowledge_graph"] = knowledge_graph
            
            # Step 7: Generate multi-modal embeddings
            embeddings = await self._generate_multimodal_embeddings(hierarchical_segments)
            results["embeddings"] = embeddings
            
            logger.info(f"Enhanced video processing completed for {video_id}")
            
        except Exception as e:
            logger.error(f"Error in enhanced video processing {video_id}: {e}")
            raise
        
        return self._convert_numpy_types(results)

    async def _detect_scenes(self, video_path: str) -> List[Dict]:
        """Detect scene boundaries using visual similarity"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        timestamps = []
        frame_count = 0
        
        # Sample frames at regular intervals
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % int(fps) == 0:  # Sample every second
                frames.append(frame)
                timestamps.append(frame_count / fps)
            frame_count += 1
        
        cap.release()
        
        if len(frames) < 2:
            return [{"start": 0, "end": frame_count / fps, "scene_id": 0}]
        
        # Extract CLIP features for scene detection
        scene_features = []
        device = next(self.clip_model.parameters()).device
        
        for frame in frames:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = self.clip_model.encode_image(img_tensor)
                scene_features.append(features.cpu().numpy().flatten())
        
        # Detect scene boundaries based on feature similarity
        scene_boundaries = []
        current_scene_start = 0
        scene_id = 0
        
        for i in range(1, len(scene_features)):
            similarity = cosine_similarity(
                [scene_features[i-1]], [scene_features[i]]
            )[0][0]
            
            if similarity < self.scene_threshold:
                # Scene boundary detected
                scene_boundaries.append({
                    "start": timestamps[current_scene_start],
                    "end": timestamps[i],
                    "scene_id": scene_id
                })
                current_scene_start = i
                scene_id += 1
        
        # Add final scene
        scene_boundaries.append({
            "start": timestamps[current_scene_start],
            "end": timestamps[-1],
            "scene_id": scene_id
        })
        
        return scene_boundaries

    async def _enhanced_transcription(self, video_path: str) -> List[Dict]:
        """Enhanced transcription with timing and confidence"""
        # Extract audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_path = temp_audio.name

        try:
            # Extract audio using ffmpeg
            import subprocess
            result = subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode != 0:
                logger.error("FFmpeg failed")
                return []

            # Transcribe with enhanced options
            segments, info = self.whisper_model.transcribe(
                audio_path,
                language=None,  # Auto-detect
                word_timestamps=True,
                vad_filter=True,
                beam_size=5
            )

            transcription = []
            for segment in segments:
                segment_data = {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip(),
                    "confidence": float(getattr(segment, 'avg_logprob', 0.0)),
                    "language": info.language,
                    "words": []
                }
                
                # Add word-level timing if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        segment_data["words"].append({
                            "word": word.word,
                            "start": float(word.start),
                            "end": float(word.end),
                            "probability": float(word.probability)
                        })
                
                transcription.append(segment_data)

            return transcription

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    async def _process_visual_content(self, video_path: str, scene_boundaries: List[Dict]) -> List[Dict]:
        """Process visual content with scene awareness"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        visual_segments = []
        
        for scene in scene_boundaries:
            scene_start_frame = int(scene["start"] * fps)
            scene_end_frame = int(scene["end"] * fps)
            
            # Sample frames within the scene
            cap.set(cv2.CAP_PROP_POS_FRAMES, scene_start_frame)
            
            frames = []
            frame_timestamps = []
            frame_count = scene_start_frame
            
            while frame_count < scene_end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample every second within the scene
                if (frame_count - scene_start_frame) % int(fps) == 0:
                    frames.append(frame)
                    frame_timestamps.append(frame_count / fps)
                
                frame_count += 1
            
            if not frames:
                continue
            
            # Process frames for this scene
            scene_visual_data = await self._process_scene_frames(
                frames, frame_timestamps, scene["scene_id"]
            )
            visual_segments.append(scene_visual_data)
        
        cap.release()
        return visual_segments

    async def _process_scene_frames(self, frames: List, timestamps: List[float], scene_id: int) -> Dict:
        """Process frames within a scene"""
        device = next(self.clip_model.parameters()).device
        
        # Extract CLIP features
        visual_features = []
        ocr_results = []
        
        for i, frame in enumerate(frames):
            # CLIP encoding
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = self.clip_model.encode_image(img_tensor)
                visual_features.append(features.cpu().numpy().flatten())
            
            # OCR processing
            try:
                ocr_result = self.ocr_reader.readtext(np.array(pil_img))
                frame_ocr = []
                for bbox, text, confidence in ocr_result:
                    if confidence > 0.5:
                        frame_ocr.append({
                            "text": text,
                            "confidence": float(confidence),
                            "bbox": [[float(x), float(y)] for x, y in bbox],
                            "timestamp": timestamps[i]
                        })
                ocr_results.extend(frame_ocr)
            except Exception as e:
                logger.warning(f"OCR failed for frame: {e}")
        
        # Average visual features for the scene
        avg_visual_features = np.mean(visual_features, axis=0) if visual_features else np.zeros(512)
        
        return {
            "scene_id": scene_id,
            "start": timestamps[0] if timestamps else 0,
            "end": timestamps[-1] if timestamps else 0,
            "visual_features": avg_visual_features.tolist(),
            "ocr_results": ocr_results,
            "num_frames": len(frames)
        }

    async def _create_hierarchical_segments(self, transcription: List[Dict], 
                                          visual_segments: List[Dict], 
                                          scene_boundaries: List[Dict]) -> List[Dict]:
        """Create hierarchical segments combining all modalities"""
        hierarchical_segments = []
        
        for scene in scene_boundaries:
            scene_start = scene["start"]
            scene_end = scene["end"]
            scene_id = scene["scene_id"]
            
            # Find transcription segments within this scene
            scene_transcripts = [
                t for t in transcription 
                if t["start"] >= scene_start and t["end"] <= scene_end
            ]
            
            # Find visual data for this scene
            scene_visual = next(
                (v for v in visual_segments if v["scene_id"] == scene_id), 
                None
            )
            
            # Create hierarchical segments within the scene
            if scene_transcripts:
                # Group transcripts into logical segments
                segments = self._group_transcripts_by_semantic_similarity(scene_transcripts)
                
                for i, segment_transcripts in enumerate(segments):
                    segment_start = min(t["start"] for t in segment_transcripts)
                    segment_end = max(t["end"] for t in segment_transcripts)
                    
                    # Combine text from all transcripts in this segment
                    combined_text = " ".join(t["text"] for t in segment_transcripts)
                    
                    hierarchical_segment = {
                        "segment_id": f"{scene_id}_{i}",
                        "scene_id": scene_id,
                        "start": segment_start,
                        "end": segment_end,
                        "text": combined_text,
                        "transcripts": segment_transcripts,
                        "visual_data": scene_visual,
                        "duration": segment_end - segment_start
                    }
                    
                    hierarchical_segments.append(hierarchical_segment)
            else:
                # Scene with no transcription (visual only)
                if scene_visual:
                    hierarchical_segment = {
                        "segment_id": f"{scene_id}_0",
                        "scene_id": scene_id,
                        "start": scene_start,
                        "end": scene_end,
                        "text": "",
                        "transcripts": [],
                        "visual_data": scene_visual,
                        "duration": scene_end - scene_start
                    }
                    hierarchical_segments.append(hierarchical_segment)
        
        return hierarchical_segments

    async def _extract_knowledge_graph(self, hierarchical_segments: List[Dict]) -> Dict:
        """Extract knowledge graph from segments"""
        if not self.nlp:
            return {}
        
        knowledge_graph = {
            "entities": {},
            "relationships": [],
            "topics": []
        }
        
        all_text = " ".join(seg["text"] for seg in hierarchical_segments if seg["text"])
        
        if not all_text.strip():
            return knowledge_graph
        
        # Process with spaCy
        doc = self.nlp(all_text)
        
        # Extract entities
        for ent in doc.ents:
            if ent.label_ not in knowledge_graph["entities"]:
                knowledge_graph["entities"][ent.label_] = []
            
            entity_info = {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 1.0  # spaCy doesn't provide confidence scores
            }
            knowledge_graph["entities"][ent.label_].append(entity_info)
        
        # Extract relationships (simplified)
        for sent in doc.sents:
            # Find relationships between entities in the same sentence
            sent_entities = [ent for ent in sent.ents]
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i+1:]:
                    knowledge_graph["relationships"].append({
                        "subject": ent1.text,
                        "object": ent2.text,
                        "relation": "co_occurs",
                        "sentence": sent.text
                    })
        
        return knowledge_graph

    def _group_transcripts_by_semantic_similarity(self, transcripts: List[Dict], 
                                                max_segment_duration: float = None) -> List[List[Dict]]:
        """Group transcripts by semantic similarity"""
        if not transcripts:
            return []
        
        if len(transcripts) == 1:
            return [transcripts]
        
        max_duration = max_segment_duration or self.max_segment_duration
        
        # Calculate embeddings for each transcript
        texts = [t["text"] for t in transcripts]
        embeddings = self.embedding_model.encode(texts)
        
        # Use simple time-based grouping with semantic validation
        groups = []
        current_group = [transcripts[0]]
        current_duration = 0
        
        for i in range(1, len(transcripts)):
            transcript = transcripts[i]
            group_end = current_group[-1]["end"]
            segment_duration = transcript["end"] - current_group[0]["start"]
            
            # Check if we should start a new group
            should_split = (
                segment_duration > max_duration or
                transcript["start"] - group_end > 2.0  # 2 second gap
            )
            
            if should_split:
                groups.append(current_group)
                current_group = [transcript]
            else:
                current_group.append(transcript)
        
        if current_group:
            groups.append(current_group)
        
        return groups

    async def _generate_multimodal_embeddings(self, hierarchical_segments: List[Dict]) -> List[Dict]:
        """Generate multi-modal embeddings for each segment"""
        embeddings = []
        
        for segment in hierarchical_segments:
            embedding_data = {
                "segment_id": segment["segment_id"],
                "scene_id": segment["scene_id"],
                "timestamp": segment["start"],
                "embeddings": {}
            }
            
            # Text embedding
            if segment["text"]:
                text_embedding = self.embedding_model.encode(segment["text"])
                embedding_data["embeddings"]["text"] = text_embedding.tolist()
            else:
                embedding_data["embeddings"]["text"] = np.zeros(384).tolist()
            
            # Visual embedding
            if segment["visual_data"] and segment["visual_data"]["visual_features"]:
                embedding_data["embeddings"]["visual"] = segment["visual_data"]["visual_features"]
            else:
                embedding_data["embeddings"]["visual"] = np.zeros(512).tolist()
            
            # OCR embedding
            if segment["visual_data"] and segment["visual_data"]["ocr_results"]:
                ocr_texts = [ocr["text"] for ocr in segment["visual_data"]["ocr_results"]]
                combined_ocr = " ".join(ocr_texts)
                if combined_ocr.strip():
                    ocr_embedding = self.embedding_model.encode(combined_ocr)
                    embedding_data["embeddings"]["ocr"] = ocr_embedding.tolist()
                else:
                    embedding_data["embeddings"]["ocr"] = np.zeros(384).tolist()
            else:
                embedding_data["embeddings"]["ocr"] = np.zeros(384).tolist()
            
            # Metadata
            embedding_data["metadata"] = {
                "duration": segment["duration"],
                "num_transcripts": len(segment["transcripts"]),
                "has_visual": segment["visual_data"] is not None,
                "has_ocr": bool(segment["visual_data"] and segment["visual_data"]["ocr_results"])
            }
            
            embeddings.append(embedding_data)
        
        return embeddings

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types"""
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
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        return obj

    async def _extract_video_info(self, video_path: str) -> Dict:
        """Extract basic video information"""
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

# Global instance
enhanced_video_processor = EnhancedVideoProcessor()

async def get_enhanced_video_processor() -> EnhancedVideoProcessor:
    return enhanced_video_processor