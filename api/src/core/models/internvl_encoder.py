# src/core/models/internvl_encoder.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image
import cv2
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

@dataclass
class InternVLOutput:
    """Output from InternVL unified encoding"""
    text_embeddings: Optional[np.ndarray] = None
    visual_embeddings: Optional[np.ndarray] = None
    unified_embedding: Optional[np.ndarray] = None
    confidence_scores: Optional[Dict[str, float]] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class InternVLUnifiedEncoder:
    """
    InternVL 2.0 Unified Multimodal Encoder
    
    Integrates with your existing VideoRAG system to provide:
    - Unified text + visual understanding
    - Superior cross-modal alignment
    - Enhanced video segment encoding
    """
    
    def __init__(self, model_name: str = "OpenGVLab/InternVL-Chat-V1-5", device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Configuration
        self.max_text_length = 512
        self.max_visual_tokens = 256
        self.batch_size = 8
        
        # Performance tracking
        self.encoding_stats = {
            "total_encodings": 0,
            "avg_processing_time": 0.0,
            "error_count": 0
        }
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.initialized = False
        
    def _setup_device(self, device: str) -> str:
        """Setup optimal device for InternVL"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    async def initialize(self):
        """Initialize InternVL model components"""
        if self.initialized:
            return
            
        try:
            logger.info(f"🔄 Loading InternVL model: {self.model_name}")
            start_time = time.time()
            
            # Load in separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Load tokenizer
            self.tokenizer = await loop.run_in_executor(
                self.executor,
                self._load_tokenizer
            )
            
            # Load processor  
            self.processor = await loop.run_in_executor(
                self.executor,
                self._load_processor
            )
            
            # Load main model
            self.model = await loop.run_in_executor(
                self.executor,
                self._load_model
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ InternVL loaded in {load_time:.2f}s on {self.device}")
            
            # Warm up with dummy data
            await self._warmup_model()
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize InternVL: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load tokenizer in thread"""
        return AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False
        )
    
    def _load_processor(self):
        """Load image processor in thread"""
        return AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
    
    def _load_model(self):
        """Load main model in thread"""
        model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        model.eval()
        return model
    
    async def _warmup_model(self):
        """Warm up model with dummy inputs"""
        try:
            dummy_text = "This is a test video segment"
            dummy_image = Image.new('RGB', (224, 224), color='red')
            
            await self.encode_unified(
                text=dummy_text,
                images=[dummy_image],
                return_attention=False
            )
            logger.info("✅ InternVL model warmed up successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed: {e}")
    
    async def encode_unified(
        self, 
        text: str, 
        images: Optional[List[Image.Image]] = None,
        return_attention: bool = False,
        normalize: bool = True
    ) -> InternVLOutput:
        """
        Unified multimodal encoding using InternVL
        
        Args:
            text: Text content to encode
            images: List of PIL images (video frames)
            return_attention: Whether to return attention weights
            normalize: Whether to normalize embeddings
            
        Returns:
            InternVLOutput with unified embeddings
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepare inputs
            inputs = await self._prepare_inputs(text, images)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
                
            # Extract embeddings
            result = await self._extract_embeddings(outputs, normalize)
            
            # Update stats
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            result.processing_time = processing_time
            result.metadata = {
                "model_name": self.model_name,
                "device": self.device,
                "input_text_length": len(text),
                "num_images": len(images) if images else 0
            }
            
            return result
            
        except Exception as e:
            self.encoding_stats["error_count"] += 1
            logger.error(f"❌ InternVL encoding failed: {e}")
            
            # Return fallback embeddings
            return InternVLOutput(
                unified_embedding=np.zeros(768),  # Fallback dimension
                confidence_scores={"unified": 0.0},
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def _prepare_inputs(
        self, 
        text: str, 
        images: Optional[List[Image.Image]] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for InternVL model"""
        
        # Tokenize text
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length
        )
        
        inputs = {
            "input_ids": text_inputs["input_ids"].to(self.device),
            "attention_mask": text_inputs["attention_mask"].to(self.device)
        }
        
        # Process images if provided
        if images:
            # Limit number of images for memory efficiency
            images = images[:8]  # Max 8 frames
            
            image_inputs = self.processor(
                images=images,
                return_tensors="pt"
            )
            
            inputs["pixel_values"] = image_inputs["pixel_values"].to(self.device)
        
        return inputs
    
    async def _extract_embeddings(
        self, 
        outputs, 
        normalize: bool = True
    ) -> InternVLOutput:
        """Extract and process embeddings from model outputs"""
        
        # Get the unified representation (usually from last hidden state)
        if hasattr(outputs, 'last_hidden_state'):
            # Pool the sequence dimension (mean pooling)
            unified_embedding = outputs.last_hidden_state.mean(dim=1)
        elif hasattr(outputs, 'pooler_output'):
            unified_embedding = outputs.pooler_output
        else:
            # Fallback: use the first token embedding
            unified_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Convert to numpy
        unified_embedding = unified_embedding.cpu().numpy().squeeze()
        
        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(unified_embedding)
            if norm > 0:
                unified_embedding = unified_embedding / norm
        
        # Calculate confidence (simplified)
        confidence = float(np.mean(np.abs(unified_embedding)))
        
        return InternVLOutput(
            unified_embedding=unified_embedding,
            confidence_scores={"unified": confidence}
        )
    
    async def encode_video_segment(
        self, 
        segment_data: Dict[str, Any]
    ) -> InternVLOutput:
        """
        Enhanced encoding for video segments using InternVL
        
        Args:
            segment_data: Dictionary containing:
                - text: Transcript text
                - images: List of video frames (PIL Images)
                - metadata: Additional segment info
        """
        
        text = segment_data.get("text", "")
        images = segment_data.get("images", [])
        metadata = segment_data.get("metadata", {})
        
        # Create enhanced prompt for better understanding
        enhanced_text = self._enhance_text_prompt(text, metadata)
        
        # Encode using unified method
        result = await self.encode_unified(
            text=enhanced_text,
            images=images,
            normalize=True
        )
        
        # Add segment-specific metadata
        if result.metadata:
            result.metadata.update({
                "segment_id": metadata.get("segment_id"),
                "scene_id": metadata.get("scene_id"),
                "temporal_position": metadata.get("start", 0)
            })
        
        return result
    
    def _enhance_text_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Enhance text with contextual information for better encoding"""
        
        enhanced_parts = []
        
        # Add temporal context
        if "start" in metadata:
            start_time = metadata["start"]
            enhanced_parts.append(f"At {start_time:.1f} seconds:")
        
        # Add scene context
        if "scene_id" in metadata:
            enhanced_parts.append(f"Scene {metadata['scene_id']}:")
        
        # Add the main text
        enhanced_parts.append(text)
        
        # Add OCR context if available
        if "ocr_texts" in metadata and metadata["ocr_texts"]:
            ocr_text = " ".join(metadata["ocr_texts"])
            enhanced_parts.append(f"Visible text: {ocr_text}")
        
        return " ".join(enhanced_parts)
    
    async def batch_encode_segments(
        self, 
        segments: List[Dict[str, Any]]
    ) -> List[InternVLOutput]:
        """Batch encode multiple segments efficiently"""
        
        results = []
        
        # Process in batches to manage memory
        for i in range(0, len(segments), self.batch_size):
            batch = segments[i:i + self.batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.encode_video_segment(segment) 
                for segment in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle any exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Batch encoding failed for segment {i+j}: {result}")
                    # Create fallback result
                    result = InternVLOutput(
                        unified_embedding=np.zeros(768),
                        confidence_scores={"unified": 0.0}
                    )
                
                results.append(result)
        
        return results
    
    def _update_stats(self, processing_time: float):
        """Update performance statistics"""
        self.encoding_stats["total_encodings"] += 1
        
        # Update running average
        count = self.encoding_stats["total_encodings"]
        current_avg = self.encoding_stats["avg_processing_time"]
        
        self.encoding_stats["avg_processing_time"] = (
            (current_avg * (count - 1) + processing_time) / count
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoding performance statistics"""
        return {
            **self.encoding_stats,
            "model_name": self.model_name,
            "device": self.device,
            "initialized": self.initialized
        }
    
    async def close(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.model:
            del self.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("🔄 InternVL encoder closed")

# Global instance
internvl_encoder = None

async def get_internvl_encoder() -> InternVLUnifiedEncoder:
    """Get singleton InternVL encoder instance"""
    global internvl_encoder
    if internvl_encoder is None:
        internvl_encoder = InternVLUnifiedEncoder()
        await internvl_encoder.initialize()
    return internvl_encoder