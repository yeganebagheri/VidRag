import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class HierarchicalVectorSearch:
    def __init__(self):
        self.embedding_model = None
        self.video_embeddings: Dict[str, List[Dict]] = {}
        self.scene_embeddings: Dict[str, List[Dict]] = {}
        
        # FAISS indices for different levels and modalities
        self.segment_indices = {
            "text": None,
            "visual": None, 
            "ocr": None,
            "multimodal": None
        }
        self.scene_indices = {
            "text": None,
            "visual": None,
            "multimodal": None
        }
        
        # Metadata storage
        self.segment_metadata: Dict[str, List[Dict]] = {}
        self.scene_metadata: Dict[str, List[Dict]] = {}
        
        # Adaptive weights
        self.default_weights = {"text": 0.4, "visual": 0.3, "ocr": 0.3}
        
    async def initialize(self):
        """Initialize search components"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Enhanced vector search initialized")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced vector search: {e}")
    
    async def index_video(self, video_id: str, processed_data: Dict):
        """Index video with hierarchical structure"""
        try:
            # Extract hierarchical segments and embeddings
            hierarchical_segments = processed_data.get("hierarchical_segments", [])
            embeddings_data = processed_data.get("embeddings", [])
            scene_boundaries = processed_data.get("scene_boundaries", [])
            
            if not hierarchical_segments or not embeddings_data:
                logger.warning(f"No hierarchical data found for video {video_id}")
                return
            
            # Index segment-level embeddings
            await self._index_segments(video_id, hierarchical_segments, embeddings_data)
            
            # Index scene-level embeddings
            await self._index_scenes(video_id, scene_boundaries, hierarchical_segments, embeddings_data)
            
            logger.info(f"Successfully indexed video {video_id} with {len(hierarchical_segments)} segments and {len(scene_boundaries)} scenes")
            
        except Exception as e:
            logger.error(f"Failed to index video {video_id}: {e}")
    
    async def _index_segments(self, video_id: str, segments: List[Dict], embeddings_data: List[Dict]):
        """Index segment-level embeddings"""
        if video_id not in self.video_embeddings:
            self.video_embeddings[video_id] = []
            self.segment_metadata[video_id] = []
        
        for i, (segment, emb_data) in enumerate(zip(segments, embeddings_data)):
            # Store segment embedding data
            segment_embedding = {
                "segment_id": segment["segment_id"],
                "scene_id": segment["scene_id"],
                "text_embedding": emb_data["embeddings"]["text"],
                "visual_embedding": emb_data["embeddings"]["visual"],
                "ocr_embedding": emb_data["embeddings"]["ocr"],
                "timestamp": segment["start"],
                "duration": segment["duration"]
            }
            
            # Store metadata
            metadata = {
                "video_id": video_id,
                "segment_id": segment["segment_id"],
                "scene_id": segment["scene_id"],
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["duration"],
                "has_visual": emb_data["metadata"]["has_visual"],
                "has_ocr": emb_data["metadata"]["has_ocr"],
                "num_transcripts": emb_data["metadata"]["num_transcripts"]
            }
            
            self.video_embeddings[video_id].append(segment_embedding)
            self.segment_metadata[video_id].append(metadata)
        
        # Rebuild FAISS indices
        await self._rebuild_segment_indices()
    
    async def _index_scenes(self, video_id: str, scenes: List[Dict], segments: List[Dict], embeddings_data: List[Dict]):
        """Index scene-level embeddings by aggregating segments"""
        if video_id not in self.scene_embeddings:
            self.scene_embeddings[video_id] = []
            self.scene_metadata[video_id] = []
        
        for scene in scenes:
            scene_id = scene["scene_id"]
            
            # Find all segments in this scene
            scene_segments = [s for s in segments if s["scene_id"] == scene_id]
            scene_embeddings_data = [e for e in embeddings_data if e["scene_id"] == scene_id]
            
            if not scene_segments:
                continue
            
            # Aggregate embeddings for the scene
            text_embeddings = [e["embeddings"]["text"] for e in scene_embeddings_data]
            visual_embeddings = [e["embeddings"]["visual"] for e in scene_embeddings_data]
            ocr_embeddings = [e["embeddings"]["ocr"] for e in scene_embeddings_data]
            
            # Average embeddings
            avg_text_emb = np.mean(text_embeddings, axis=0) if text_embeddings else np.zeros(384)
            avg_visual_emb = np.mean(visual_embeddings, axis=0) if visual_embeddings else np.zeros(512)
            avg_ocr_emb = np.mean(ocr_embeddings, axis=0) if ocr_embeddings else np.zeros(384)
            
            scene_embedding = {
                "scene_id": scene_id,
                "text_embedding": avg_text_emb.tolist(),
                "visual_embedding": avg_visual_emb.tolist(),
                "ocr_embedding": avg_ocr_emb.tolist(),
                "start": scene["start"],
                "end": scene["end"]
            }
            
            # Scene metadata
            scene_text = " ".join(s["text"] for s in scene_segments if s["text"])
            metadata = {
                "video_id": video_id,
                "scene_id": scene_id,
                "text": scene_text,
                "start": scene["start"],
                "end": scene["end"],
                "num_segments": len(scene_segments),
                "duration": scene["end"] - scene["start"]
            }
            
            self.scene_embeddings[video_id].append(scene_embedding)
            self.scene_metadata[video_id].append(metadata)
        
        # Rebuild scene indices
        await self._rebuild_scene_indices()
    
    async def _rebuild_segment_indices(self):
        """Rebuild FAISS indices for segments"""
        all_text_embeddings = []
        all_visual_embeddings = []
        all_ocr_embeddings = []
        
        for video_id, embeddings in self.video_embeddings.items():
            for emb in embeddings:
                all_text_embeddings.append(emb["text_embedding"])
                all_visual_embeddings.append(emb["visual_embedding"])
                all_ocr_embeddings.append(emb["ocr_embedding"])
        
        if not all_text_embeddings:
            return
        
        # Build indices
        text_matrix = np.array(all_text_embeddings).astype('float32')
        visual_matrix = np.array(all_visual_embeddings).astype('float32')
        ocr_matrix = np.array(all_ocr_embeddings).astype('float32')
        
        # Create FAISS indices
        self.segment_indices["text"] = faiss.IndexFlatIP(text_matrix.shape[1])
        self.segment_indices["visual"] = faiss.IndexFlatIP(visual_matrix.shape[1])
        self.segment_indices["ocr"] = faiss.IndexFlatIP(ocr_matrix.shape[1])
        
        # Add embeddings to indices
        self.segment_indices["text"].add(text_matrix)
        self.segment_indices["visual"].add(visual_matrix)
        self.segment_indices["ocr"].add(ocr_matrix)
        
        # Create multimodal index by concatenating embeddings
        multimodal_matrix = np.concatenate([text_matrix, visual_matrix, ocr_matrix], axis=1)
        self.segment_indices["multimodal"] = faiss.IndexFlatIP(multimodal_matrix.shape[1])
        self.segment_indices["multimodal"].add(multimodal_matrix)
    
    async def _rebuild_scene_indices(self):
        """Rebuild FAISS indices for scenes"""
        all_text_embeddings = []
        all_visual_embeddings = []
        all_ocr_embeddings = []
        
        for video_id, embeddings in self.scene_embeddings.items():
            for emb in embeddings:
                all_text_embeddings.append(emb["text_embedding"])
                all_visual_embeddings.append(emb["visual_embedding"])
                all_ocr_embeddings.append(emb["ocr_embedding"])
        
        if not all_text_embeddings:
            return
        
        # Build scene indices
        text_matrix = np.array(all_text_embeddings).astype('float32')
        visual_matrix = np.array(all_visual_embeddings).astype('float32')
        ocr_matrix = np.array(all_ocr_embeddings).astype('float32')
        
        self.scene_indices["text"] = faiss.IndexFlatIP(text_matrix.shape[1])
        self.scene_indices["visual"] = faiss.IndexFlatIP(visual_matrix.shape[1])
        
        self.scene_indices["text"].add(text_matrix)
        self.scene_indices["visual"].add(visual_matrix)
        
        # Multimodal scene index
        multimodal_matrix = np.concatenate([text_matrix, visual_matrix, ocr_matrix], axis=1)
        self.scene_indices["multimodal"] = faiss.IndexFlatIP(multimodal_matrix.shape[1])
        self.scene_indices["multimodal"].add(multimodal_matrix)
    
    async def hierarchical_search(self, query: str, top_k: int = 10, 
                                search_level: str = "both") -> List[Dict]:
        """
        Hierarchical search starting from scenes then drilling down to segments
        
        Args:
            query: Search query
            top_k: Number of results to return
            search_level: "scenes", "segments", or "both"
        """
        if not self.embedding_model:
            return []
        
        results = []
        
        try:
            # Generate query embeddings
            query_text_emb = self.embedding_model.encode(query).astype('float32')
            
            if search_level in ["scenes", "both"]:
                scene_results = await self._search_scenes(query_text_emb, top_k)
                
                if search_level == "both":
                    # For each relevant scene, find the best segments
                    for scene_result in scene_results[:5]:  # Top 5 scenes
                        scene_segments = await self._search_segments_in_scene(
                            query_text_emb, scene_result["scene_id"], 3
                        )
                        results.extend(scene_segments)
                else:
                    results = scene_results
            
            if search_level == "segments":
                results = await self._search_segments(query_text_emb, top_k)
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Hierarchical search failed: {e}")
            return []
    
    async def _search_scenes(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """Search at scene level"""
        if not self.scene_indices["text"]:
            return []
        
        # Search in scene text index
        scores, indices = self.scene_indices["text"].search(
            query_embedding.reshape(1, -1), k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # Invalid index
                continue
                
            # Find corresponding scene metadata
            scene_meta = self._get_scene_metadata_by_index(idx)
            if scene_meta:
                results.append({
                    "type": "scene",
                    "scene_id": scene_meta["scene_id"],
                    "video_id": scene_meta["video_id"],
                    "text": scene_meta["text"],
                    "start": scene_meta["start"],
                    "end": scene_meta["end"],
                    "score": float(score),
                    "num_segments": scene_meta["num_segments"]
                })
        
        return results
    
    async def _search_segments(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """Search at segment level"""
        if not self.segment_indices["text"]:
            return []
        
        scores, indices = self.segment_indices["text"].search(
            query_embedding.reshape(1, -1), k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
                
            segment_meta = self._get_segment_metadata_by_index(idx)
            if segment_meta:
                results.append({
                    "type": "segment",
                    "segment_id": segment_meta["segment_id"],
                    "scene_id": segment_meta["scene_id"],
                    "video_id": segment_meta["video_id"],
                    "text": segment_meta["text"],
                    "start": segment_meta["start"],
                    "end": segment_meta["end"],
                    "score": float(score),
                    "duration": segment_meta["duration"]
                })
        
        return results
    
    async def _search_segments_in_scene(self, query_embedding: np.ndarray, 
                                      scene_id: str, k: int) -> List[Dict]:
        """Search segments within a specific scene"""
        if not self.segment_indices["text"]:
            return []
        
        # Get all segments in the scene
        scene_segment_indices = []
        current_idx = 0
        
        for video_id, metadata_list in self.segment_metadata.items():
            for meta in metadata_list:
                if meta["scene_id"] == scene_id:
                    scene_segment_indices.append(current_idx)
                current_idx += 1
        
        if not scene_segment_indices:
            return []
        
        # Search among scene segments
        all_scores, all_indices = self.segment_indices["text"].search(
            query_embedding.reshape(1, -1), len(scene_segment_indices) * 2
        )
        
        results = []
        found_count = 0
        
        for score, idx in zip(all_scores[0], all_indices[0]):
            if idx in scene_segment_indices and found_count < k:
                segment_meta = self._get_segment_metadata_by_index(idx)
                if segment_meta:
                    results.append({
                        "type": "segment",
                        "segment_id": segment_meta["segment_id"],
                        "scene_id": segment_meta["scene_id"],
                        "video_id": segment_meta["video_id"],
                        "text": segment_meta["text"],
                        "start": segment_meta["start"],
                        "end": segment_meta["end"],
                        "score": float(score),
                        "duration": segment_meta["duration"]
                    })
                    found_count += 1
        
        return results
    
    async def multimodal_search(self, query: str, modality_weights: Optional[Dict] = None, 
                              top_k: int = 10) -> List[Dict]:
        """Advanced multimodal search with adaptive weighting"""
        if not self.embedding_model:
            return []
        
        weights = modality_weights or self.default_weights
        
        # Analyze query to determine modality preference
        adaptive_weights = self._analyze_query_modality(query, weights)
        
        try:
            query_text_emb = self.embedding_model.encode(query).astype('float32')
            
            # Search in each modality
            text_results = await self._search_by_modality("text", query_text_emb, top_k * 2)
            visual_results = await self._search_by_modality("visual", query_text_emb, top_k * 2)
            ocr_results = await self._search_by_modality("ocr", query_text_emb, top_k * 2)
            
            # Combine and re-rank results
            combined_results = self._combine_multimodal_results(
                text_results, visual_results, ocr_results, adaptive_weights
            )
            
            # Add contextual ranking
            final_results = await self._contextual_ranking(combined_results, query)
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Multimodal search failed: {e}")
            return []
    
    async def _search_by_modality(self, modality: str, query_embedding: np.ndarray, 
                                k: int) -> List[Dict]:
        """Search by specific modality"""
        if modality not in self.segment_indices or not self.segment_indices[modality]:
            return []
        
        scores, indices = self.segment_indices[modality].search(
            query_embedding.reshape(1, -1), k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
                
            segment_meta = self._get_segment_metadata_by_index(idx)
            if segment_meta:
                result = segment_meta.copy()
                result["modality"] = modality
                result["modality_score"] = float(score)
                results.append(result)
        
        return results
    
    def _combine_multimodal_results(self, text_results: List[Dict], 
                                  visual_results: List[Dict], 
                                  ocr_results: List[Dict],
                                  weights: Dict[str, float]) -> List[Dict]:
        """Combine results from multiple modalities"""
        # Group results by segment_id
        segment_scores = defaultdict(lambda: {"scores": {}, "metadata": None})
        
        for result in text_results:
            seg_id = result["segment_id"]
            segment_scores[seg_id]["scores"]["text"] = result["modality_score"]
            segment_scores[seg_id]["metadata"] = result
        
        for result in visual_results:
            seg_id = result["segment_id"]
            segment_scores[seg_id]["scores"]["visual"] = result["modality_score"]
            if not segment_scores[seg_id]["metadata"]:
                segment_scores[seg_id]["metadata"] = result
        
        for result in ocr_results:
            seg_id = result["segment_id"]
            segment_scores[seg_id]["scores"]["ocr"] = result["modality_score"]
            if not segment_scores[seg_id]["metadata"]:
                segment_scores[seg_id]["metadata"] = result
        
        # Calculate combined scores
        combined_results = []
        for seg_id, data in segment_scores.items():
            if not data["metadata"]:
                continue
                
            scores = data["scores"]
            combined_score = (
                scores.get("text", 0) * weights["text"] +
                scores.get("visual", 0) * weights["visual"] +
                scores.get("ocr", 0) * weights["ocr"]
            )
            
            result = data["metadata"].copy()
            result["combined_score"] = combined_score
            result["modality_scores"] = scores
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return combined_results
    
    async def _contextual_ranking(self, results: List[Dict], query: str) -> List[Dict]:
        """Apply contextual ranking based on temporal coherence and semantic clusters"""
        if len(results) <= 1:
            return results
        
        # Add temporal coherence bonus
        for i, result in enumerate(results):
            temporal_bonus = 0
            
            # Check for adjacent high-scoring segments
            for j, other_result in enumerate(results):
                if i == j:
                    continue
                    
                time_diff = abs(result["start"] - other_result["start"])
                if time_diff < 30:  # Within 30 seconds
                    proximity_score = max(0, 1 - time_diff / 30)
                    temporal_bonus += proximity_score * other_result["combined_score"] * 0.1
            
            result["final_score"] = result["combined_score"] + temporal_bonus
        
        # Re-sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
    
    def _analyze_query_modality(self, query: str, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Analyze query to determine modality preferences"""
        query_lower = query.lower()
        
        # Visual indicators
        visual_keywords = ["show", "display", "see", "look", "visual", "image", "picture", "scene"]
        visual_count = sum(1 for kw in visual_keywords if kw in query_lower)
        
        # Text indicators
        text_keywords = ["say", "speak", "mention", "discuss", "talk", "explain", "describe"]
        text_count = sum(1 for kw in text_keywords if kw in query_lower)
        
        # OCR indicators
        ocr_keywords = ["text", "read", "written", "title", "subtitle", "caption", "sign"]
        ocr_count = sum(1 for kw in ocr_keywords if kw in query_lower)
        
        # Adjust weights based on keyword presence
        total_keywords = visual_count + text_count + ocr_count
        
        if total_keywords == 0:
            return base_weights
        
        # Calculate adaptive weights
        text_boost = 0.2 * (text_count / total_keywords)
        visual_boost = 0.2 * (visual_count / total_keywords)
        ocr_boost = 0.2 * (ocr_count / total_keywords)
        
        adaptive_weights = {
            "text": min(0.8, base_weights["text"] + text_boost),
            "visual": min(0.8, base_weights["visual"] + visual_boost),
            "ocr": min(0.8, base_weights["ocr"] + ocr_boost)
        }
        
        # Normalize weights
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            adaptive_weights = {k: v / total_weight for k, v in adaptive_weights.items()}
        
        return adaptive_weights
    
    def _get_segment_metadata_by_index(self, index: int) -> Optional[Dict]:
        """Get segment metadata by global index"""
        current_idx = 0
        for video_id, metadata_list in self.segment_metadata.items():
            if index < current_idx + len(metadata_list):
                return metadata_list[index - current_idx]
            current_idx += len(metadata_list)
        return None
    
    def _get_scene_metadata_by_index(self, index: int) -> Optional[Dict]:
        """Get scene metadata by global index"""
        current_idx = 0
        for video_id, metadata_list in self.scene_metadata.items():
            if index < current_idx + len(metadata_list):
                return metadata_list[index - current_idx]
            current_idx += len(metadata_list)
        return None
    
    async def get_video_structure(self, video_id: str) -> Dict:
        """Get hierarchical structure of a video"""
        if video_id not in self.video_embeddings:
            return {}
        
        structure = {
            "video_id": video_id,
            "scenes": [],
            "segments": self.segment_metadata.get(video_id, []),
            "total_segments": len(self.segment_metadata.get(video_id, [])),
            "total_scenes": len(self.scene_metadata.get(video_id, []))
        }
        
        # Group segments by scene
        scene_segments = defaultdict(list)
        for segment in structure["segments"]:
            scene_segments[segment["scene_id"]].append(segment)
        
        for scene_meta in self.scene_metadata.get(video_id, []):
            scene_data = scene_meta.copy()
            scene_data["segments"] = scene_segments[scene_meta["scene_id"]]
            structure["scenes"].append(scene_data)
        
        return structure

# Global instance
enhanced_vector_search = HierarchicalVectorSearch()

async def get_enhanced_vector_search() -> HierarchicalVectorSearch:
    return enhanced_vector_search