# src/core/search/internvl_enhanced_vector_search.py

import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json
from collections import defaultdict
import asyncio

# Import InternVL encoder
from src.core.models.internvl_encoder import get_internvl_encoder

logger = logging.getLogger(__name__)

class InternVLEnhancedVectorSearch:
    """
    Enhanced Vector Search with InternVL 2.0 Integration
    
    Key Improvements:
    1. InternVL unified search queries
    2. Cross-modal query understanding
    3. Enhanced retrieval accuracy
    4. Backward compatible with existing system
    """
    
    def __init__(self):
        # Existing components (keep for compatibility)
        self.embedding_model = None
        self.video_embeddings: Dict[str, List[Dict]] = {}
        self.scene_embeddings: Dict[str, List[Dict]] = {}
        
        # FAISS indices for different levels and modalities
        self.segment_indices = {
            "text": None,
            "visual": None, 
            "ocr": None,
            "multimodal": None,
            "unified": None  # NEW: InternVL unified index
        }
        self.scene_indices = {
            "text": None,
            "visual": None,
            "multimodal": None,
            "unified": None  # NEW: InternVL unified index
        }
        
        # Metadata storage
        self.segment_metadata: Dict[str, List[Dict]] = {}
        self.scene_metadata: Dict[str, List[Dict]] = {}
        
        # Adaptive weights
        self.default_weights = {"text": 0.4, "visual": 0.3, "ocr": 0.3}
        
        # NEW: InternVL components
        self.internvl_encoder = None
        self.use_internvl = True
        
        # Knowledge graph for enhanced retrieval
        self.knowledge_graphs: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize search components including InternVL"""
        try:
            # Initialize existing embedding model (keep for fallback)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded")
            
            # Initialize InternVL encoder
            if self.use_internvl:
                try:
                    self.internvl_encoder = await get_internvl_encoder()
                    logger.info("✅ InternVL encoder integrated with search")
                except Exception as e:
                    logger.error(f"❌ InternVL integration failed: {e}")
                    self.use_internvl = False
            
            logger.info("✅ Enhanced vector search initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced vector search: {e}")
    
    async def index_video(self, video_id: str, processed_data: Dict):
        """Enhanced video indexing with InternVL support"""
        try:
            # Extract hierarchical segments and embeddings
            hierarchical_segments = processed_data.get("hierarchical_segments", [])
            embeddings_data = processed_data.get("embeddings", [])
            scene_boundaries = processed_data.get("scene_boundaries", [])
            knowledge_graph = processed_data.get("knowledge_graph", {})
            
            if not hierarchical_segments or not embeddings_data:
                logger.warning(f"No hierarchical data found for video {video_id}")
                return
            
            # Store knowledge graph for enhanced retrieval
            self.knowledge_graphs[video_id] = knowledge_graph
            
            # Index segment-level embeddings with InternVL support
            await self._index_segments_enhanced(video_id, hierarchical_segments, embeddings_data)
            
            # Index scene-level embeddings
            await self._index_scenes_enhanced(video_id, scene_boundaries, hierarchical_segments, embeddings_data)
            
            logger.info(f"✅ Enhanced indexing completed for video {video_id}: {len(hierarchical_segments)} segments, {len(scene_boundaries)} scenes")
            
        except Exception as e:
            logger.error(f"❌ Failed to index video {video_id}: {e}")
    
    async def _index_segments_enhanced(self, video_id: str, segments: List[Dict], embeddings_data: List[Dict]):
        """Enhanced segment indexing with unified embeddings"""
        
        if video_id not in self.video_embeddings:
            self.video_embeddings[video_id] = []
            self.segment_metadata[video_id] = []
        
        for i, (segment, emb_data) in enumerate(zip(segments, embeddings_data)):
            # Store segment embedding data
            segment_embedding = {
                "segment_id": segment["segment_id"],
                "scene_id": segment["scene_id"],
                "timestamp": segment["start"],
                "duration": segment["duration"]
            }
            
            # Add existing embeddings
            embeddings = emb_data.get("embeddings", {})
            segment_embedding["text_embedding"] = embeddings.get("text", [])
            segment_embedding["visual_embedding"] = embeddings.get("visual", [])
            segment_embedding["ocr_embedding"] = embeddings.get("ocr", [])
            
            # NEW: Add unified InternVL embedding
            if "unified" in embeddings:
                segment_embedding["unified_embedding"] = embeddings["unified"]
                segment_embedding["internvl_enhanced"] = True
            else:
                # Create fallback unified embedding
                text_emb = np.array(embeddings.get("text", np.zeros(384)))
                visual_emb = np.array(embeddings.get("visual", np.zeros(512)))
                ocr_emb = np.array(embeddings.get("ocr", np.zeros(384)))
                
                # Simple concatenation for fallback
                unified_emb = np.concatenate([text_emb[:256], visual_emb[:256], ocr_emb[:256]])
                segment_embedding["unified_embedding"] = unified_emb.tolist()
                segment_embedding["internvl_enhanced"] = False
            
            # Store enhanced metadata
            metadata = {
                "video_id": video_id,
                "segment_id": segment["segment_id"],
                "scene_id": segment["scene_id"],
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["duration"],
                "internvl_enhanced": segment_embedding["internvl_enhanced"],
                "confidence": emb_data.get("confidence", 1.0)
            }
            
            # Add visual metadata
            if segment.get("visual_data"):
                visual_data = segment["visual_data"]
                metadata.update({
                    "has_visual": visual_data is not None,
                    "has_ocr": bool(visual_data.get("ocr_results")),
                    "num_frames": visual_data.get("num_frames", 0)
                })
            
            self.video_embeddings[video_id].append(segment_embedding)
            self.segment_metadata[video_id].append(metadata)
        
        # Rebuild enhanced FAISS indices
        await self._rebuild_segment_indices_enhanced()
    
    async def _rebuild_segment_indices_enhanced(self):
        """Rebuild FAISS indices with unified embeddings"""
        
        all_text_embeddings = []
        all_visual_embeddings = []
        all_ocr_embeddings = []
        all_unified_embeddings = []  # NEW
        
        for video_id, embeddings in self.video_embeddings.items():
            for emb in embeddings:
                all_text_embeddings.append(emb["text_embedding"])
                all_visual_embeddings.append(emb["visual_embedding"])
                all_ocr_embeddings.append(emb["ocr_embedding"])
                all_unified_embeddings.append(emb["unified_embedding"])  # NEW
        
        if not all_text_embeddings:
            return
        
        # Build existing indices
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
        
        # NEW: Create unified InternVL index
        unified_matrix = np.array(all_unified_embeddings).astype('float32')
        self.segment_indices["unified"] = faiss.IndexFlatIP(unified_matrix.shape[1])
        self.segment_indices["unified"].add(unified_matrix)
        
        logger.info(f"✅ Enhanced indices built: unified={unified_matrix.shape}")
    
    async def _index_scenes_enhanced(self, video_id: str, scenes: List[Dict], segments: List[Dict], embeddings_data: List[Dict]):
        """Enhanced scene indexing with unified embeddings"""
        
        if video_id not in self.scene_embeddings:
            self.scene_embeddings[video_id] = []
            self.scene_metadata[video_id] = []
        
        for scene in scenes:
            scene_id = scene["scene_id"]
            
            # Find all segments in this scene
            scene_segments = [s for s in segments if s["scene_id"] == scene_id]
            scene_embeddings_data = [e for e in embeddings_data if e.get("scene_id") == scene_id]
            
            if not scene_segments:
                continue
            
            # Aggregate embeddings for the scene
            text_embeddings = []
            visual_embeddings = []
            ocr_embeddings = []
            unified_embeddings = []  # NEW
            
            for emb_data in scene_embeddings_data:
                embeddings = emb_data.get("embeddings", {})
                if "text" in embeddings:
                    text_embeddings.append(embeddings["text"])
                if "visual" in embeddings:
                    visual_embeddings.append(embeddings["visual"])
                if "ocr" in embeddings:
                    ocr_embeddings.append(embeddings["ocr"])
                if "unified" in embeddings:  # NEW
                    unified_embeddings.append(embeddings["unified"])
            
            # Average embeddings
            avg_text_emb = np.mean(text_embeddings, axis=0) if text_embeddings else np.zeros(384)
            avg_visual_emb = np.mean(visual_embeddings, axis=0) if visual_embeddings else np.zeros(512)
            avg_ocr_emb = np.mean(ocr_embeddings, axis=0) if ocr_embeddings else np.zeros(384)
            
            # NEW: Average unified embeddings
            if unified_embeddings:
                avg_unified_emb = np.mean(unified_embeddings, axis=0)
                scene_has_internvl = True
            else:
                # Create fallback unified embedding
                avg_unified_emb = np.concatenate([avg_text_emb[:256], avg_visual_emb[:256], avg_ocr_emb[:256]])
                scene_has_internvl = False
            
            scene_embedding = {
                "scene_id": scene_id,
                "text_embedding": avg_text_emb.tolist(),
                "visual_embedding": avg_visual_emb.tolist(),
                "ocr_embedding": avg_ocr_emb.tolist(),
                "unified_embedding": avg_unified_emb.tolist(),  # NEW
                "start": scene["start"],
                "end": scene["end"],
                "internvl_enhanced": scene_has_internvl  # NEW
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
                "duration": scene["end"] - scene["start"],
                "internvl_enhanced": scene_has_internvl  # NEW
            }
            
            self.scene_embeddings[video_id].append(scene_embedding)
            self.scene_metadata[video_id].append(metadata)
        
        # Rebuild scene indices
        await self._rebuild_scene_indices_enhanced()
    
    async def _rebuild_scene_indices_enhanced(self):
        """Rebuild FAISS indices for scenes with unified support"""
        
        all_text_embeddings = []
        all_visual_embeddings = []
        all_ocr_embeddings = []
        all_unified_embeddings = []  # NEW
        
        for video_id, embeddings in self.scene_embeddings.items():
            for emb in embeddings:
                all_text_embeddings.append(emb["text_embedding"])
                all_visual_embeddings.append(emb["visual_embedding"])
                all_ocr_embeddings.append(emb["ocr_embedding"])
                all_unified_embeddings.append(emb["unified_embedding"])  # NEW
        
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
        
        # NEW: Unified scene index
        unified_matrix = np.array(all_unified_embeddings).astype('float32')
        self.scene_indices["unified"] = faiss.IndexFlatIP(unified_matrix.shape[1])
        self.scene_indices["unified"].add(unified_matrix)
    
    async def enhanced_search(
        self, 
        query: str, 
        top_k: int = 10, 
        search_level: str = "both",
        use_knowledge_enhancement: bool = True
    ) -> List[Dict]:
        """
        Enhanced search with InternVL and Video-RAG capabilities
        
        Args:
            query: Search query
            top_k: Number of results
            search_level: "scenes", "segments", or "both"
            use_knowledge_enhancement: Enable knowledge graph enhancement
        """
        
        if not self.embedding_model:
            return []
        
        # Phase 1: Enhanced Query Encoding
        query_embeddings = await self._encode_query_enhanced(query)
        
        # Phase 2: Multi-modal Retrieval
        retrieval_results = await self._multi_modal_retrieval(
            query_embeddings, top_k * 3, search_level
        )
        
        # Phase 3: Knowledge-Enhanced Re-ranking
        if use_knowledge_enhancement:
            enhanced_results = await self._knowledge_enhanced_reranking(
                query, retrieval_results, top_k
            )
        else:
            enhanced_results = retrieval_results[:top_k]
        
        # Phase 4: Temporal Context Enhancement
        final_results = await self._temporal_context_enhancement(enhanced_results)
        
        return final_results[:top_k]
    
    async def _encode_query_enhanced(self, query: str) -> Dict[str, np.ndarray]:
        """Enhanced query encoding using InternVL and fallback methods"""
        
        query_embeddings = {}
        
        # Method 1: InternVL unified encoding (primary)
        if self.use_internvl and self.internvl_encoder:
            try:
                internvl_output = await self.internvl_encoder.encode_unified(
                    text=f"Search query: {query}",
                    normalize=True
                )
                
                if internvl_output.unified_embedding is not None:
                    query_embeddings["unified"] = internvl_output.unified_embedding
                    query_embeddings["confidence"] = internvl_output.confidence_scores.get("unified", 1.0)
                    
            except Exception as e:
                logger.warning(f"⚠️ InternVL query encoding failed: {e}")
        
        # Method 2: Traditional embeddings (fallback/compatibility)
        if self.embedding_model:
            traditional_emb = self.embedding_model.encode(query).astype('float32')
            query_embeddings["text"] = traditional_emb
            
            # Create fallback unified if InternVL failed
            if "unified" not in query_embeddings:
                # Simple approach: repeat text embedding to match unified dimension
                unified_fallback = np.concatenate([traditional_emb[:256], traditional_emb[:256], traditional_emb[:256]])
                query_embeddings["unified"] = unified_fallback
                query_embeddings["confidence"] = 0.7
        
        return query_embeddings
    
    async def _multi_modal_retrieval(
        self, 
        query_embeddings: Dict[str, np.ndarray], 
        k: int, 
        search_level: str
    ) -> List[Dict]:
        """Multi-modal retrieval with InternVL prioritization"""
        
        all_results = []
        
        # Priority 1: InternVL unified search (best results)
        if "unified" in query_embeddings and self.segment_indices.get("unified"):
            unified_results = await self._search_unified_index(
                query_embeddings["unified"], k, search_level
            )
            
            # Mark as high-confidence InternVL results
            for result in unified_results:
                result["search_method"] = "internvl_unified"
                result["base_confidence"] = query_embeddings.get("confidence", 1.0)
                result["score"] = result["score"] * 1.2  # Boost InternVL results
            
            all_results.extend(unified_results)
        
        # Priority 2: Traditional multimodal search (fallback)
        if len(all_results) < k and "text" in query_embeddings:
            traditional_results = await self._search_traditional_multimodal(
                query_embeddings["text"], k - len(all_results), search_level
            )
            
            # Mark as traditional results
            for result in traditional_results:
                result["search_method"] = "traditional_multimodal"
                result["base_confidence"] = 0.8
            
            all_results.extend(traditional_results)
        
        # Remove duplicates and sort by score
        unique_results = self._deduplicate_results(all_results)
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        
        return unique_results[:k]
    
    async def _search_unified_index(
        self, 
        query_embedding: np.ndarray, 
        k: int, 
        search_level: str
    ) -> List[Dict]:
        """Search using InternVL unified index"""
        
        results = []
        
        # Search segments
        if search_level in ["segments", "both"] and self.segment_indices.get("unified"):
            segment_results = await self._search_segments_unified(query_embedding, k)
            results.extend(segment_results)
        
        # Search scenes
        if search_level in ["scenes", "both"] and self.scene_indices.get("unified"):
            scene_results = await self._search_scenes_unified(query_embedding, k)
            results.extend(scene_results)
        
        return results
    
    async def _search_segments_unified(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """Search segments using unified InternVL index"""
        
        if not self.segment_indices.get("unified"):
            return []
        
        scores, indices = self.segment_indices["unified"].search(
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
                    "duration": segment_meta["duration"],
                    "internvl_enhanced": segment_meta.get("internvl_enhanced", False),
                    "confidence": segment_meta.get("confidence", 1.0)
                })
        
        return results
    
    async def _search_scenes_unified(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """Search scenes using unified InternVL index"""
        
        if not self.scene_indices.get("unified"):
            return []
        
        scores, indices = self.scene_indices["unified"].search(
            query_embedding.reshape(1, -1), k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
                
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
                    "num_segments": scene_meta["num_segments"],
                    "internvl_enhanced": scene_meta.get("internvl_enhanced", False)
                })
        
        return results
    
    async def _search_traditional_multimodal(
        self, 
        query_embedding: np.ndarray, 
        k: int, 
        search_level: str
    ) -> List[Dict]:
        """Traditional multimodal search (fallback method)"""
        
        results = []
        
        if search_level in ["segments", "both"]:
            segment_results = await self._search_segments_traditional(query_embedding, k)
            results.extend(segment_results)
        
        if search_level in ["scenes", "both"]:
            scene_results = await self._search_scenes_traditional(query_embedding, k)
            results.extend(scene_results)
        
        return results
    
    async def _search_segments_traditional(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """Traditional segment search"""
        
        if not self.segment_indices.get("text"):
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
    
    async def _search_scenes_traditional(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """Traditional scene search"""
        
        if not self.scene_indices.get("text"):
            return []
        
        scores, indices = self.scene_indices["text"].search(
            query_embedding.reshape(1, -1), k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
                
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
    
    async def _knowledge_enhanced_reranking(
        self, 
        query: str, 
        results: List[Dict], 
        top_k: int
    ) -> List[Dict]:
        """Video-RAG: Knowledge-enhanced re-ranking using extracted knowledge graphs"""
        
        if not results:
            return results
        
        enhanced_results = []
        
        for result in results:
            video_id = result["video_id"]
            knowledge_graph = self.knowledge_graphs.get(video_id, {})
            
            # Calculate knowledge relevance score
            knowledge_score = self._calculate_knowledge_relevance(query, knowledge_graph, result)
            
            # Calculate temporal context score
            temporal_score = self._calculate_temporal_context_score(result, results)
            
            # Combine scores with weights
            original_score = result["score"]
            base_confidence = result.get("base_confidence", 1.0)
            
            # Enhanced scoring formula
            enhanced_score = (
                original_score * 0.6 +           # Original similarity
                knowledge_score * 0.25 +         # Knowledge graph relevance
                temporal_score * 0.15            # Temporal context
            ) * base_confidence                   # Confidence multiplier
            
            result["enhanced_score"] = enhanced_score
            result["knowledge_score"] = knowledge_score
            result["temporal_score"] = temporal_score
            
            enhanced_results.append(result)
        
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x["enhanced_score"], reverse=True)
        
        return enhanced_results[:top_k]
    
    def _calculate_knowledge_relevance(
        self, 
        query: str, 
        knowledge_graph: Dict, 
        result: Dict
    ) -> float:
        """Calculate relevance based on knowledge graph entities and relationships"""
        
        if not knowledge_graph.get("entities"):
            return 0.0
        
        query_lower = query.lower()
        relevance_score = 0.0
        
        # Check entity matches
        for entity_type, entities in knowledge_graph["entities"].items():
            for entity in entities:
                entity_text = entity["text"].lower()
                if entity_text in query_lower or any(word in entity_text for word in query_lower.split()):
                    # Entity type weights
                    type_weights = {
                        "PERSON": 0.8,
                        "ORG": 0.7,
                        "GPE": 0.6,  # Geopolitical entity
                        "EVENT": 0.9,
                        "PRODUCT": 0.7
                    }
                    weight = type_weights.get(entity_type, 0.5)
                    relevance_score += weight * entity.get("confidence", 1.0)
        
        # Check relationship relevance
        relationships = knowledge_graph.get("relationships", [])
        for rel in relationships:
            rel_text = f"{rel['subject']} {rel['relation']} {rel['object']}".lower()
            if any(word in rel_text for word in query_lower.split()):
                relevance_score += 0.3
        
        # Normalize score
        return min(1.0, relevance_score / 2.0)
    
    def _calculate_temporal_context_score(self, result: Dict, all_results: List[Dict]) -> float:
        """Calculate temporal context score based on nearby relevant segments"""
        
        video_id = result["video_id"]
        result_start = result["start"]
        
        # Find nearby segments from same video
        nearby_segments = [
            r for r in all_results 
            if r["video_id"] == video_id and 
               abs(r["start"] - result_start) < 60 and  # Within 60 seconds
               r["segment_id"] != result.get("segment_id")
        ]
        
        if not nearby_segments:
            return 0.0
        
        # Calculate context score based on nearby segment relevance
        context_score = 0.0
        for nearby in nearby_segments:
            time_diff = abs(nearby["start"] - result_start)
            proximity_weight = max(0, 1 - (time_diff / 60))  # Closer = higher weight
            context_score += nearby["score"] * proximity_weight * 0.1
        
        return min(1.0, context_score)
    
    async def _temporal_context_enhancement(self, results: List[Dict]) -> List[Dict]:
        """Enhanced temporal context processing"""
        
        if len(results) <= 1:
            return results
        
        # Group results by video for temporal analysis
        video_groups = defaultdict(list)
        for result in results:
            video_groups[result["video_id"]].append(result)
        
        enhanced_results = []
        
        for video_id, video_results in video_groups.items():
            # Sort by timestamp
            video_results.sort(key=lambda x: x["start"])
            
            # Add temporal coherence bonuses
            for i, result in enumerate(video_results):
                temporal_bonus = 0.0
                
                # Check for narrative flow (consecutive segments)
                if i > 0:
                    prev_result = video_results[i-1]
                    time_gap = result["start"] - prev_result["end"]
                    if time_gap < 10:  # Within 10 seconds
                        temporal_bonus += 0.05
                
                if i < len(video_results) - 1:
                    next_result = video_results[i+1]
                    time_gap = next_result["start"] - result["end"]
                    if time_gap < 10:
                        temporal_bonus += 0.05
                
                # Apply temporal bonus
                if "enhanced_score" in result:
                    result["enhanced_score"] += temporal_bonus
                else:
                    result["score"] += temporal_bonus
                
                result["temporal_bonus"] = temporal_bonus
                enhanced_results.append(result)
        
        # Final sort
        sort_key = "enhanced_score" if enhanced_results and "enhanced_score" in enhanced_results[0] else "score"
        enhanced_results.sort(key=lambda x: x[sort_key], reverse=True)
        
        return enhanced_results
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on segment/scene ID"""
        
        seen_ids = set()
        unique_results = []
        
        for result in results:
            result_id = result.get("segment_id") or result.get("scene_id")
            if result_id and result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        return unique_results
    
    # === EXISTING METHODS (keep for compatibility) ===
    
    async def hierarchical_search(self, query: str, top_k: int = 10, 
                                search_level: str = "both") -> List[Dict]:
        """Backward compatible hierarchical search (enhanced internally)"""
        return await self.enhanced_search(query, top_k, search_level)
    
    async def multimodal_search(self, query: str, modality_weights: Optional[Dict] = None, 
                              top_k: int = 10) -> List[Dict]:
        """Enhanced multimodal search"""
        # Use enhanced search but focus on unified results
        results = await self.enhanced_search(query, top_k, "segments")
        
        # Add modality information for backward compatibility
        for result in results:
            result["modality_scores"] = {
                "unified": result.get("enhanced_score", result["score"]),
                "confidence": result.get("base_confidence", 1.0)
            }
        
        return results
    
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
        """Get hierarchical structure of a video (enhanced)"""
        if video_id not in self.video_embeddings:
            return {}
        
        structure = {
            "video_id": video_id,
            "scenes": [],
            "segments": self.segment_metadata.get(video_id, []),
            "total_segments": len(self.segment_metadata.get(video_id, [])),
            "total_scenes": len(self.scene_metadata.get(video_id, [])),
            "internvl_enhanced": any(
                seg.get("internvl_enhanced", False) 
                for seg in self.segment_metadata.get(video_id, [])
            ),
            "knowledge_graph": self.knowledge_graphs.get(video_id, {})
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
internvl_enhanced_vector_search = InternVLEnhancedVectorSearch()

async def get_internvl_enhanced_vector_search() -> InternVLEnhancedVectorSearch:
    """Get the enhanced vector search instance"""
    return internvl_enhanced_vector_search