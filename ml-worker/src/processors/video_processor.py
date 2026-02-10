# src/core/services/enhanced_video_processor.py
# Minimal version to get your server running

# import os
# import cv2
# import tempfile
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# import logging
# import numpy as np
# from PIL import Image
# import asyncio
# from processors.advanced_nlp import AdvancedNERProcessor, RelationExtractor
# from processors.temporal_analyzer import TemporalKnowledgeGraph 
# from processors.gnn_embeddings import GNNKnowledgeGraphEmbedder
# from processors.relationship_miner import RelationshipMiner
# from utils.database import get_db_session
# from utils.knowledge_graph_repository import KnowledgeGraphRepository
# import time

# ==============================================================================
# COMPLETE WORKING IMPORTS for ml-worker/src/processors/video_processor.py
# Copy this entire section to replace your imports
# ==============================================================================

import os
import cv2
import numpy as np
import tempfile
import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from pathlib import Path
import time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# ==============================================================================
# CORE DEPENDENCIES (Required)
# ==============================================================================

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("⚠️ Faster-Whisper not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ Sentence-Transformers not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch not available")

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("⚠️ EasyOCR not available")

# ==============================================================================
# OPTIONAL ADVANCED COMPONENTS
# ==============================================================================

# Advanced NLP
try:
    from processors.advanced_nlp import AdvancedNERProcessor, RelationExtractor
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    logger.warning("⚠️ Advanced NLP not available - using minimal NER")

# Temporal Analyzer
try:
    from processors.temporal_analyzer import TemporalKnowledgeGraph
    TEMPORAL_ANALYZER_AVAILABLE = True
except ImportError:
    TEMPORAL_ANALYZER_AVAILABLE = False
    logger.warning("⚠️ Temporal Analyzer not available - skipping temporal graphs")

# GNN Embeddings
try:
    from processors.gnn_embeddings import GNNKnowledgeGraphEmbedder
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    logger.warning("⚠️ GNN Embeddings not available - skipping GNN features")

# Relationship Miner
try:
    from processors.relationship_miner import RelationshipMiner
    RELATIONSHIP_MINER_AVAILABLE = True
except ImportError:
    RELATIONSHIP_MINER_AVAILABLE = False
    logger.warning("⚠️ Relationship Miner not available - skipping pattern mining")

# ==============================================================================
# DATABASE - WITH MULTIPLE FALLBACK STRATEGIES
# ==============================================================================

DATABASE_AVAILABLE = False
KnowledgeGraphRepository = None
get_db_session = None

try:
    # Try to import database manager
    from utils.database import DatabaseManager
    
    # Try to import get_db_session
    try:
        from utils.database import get_db_session as _existing_get_db_session
        get_db_session = _existing_get_db_session
        logger.info("✅ Found existing get_db_session")
    except ImportError:
        # Create get_db_session if it doesn't exist
        logger.info("Creating get_db_session helper...")
        
        @asynccontextmanager
        async def get_db_session() -> AsyncGenerator:
            """Get async database session with automatic cleanup"""
            db_manager = DatabaseManager()
            
            if not db_manager.initialized:
                await db_manager.initialize()
            
            session = await db_manager.get_session()
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    # Try to import repository
    try:
        from utils.knowledge_graph_repository import KnowledgeGraphRepository as _KGRepo
        KnowledgeGraphRepository = _KGRepo
        DATABASE_AVAILABLE = True
        logger.info("✅ Database and KG Repository available")
    except ImportError:
        logger.warning("⚠️ KnowledgeGraphRepository not found - database save disabled")
        DATABASE_AVAILABLE = False
        
except ImportError as e:
    logger.warning(f"⚠️ Database module not available: {e}")
    DATABASE_AVAILABLE = False

# ==============================================================================
# LOG COMPONENT AVAILABILITY
# ==============================================================================

def _log_component_availability():
    """Log which components are available on startup"""
    logger.info("=" * 70)
    logger.info("VideoRAG Component Availability Check")
    logger.info("=" * 70)
    logger.info(f"Core Components:")
    logger.info(f"  • Whisper (transcription):      {'✅' if WHISPER_AVAILABLE else '❌'}")
    logger.info(f"  • Sentence Transformers:        {'✅' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌'}")
    logger.info(f"  • PyTorch:                      {'✅' if TORCH_AVAILABLE else '❌'}")
    logger.info(f"  • EasyOCR:                      {'✅' if OCR_AVAILABLE else '❌'}")
    logger.info(f"")
    logger.info(f"Advanced Knowledge Graph Components:")
    logger.info(f"  • Advanced NLP (Transformers):  {'✅' if ADVANCED_NLP_AVAILABLE else '❌'}")
    logger.info(f"  • Temporal Analyzer:            {'✅' if TEMPORAL_ANALYZER_AVAILABLE else '❌'}")
    logger.info(f"  • GNN Embeddings:               {'✅' if GNN_AVAILABLE else '❌'}")
    logger.info(f"  • Relationship Miner:           {'✅' if RELATIONSHIP_MINER_AVAILABLE else '❌'}")
    logger.info(f"")
    logger.info(f"Infrastructure:")
    logger.info(f"  • Database Repository:          {'✅' if DATABASE_AVAILABLE else '❌'}")
    logger.info("=" * 70)
    
    # Determine processing mode
    if ADVANCED_NLP_AVAILABLE and GNN_AVAILABLE:
        mode = "ADVANCED (Full Features)"
    elif WHISPER_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
        mode = "STANDARD (Basic Features)"
    else:
        mode = "MINIMAL (Limited Features)"
    
    logger.info(f"Processing Mode: {mode}")
    logger.info("=" * 70)

# Run availability check
_log_component_availability()


logger = logging.getLogger(__name__)

class InternVLEnhancedVideoProcessor:
    """
    Minimal Enhanced Video Processor - Core functionality only
    Heavy ML dependencies are loaded lazily to avoid startup crashes
    """
    
    def __init__(self):
        # Basic components
        self.whisper_model = None
        self.embedding_model = None
        self.ocr_reader = None
        
        # Advanced components (may be None)
        self.advanced_ner = None
        self.relation_extractor = None
        self.gnn_embedder = None
        
        # Component availability tracking
        self.components_loaded = {
            "whisper": False,
            "embeddings": False,
            "ocr": False,
            "advanced_ner": False,
            "relation_extraction": False,
            "gnn_embeddings": False,
            "temporal_analyzer": TEMPORAL_ANALYZER_AVAILABLE,
            "relationship_miner": RELATIONSHIP_MINER_AVAILABLE,
            "database": DATABASE_AVAILABLE
        }
        
        # Configuration
        self.nlp_method = "minimal"
        self.worker_id = f"worker_{os.getpid()}"
        
    async def initialize(self):
        # Basic components
        self.whisper_model = None
        self.embedding_model = None
        self.ocr_reader = None
        
        # Advanced components (may be None)
        self.advanced_ner = None
        self.relation_extractor = None
        self.gnn_embedder = None
        
        # Component availability tracking
        self.components_loaded = {
            "whisper": False,
            "embeddings": False,
            "ocr": False,
            "advanced_ner": False,
            "relation_extraction": False,
            "gnn_embeddings": False,
            "temporal_analyzer": TEMPORAL_ANALYZER_AVAILABLE,
            "relationship_miner": RELATIONSHIP_MINER_AVAILABLE,
            "database": DATABASE_AVAILABLE
        }
        
        # Configuration
        self.nlp_method = "minimal"
        self.worker_id = f"worker_{os.getpid()}"
    
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
        """
        Complete video processing with advanced knowledge graph extraction
        Includes graceful degradation if components are unavailable
        """
        logger.info(f"🎬 Processing video: {video_path}")
        start_time = time.time()
        
        # Initialize results structure
        results = {
            "video_id": video_id,
            "transcription": [],
            "visual_segments": [],
            "scene_boundaries": [],
            "hierarchical_segments": [],
            "knowledge_graph": {},
            "temporal_knowledge_graph": {},
            "relationship_patterns": {},
            "gnn_entity_embeddings": None,
            "embeddings": [],
            "processing_mode": "advanced" if self.components_loaded.get("advanced_ner") else "minimal",
            "components_used": [k for k, v in self.components_loaded.items() if v]
        }
        
        # Initialize variables for later use
        gnn_embeddings = None
        
        try:
            # ========================================================================
            # STEP 1: Extract Basic Video Information
            # ========================================================================
            logger.info("Step 1/9: Extracting video info...")
            video_info = await self._extract_video_info(video_path)
            results["video_info"] = video_info
            logger.info(f"✅ Video info: {video_info.get('duration', 0):.1f}s, {video_info.get('fps', 0):.1f} fps")
            
            # ========================================================================
            # STEP 2: Scene Detection
            # ========================================================================
            logger.info("Step 2/9: Detecting scenes...")
            scene_boundaries = await self._detect_scenes_minimal(video_path)
            results["scene_boundaries"] = scene_boundaries
            logger.info(f"✅ Detected {len(scene_boundaries)} scenes")
            
            # ========================================================================
            # STEP 3: Audio Transcription
            # ========================================================================
            logger.info("Step 3/9: Transcribing audio...")
            if self.components_loaded["whisper"]:
                transcription = await self._enhanced_transcription(video_path)
                results["transcription"] = transcription
                logger.info(f"✅ Transcribed {len(transcription)} segments")
            else:
                logger.warning("⚠️ Whisper not available - skipping transcription")
                transcription = []
                results["transcription"] = []
            
            # ========================================================================
            # STEP 4: Visual Content Processing
            # ========================================================================
            logger.info("Step 4/9: Processing visual content...")
            visual_segments = await self._process_visual_content_minimal(
                video_path, 
                scene_boundaries
            )
            results["visual_segments"] = visual_segments
            logger.info(f"✅ Processed {len(visual_segments)} visual segments")
            
            # ========================================================================
            # STEP 5: Create Hierarchical Segments
            # ========================================================================
            logger.info("Step 5/9: Creating hierarchical segments...")
            hierarchical_segments = await self._create_hierarchical_segments_minimal(
                transcription, 
                visual_segments, 
                scene_boundaries
            )
            results["hierarchical_segments"] = hierarchical_segments
            logger.info(f"✅ Created {len(hierarchical_segments)} hierarchical segments")
            
            # ========================================================================
            # STEP 6: Knowledge Graph Extraction
            # ========================================================================
            logger.info("Step 6/9: Extracting knowledge graph...")
            if self.components_loaded.get("advanced_ner"):
                # Use advanced transformer-based NER
                knowledge_graph = await self._extract_knowledge_graph_advanced(
                    hierarchical_segments
                )
                logger.info(f"✅ Advanced KG: {knowledge_graph.get('total_entities_found', 0)} entities")
            else:
                # Fallback to minimal regex-based extraction
                knowledge_graph = await self._extract_knowledge_graph_minimal(
                    hierarchical_segments
                )
                logger.info(f"✅ Minimal KG: entities extracted")
            
            results["knowledge_graph"] = knowledge_graph
            
            # ========================================================================
            # STEP 7: Temporal Graph Construction
            # ========================================================================
            logger.info("Step 7/9: Building temporal knowledge graph...")
            try:
                from processors.temporal_analyzer import TemporalKnowledgeGraph
                
                temporal_kg = TemporalKnowledgeGraph()
                temporal_graph = temporal_kg.build_temporal_graph(
                    knowledge_graph,
                    hierarchical_segments
                )
                
                results["temporal_knowledge_graph"] = temporal_graph
                results["graph_visualization"] = temporal_kg.export_graph_for_visualization()
                
                logger.info(
                    f"✅ Temporal graph: {temporal_graph['graph_summary']['total_nodes']} nodes, "
                    f"{temporal_graph['graph_summary']['total_edges']} edges"
                )
            except Exception as e:
                logger.warning(f"⚠️ Temporal graph construction failed: {e}")
                results["temporal_knowledge_graph"] = {"graph_summary": {"total_nodes": 0, "total_edges": 0}}
                results["graph_visualization"] = {"nodes": [], "links": []}
                temporal_graph = results["temporal_knowledge_graph"]
            
            # ========================================================================
            # STEP 8: GNN Embeddings (if available)
            # ========================================================================
            logger.info("Step 8/9: Generating GNN embeddings...")
            if self.components_loaded.get("gnn_embeddings") and temporal_graph.get("graph_summary", {}).get("total_nodes", 0) > 0:
                try:
                    gnn_embeddings = self.gnn_embedder.create_graph_embeddings(
                        temporal_graph,
                        knowledge_graph
                    )
                    
                    results["gnn_entity_embeddings"] = {
                        entity: embedding.tolist()
                        for entity, embedding in gnn_embeddings.items()
                    }
                    
                    logger.info(f"✅ Generated GNN embeddings for {len(gnn_embeddings)} entities")
                except Exception as e:
                    logger.warning(f"⚠️ GNN embedding generation failed: {e}")
                    gnn_embeddings = None
                    results["gnn_entity_embeddings"] = None
            else:
                logger.info("⚠️ Skipping GNN embeddings (not available or no entities)")
                gnn_embeddings = None
                results["gnn_entity_embeddings"] = None
            
            # ========================================================================
            # STEP 9: Relationship Mining
            # ========================================================================
            logger.info("Step 9/9: Mining relationship patterns...")
            try:
                from processors.relationship_miner import RelationshipMiner
                
                relationship_miner = RelationshipMiner()
                relationship_patterns = relationship_miner.mine_relationships(
                    temporal_graph,
                    knowledge_graph
                )
                
                results["relationship_patterns"] = relationship_patterns
                
                logger.info(
                    f"✅ Found {len(relationship_patterns.get('entity_clusters', []))} clusters, "
                    f"{len(relationship_patterns.get('causal_chains', []))} causal chains"
                )
            except Exception as e:
                logger.warning(f"⚠️ Relationship mining failed: {e}")
                results["relationship_patterns"] = {}
                relationship_patterns = {}
            
            # ========================================================================
            # STEP 10: Generate Standard Embeddings (if available)
            # ========================================================================
            if self.components_loaded["embeddings"]:
                logger.info("Generating standard embeddings...")
                embeddings = await self._generate_embeddings_minimal(hierarchical_segments)
                results["embeddings"] = embeddings
                logger.info(f"✅ Generated {len(embeddings)} embedding vectors")
            else:
                logger.info("⚠️ Skipping standard embeddings (not available)")
                results["embeddings"] = []
            
            # ========================================================================
            # STEP 11: Save to Database
            # ========================================================================
            processing_time = time.time() - start_time
            logger.info(f"Saving knowledge graph to database...")
            
            try:
                from utils.database import get_db_session
                from utils.knowledge_graph_repository import KnowledgeGraphRepository
                
                async with get_db_session() as session:
                    kg_repo = KnowledgeGraphRepository(session)
                    
                    await kg_repo.save_complete_knowledge_graph(
                        video_id=video_id,
                        knowledge_graph=knowledge_graph,
                        temporal_graph=temporal_graph,
                        gnn_embeddings=gnn_embeddings,
                        relationship_patterns=relationship_patterns,
                        processing_metadata={
                            "gnn_enabled": self.components_loaded.get("gnn_embeddings", False),
                            "processing_time": processing_time,
                            "nlp_method": knowledge_graph.get("nlp_method_used", "minimal"),
                            "total_segments": len(hierarchical_segments),
                            "total_scenes": len(scene_boundaries)
                        }
                    )
                    
                logger.info(f"✅ Knowledge graph saved to database")
                
            except ImportError as e:
                logger.warning(f"⚠️ Database save skipped - missing dependencies: {e}")
            except Exception as e:
                logger.error(f"❌ Failed to save knowledge graph to database: {e}", exc_info=True)
                # Don't fail the whole processing - just log the error
            
            # ========================================================================
            # Final Summary
            # ========================================================================
            logger.info(
                f"✅ Video processing completed for {video_id} in {processing_time:.1f}s\n"
                f"  • Entities: {knowledge_graph.get('total_entities_found', 0)}\n"
                f"  • Relationships: {len(temporal_graph.get('relationship_edges', []))}\n"
                f"  • Patterns: {len(relationship_patterns.get('entity_clusters', []))}\n"
                f"  • Mode: {results['processing_mode']}"
            )
            
        except Exception as e:
            logger.error(f"❌ Video processing failed for {video_id}: {e}", exc_info=True)
            raise
        
        # Convert numpy types for JSON serialization
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
    

    async def _extract_knowledge_graph_advanced(
        self, 
        hierarchical_segments: List[Dict]
    ) -> Dict[str, Any]:
        """
        ADVANCED knowledge graph extraction with transformers
        """
        knowledge_graph = {
            "entities": {},
            "relationships": [],
            "entity_timeline": {},
            "entity_cooccurrence": {},
            "topics": [],
            "sentiment_analysis": {"polarity": 0.0, "subjectivity": 0.0},
            "nlp_method_used": "transformer_advanced",
            "extraction_metadata": {
                "model": self.advanced_ner.model_name if self.advanced_ner else "none",
                "total_segments_processed": len(hierarchical_segments)
            }
        }
        
        if not self.components_loaded.get("advanced_ner"):
            logger.warning("Advanced NER not available, falling back to minimal")
            return await self._extract_knowledge_graph_minimal(hierarchical_segments)
        
        # Extract entities with transformer
        kg_entities = self.advanced_ner.extract_entities_from_segments(
            hierarchical_segments
        )
        
        # Merge into knowledge graph
        knowledge_graph["entities"] = kg_entities["entities"]
        knowledge_graph["entity_timeline"] = kg_entities["entity_timeline"]
        knowledge_graph["entity_cooccurrence"] = kg_entities["entity_cooccurrence"]
        knowledge_graph["entity_statistics"] = kg_entities["entity_statistics"]
        knowledge_graph["total_entities_found"] = kg_entities["total_entities_found"]
        
        # Extract relationships if available
        if self.components_loaded.get("relation_extraction"):
            for segment in hierarchical_segments:
                text = segment.get("text", "")
                if not text:
                    continue
                
                # Get entities in this segment
                segment_entities = self.advanced_ner.extract_entities(text)
                
                # Extract relations
                relations = self.relation_extractor.extract_relations(
                    text, 
                    segment_entities
                )
                
                # Add temporal context to relations
                for relation in relations:
                    relation["segment_id"] = segment["segment_id"]
                    relation["timestamp"] = segment["start"]
                    relation["scene_id"] = segment["scene_id"]
                    knowledge_graph["relationships"].append(relation)
        
        logger.info(
            f"Advanced KG: {knowledge_graph['total_entities_found']} entities, "
            f"{len(knowledge_graph['relationships'])} relationships extracted"
        )
        
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