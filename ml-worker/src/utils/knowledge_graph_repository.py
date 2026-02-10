# ml-worker/src/utils/knowledge_graph_repository.py
"""
Repository for saving advanced knowledge graph data to database
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)


class KnowledgeGraphRepository:
    """
    Handles saving knowledge graph data to the new database tables:
    - entity_embeddings
    - entity_relationships
    - temporal_patterns
    - knowledge_graph_metadata
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save_complete_knowledge_graph(
        self,
        video_id: str,
        knowledge_graph: Dict[str, Any],
        temporal_graph: Dict[str, Any],
        gnn_embeddings: Optional[Dict[str, np.ndarray]] = None,
        relationship_patterns: Optional[Dict[str, Any]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save complete knowledge graph data for a video
        
        This is the main method you'll call from your video processor
        """
        try:
            logger.info(f"Saving knowledge graph for video {video_id}")
            
            # 1. Save entities
            await self.save_entities(video_id, knowledge_graph, gnn_embeddings)
            
            # 2. Save relationships
            await self.save_relationships(video_id, knowledge_graph, temporal_graph)
            
            # 3. Save temporal patterns
            await self.save_temporal_patterns(video_id, relationship_patterns, temporal_graph)
            
            # 4. Save metadata
            await self.save_metadata(video_id, knowledge_graph, temporal_graph, processing_metadata)
            
            await self.session.commit()
            logger.info(f"✅ Knowledge graph saved successfully for video {video_id}")
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to save knowledge graph: {e}")
            raise
    
    async def save_entities(
        self,
        video_id: str,
        knowledge_graph: Dict[str, Any],
        gnn_embeddings: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Save entities to entity_embeddings table
        """
        entities_data = knowledge_graph.get("entities", {})
        entity_timeline = knowledge_graph.get("entity_timeline", {})
        
        for entity_type, entities in entities_data.items():
            for entity in entities:
                entity_text = entity["text"]
                
                # Get timeline information for this entity
                timeline = entity_timeline.get(entity_text, [])
                timestamps = [t["timestamp"] for t in timeline]
                segment_ids = [t["segment_id"] for t in timeline]
                
                # Get embeddings
                semantic_embedding = None
                gnn_embedding = None
                
                if gnn_embeddings and entity_text in gnn_embeddings:
                    gnn_embedding = gnn_embeddings[entity_text].tolist()
                
                # Prepare data
                entity_data = {
                    "video_id": video_id,
                    "entity_text": entity_text,
                    "entity_type": entity_type,
                    "first_appearance": min(timestamps) if timestamps else entity.get("timestamp", 0),
                    "last_appearance": max(timestamps) if timestamps else entity.get("timestamp", 0),
                    "total_appearances": len(timestamps),
                    "confidence": entity.get("confidence", 0.0),
                    "appearance_timestamps": timestamps,
                    "segment_ids": segment_ids,
                    "semantic_embedding": semantic_embedding,
                    "gnn_embedding": gnn_embedding
                }
                
                # Insert entity
                query = text("""
                    INSERT INTO entity_embeddings (
                        video_id, entity_text, entity_type,
                        first_appearance, last_appearance, total_appearances,
                        confidence, appearance_timestamps, segment_ids,
                        semantic_embedding, gnn_embedding
                    )
                    VALUES (
                        :video_id, :entity_text, :entity_type,
                        :first_appearance, :last_appearance, :total_appearances,
                        :confidence, :appearance_timestamps, :segment_ids,
                        :semantic_embedding::vector, :gnn_embedding::vector
                    )
                    ON CONFLICT (video_id, entity_text) 
                    DO UPDATE SET
                        total_appearances = EXCLUDED.total_appearances,
                        last_appearance = EXCLUDED.last_appearance,
                        confidence = EXCLUDED.confidence,
                        appearance_timestamps = EXCLUDED.appearance_timestamps,
                        segment_ids = EXCLUDED.segment_ids,
                        gnn_embedding = EXCLUDED.gnn_embedding,
                        updated_at = NOW()
                """)
                
                await self.session.execute(query, entity_data)
        
        logger.info(f"Saved entities for video {video_id}")
    
    async def save_relationships(
        self,
        video_id: str,
        knowledge_graph: Dict[str, Any],
        temporal_graph: Dict[str, Any]
    ):
        """
        Save relationships to entity_relationships table
        """
        # Save semantic relationships (from relation extraction)
        for relation in knowledge_graph.get("relationships", []):
            relationship_data = {
                "video_id": video_id,
                "source_entity": relation.get("subject"),
                "target_entity": relation.get("object"),
                "relationship_type": "semantic_relation",
                "predicate": relation.get("predicate"),
                "confidence": relation.get("confidence", 0.0),
                "timestamp": relation.get("timestamp"),
                "segment_id": relation.get("segment_id"),
                "scene_id": relation.get("scene_id"),
                "co_occurrence_count": 1
            }
            
            query = text("""
                INSERT INTO entity_relationships (
                    video_id, source_entity, target_entity,
                    relationship_type, predicate, confidence,
                    timestamp, segment_id, scene_id, co_occurrence_count
                )
                VALUES (
                    :video_id, :source_entity, :target_entity,
                    :relationship_type, :predicate, :confidence,
                    :timestamp, :segment_id, :scene_id, :co_occurrence_count
                )
            """)
            
            await self.session.execute(query, relationship_data)
        
        # Save temporal relationships
        for edge in temporal_graph.get("temporal_edges", []):
            relationship_data = {
                "video_id": video_id,
                "source_entity": edge["source"],
                "target_entity": edge["target"],
                "relationship_type": edge["type"],
                "predicate": None,
                "confidence": 0.8,
                "timestamp": edge.get("timestamp"),
                "time_difference": edge.get("time_difference"),
                "co_occurrence_count": 1
            }
            
            query = text("""
                INSERT INTO entity_relationships (
                    video_id, source_entity, target_entity,
                    relationship_type, predicate, confidence,
                    timestamp, time_difference, co_occurrence_count
                )
                VALUES (
                    :video_id, :source_entity, :target_entity,
                    :relationship_type, :predicate, :confidence,
                    :timestamp, :time_difference, :co_occurrence_count
                )
            """)
            
            await self.session.execute(query, relationship_data)
        
        # Save co-occurrence relationships
        for edge in temporal_graph.get("cooccurrence_edges", []):
            relationship_data = {
                "video_id": video_id,
                "source_entity": edge["source"],
                "target_entity": edge["target"],
                "relationship_type": edge["type"],
                "predicate": None,
                "confidence": 0.9,
                "timestamp": edge.get("timestamps", [0])[0] if edge.get("timestamps") else None,
                "co_occurrence_count": edge.get("count", 1)
            }
            
            query = text("""
                INSERT INTO entity_relationships (
                    video_id, source_entity, target_entity,
                    relationship_type, predicate, confidence,
                    timestamp, co_occurrence_count
                )
                VALUES (
                    :video_id, :source_entity, :target_entity,
                    :relationship_type, :predicate, :confidence,
                    :timestamp, :co_occurrence_count
                )
            """)
            
            await self.session.execute(query, relationship_data)
        
        logger.info(f"Saved relationships for video {video_id}")
    
    async def save_temporal_patterns(
        self,
        video_id: str,
        relationship_patterns: Optional[Dict[str, Any]],
        temporal_graph: Dict[str, Any]
    ):
        """
        Save temporal patterns to temporal_patterns table
        """
        if not relationship_patterns:
            return
        
        # Save entity clusters
        for cluster in relationship_patterns.get("entity_clusters", []):
            pattern_data = {
                "video_id": video_id,
                "pattern_type": "entity_cluster",
                "pattern_name": f"Cluster {cluster['cluster_id']}",
                "pattern_data": cluster,
                "confidence": 0.8,
                "importance_score": cluster.get("density", 0.5),
                "support_count": cluster.get("size", 0),
                "entities": cluster.get("entities", [])
            }
            
            query = text("""
                INSERT INTO temporal_patterns (
                    video_id, pattern_type, pattern_name,
                    pattern_data, confidence, importance_score,
                    support_count, entities
                )
                VALUES (
                    :video_id, :pattern_type, :pattern_name,
                    :pattern_data::jsonb, :confidence, :importance_score,
                    :support_count, :entities
                )
            """)
            
            await self.session.execute(query, pattern_data)
        
        # Save causal chains
        for chain in relationship_patterns.get("causal_chains", [])[:20]:  # Limit to top 20
            pattern_data = {
                "video_id": video_id,
                "pattern_type": "causal_chain",
                "pattern_name": chain.get("chain_description", "Unknown Chain"),
                "pattern_data": chain,
                "confidence": 0.7,
                "importance_score": min(chain.get("length", 0) / 10.0, 1.0),
                "support_count": 1,
                "entities": chain.get("chain", [])
            }
            
            query = text("""
                INSERT INTO temporal_patterns (
                    video_id, pattern_type, pattern_name,
                    pattern_data, confidence, importance_score,
                    support_count, entities
                )
                VALUES (
                    :video_id, :pattern_type, :pattern_name,
                    :pattern_data::jsonb, :confidence, :importance_score,
                    :support_count, :entities
                )
            """)
            
            await self.session.execute(query, pattern_data)
        
        # Save recurring patterns
        for pattern in relationship_patterns.get("recurring_patterns", []):
            pattern_data = {
                "video_id": video_id,
                "pattern_type": "recurring_pattern",
                "pattern_name": f"Pattern: {' & '.join(pattern.get('entities', [])[:2])}",
                "pattern_data": pattern,
                "confidence": 0.85,
                "importance_score": min(pattern.get("occurrences", 0) / 10.0, 1.0),
                "support_count": pattern.get("occurrences", 1),
                "entities": pattern.get("entities", []),
                "start_timestamp": min(pattern.get("timestamps", [0])) if pattern.get("timestamps") else None,
                "end_timestamp": max(pattern.get("timestamps", [0])) if pattern.get("timestamps") else None
            }
            
            query = text("""
                INSERT INTO temporal_patterns (
                    video_id, pattern_type, pattern_name,
                    pattern_data, confidence, importance_score,
                    support_count, entities,
                    start_timestamp, end_timestamp
                )
                VALUES (
                    :video_id, :pattern_type, :pattern_name,
                    :pattern_data::jsonb, :confidence, :importance_score,
                    :support_count, :entities,
                    :start_timestamp, :end_timestamp
                )
            """)
            
            await self.session.execute(query, pattern_data)
        
        # Save temporal bursts from temporal_graph
        for pattern in temporal_graph.get("temporal_patterns", []):
            if pattern.get("pattern_type") == "temporal_burst":
                pattern_data = {
                    "video_id": video_id,
                    "pattern_type": "temporal_burst",
                    "pattern_name": f"Burst: {pattern.get('entity', 'Unknown')}",
                    "pattern_data": pattern,
                    "confidence": 0.8,
                    "importance_score": 0.7,
                    "support_count": pattern.get("num_appearances", 1),
                    "entities": [pattern.get("entity")],
                    "start_timestamp": pattern.get("start_time"),
                    "duration": pattern.get("timespan")
                }
                
                query = text("""
                    INSERT INTO temporal_patterns (
                        video_id, pattern_type, pattern_name,
                        pattern_data, confidence, importance_score,
                        support_count, entities,
                        start_timestamp, duration
                    )
                    VALUES (
                        :video_id, :pattern_type, :pattern_name,
                        :pattern_data::jsonb, :confidence, :importance_score,
                        :support_count, :entities,
                        :start_timestamp, :duration
                    )
                """)
                
                await self.session.execute(query, pattern_data)
        
        logger.info(f"Saved temporal patterns for video {video_id}")
    
    async def save_metadata(
        self,
        video_id: str,
        knowledge_graph: Dict[str, Any],
        temporal_graph: Dict[str, Any],
        processing_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save knowledge graph metadata to knowledge_graph_metadata table
        """
        metadata = {
            "video_id": video_id,
            "total_entities": knowledge_graph.get("total_entities_found", 0),
            "total_relationships": len(knowledge_graph.get("relationships", [])),
            "entity_counts_by_type": knowledge_graph.get("entity_statistics", {}).get("entities_by_type_count", {}),
            "graph_density": temporal_graph.get("graph_summary", {}).get("density"),
            "num_communities": len(temporal_graph.get("graph_metrics", {}).get("communities", [])),
            "is_connected": temporal_graph.get("graph_summary", {}).get("is_connected"),
            "top_entities": temporal_graph.get("graph_metrics", {}).get("most_central_entities", []),
            "nlp_method_used": knowledge_graph.get("nlp_method_used", "unknown"),
            "gnn_enabled": processing_metadata.get("gnn_enabled", False) if processing_metadata else False,
            "processing_time_seconds": processing_metadata.get("processing_time", 0) if processing_metadata else 0,
            "graph_visualization_data": temporal_graph.get("graph_visualization", {})
        }
        
        # Calculate total relationships from temporal graph
        total_relationships = (
            len(temporal_graph.get("temporal_edges", [])) +
            len(temporal_graph.get("cooccurrence_edges", [])) +
            len(temporal_graph.get("relationship_edges", []))
        )
        metadata["total_relationships"] = total_relationships
        
        query = text("""
            INSERT INTO knowledge_graph_metadata (
                video_id, total_entities, total_relationships,
                entity_counts_by_type, graph_density, num_communities,
                is_connected, top_entities, nlp_method_used,
                gnn_enabled, processing_time_seconds, graph_visualization_data
            )
            VALUES (
                :video_id, :total_entities, :total_relationships,
                :entity_counts_by_type::jsonb, :graph_density, :num_communities,
                :is_connected, :top_entities::jsonb, :nlp_method_used,
                :gnn_enabled, :processing_time_seconds, :graph_visualization_data::jsonb
            )
            ON CONFLICT (video_id) DO UPDATE SET
                total_entities = EXCLUDED.total_entities,
                total_relationships = EXCLUDED.total_relationships,
                entity_counts_by_type = EXCLUDED.entity_counts_by_type,
                graph_density = EXCLUDED.graph_density,
                num_communities = EXCLUDED.num_communities,
                is_connected = EXCLUDED.is_connected,
                top_entities = EXCLUDED.top_entities,
                nlp_method_used = EXCLUDED.nlp_method_used,
                gnn_enabled = EXCLUDED.gnn_enabled,
                processing_time_seconds = EXCLUDED.processing_time_seconds,
                graph_visualization_data = EXCLUDED.graph_visualization_data,
                updated_at = NOW()
        """)
        
        await self.session.execute(query, metadata)
        
        logger.info(f"Saved knowledge graph metadata for video {video_id}")
    
    async def get_video_entities(
        self,
        video_id: str,
        entity_type: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve entities for a video
        """
        query = text("""
            SELECT 
                entity_text,
                entity_type,
                total_appearances,
                confidence,
                first_appearance,
                last_appearance,
                appearance_timestamps,
                segment_ids
            FROM entity_embeddings
            WHERE video_id = :video_id
                AND (:entity_type IS NULL OR entity_type = :entity_type)
                AND confidence >= :min_confidence
            ORDER BY total_appearances DESC, confidence DESC
        """)
        
        result = await self.session.execute(
            query,
            {"video_id": video_id, "entity_type": entity_type, "min_confidence": min_confidence}
        )
        
        return [dict(row._mapping) for row in result]
    
    async def search_similar_entities(
        self,
        query_embedding: List[float],
        similarity_threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar entities using semantic embeddings
        """
        query = text("""
            SELECT 
                entity_text,
                entity_type,
                video_id,
                1 - (semantic_embedding <=> :query_embedding::vector) as similarity
            FROM entity_embeddings
            WHERE semantic_embedding IS NOT NULL
                AND 1 - (semantic_embedding <=> :query_embedding::vector) >= :similarity_threshold
            ORDER BY semantic_embedding <=> :query_embedding::vector
            LIMIT :max_results
        """)
        
        result = await self.session.execute(
            query,
            {
                "query_embedding": query_embedding,
                "similarity_threshold": similarity_threshold,
                "max_results": max_results
            }
        )
        
        return [dict(row._mapping) for row in result]


# Example usage in your video processor:
"""
# In ml-worker/src/processors/video_processor.py

async def process_video(self, video_path: str, video_id: str) -> Dict[str, Any]:
    import time
    start_time = time.time()
    
    # ... existing processing code ...
    
    # After generating all knowledge graph data:
    
    # Save to database
    from utils.database import get_db_session
    from utils.knowledge_graph_repository import KnowledgeGraphRepository
    
    async with get_db_session() as session:
        kg_repo = KnowledgeGraphRepository(session)
        
        await kg_repo.save_complete_knowledge_graph(
            video_id=video_id,
            knowledge_graph=knowledge_graph,
            temporal_graph=temporal_graph,
            gnn_embeddings=gnn_embeddings if self.components_loaded.get("gnn_embeddings") else None,
            relationship_patterns=relationship_patterns,
            processing_metadata={
                "gnn_enabled": self.components_loaded.get("gnn_embeddings", False),
                "processing_time": time.time() - start_time
            }
        )
    
    return results
"""