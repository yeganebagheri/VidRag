import networkx as nx
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

class TemporalKnowledgeGraph:
    """
    Builds and analyzes temporal relationships in video content
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        self.temporal_edges = []
        self.spatial_edges = []
        
    def build_temporal_graph(
        self, 
        knowledge_graph: Dict[str, Any],
        hierarchical_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build temporal knowledge graph from extracted entities and segments
        """
        # Add entity nodes
        entity_nodes = self._add_entity_nodes(knowledge_graph)
        
        # Add temporal edges (entities appearing in sequence)
        temporal_edges = self._add_temporal_edges(
            knowledge_graph, 
            hierarchical_segments
        )
        
        # Add co-occurrence edges (entities in same segment)
        cooccurrence_edges = self._add_cooccurrence_edges(knowledge_graph)
        
        # Add relationship edges (from relation extraction)
        relationship_edges = self._add_relationship_edges(knowledge_graph)
        
        # Analyze graph properties
        graph_metrics = self._compute_graph_metrics()
        
        # Find important patterns
        temporal_patterns = self._find_temporal_patterns()
        
        return {
            "nodes": entity_nodes,
            "temporal_edges": temporal_edges,
            "cooccurrence_edges": cooccurrence_edges,
            "relationship_edges": relationship_edges,
            "graph_metrics": graph_metrics,
            "temporal_patterns": temporal_patterns,
            "graph_summary": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "is_connected": nx.is_weakly_connected(self.graph)
            }
        }
    
    def _add_entity_nodes(self, knowledge_graph: Dict) -> List[Dict]:
        """Add entities as nodes in the graph"""
        entity_nodes = []
        
        for entity_type, entities in knowledge_graph.get("entities", {}).items():
            for entity in entities:
                entity_text = entity["text"]
                
                # Add node if doesn't exist
                if not self.graph.has_node(entity_text):
                    self.graph.add_node(
                        entity_text,
                        entity_type=entity_type,
                        confidence=entity["confidence"],
                        first_appearance=entity.get("timestamp", 0),
                        appearances=[]
                    )
                
                # Track all appearances
                self.graph.nodes[entity_text]["appearances"].append({
                    "timestamp": entity.get("timestamp", 0),
                    "segment_id": entity.get("segment_id"),
                    "confidence": entity["confidence"]
                })
                
                entity_nodes.append({
                    "entity": entity_text,
                    "type": entity_type,
                    "total_appearances": len(self.graph.nodes[entity_text]["appearances"])
                })
        
        return entity_nodes
    
    def _add_temporal_edges(
        self, 
        knowledge_graph: Dict,
        hierarchical_segments: List[Dict]
    ) -> List[Dict]:
        """
        Add edges between entities based on temporal proximity
        If two entities appear within N seconds, connect them
        """
        temporal_edges = []
        TEMPORAL_WINDOW = 30.0  # 30 seconds window
        
        # Get all entity timeline data
        entity_timeline = knowledge_graph.get("entity_timeline", {})
        
        # For each entity, find other entities nearby in time
        for entity1, timeline1 in entity_timeline.items():
            for entity2, timeline2 in entity_timeline.items():
                if entity1 >= entity2:  # Avoid duplicates and self-loops
                    continue
                
                # Check temporal proximity
                for appearance1 in timeline1:
                    for appearance2 in timeline2:
                        time_diff = abs(appearance1["timestamp"] - appearance2["timestamp"])
                        
                        if time_diff <= TEMPORAL_WINDOW:
                            # Add temporal edge
                            self.graph.add_edge(
                                entity1,
                                entity2,
                                edge_type="temporal_proximity",
                                time_difference=time_diff,
                                timestamp=min(appearance1["timestamp"], appearance2["timestamp"]),
                                segment_ids=[appearance1["segment_id"], appearance2["segment_id"]]
                            )
                            
                            temporal_edges.append({
                                "source": entity1,
                                "target": entity2,
                                "type": "temporal_proximity",
                                "time_difference": time_diff,
                                "timestamp": min(appearance1["timestamp"], appearance2["timestamp"])
                            })
        
        return temporal_edges
    
    def _add_cooccurrence_edges(self, knowledge_graph: Dict) -> List[Dict]:
        """Add edges for entities appearing in the same segment"""
        cooccurrence_edges = []
        
        entity_cooccurrence = knowledge_graph.get("entity_cooccurrence", {})
        
        for (entity1, entity2), occurrences in entity_cooccurrence.items():
            # Add edge with co-occurrence count
            self.graph.add_edge(
                entity1,
                entity2,
                edge_type="co_occurrence",
                co_occurrence_count=len(occurrences),
                segments=occurrences
            )
            
            cooccurrence_edges.append({
                "source": entity1,
                "target": entity2,
                "type": "co_occurrence",
                "count": len(occurrences),
                "timestamps": [occ["timestamp"] for occ in occurrences]
            })
        
        return cooccurrence_edges
    
    def _add_relationship_edges(self, knowledge_graph: Dict) -> List[Dict]:
        """Add edges from extracted relationships (subject-predicate-object)"""
        relationship_edges = []
        
        for relation in knowledge_graph.get("relationships", []):
            subject = relation.get("subject")
            obj = relation.get("object")
            predicate = relation.get("predicate")
            
            if subject and obj:
                self.graph.add_edge(
                    subject,
                    obj,
                    edge_type="semantic_relation",
                    predicate=predicate,
                    confidence=relation.get("confidence", 0.5),
                    timestamp=relation.get("timestamp", 0),
                    segment_id=relation.get("segment_id")
                )
                
                relationship_edges.append({
                    "source": subject,
                    "target": obj,
                    "type": "semantic_relation",
                    "predicate": predicate,
                    "confidence": relation.get("confidence", 0.5),
                    "timestamp": relation.get("timestamp", 0)
                })
        
        return relationship_edges
    
    def _compute_graph_metrics(self) -> Dict[str, Any]:
        """Compute important graph metrics"""
        metrics = {}
        
        if self.graph.number_of_nodes() == 0:
            return metrics
        
        # Centrality measures (which entities are most important)
        try:
            metrics["degree_centrality"] = nx.degree_centrality(self.graph)
            metrics["betweenness_centrality"] = nx.betweenness_centrality(self.graph)
            
            # Top central entities
            top_central = sorted(
                metrics["degree_centrality"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            metrics["most_central_entities"] = [
                {"entity": entity, "centrality": score}
                for entity, score in top_central
            ]
        except:
            pass
        
        # Community detection
        try:
            undirected = self.graph.to_undirected()
            communities = nx.community.greedy_modularity_communities(undirected)
            metrics["communities"] = [
                {"community_id": i, "entities": list(community)}
                for i, community in enumerate(communities)
            ]
            metrics["num_communities"] = len(communities)
        except:
            pass
        
        return metrics
    
    def _find_temporal_patterns(self) -> List[Dict]:
        """
        Find interesting temporal patterns:
        - Entity sequences that appear multiple times
        - Entities that always appear together
        - Temporal bursts (entity appears frequently in short time)
        """
        patterns = []
        
        # Find temporal bursts
        for node in self.graph.nodes():
            appearances = self.graph.nodes[node].get("appearances", [])
            if len(appearances) < 3:
                continue
            
            timestamps = [app["timestamp"] for app in appearances]
            timestamps.sort()
            
            # Check for bursts (3+ appearances within 60 seconds)
            for i in range(len(timestamps) - 2):
                if timestamps[i+2] - timestamps[i] <= 60:
                    patterns.append({
                        "pattern_type": "temporal_burst",
                        "entity": node,
                        "timespan": timestamps[i+2] - timestamps[i],
                        "num_appearances": 3,
                        "start_time": timestamps[i]
                    })
                    break
        
        return patterns
    
    def export_graph_for_visualization(self) -> Dict:
        """Export graph in format suitable for visualization (e.g., D3.js)"""
        nodes = []
        links = []
        
        # Export nodes
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            nodes.append({
                "id": node,
                "entity_type": node_data.get("entity_type"),
                "appearances": len(node_data.get("appearances", [])),
                "first_appearance": node_data.get("first_appearance", 0)
            })
        
        # Export edges
        for source, target, data in self.graph.edges(data=True):
            links.append({
                "source": source,
                "target": target,
                "edge_type": data.get("edge_type"),
                "weight": data.get("co_occurrence_count", 1)
            })
        
        return {"nodes": nodes, "links": links}