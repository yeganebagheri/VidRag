from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import networkx as nx

class RelationshipMiner:
    """
    Mine complex relationship patterns from knowledge graph
    """
    
    def __init__(self):
        self.graph = None
        
    def mine_relationships(
        self,
        temporal_graph: Dict[str, Any],
        knowledge_graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Mine various relationship patterns
        """
        # Build NetworkX graph from temporal graph
        self.graph = self._build_networkx_graph(temporal_graph)
        
        patterns = {
            "entity_clusters": self._find_entity_clusters(),
            "causal_chains": self._find_causal_chains(),
            "recurring_patterns": self._find_recurring_patterns(knowledge_graph),
            "entity_roles": self._classify_entity_roles(),
            "temporal_dependencies": self._find_temporal_dependencies(knowledge_graph),
            "relationship_statistics": self._compute_relationship_statistics()
        }
        
        return patterns
    
    def _build_networkx_graph(self, temporal_graph: Dict) -> nx.DiGraph:
        """Build NetworkX graph from temporal graph data"""
        G = nx.DiGraph()
        
        # Add nodes
        for node in temporal_graph.get("nodes", []):
            G.add_node(
                node["entity"],
                entity_type=node["type"],
                appearances=node.get("total_appearances", 1)
            )
        
        # Add edges
        for edge in temporal_graph.get("relationship_edges", []):
            G.add_edge(
                edge["source"],
                edge["target"],
                edge_type="semantic",
                predicate=edge.get("predicate"),
                weight=edge.get("confidence", 1.0)
            )
        
        for edge in temporal_graph.get("cooccurrence_edges", []):
            if not G.has_edge(edge["source"], edge["target"]):
                G.add_edge(
                    edge["source"],
                    edge["target"],
                    edge_type="cooccurrence",
                    weight=edge.get("count", 1)
                )
        
        return G
    
    def _find_entity_clusters(self) -> List[Dict]:
        """
        Find clusters of tightly connected entities
        These represent topics or themes in the video
        """
        if self.graph.number_of_nodes() < 3:
            return []
        
        # Convert to undirected for clustering
        undirected = self.graph.to_undirected()
        
        # Find communities
        communities = nx.community.greedy_modularity_communities(undirected)
        
        clusters = []
        for i, community in enumerate(communities):
            if len(community) < 2:
                continue
            
            # Analyze cluster
            subgraph = self.graph.subgraph(community)
            
            clusters.append({
                "cluster_id": i,
                "entities": list(community),
                "size": len(community),
                "density": nx.density(subgraph),
                "entity_types": list(set(
                    self.graph.nodes[node].get("entity_type", "UNKNOWN")
                    for node in community
                ))
            })
        
        return clusters
    
    def _find_causal_chains(self) -> List[Dict]:
        """
        Find potential causal chains: A -> B -> C
        """
        chains = []
        
        # Find all paths of length 2-4
        for source in self.graph.nodes():
            for target in self.graph.nodes():
                if source == target:
                    continue
                
                try:
                    # Find all simple paths up to length 4
                    paths = list(nx.all_simple_paths(
                        self.graph, 
                        source, 
                        target, 
                        cutoff=4
                    ))
                    
                    for path in paths:
                        if len(path) >= 3:  # At least 3 entities
                            # Extract predicates if available
                            predicates = []
                            for i in range(len(path) - 1):
                                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                                if edge_data:
                                    predicates.append(edge_data.get("predicate", "related_to"))
                            
                            chains.append({
                                "chain": path,
                                "length": len(path),
                                "predicates": predicates,
                                "chain_description": " -> ".join(path)
                            })
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by length and return top chains
        chains.sort(key=lambda x: x["length"], reverse=True)
        return chains[:20]  # Top 20 chains
    
    def _find_recurring_patterns(self, knowledge_graph: Dict) -> List[Dict]:
        """
        Find patterns that occur multiple times (e.g., same entity pair appearing together)
        """
        patterns = []
        
        # Find frequently co-occurring entity pairs
        cooccurrence = knowledge_graph.get("entity_cooccurrence", {})
        
        frequent_pairs = [
            {
                "entities": list(entities),
                "occurrences": len(occurrences),
                "timestamps": [occ["timestamp"] for occ in occurrences]
            }
            for entities, occurrences in cooccurrence.items()
            if len(occurrences) >= 2
        ]
        
        frequent_pairs.sort(key=lambda x: x["occurrences"], reverse=True)
        patterns.extend(frequent_pairs[:10])
        
        return patterns
    
    def _classify_entity_roles(self) -> Dict[str, List[str]]:
        """
        Classify entities by their role in the graph:
        - Hubs: highly connected entities
        - Bridges: entities connecting different clusters
        - Periphery: entities with few connections
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Calculate centrality
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        roles = {
            "hubs": [],  # High degree centrality
            "bridges": [],  # High betweenness centrality
            "periphery": []  # Low centrality
        }
        
        for node in self.graph.nodes():
            degree = degree_centrality.get(node, 0)
            betweenness = betweenness_centrality.get(node, 0)
            
            if degree > 0.5:  # Top 50% degree
                roles["hubs"].append(node)
            elif betweenness > 0.5:  # High betweenness
                roles["bridges"].append(node)
            elif degree < 0.2:  # Low degree
                roles["periphery"].append(node)
        
        return roles
    
    def _find_temporal_dependencies(self, knowledge_graph: Dict) -> List[Dict]:
        """
        Find entities that consistently appear before/after others
        """
        dependencies = []
        
        entity_timeline = knowledge_graph.get("entity_timeline", {})
        
        # Compare all entity pairs
        entities = list(entity_timeline.keys())
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                timeline1 = entity_timeline[entity1]
                timeline2 = entity_timeline[entity2]
                
                # Count how often entity1 appears before entity2
                before_count = sum(
                    1 for t1 in timeline1 for t2 in timeline2
                    if t1["timestamp"] < t2["timestamp"]
                )
                
                after_count = sum(
                    1 for t1 in timeline1 for t2 in timeline2
                    if t1["timestamp"] > t2["timestamp"]
                )
                
                # If there's a strong temporal bias
                if before_count > after_count * 2:  # Entity1 usually before entity2
                    dependencies.append({
                        "entity_before": entity1,
                        "entity_after": entity2,
                        "confidence": before_count / (before_count + after_count),
                        "occurrences": before_count
                    })
                elif after_count > before_count * 2:
                    dependencies.append({
                        "entity_before": entity2,
                        "entity_after": entity1,
                        "confidence": after_count / (before_count + after_count),
                        "occurrences": after_count
                    })
        
        # Sort by confidence
        dependencies.sort(key=lambda x: x["confidence"], reverse=True)
        return dependencies[:15]
    
    def _compute_relationship_statistics(self) -> Dict:
        """Compute overall relationship statistics"""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Count relationship types
        edge_types = defaultdict(int)
        predicates = defaultdict(int)
        
        for _, _, data in self.graph.edges(data=True):
            edge_types[data.get("edge_type", "unknown")] += 1
            if "predicate" in data:
                predicates[data["predicate"]] += 1
        
        return {
            "total_relationships": self.graph.number_of_edges(),
            "relationships_by_type": dict(edge_types),
            "most_common_predicates": [
                {"predicate": pred, "count": count}
                for pred, count in Counter(predicates).most_common(10)
            ],
            "average_connections_per_entity": (
                2 * self.graph.number_of_edges() / max(1, self.graph.number_of_nodes())
            )
        }