import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from typing import Dict, Any, List

class KnowledgeGraphGNN(nn.Module):
    """
    Graph Neural Network for learning entity embeddings
    from the knowledge graph structure
    """
    
    def __init__(
        self, 
        input_dim: int = 384,  # sentence-transformer embedding size
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        use_gat: bool = True  # Graph Attention Network
    ):
        super().__init__()
        
        self.use_gat = use_gat
        self.num_layers = num_layers
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        if use_gat:
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
            current_dim = hidden_dim * 4
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            current_dim = hidden_dim
        self.batch_norms.append(nn.BatchNorm1d(current_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if use_gat:
                self.convs.append(GATConv(current_dim, hidden_dim, heads=4, concat=True))
                current_dim = hidden_dim * 4
            else:
                self.convs.append(GCNConv(current_dim, hidden_dim))
                current_dim = hidden_dim
            self.batch_norms.append(nn.BatchNorm1d(current_dim))
        
        # Output layer
        if use_gat:
            self.convs.append(GATConv(current_dim, output_dim, heads=1, concat=False))
        else:
            self.convs.append(GCNConv(current_dim, output_dim))
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Apply GNN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer (no activation)
        x = self.convs[-1](x, edge_index)
        
        return x


class GNNKnowledgeGraphEmbedder:
    """
    Uses GNN to create enhanced entity embeddings
    based on graph structure
    """
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model  # sentence-transformers model
        self.gnn_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def initialize(self):
        """Initialize GNN model"""
        self.gnn_model = KnowledgeGraphGNN(
            input_dim=384,  # Sentence transformer dim
            hidden_dim=256,
            output_dim=128,
            num_layers=3,
            use_gat=True
        )
        self.gnn_model = self.gnn_model.to(self.device)
        self.gnn_model.eval()
        
    def create_graph_embeddings(
        self,
        temporal_graph: Dict[str, Any],
        knowledge_graph: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Create GNN-enhanced embeddings for all entities
        
        Returns:
            Dictionary mapping entity text to embedding vector
        """
        # Convert NetworkX graph to PyTorch Geometric format
        pyg_data = self._convert_to_pyg_data(temporal_graph, knowledge_graph)
        
        if pyg_data is None:
            return {}
        
        # Run GNN
        with torch.no_grad():
            pyg_data = pyg_data.to(self.device)
            entity_embeddings = self.gnn_model(pyg_data.x, pyg_data.edge_index)
            entity_embeddings = entity_embeddings.cpu().numpy()
        
        # Map back to entity names
        entity_to_embedding = {}
        for i, entity_name in enumerate(pyg_data.entity_names):
            entity_to_embedding[entity_name] = entity_embeddings[i]
        
        return entity_to_embedding
    
    def _convert_to_pyg_data(
        self, 
        temporal_graph: Dict,
        knowledge_graph: Dict
    ) -> Data:
        """Convert graph to PyTorch Geometric Data object"""
        nodes = temporal_graph.get("nodes", [])
        if not nodes:
            return None
        
        # Create entity name to index mapping
        entity_to_idx = {node["entity"]: i for i, node in enumerate(nodes)}
        
        # Create initial node features using sentence-transformers
        node_features = []
        entity_names = []
        
        for node in nodes:
            entity_text = node["entity"]
            entity_names.append(entity_text)
            
            # Get semantic embedding for this entity
            if self.embedding_model:
                embedding = self.embedding_model.encode(entity_text)
            else:
                # Random initialization if no embedding model
                embedding = np.random.randn(384)
            
            node_features.append(embedding)
        
        # Convert to tensor
        x = torch.FloatTensor(np.array(node_features))
        
        # Create edge index
        edge_list = []
        
        # Add all edges (temporal, co-occurrence, relationships)
        for edge_type in ["temporal_edges", "cooccurrence_edges", "relationship_edges"]:
            for edge in temporal_graph.get(edge_type, []):
                source = edge["source"]
                target = edge["target"]
                
                if source in entity_to_idx and target in entity_to_idx:
                    edge_list.append([entity_to_idx[source], entity_to_idx[target]])
                    # Add reverse edge for undirected learning
                    edge_list.append([entity_to_idx[target], entity_to_idx[source]])
        
        if not edge_list:
            # Create self-loops if no edges
            edge_list = [[i, i] for i in range(len(nodes))]
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)
        data.entity_names = entity_names
        
        return data