import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union

from src.data.datasets import GraphData
from src.models.encoder.base import BaseEncoder
from src.models.encoder.hyperbolic import (
    HyperbolicTangentSpace,
    EuclideanToHyperbolic,
    HyperbolicToEuclidean,
    HyperbolicLinear
)


class HyperbolicEncoder(BaseEncoder):
    """
    Simple hyperbolic encoder for citation networks.
    
    This encoder transforms node features into hyperbolic space and applies
    a series of hyperbolic transformations to create node embeddings.
    """
    
    def __init__(self,
                node_dim: int,
                edge_dim: int,
                embed_dim: int,
                curvature: float = 1.0,
                use_tangent_space: bool = True,
                use_lorentz: bool = False,
                **kwargs):
        """
        Initialize the hyperbolic encoder.
        
        Args:
            node_dim: Dimensionality of node features
            edge_dim: Dimensionality of edge features
            embed_dim: Dimensionality of output embeddings
            curvature: Curvature of the hyperbolic space
            use_tangent_space: Whether to use tangent space operations
            use_lorentz: Whether to use Lorentz model instead of Poincaré
        """
        super().__init__(node_dim, edge_dim, embed_dim)
        
        # Store parameters
        self.curvature = curvature
        self.use_tangent_space = use_tangent_space
        self.use_lorentz = use_lorentz
        
        # Create hyperbolic components
        self.hyperbolic_space = HyperbolicTangentSpace(curvature=curvature)
        
        # Create layers
        hidden_dim = 2 * embed_dim
        self.initial_transform = nn.Linear(node_dim, hidden_dim)
        self.to_hyperbolic = EuclideanToHyperbolic(hidden_dim, hidden_dim, curvature=curvature)
        self.hyperbolic_layer = HyperbolicLinear(hidden_dim, embed_dim, curvature=curvature)
        
        # Activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, graph: GraphData) -> torch.Tensor:
        """
        Transform graph data into node embeddings.
        
        Args:
            graph: Graph data object containing the citation network
        
        Returns:
            Node embeddings of shape [num_nodes, embed_dim]
        """
        # Get node features
        if not hasattr(graph, 'x') or graph.x is None:
            raise ValueError("Graph must have node features (x)")
        
        x = graph.x
        
        # Initial transformation in Euclidean space
        x = self.initial_transform(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Transform to hyperbolic space
        x = self.to_hyperbolic(x)
        
        # Apply hyperbolic transformation
        x = self.hyperbolic_layer(x)
        
        # Project back to unit ball if needed
        x = self.hyperbolic_space.proj(x)
        
        return x
    
    def encode_subgraph(self, graph: GraphData, node_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a subgraph of the citation network.
        
        Args:
            graph: The complete graph data
            node_indices: Indices of nodes to include in the subgraph
        
        Returns:
            Node embeddings for the subgraph
        """
        if node_indices is None:
            # Encode the entire graph
            return self.forward(graph)
        
        # Create a subgraph with only the specified nodes
        if not hasattr(graph, 'x') or graph.x is None:
            raise ValueError("Graph must have node features (x)")
        
        # Get embeddings for all nodes and then select the ones we want
        all_embeddings = self.forward(graph)
        return all_embeddings[node_indices]
    
    def encode_temporal_snapshot(self, graph: GraphData, time_threshold: float) -> torch.Tensor:
        """
        Encode a temporal snapshot of the citation network.
        
        Args:
            graph: The complete graph data
            time_threshold: The timestamp threshold to use for the snapshot
        
        Returns:
            Node embeddings for the temporal snapshot
        """
        # Check if the graph has timestamps
        if not hasattr(graph, 'node_timestamps') or graph.node_timestamps is None:
            raise ValueError("Graph must have node timestamps for temporal encoding")
        
        # Get nodes with timestamps <= threshold
        node_indices = torch.where(graph.node_timestamps <= time_threshold)[0]
        
        # Encode the subgraph
        return self.encode_subgraph(graph, node_indices)
    
    def get_embedding_metric(self) -> str:
        """
        Get the name of the distance metric used in the embedding space.
        
        Returns:
            The name of the distance metric
        """
        return 'hyperbolic'
    
    def compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between embeddings in hyperbolic space.
        
        Args:
            x1: First set of embeddings
            x2: Second set of embeddings
        
        Returns:
            Pairwise distances between embeddings
        """
        # Reshape if needed
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
            
        # Compute pairwise distances
        n1, n2 = x1.size(0), x2.size(0)
        distances = torch.zeros(n1, n2, device=x1.device)
        
        for i in range(n1):
            for j in range(n2):
                distances[i, j] = self.hyperbolic_space.distance(x1[i], x2[j])
                
        return distances
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration parameters.
        
        Returns:
            Dictionary containing configuration parameters
        """
        config = super().get_config()
        config.update({
            'curvature': self.curvature,
            'use_tangent_space': self.use_tangent_space,
            'use_lorentz': self.use_lorentz
        })
        return config 