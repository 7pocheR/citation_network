import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union

from src.data.datasets import GraphData


class BaseEncoder(nn.Module, ABC):
    """Base interface for all graph encoders in the citation network project.
    
    The encoder transforms the graph structure and node features into node embeddings.
    These embeddings capture the structural and semantic information of papers in the
    citation network, accounting for both the graph topology and node attributes.
    
    All encoder implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int,
                 embed_dim: int,
                 **kwargs):
        """Initialize the base encoder.
        
        Args:
            node_dim (int): Dimensionality of node features
            edge_dim (int): Dimensionality of edge features
            embed_dim (int): Dimensionality of output embeddings
            **kwargs: Additional parameters specific to the encoder implementation
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.embed_dim = embed_dim
        
    @abstractmethod
    def forward(self, 
                graph: GraphData) -> torch.Tensor:
        """Transform graph data into node embeddings.
        
        Args:
            graph (GraphData): Graph data object containing the citation network
                information, including nodes, edges, and their attributes.
                
        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, embed_dim]
        """
        pass
    
    @abstractmethod
    def encode_subgraph(self, 
                        graph: GraphData, 
                        node_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a subgraph of the citation network.
        
        Used for encoding specific parts of the graph, such as time-based
        snapshots or other subnetworks.
        
        Args:
            graph (GraphData): The complete graph data
            node_indices (Optional[torch.Tensor]): Indices of nodes to include in the subgraph.
                If None, encodes the entire graph.
                
        Returns:
            torch.Tensor: Node embeddings for the subgraph [num_selected_nodes, embed_dim]
        """
        pass
    
    @abstractmethod
    def encode_temporal_snapshot(self, 
                                graph: GraphData, 
                                time_threshold: float) -> torch.Tensor:
        """Encode a temporal snapshot of the citation network.
        
        Encodes only the part of the graph with timestamps less than or equal
        to the specified threshold, representing the state of the citation network
        at a particular point in time.
        
        Args:
            graph (GraphData): The complete graph data
            time_threshold (float): The timestamp threshold to use for the snapshot
                
        Returns:
            torch.Tensor: Node embeddings for the temporal snapshot
        """
        pass
    
    def get_embedding_metric(self) -> str:
        """Get the name of the distance metric used in the embedding space.
        
        Returns:
            str: The name of the distance metric ('euclidean', 'hyperbolic', etc.)
        """
        return 'euclidean'  # Default metric
    
    def compute_distance(self, 
                        x1: torch.Tensor, 
                        x2: torch.Tensor) -> torch.Tensor:
        """Compute distance between embeddings in the encoder's space.
        
        Args:
            x1 (torch.Tensor): First set of embeddings
            x2 (torch.Tensor): Second set of embeddings
                
        Returns:
            torch.Tensor: Pairwise distances between embeddings
        """
        # Default Euclidean distance implementation
        return torch.cdist(x1, x2, p=2)
    
    def get_config(self) -> Dict[str, Any]:
        """Get encoder configuration parameters.
        
        Returns:
            Dict[str, Any]: Dictionary containing configuration parameters
        """
        return {
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'embed_dim': self.embed_dim,
            'embedding_metric': self.get_embedding_metric(),
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseEncoder':
        """Create an encoder instance from configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
                
        Returns:
            BaseEncoder: An instance of the encoder
        """
        return cls(**config) 