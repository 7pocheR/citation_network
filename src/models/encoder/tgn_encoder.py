import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union, List
import time
import numpy as np

from src.data.datasets import GraphData
from src.models.encoder.base import BaseEncoder
from src.models.encoder.tgn import TemporalGraphNetwork, MemoryModule, MessageFunction, TimeEncoder

class TGNEncoder(BaseEncoder):
    """
    Temporal Graph Network (TGN) encoder implementation that conforms to the BaseEncoder interface.
    
    This encoder transforms graph structure and node features into embeddings using temporal dynamics 
    and message passing over the citation network. It captures both structural relationships and 
    the temporal evolution of the network.
    """
    
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int,
                 embed_dim: int,
                 memory_dim: int = 100,
                 time_dim: int = 10,
                 message_dim: int = 100,
                 num_nodes: int = None,
                 num_gnn_layers: int = 2,
                 use_memory: bool = True,
                 aggregator_type: str = 'last',
                 hyperbolic: bool = False,
                 curvature: float = 1.0,
                 dropout: float = 0.1):
        """
        Initialize the TGN encoder.
        
        Args:
            node_dim (int): Dimensionality of node features
            edge_dim (int): Dimensionality of edge features
            embed_dim (int): Dimensionality of output embeddings
            memory_dim (int, optional): Dimension of memory vectors. Defaults to 100.
            time_dim (int, optional): Dimension of time encoding. Defaults to 10.
            message_dim (int, optional): Dimension of message vectors. Defaults to 100.
            num_nodes (int, optional): Number of nodes in the graph. If None, determined at runtime.
            num_gnn_layers (int, optional): Number of GNN layers. Defaults to 2.
            use_memory (bool, optional): Whether to use memory module. Defaults to True.
            aggregator_type (str, optional): Aggregation type ('mean', 'max', or 'last'). Defaults to 'last'.
            hyperbolic (bool, optional): Whether to use hyperbolic embeddings. Defaults to False.
            curvature (float, optional): Curvature of hyperbolic space. Defaults to 1.0.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__(node_dim, edge_dim, embed_dim)
        
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.message_dim = message_dim
        self.num_gnn_layers = num_gnn_layers
        self.use_memory = use_memory
        self.aggregator_type = aggregator_type
        self.hyperbolic = hyperbolic
        self.curvature = curvature
        self.dropout = dropout
        
        # If num_nodes is not provided, it will be determined when first processing a graph
        self.num_nodes = num_nodes
        self.tgn_initialized = False
        self.tgn = None
        
    def _ensure_tgn_initialized(self, graph: GraphData):
        """Initialize the TGN if not already done."""
        if not self.tgn_initialized:
            # Get the number of nodes from the graph if not provided
            if self.num_nodes is None and graph is not None:
                self.num_nodes = graph.x.size(0)
            
            # Initialize the TGN
            self.tgn = TemporalGraphNetwork(
                num_nodes=self.num_nodes,
                node_feature_dim=self.node_dim,
                memory_dim=self.memory_dim,
                time_dim=self.time_dim,
                embedding_dim=self.embed_dim,
                message_dim=self.message_dim,
                edge_dim=self.edge_dim,
                num_gnn_layers=self.num_gnn_layers,
                use_memory=self.use_memory,
                aggregator_type=self.aggregator_type,
                hyperbolic=self.hyperbolic,
                curvature=self.curvature,
                dropout=self.dropout
            )
            
            # Ensure model parameters are on the correct device
            device = graph.x.device if graph is not None and hasattr(graph, 'x') else next(self.parameters()).device
            self.tgn = self.tgn.to(device)
            
            self.tgn_initialized = True
    
    def forward(self, graph: GraphData) -> torch.Tensor:
        """
        Transform graph data into node embeddings.
        
        Args:
            graph (GraphData): Graph data object containing the citation network
                information, including nodes, edges, and their attributes.
                
        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, embed_dim]
        """
        # Ensure TGN is initialized
        self._ensure_tgn_initialized(graph)
        
        # For forward, we'll treat the graph as a single snapshot
        return self.tgn([graph])
    
    def encode_subgraph(self, 
                        graph: GraphData, 
                        node_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a subgraph of the citation network.
        
        Used for encoding specific parts of the graph, such as time-based
        snapshots or other subnetworks.
        
        Args:
            graph (GraphData): The complete graph data
            node_indices (Optional[torch.Tensor]): Indices of nodes to include in the subgraph.
                If None, encodes the entire graph.
                
        Returns:
            torch.Tensor: Node embeddings for the subgraph [num_selected_nodes, embed_dim]
        """
        # First encode the entire graph
        all_embeddings = self.forward(graph)
        
        # Then select the embeddings for the specified nodes
        if node_indices is not None:
            return all_embeddings[node_indices]
        else:
            return all_embeddings
    
    def encode_temporal_snapshot(self, 
                                graph: GraphData, 
                                time_threshold: float) -> torch.Tensor:
        """
        Encode a temporal snapshot of the citation network.
        
        Encodes only the part of the graph with timestamps less than or equal
        to the specified threshold, representing the state of the citation network
        at a particular point in time.
        
        Args:
            graph (GraphData): The complete graph data
            time_threshold (float): The timestamp threshold to use for the snapshot
                
        Returns:
            torch.Tensor: Node embeddings for the temporal snapshot
        """
        # Ensure TGN is initialized
        self._ensure_tgn_initialized(graph)
        
        # Extract edge timestamps using standardized field name first
        edge_timestamps = None
        if hasattr(graph, 'edge_timestamps') and graph.edge_timestamps is not None:
            edge_timestamps = graph.edge_timestamps
        # Fall back to legacy field access patterns if needed
        elif isinstance(graph.edge_attr, dict) and 'time' in graph.edge_attr:
            edge_timestamps = graph.edge_attr['time']
        elif hasattr(graph, 'edge_time') and graph.edge_time is not None:
            edge_timestamps = graph.edge_time
        
        # If we can't find timestamps, return regular embeddings
        if edge_timestamps is None:
            return self.forward(graph)
        
        # Create a mask for edges with timestamps <= threshold
        time_mask = edge_timestamps <= time_threshold
        
        # Filter edges based on the mask
        filtered_edge_index = graph.edge_index[:, time_mask]
        
        # Get the set of nodes in the filtered graph
        filtered_nodes = torch.unique(filtered_edge_index)
        
        # Encode the subgraph
        return self.encode_subgraph(graph, filtered_nodes)
    
    def get_embedding_metric(self) -> str:
        """
        Get the name of the distance metric used in the embedding space.
        
        Returns:
            str: The name of the distance metric ('euclidean', 'hyperbolic', etc.)
        """
        return 'hyperbolic' if self.hyperbolic else 'euclidean'
    
    def compute_distance(self, 
                        x1: torch.Tensor, 
                        x2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between embeddings in the encoder's space.
        
        Args:
            x1 (torch.Tensor): First set of embeddings
            x2 (torch.Tensor): Second set of embeddings
                
        Returns:
            torch.Tensor: Pairwise distances between embeddings
        """
        if self.hyperbolic and self.tgn_initialized:
            # Use hyperbolic distance if using hyperbolic embeddings
            # We need the tangent space from the TGN to compute this
            hyp_space = self.tgn.hyp_tangent_space
            
            # Compute pairwise hyperbolic distances
            n1, n2 = x1.size(0), x2.size(0)
            all_dists = torch.zeros((n1, n2), device=x1.device)
            
            for i in range(n1):
                for j in range(n2):
                    all_dists[i, j] = hyp_space.distance(x1[i:i+1], x2[j:j+1])
            
            return all_dists
        else:
            # Default to Euclidean distance
            return torch.cdist(x1, x2, p=2)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration parameters.
        
        Returns:
            Dict[str, Any]: Dictionary containing configuration parameters
        """
        config = super().get_config()
        config.update({
            'memory_dim': self.memory_dim,
            'time_dim': self.time_dim,
            'message_dim': self.message_dim,
            'num_nodes': self.num_nodes,
            'num_gnn_layers': self.num_gnn_layers,
            'use_memory': self.use_memory,
            'aggregator_type': self.aggregator_type,
            'hyperbolic': self.hyperbolic,
            'curvature': self.curvature,
            'dropout': self.dropout
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TGNEncoder':
        """
        Create an encoder instance from configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
                
        Returns:
            TGNEncoder: An instance of the TGN encoder
        """
        return cls(**config) 