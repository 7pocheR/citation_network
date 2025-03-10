import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List
import math

from src.data.datasets import GraphData
from src.models.encoder.base import BaseEncoder
from src.models.encoder.hyperbolic import (
    HyperbolicTangentSpace, 
    HyperbolicLinear, 
    HyperbolicActivation,
    HyperbolicGRU,
    EuclideanToHyperbolic
)
from src.models.encoder.tgn import TimeEncoder

class HyperbolicTemporalEncoder(BaseEncoder):
    """
    Combined Hyperbolic and Temporal Graph Network encoder implementation.
    
    This encoder combines the benefits of both hyperbolic embeddings for hierarchical
    structures and temporal graph networks for dynamic evolution, making it
    particularly well-suited for evolving citation networks.
    """
    
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int,
                 embed_dim: int,
                 memory_dim: int = 100,
                 time_dim: int = 10,
                 message_dim: int = 100,
                 num_layers: int = 2,
                 curvature: float = 1.0,
                 dropout: float = 0.1,
                 use_gnn: bool = True):
        """
        Initialize the hyperbolic temporal encoder.
        
        Args:
            node_dim (int): Dimensionality of node features
            edge_dim (int): Dimensionality of edge features
            embed_dim (int): Dimensionality of output embeddings
            memory_dim (int, optional): Dimension of memory vectors. Defaults to 100.
            time_dim (int, optional): Dimension of time encoding. Defaults to 10.
            message_dim (int, optional): Dimension of message vectors. Defaults to 100.
            num_layers (int, optional): Number of hyperbolic layers. Defaults to 2.
            curvature (float, optional): Curvature of the hyperbolic space. Defaults to 1.0.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            use_gnn (bool, optional): Whether to use GNN for neighbor aggregation. Defaults to True.
        """
        super().__init__(node_dim, edge_dim, embed_dim)
        
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.message_dim = message_dim
        self.num_layers = num_layers
        self.curvature = curvature
        self.dropout = dropout
        self.use_gnn = use_gnn
        
        # Hyperbolic space operations
        self.hyp = HyperbolicTangentSpace(curvature)
        
        # Initial embedding layer to project node features to embedding space
        self.feature_projector = nn.Linear(node_dim, embed_dim)
        
        # Euclidean to Hyperbolic mapping
        self.e2h = EuclideanToHyperbolic(embed_dim, embed_dim, curvature)
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)
        
        # Message processing
        self.message_processor = nn.Sequential(
            nn.Linear(edge_dim + 2 * embed_dim + time_dim, message_dim * 2),
            nn.ReLU(),
            nn.Linear(message_dim * 2, message_dim)
        )
        
        # Hyperbolic GRU for temporal updates
        self.hyp_gru = HyperbolicGRU(message_dim, embed_dim, curvature)
        
        # Hyperbolic transformation layers
        self.hyp_layers = nn.ModuleList()
        for i in range(num_layers):
            self.hyp_layers.append(HyperbolicLinear(embed_dim, embed_dim, curvature=curvature))
            
        # Hyperbolic activation function
        self.hyp_activation = HyperbolicActivation(nn.ReLU(), curvature=curvature)
        
        # GNN layer for neighbor aggregation
        self.gnn = None
        if use_gnn:
            from torch_geometric.nn import GCNConv
            self.gnn = GCNConv(embed_dim, embed_dim)
        
        # Memory for node states
        self.memory = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def _resize_memory(self, num_nodes: int):
        """
        Resize the memory to accommodate the number of nodes.
        
        Args:
            num_nodes (int): Number of nodes in the graph
        """
        if self.memory.size(0) < num_nodes:
            device = self.memory.device
            
            # Create new memory tensors
            new_memory = torch.zeros(num_nodes, self.embed_dim, device=device)
            new_last_update = torch.zeros(num_nodes, device=device)
            
            # Copy existing data
            new_memory[:self.memory.size(0)] = self.memory
            new_last_update[:self.last_update.size(0)] = self.last_update
            
            # Replace old tensors
            self.memory = nn.Parameter(new_memory, requires_grad=False)
            self.last_update = nn.Parameter(new_last_update, requires_grad=False)
    
    def _process_messages(self, graph: GraphData) -> torch.Tensor:
        """
        Process messages for each node based on its neighbors.
        
        Args:
            graph (GraphData): Graph data object
            
        Returns:
            torch.Tensor: Message vectors for each node
        """
        device = graph.x.device
        edge_index = graph.edge_index
        num_nodes = graph.x.size(0)
        
        # Ensure memory is properly sized
        self._resize_memory(num_nodes)
        
        # Extract timestamps
        edge_times = None
        if hasattr(graph, 'edge_timestamps') and graph.edge_timestamps is not None:
            edge_times = graph.edge_timestamps
        elif isinstance(graph.edge_attr, dict) and 'time' in graph.edge_attr:
            edge_times = graph.edge_attr['time']
        elif hasattr(graph, 'edge_time') and graph.edge_time is not None:
            edge_times = graph.edge_time
        else:
            # If no timestamps, use fake timestamps
            edge_times = torch.ones(edge_index.size(1), device=device)
        
        # Extract edge features
        edge_features = None
        if isinstance(graph.edge_attr, dict) and 'attr' in graph.edge_attr:
            edge_features = graph.edge_attr['attr']
        elif graph.edge_attr is not None and not isinstance(graph.edge_attr, dict):
            edge_features = graph.edge_attr
        
        if edge_features is None:
            # If no edge features, use dummy features
            edge_features = torch.zeros(edge_index.size(1), self.edge_dim, device=device)
        
        # Calculate time delta
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        src_last_update = self.last_update[src_nodes]
        time_delta = edge_times - src_last_update
        
        # Encode time delta
        time_encoding = self.time_encoder(time_delta.unsqueeze(1)).squeeze(1)
        
        # Get node embeddings from memory
        src_embeddings = self.memory[src_nodes]
        dst_embeddings = self.memory[dst_nodes]
        
        # Create message inputs by concatenating source and destination embeddings,
        # edge features, and time encoding
        message_inputs = torch.cat([
            src_embeddings,
            dst_embeddings,
            edge_features,
            time_encoding
        ], dim=1)
        
        # Process messages
        messages = self.message_processor(message_inputs)
        
        # Aggregate messages for each target node (using mean)
        node_messages = torch.zeros(num_nodes, self.message_dim, device=device)
        node_counts = torch.zeros(num_nodes, device=device)
        
        # Aggregate messages for each destination node
        for i in range(edge_index.size(1)):
            dst = dst_nodes[i]
            node_messages[dst] += messages[i]
            node_counts[dst] += 1
        
        # Average the messages
        node_counts = node_counts.clamp(min=1).unsqueeze(1)
        node_messages = node_messages / node_counts
        
        # Update last update time for all nodes that received messages
        received_mask = node_counts.squeeze() > 0
        max_time = edge_times.max()
        self.last_update[received_mask] = max_time
        
        return node_messages
    
    def _apply_gnn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Apply GNN for neighbor aggregation.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge connectivity
            
        Returns:
            torch.Tensor: Updated node features
        """
        if self.gnn is None:
            return x
        
        # First convert from hyperbolic to Euclidean space
        x_euclidean = self.hyp.logmap0(x)
        
        # Apply GNN
        x_updated = self.gnn(x_euclidean, edge_index)
        x_updated = F.relu(x_updated)
        x_updated = F.dropout(x_updated, p=self.dropout, training=self.training)
        
        # Convert back to hyperbolic space
        return self.hyp.expmap0(x_updated)
    
    def forward(self, graph: GraphData) -> torch.Tensor:
        """
        Transform graph data into node embeddings.
        
        Args:
            graph (GraphData): Graph data object containing the citation network
                information, including nodes, edges, and their attributes.
                
        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, embed_dim]
        """
        # Extract node features and edge connectivity
        x = graph.x 
        edge_index = graph.edge_index
        num_nodes = x.size(0)
        
        # Ensure memory is properly sized
        self._resize_memory(num_nodes)
        
        # Process messages for temporal updates
        node_messages = self._process_messages(graph)
        
        # Project node features to embedding space
        x = self.feature_projector(x)
        
        # Map to hyperbolic space
        x = self.e2h(x)
        
        # Update memory with GRU using messages
        # First convert messages to hyperbolic space
        h_messages = self.e2h(node_messages)
        
        # Reshape for GRU (expecting batch_size, seq_len, input_dim)
        h_messages = h_messages.unsqueeze(0)
        current_memory = self.memory.unsqueeze(0)
        
        # Update memory with hyperbolic GRU
        updated_memory = self.hyp_gru(h_messages, current_memory)
        
        # Store updated memory
        self.memory.data = updated_memory.squeeze(0)
        
        # Apply hyperbolic transformations
        for layer in self.hyp_layers:
            x = layer(x)
            x = self.hyp_activation(x)
            
            # Apply GNN for neighbor aggregation if requested
            if self.use_gnn:
                x = self._apply_gnn(x, edge_index)
        
        # Combine with memory using hyperbolic addition
        for i in range(num_nodes):
            x[i] = self.hyp.mobius_addition(x[i], self.memory[i])
        
        return x
    
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
        if node_indices is None:
            # If no indices provided, encode the entire graph
            return self.forward(graph)
        
        # Create a subgraph with only the specified nodes
        device = graph.x.device
        
        # Extract features for specified nodes
        subgraph_x = graph.x[node_indices]
        
        # Filter edges to include only those connecting the specified nodes
        edge_mask = torch.zeros(graph.edge_index.shape[1], dtype=torch.bool, device=device)
        
        # Create a set of node indices for efficient lookup
        node_set = set(node_indices.cpu().tolist())
        
        # Check each edge to see if both endpoints are in the node set
        for i in range(graph.edge_index.shape[1]):
            src, dst = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
            if src in node_set and dst in node_set:
                edge_mask[i] = True
        
        # Extract filtered edges
        subgraph_edge_index = graph.edge_index[:, edge_mask]
        
        # Create a mapping from original indices to new indices
        idx_map = {idx.item(): i for i, idx in enumerate(node_indices)}
        
        # Reindex the edges
        for i in range(subgraph_edge_index.shape[1]):
            subgraph_edge_index[0, i] = idx_map[subgraph_edge_index[0, i].item()]
            subgraph_edge_index[1, i] = idx_map[subgraph_edge_index[1, i].item()]
        
        # Extract edge attributes for the subgraph
        subgraph_edge_attr = None
        if graph.edge_attr is not None:
            if isinstance(graph.edge_attr, dict):
                subgraph_edge_attr = {}
                for key, value in graph.edge_attr.items():
                    subgraph_edge_attr[key] = value[edge_mask]
            else:
                subgraph_edge_attr = graph.edge_attr[edge_mask]
        
        # Create a new GraphData object for the subgraph
        subgraph = GraphData(
            x=subgraph_x,
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr
        )
        
        # Now encode the subgraph
        return self.forward(subgraph)
    
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
        # Extract edge timestamps
        edge_times = None
        if hasattr(graph, 'edge_timestamps') and graph.edge_timestamps is not None:
            edge_times = graph.edge_timestamps
        elif isinstance(graph.edge_attr, dict) and 'time' in graph.edge_attr:
            edge_times = graph.edge_attr['time']
        elif hasattr(graph, 'edge_time') and graph.edge_time is not None:
            edge_times = graph.edge_time
        
        # If we can't find timestamps, return regular embeddings
        if edge_times is None:
            return self.forward(graph)
        
        # Create a mask for edges with timestamps <= threshold
        time_mask = edge_times <= time_threshold
        
        # Filter edges based on the mask
        filtered_edge_index = graph.edge_index[:, time_mask]
        
        # Create filtered edge attributes
        filtered_edge_attr = None
        if graph.edge_attr is not None:
            if isinstance(graph.edge_attr, dict):
                filtered_edge_attr = {}
                for key, value in graph.edge_attr.items():
                    filtered_edge_attr[key] = value[time_mask]
            else:
                filtered_edge_attr = graph.edge_attr[time_mask]
        
        # Create a filtered graph
        filtered_graph = GraphData(
            x=graph.x,
            edge_index=filtered_edge_index,
            edge_attr=filtered_edge_attr,
            node_timestamps=graph.node_timestamps if hasattr(graph, 'node_timestamps') else (
                graph.paper_times if hasattr(graph, 'paper_times') else None
            ),
            snapshot_time=torch.tensor([time_threshold])
        )
        
        # Now encode the filtered graph
        return self.forward(filtered_graph)
    
    def get_embedding_metric(self) -> str:
        """
        Get the name of the distance metric used in the embedding space.
        
        Returns:
            str: The name of the distance metric ('euclidean', 'hyperbolic', etc.)
        """
        return 'hyperbolic'
    
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
        # Compute pairwise hyperbolic distances
        n1, n2 = x1.size(0), x2.size(0)
        all_dists = torch.zeros((n1, n2), device=x1.device)
        
        for i in range(n1):
            for j in range(n2):
                all_dists[i, j] = self.hyp.distance(x1[i:i+1], x2[j:j+1])
        
        return all_dists
    
    def reset_memory(self):
        """Reset the memory to the initial state."""
        self.memory.data.zero_()
        self.last_update.data.zero_()
    
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
            'num_layers': self.num_layers,
            'curvature': self.curvature,
            'dropout': self.dropout,
            'use_gnn': self.use_gnn
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'HyperbolicTemporalEncoder':
        """
        Create an encoder instance from configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
                
        Returns:
            HyperbolicTemporalEncoder: An instance of the hyperbolic temporal encoder
        """
        return cls(**config) 