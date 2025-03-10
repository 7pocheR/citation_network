import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import time
import numpy as np
from torch_geometric.nn import GCNConv, GATConv
from .hyperbolic import HyperbolicTangentSpace, EuclideanToHyperbolic, HyperbolicGRU
from src.data.dataset import GraphData


class MemoryModule(nn.Module):
    """
    Memory module for Temporal Graph Networks that maintains and updates node states.
    """
    
    def __init__(self, 
                 num_nodes: int, 
                 memory_dim: int, 
                 time_dim: int = 10,
                 message_dim: int = 100):
        """
        Initialize the memory module.
        
        Args:
            num_nodes: Number of nodes in the graph
            memory_dim: Dimension of memory vectors
            time_dim: Dimension of time encoding
            message_dim: Dimension of messages
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        
        # Memory state for each node
        self.memory = nn.Parameter(torch.zeros(num_nodes, memory_dim), requires_grad=False)
        # Last update time for each node
        self.last_update = nn.Parameter(torch.zeros(num_nodes), requires_grad=False)
        
        # GRU for memory updates
        self.gru = nn.GRUCell(message_dim + time_dim, memory_dim)
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)
    
    def get_memory(self, node_idxs: torch.Tensor) -> torch.Tensor:
        """Get memory for selected nodes."""
        return self.memory[node_idxs]
    
    def get_last_update(self, node_idxs: torch.Tensor) -> torch.Tensor:
        """Get last update time for selected nodes."""
        return self.last_update[node_idxs]
    
    def update_memory(self, 
                     node_idxs: torch.Tensor, 
                     messages: torch.Tensor, 
                     timestamps: torch.Tensor):
        """
        Update memory for nodes with new messages.
        
        Args:
            node_idxs: Indices of nodes to update
            messages: Message vectors for each node
            timestamps: Timestamps for each update
        """
        # Get current memory state for these nodes
        curr_memory = self.memory[node_idxs]
        
        # Compute time delta since last update
        last_update = self.last_update[node_idxs]
        delta_t = timestamps - last_update
        
        # Encode the time difference
        time_encoding = self.time_encoder(delta_t.unsqueeze(1)).squeeze(1)
        
        # Concatenate message and time encoding
        inputs = torch.cat([messages, time_encoding], dim=1)
        
        # Update memory using GRU
        new_memory = self.gru(inputs, curr_memory)
        
        # Update memory and last_update time
        self.memory[node_idxs] = new_memory
        self.last_update[node_idxs] = timestamps
    
    def reset_memory(self):
        """Reset all memory states to initial values."""
        self.memory.data.zero_()
        self.last_update.data.zero_()


class MessageFunction(nn.Module):
    """
    Message function for TGN that computes messages from source to target nodes.
    """
    
    def __init__(self, 
                 memory_dim: int, 
                 message_dim: int, 
                 edge_dim: Optional[int] = None):
        """
        Initialize the message function.
        
        Args:
            memory_dim: Dimension of memory vectors
            message_dim: Dimension of message vectors
            edge_dim: Dimension of edge features (if available)
        """
        super().__init__()
        input_dim = 2 * memory_dim  # Source and target memory
        if edge_dim is not None:
            input_dim += edge_dim
            
        # MLP for computing messages
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * message_dim),
            nn.ReLU(),
            nn.Linear(2 * message_dim, message_dim)
        )
    
    def forward(self, 
                source_memory: torch.Tensor, 
                target_memory: torch.Tensor, 
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute messages from source to target nodes.
        
        Args:
            source_memory: Memory vectors of source nodes
            target_memory: Memory vectors of target nodes
            edge_features: Edge feature vectors (optional)
            
        Returns:
            Message vectors for each edge
        """
        if edge_features is not None:
            inputs = torch.cat([source_memory, target_memory, edge_features], dim=1)
        else:
            inputs = torch.cat([source_memory, target_memory], dim=1)
            
        return self.mlp(inputs)


class TimeEncoder(nn.Module):
    """
    Time encoder that embeds time values into fixed-dimensional vectors.
    Uses sinusoidal position encoding similar to transformers.
    """
    
    def __init__(self, dim: int):
        """
        Initialize the time encoder.
        
        Args:
            dim: Dimension of time encoding
        """
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.ones(1, dim))
        self.b = nn.Parameter(torch.zeros(1, dim))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time values.
        
        Args:
            t: Time tensor of shape [batch_size, 1]
            
        Returns:
            Time encoding of shape [batch_size, dim]
        """
        # Transform time using sine function
        t = t.unsqueeze(2) if t.dim() == 2 else t.unsqueeze(1)
        t = t * self.w + self.b
        return torch.sin(t)


class TemporalGraphNetwork(nn.Module):
    """
    Temporal Graph Network (TGN) with hyperbolic embeddings for dynamic citation networks.
    
    This model handles dynamic graph evolution over time and outputs temporally-aware
    node embeddings in hyperbolic space.
    """
    
    def __init__(self, 
                 num_nodes: int,
                 node_feature_dim: int,
                 memory_dim: int = 100,
                 time_dim: int = 10,
                 embedding_dim: int = 128,
                 message_dim: int = 100,
                 edge_dim: Optional[int] = None,
                 num_gnn_layers: int = 2,
                 use_memory: bool = True,
                 aggregator_type: str = 'last',
                 hyperbolic: bool = True,
                 curvature: float = 1.0,
                 dropout: float = 0.1):
        """
        Initialize the TGN model.
        
        Args:
            num_nodes: Number of nodes in the graph
            node_feature_dim: Dimension of node features
            memory_dim: Dimension of memory vectors
            time_dim: Dimension of time encoding
            embedding_dim: Dimension of output node embeddings
            message_dim: Dimension of message vectors
            edge_dim: Dimension of edge features (if available)
            num_gnn_layers: Number of GNN layers
            use_memory: Whether to use memory module or not
            aggregator_type: Aggregation type ('mean', 'max', or 'last')
            hyperbolic: Whether to use hyperbolic embeddings
            curvature: Curvature of hyperbolic space
            dropout: Dropout probability
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.use_memory = use_memory
        self.aggregator_type = aggregator_type
        self.hyperbolic = hyperbolic
        self.dropout = dropout
        
        # Node embedding layer
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # Feature projection
        self.feature_projector = nn.Linear(node_feature_dim, embedding_dim)
        
        # Memory module
        if use_memory:
            self.memory = MemoryModule(
                num_nodes=num_nodes,
                memory_dim=memory_dim,
                time_dim=time_dim,
                message_dim=message_dim
            )
            
            # Message function
            self.message_function = MessageFunction(
                memory_dim=memory_dim,
                message_dim=message_dim,
                edge_dim=edge_dim
            )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        input_dim = embedding_dim + memory_dim if use_memory else embedding_dim
        
        for i in range(num_gnn_layers):
            if i == 0:
                self.gnn_layers.append(GATConv(input_dim, embedding_dim))
            else:
                self.gnn_layers.append(GATConv(embedding_dim, embedding_dim))
        
        # Hyperbolic components
        if hyperbolic:
            self.hyp_tangent_space = HyperbolicTangentSpace(curvature)
            self.euclidean_to_hyperbolic = EuclideanToHyperbolic(embedding_dim, embedding_dim, curvature)
            self.hyp_gru = HyperbolicGRU(embedding_dim, embedding_dim, curvature)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset model parameters."""
        if hasattr(self, 'memory') and self.use_memory:
            self.memory.reset_memory()
        
        nn.init.xavier_uniform_(self.node_embedding.weight)
    
    def compute_temporal_embeddings(self, 
                                   node_features: torch.Tensor, 
                                   edge_index: torch.Tensor, 
                                   edge_timestamps: Optional[torch.Tensor] = None,
                                   hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute temporal embeddings for nodes in a graph snapshot.
        
        Args:
            node_features: Node features tensor
            edge_index: Edge index tensor
            edge_timestamps: Edge timestamps tensor or dictionary containing time information
            hidden_state: Previous hidden state
            
        Returns:
            torch.Tensor: Temporal node embeddings
        """
        device = node_features.device
        
        # Project node features
        x = self.feature_projector(node_features)
        
        # Get initial node embeddings
        node_emb = self.node_embedding.weight
        
        # Combine node embeddings with projected features
        x = x + node_emb[:node_features.size(0)]
        
        # Handle different forms of edge timestamps
        if edge_timestamps is None:
            # No timestamp information available
            pass
        elif isinstance(edge_timestamps, dict):
            # Try to find standardized timestamp data first
            if 'edge_timestamps' in edge_timestamps:
                edge_timestamps = edge_timestamps['edge_timestamps']
            # Fall back to legacy field name
            elif 'time' in edge_timestamps:
                edge_timestamps = edge_timestamps['time']
        
        # Apply time encoding to edge timestamps if using memory
        if self.use_memory and edge_timestamps is not None:
            # Encode the timestamps
            time_encodings = self.memory.time_encoder(edge_timestamps)
            
            # Use time encodings in message passing or attention mechanism
            # This would need to be integrated with the GATConv implementation
        
        # Apply GNN layers
        for gnn in self.gnn_layers:
            x = gnn(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Convert to hyperbolic space if needed
        if self.hyperbolic:
            x = self.euclidean_to_hyperbolic(x)
        
        # Update hidden state with GRU if provided
        if hidden_state is not None:
            # Reshape x to match hidden state dimensions
            x_expanded = x.unsqueeze(0)  # Add batch dimension
            
            if self.hyperbolic:
                # Update in hyperbolic space
                hidden_state = self.hyp_gru(x_expanded, hidden_state)
            else:
                # Update in Euclidean space (would need a regular GRU)
                pass
            
            return hidden_state
        else:
            # If no hidden state provided, just return the current embeddings
            # with a batch dimension added
            return x.unsqueeze(0)
    
    def forward(self, snapshots: List[GraphData]) -> torch.Tensor:
        """
        Forward pass through the temporal graph network.
        
        Args:
            snapshots: List of graph snapshots in temporal order
            
        Returns:
            torch.Tensor: Node embeddings
        """
        device = next(self.parameters()).device
        
        # Reset memory at the beginning of a new sequence
        if self.use_memory:
            self.memory.reset_memory()
        
        # Process each snapshot in temporal order
        all_embeddings = []
        for snapshot in snapshots:
            # Move data to device
            snapshot.x = snapshot.x.to(device)
            snapshot.edge_index = snapshot.edge_index.to(device)
            
            # Handle edge attributes - could be dict or tensor
            if isinstance(snapshot.edge_attr, dict):
                # Move each tensor in the dict to the device
                for key, value in snapshot.edge_attr.items():
                    if isinstance(value, torch.Tensor):
                        snapshot.edge_attr[key] = value.to(device)
            elif snapshot.edge_attr is not None:
                # If it's a tensor, move it directly
                snapshot.edge_attr = snapshot.edge_attr.to(device)
            
            # Get node features and edge structure
            node_features = snapshot.x
            edge_index = snapshot.edge_index

            # Get edge timestamps using standardized field name
            if hasattr(snapshot, 'edge_timestamps') and snapshot.edge_timestamps is not None:
                edge_timestamps = snapshot.edge_timestamps
            else:
                # Fall back to legacy field access patterns
                edge_timestamps = snapshot.edge_attr
            
            # Compute temporal embeddings
            h = self.compute_temporal_embeddings(node_features, edge_index, edge_timestamps)
            
            all_embeddings.append(h)
        
        # Return final embeddings (squeeze batch dimension)
        return all_embeddings[-1]
    
    def encode_node_history(self, 
                           node_idxs: torch.Tensor, 
                           temporal_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode the history of nodes across temporal embeddings.
        
        Args:
            node_idxs: Indices of nodes to encode
            temporal_embeddings: List of node embeddings at different time steps
            
        Returns:
            Historical encoding for specified nodes
        """
        node_embeddings = [emb[node_idxs] for emb in temporal_embeddings]
        
        # Aggregate temporal embeddings based on specified strategy
        if self.aggregator_type == 'mean':
            return torch.stack(node_embeddings).mean(dim=0)
        elif self.aggregator_type == 'max':
            return torch.stack(node_embeddings).max(dim=0)[0]
        elif self.aggregator_type == 'last':
            return node_embeddings[-1]
        else:
            raise ValueError(f"Unknown aggregator type: {self.aggregator_type}")
            
    def get_embedding(self, node_idx: int) -> torch.Tensor:
        """Get the current embedding for a specific node."""
        if self.hyperbolic:
            # Return node embedding in hyperbolic space
            return self.euclidean_to_hyperbolic(self.node_embedding.weight[node_idx].unsqueeze(0)).squeeze(0)
        else:
            return self.node_embedding.weight[node_idx] 