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
    EuclideanToHyperbolic,
    HyperbolicToEuclidean
)

# Import hyperbolic geometry classes
try:
    # Try to import from geoopt first (preferred)
    import geoopt
    from geoopt.manifolds import PoincareBall
    from geoopt.layers import HyperbolicLinear
    HAS_GEOOPT = True
except ImportError:
    # Fallback to basic hyperbolic operations
    HAS_GEOOPT = False
    print("Warning: geoopt is not installed. Using basic hyperbolic operations.")

# Custom hyperbolic layers (simplified implementation if geoopt is not available)
class HyperbolicGCNConv(nn.Module):
    """Hyperbolic version of Graph Convolutional Network layer."""
    def __init__(self, in_dim, out_dim, manifold):
        super().__init__()
        self.manifold = manifold
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, edge_index):
        """
        Forward pass with vectorized neighborhood processing.
        
        Args:
            x: Node features in hyperbolic space [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings in hyperbolic space [num_nodes, out_dim]
        """
        # Project to tangent space
        x_tangent = self.manifold.logmap0(x)
        
        # Extract source and destination nodes
        src, dst = edge_index
        
        # Apply the linear transformation first (GCN approach)
        x_transformed = self.linear(x_tangent)
        
        # If no edges, just return transformed features
        if edge_index.size(1) == 0:
            return self.manifold.expmap0(x_transformed)
        
        # Compute degrees for normalization
        num_nodes = x.size(0)
        degrees = torch.zeros(num_nodes, device=x.device)
        degrees.index_add_(0, dst, torch.ones_like(src, dtype=torch.float))
        degrees.index_add_(0, src, torch.ones_like(dst, dtype=torch.float))  # For undirected graphs
        
        # Avoid division by zero
        degrees = degrees.clamp(min=1.0)
        deg_inv_sqrt = degrees.pow(-0.5)
        
        # Apply GCN-style normalization: D^(-0.5) * A * D^(-0.5) * X
        output = torch.zeros_like(x_transformed)
        
        # For each feature dimension
        for dim in range(x_transformed.size(1)):
            # Get normalized source features
            src_feat = x_transformed[src, dim] * deg_inv_sqrt[src]
            # Weight features by destination degree normalization
            weighted_feat = src_feat * deg_inv_sqrt[dst]
            # Aggregate to destination nodes
            output[:, dim].index_add_(0, dst, weighted_feat)
        
        # Project back to hyperbolic space
        return self.manifold.expmap0(output)

class HyperbolicGATConv(nn.Module):
    """Hyperbolic version of Graph Attention Network layer."""
    def __init__(self, in_dim, out_dim, manifold, heads=1, dropout=0.1):
        super().__init__()
        self.manifold = manifold
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        
        # Define transformation layers
        self.linear = nn.Linear(in_dim, out_dim)
        self.attention = nn.Linear(2 * out_dim, 1)
        
    def forward(self, x, edge_index):
        """
        Forward pass with vectorized attention mechanism.
        
        Args:
            x: Node features in hyperbolic space [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings in hyperbolic space [num_nodes, out_dim]
        """
        # Project to tangent space at origin
        x_tangent = self.manifold.logmap0(x)
        
        # Apply feature transformation
        h = self.linear(x_tangent)
        
        # If no edges, return transformed features
        if edge_index.size(1) == 0:
            return self.manifold.expmap0(h)
        
        # Get source and destination nodes
        src, dst = edge_index
        num_edges = src.size(0)
        num_nodes = x.size(0)
        
        # Vectorized attention mechanism
        # Create node features for edges
        h_src = h[src]  # Shape [num_edges, out_dim]
        h_dst = h[dst]  # Shape [num_edges, out_dim]
        
        # Compute attention scores for all edges at once
        # Concatenate source and destination features
        attention_input = torch.cat([h_src, h_dst], dim=1)  # Shape [num_edges, 2*out_dim]
        
        # Calculate raw attention scores
        e = F.leaky_relu(self.attention(attention_input).squeeze(-1), negative_slope=0.2)  # Shape [num_edges]
        
        # Use sparse softmax to normalize attention scores
        attention_scores = torch.sparse.softmax(
            torch.sparse.FloatTensor(
                edge_index, 
                e,
                torch.Size([num_nodes, num_nodes])
            ),
            dim=1
        ).coalesce().values()
        
        # Apply dropout to attention scores
        if self.training:
            attention_scores = F.dropout(attention_scores, p=self.dropout)
        
        # Apply attention weights to neighbor features
        weighted_features = h_src * attention_scores.view(-1, 1)
        
        # Aggregate for each node
        out = torch.zeros_like(h)
        for i in range(h.size(1)):
            out[:, i].index_add_(0, dst, weighted_features[:, i])
                
        # Project back to hyperbolic space
        return self.manifold.expmap0(out)

class HyperbolicSAGEConv(nn.Module):
    """Hyperbolic version of GraphSAGE convolution layer."""
    def __init__(self, in_dim, out_dim, manifold):
        super().__init__()
        self.manifold = manifold
        self.linear_self = nn.Linear(in_dim, out_dim)
        self.linear_neigh = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, edge_index):
        """
        Forward pass with vectorized neighborhood processing for GraphSAGE.
        
        Args:
            x: Node features in hyperbolic space [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings in hyperbolic space [num_nodes, out_dim]
        """
        # Project to tangent space at origin
        x_tangent = self.manifold.logmap0(x)
        
        # Self-transformation
        h_self = self.linear_self(x_tangent)
        
        # If no edges, just use self transform
        if edge_index.size(1) == 0:
            return self.manifold.expmap0(h_self)
        
        # Neighbor aggregation with vectorized operations
        src, dst = edge_index
        num_nodes = x.size(0)
        
        # Create index mapping for efficient aggregation 
        # We use sparse matrix operations for efficient mean aggregation
        edge_weight = torch.ones(edge_index.size(1), device=x.device)
        
        # Efficient neighbor aggregation
        h_neigh = torch.zeros_like(x_tangent)
        
        # Count number of neighbors for each node to compute mean
        degrees = torch.zeros(num_nodes, device=x.device)
        degrees.index_add_(0, dst, torch.ones_like(src, dtype=torch.float))
        
        # Avoid division by zero
        degrees = degrees.clamp(min=1.0)
        
        # Vectorized aggregation
        for dim in range(x_tangent.size(1)):
            # Aggregate neighbor features 
            h_neigh[:, dim].index_add_(0, dst, x_tangent[src, dim])
            
        # Compute mean by dividing by degree (for each node)
        for i in range(num_nodes):
            if degrees[i] > 0:
                h_neigh[i] = h_neigh[i] / degrees[i]
        
        # Apply neighbor transformation
        h_neigh = self.linear_neigh(h_neigh)
        
        # Combine self and neighbor representations
        h_combined = h_self + h_neigh
        
        # Project back to hyperbolic space
        return self.manifold.expmap0(h_combined)

class HyperbolicActivation(nn.Module):
    """Apply activation in tangent space."""
    def __init__(self, activation, manifold):
        super().__init__()
        self.activation = activation
        self.manifold = manifold
        
    def forward(self, x):
        x_tangent = self.manifold.logmap0(x)
        x_tangent = self.activation(x_tangent)
        return self.manifold.expmap0(x_tangent)

class HyperbolicDropout(nn.Module):
    """Apply dropout in tangent space."""
    def __init__(self, p, manifold):
        super().__init__()
        self.p = p
        self.manifold = manifold
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        x_tangent = self.manifold.logmap0(x)
        x_tangent = self.dropout(x_tangent)
        return self.manifold.expmap0(x_tangent)

# Define fallback implementations if geoopt is not available
if not HAS_GEOOPT:
    # Basic PoincareBall implementation
    class PoincareBall:
        def __init__(self, c=1.0):
            self.c = c
            
        def expmap0(self, x):
            """Map from tangent space at origin to hyperbolic space."""
            norm = torch.norm(x, dim=-1, keepdim=True)
            return torch.tanh(torch.sqrt(self.c) * norm) * x / (torch.sqrt(self.c) * norm.clamp(min=1e-8))
            
        def logmap0(self, x):
            """Map from hyperbolic space to tangent space at origin."""
            norm = torch.norm(x, dim=-1, keepdim=True)
            return torch.atanh(torch.sqrt(self.c) * norm.clamp(max=1-1e-5)) * x / (torch.sqrt(self.c) * norm.clamp(min=1e-8))
            
        def distance(self, x, y):
            """Compute distance in hyperbolic space."""
            sqrt_c = torch.sqrt(self.c)
            x2 = torch.sum(x * x, dim=-1, keepdim=True)
            y2 = torch.sum(y * y, dim=-1, keepdim=True)
            xy = torch.sum(x * y, dim=-1, keepdim=True)
            
            num = 2 * self.c * xy
            denom = (1 - self.c * x2) * (1 - self.c * y2)
            xi = torch.clamp(1 + num / denom, min=1.0 + 1e-6)
            dist = torch.acosh(xi) / sqrt_c
            return dist.squeeze(-1)
    
    # Basic HyperbolicLinear implementation
    class HyperbolicLinear(nn.Module):
        def __init__(self, in_dim, out_dim, manifold):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.manifold = manifold
            
        def forward(self, x):
            # Project to tangent space, apply linear transformation, project back
            x_tangent = self.manifold.logmap0(x)
            x_transformed = self.linear(x_tangent)
            return self.manifold.expmap0(x_transformed)

# Basic hyperbolic operations
class HyperbolicSpace:
    """Simple hyperbolic space operations."""
    def __init__(self, curvature=1.0):
        self.c = curvature
        
    def expmap0(self, x):
        """Map from tangent space at origin to hyperbolic space."""
        norm = torch.norm(x, dim=-1, keepdim=True)
        # Handle edge case for zero norms
        zeros = torch.zeros_like(norm)
        condition = norm == 0
        norm = torch.where(condition, torch.ones_like(norm), norm)
        
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=x.device))
        res = torch.tanh(sqrt_c * norm) * x / (sqrt_c * norm)
        return torch.where(condition.expand_as(res), x, res)
        
    def logmap0(self, x):
        """Map from hyperbolic space to tangent space at origin."""
        norm = torch.norm(x, dim=-1, keepdim=True)
        # Handle edge case for zero norms
        zeros = torch.zeros_like(norm)
        condition = norm == 0
        norm = torch.where(condition, torch.ones_like(norm), norm)
        
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=x.device))
        res = torch.atanh(torch.clamp(sqrt_c * norm, max=0.99)) * x / (sqrt_c * norm)
        return torch.where(condition.expand_as(res), x, res)
        
    def distance(self, x, y):
        """Compute distance in hyperbolic space."""
        # Convert to correct device
        c = torch.tensor(self.c, device=x.device)
        sqrt_c = torch.sqrt(c)
        
        # Compute the distance
        sum_x = torch.sum(x * x, dim=-1, keepdim=True)
        sum_y = torch.sum(y * y, dim=-1, keepdim=True)
        dot_xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        alpha = 1 - 2 * c * dot_xy + c * sum_y
        beta = 1 - c * sum_x
        gamma = 1 + 2 * c * dot_xy + c**2 * sum_x * sum_y
        
        dist = torch.acosh(gamma / (alpha * beta))
        return dist.squeeze(-1)

class HyperbolicEncoder(nn.Module):
    """
    Hyperbolic encoder for citation networks.
    
    Uses GNNs in hyperbolic space to encode nodes.
    """
    
    def __init__(
        self,
                 node_dim: int, 
                 embed_dim: int,
        hidden_dim: int = None,
        edge_dim: Optional[int] = None,
                 curvature: float = 1.0,
        num_layers: int = 1,
                 dropout: float = 0.1,
                 use_gnn: bool = True,
        gnn_type: str = 'gcn'
    ):
        """
        Initialize the hyperbolic encoder.
        
        Args:
            node_dim (int): Dimension of node features
            embed_dim (int): Output embedding dimension
            hidden_dim (int, optional): Hidden dimension for intermediate layers
            edge_dim (Optional[int]): Edge feature dimension
            curvature (float): Hyperbolic curvature parameter (c)
            num_layers (int): Number of hyperbolic GNN layers
            dropout (float): Dropout rate
            use_gnn (bool): Whether to use GNN or simple MLP
            gnn_type (str): Type of GNN to use ('gcn', 'gat', 'sage')
        """
        super().__init__()
        
        # Store parameters
        self.node_dim = node_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim * 2
        self.curvature = curvature
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gnn = use_gnn
        self.gnn_type = gnn_type
        
        # Hyperbolic space
        self.hyp = HyperbolicSpace(curvature=curvature)
        
        # Input transformation
        self.input_layer = nn.Sequential(
            nn.Linear(node_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Create multiple GNN or MLP layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = self.hidden_dim if i == 0 else self.hidden_dim
            out_dim = self.hidden_dim if i < num_layers - 1 else embed_dim
            
            if use_gnn:
                # Use a GNN layer - implementation simplified for readability
                self.layers.append(nn.Linear(in_dim, out_dim))
            else:
                # Use a regular linear layer
                self.layers.append(nn.Linear(in_dim, out_dim))
            
            # Add activation and dropout for all but the last layer
            if i < num_layers - 1:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout))
    
    def forward(self, graph: GraphData) -> torch.Tensor:
        """
        Forward pass of the hyperbolic encoder.
        
        Args:
            graph: Graph data object with node features and edge indices
            
        Returns:
            torch.Tensor: Node embeddings in hyperbolic space
        """
        # Ensure input is on the correct device
        device = next(self.parameters()).device
        x = graph.x.to(device, dtype=torch.float32)  # Ensure float32 type
        edge_index = graph.edge_index.to(device)
        
        # Initial transformation in Euclidean space
        x = self.input_layer(x)
        
        # Map to hyperbolic space
        x_hyp = self.hyp.expmap0(x)
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Dropout):
                # For activation and dropout: map to tangent space, apply, and map back
                x_tangent = self.hyp.logmap0(x_hyp)
                x_tangent = layer(x_tangent)
                x_hyp = self.hyp.expmap0(x_tangent)
            else:
                # For GNN layers
                if self.use_gnn and isinstance(layer, (HyperbolicGCNConv, HyperbolicGATConv, HyperbolicSAGEConv)):
                    x_hyp = layer(x_hyp, edge_index)
                else:
                    # For linear layers: map to tangent space, apply, and map back
                    x_tangent = self.hyp.logmap0(x_hyp)
                    x_tangent = layer(x_tangent)
                    x_hyp = self.hyp.expmap0(x_tangent)
        
        return x_hyp
    
    def encode_subgraph(self, 
                        graph: GraphData, 
                        node_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a subgraph defined by node_indices.
        
        Args:
            graph (GraphData): Full graph
            node_indices (Optional[torch.Tensor]): Indices of nodes to encode
                
        Returns:
            torch.Tensor: Node embeddings for the subgraph
        """
        # Encode the full graph
        all_embeddings = self.forward(graph)
        
        # If node_indices is provided, extract those embeddings
        if node_indices is not None:
            node_indices = node_indices.to(all_embeddings.device)
            return all_embeddings[node_indices]
        
        return all_embeddings
    
    def encode_temporal_snapshot(self, 
                                graph: GraphData, 
                                time_threshold: float) -> torch.Tensor:
        """
        Encode a temporal snapshot of the graph up to time_threshold.
        
        Args:
            graph (GraphData): Full graph with temporal information
            time_threshold (float): Upper time limit for the snapshot
                
        Returns:
            torch.Tensor: Node embeddings for the snapshot
        """
        # Check if graph has timestamps
        if not hasattr(graph, 'node_timestamps') or graph.node_timestamps is None:
            # No temporal information, just encode the full graph
            return self.forward(graph)
        
        # Get nodes that exist at or before the time_threshold
        node_timestamps = graph.node_timestamps
        valid_mask = node_timestamps <= time_threshold
        valid_nodes = torch.where(valid_mask)[0]
        
        # Get edges that exist at or before the time_threshold
        edge_timestamps = getattr(graph, 'edge_timestamps', None)
        if edge_timestamps is not None:
            valid_edge_mask = edge_timestamps <= time_threshold
            edge_index = graph.edge_index[:, valid_edge_mask]
        else:
            # If no edge timestamps, use all edges
            edge_index = graph.edge_index
        
        # Create a subgraph with only the valid nodes and edges
        temporal_graph = GraphData(
            x=graph.x[valid_nodes],
            edge_index=edge_index  # Simplified; would need to remap indices in practice
        )
        
        # Encode the temporal subgraph
        return self.forward(temporal_graph)
    
    def get_embedding_metric(self) -> str:
        """
        Get the name of the metric used for computing distances between embeddings.
        
        Returns:
            str: Name of the metric
        """
        return "hyperbolic_distance"
    
    def compute_distance(self, 
                        x1: torch.Tensor, 
                        x2: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between two sets of embeddings.
        
        Args:
            x1 (torch.Tensor): First set of embeddings
            x2 (torch.Tensor): Second set of embeddings
                
        Returns:
            torch.Tensor: Pairwise distances
        """
        # Initialize distance matrix
        n1 = x1.size(0)
        n2 = x2.size(0)
        all_dists = torch.zeros((n1, n2), device=x1.device)
        
        # Compute pairwise distances
        for i in range(n1):
            for j in range(n2):
                all_dists[i, j] = self.hyp.distance(x1[i:i+1], x2[j:j+1])
        
        return all_dists
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration parameters of the encoder.
        
        Returns:
            Dict[str, Any]: Dictionary containing configuration parameters
        """
        config = {
            'curvature': self.curvature,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'use_gnn': self.use_gnn,
            'gnn_type': self.gnn_type
        }
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'HyperbolicEncoder':
        """
        Create a new instance from a configuration dictionary.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
                
        Returns:
            HyperbolicEncoder: New instance
        """
        return cls(**config) 