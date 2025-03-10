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


class HyperbolicMessagePassing(nn.Module):
    """
    Hyperbolic message passing layer for graph neural networks.
    
    This layer implements message passing in hyperbolic space by:
    1. Mapping node embeddings from hyperbolic to tangent space
    2. Computing messages between connected nodes
    3. Aggregating messages using permutation-invariant operations
    4. Mapping aggregated messages back to hyperbolic space
    
    The implementation supports both Poincaré ball and Lorentz models.
    """
    
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 curvature: float = 1.0, 
                 use_attention: bool = True,
                 heads: int = 4,
                 dropout: float = 0.1,
                 model: str = 'poincare'):
        """
        Initialize the hyperbolic message passing layer.
        
        Args:
            in_dim (int): Input feature dimension
            out_dim (int): Output feature dimension
            curvature (float, optional): Curvature of hyperbolic space. Defaults to 1.0.
            use_attention (bool, optional): Whether to use attention for message passing. Defaults to True.
            heads (int, optional): Number of attention heads if attention is used. Defaults to 4.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            model (str, optional): Hyperbolic model to use ('poincare' or 'lorentz'). Defaults to 'poincare'.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.curvature = curvature
        self.use_attention = use_attention
        self.heads = heads
        self.dropout = dropout
        self.model = model
        
        # Set up hyperbolic space operations based on model
        if model == 'poincare':
            self.hyp = HyperbolicTangentSpace(curvature)
        elif model == 'lorentz':
            # For future implementation
            self.hyp = HyperbolicTangentSpace(curvature)  # Placeholder, to be replaced with Lorentz model
            raise NotImplementedError("Lorentz model is not yet implemented")
        else:
            raise ValueError(f"Unknown hyperbolic model: {model}")
        
        # Message computation layers
        if use_attention:
            # For multi-head attention, we split the dimensions
            self.head_dim = out_dim // heads
            self.query = nn.Linear(in_dim, self.head_dim * heads)
            self.key = nn.Linear(in_dim, self.head_dim * heads)
            self.value = nn.Linear(in_dim, self.head_dim * heads)
            self.combine_heads = nn.Linear(self.head_dim * heads, out_dim)
        else:
            # Simple message function
            self.message_func = nn.Linear(in_dim * 2, out_dim)
        
        # Update function after aggregation
        self.update_func = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),  # SiLU/Swish activation for better gradient flow
            nn.Dropout(dropout)
        )
        
        # Skip connection scaling factor (learnable)
        self.skip_scale = nn.Parameter(torch.tensor(0.5))
        
    def _compute_attention_messages(self, x_tangent: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute messages using multi-head attention.
        
        Args:
            x_tangent (torch.Tensor): Node features in tangent space [num_nodes, in_dim]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            
        Returns:
            torch.Tensor: Computed messages [num_nodes, out_dim]
        """
        num_nodes = x_tangent.size(0)
        
        # Compute query, key, value projections
        q = self.query(x_tangent).view(num_nodes, self.heads, self.head_dim)
        k = self.key(x_tangent).view(num_nodes, self.heads, self.head_dim)
        v = self.value(x_tangent).view(num_nodes, self.heads, self.head_dim)
        
        # Initialize messages tensor
        messages = torch.zeros(num_nodes, self.heads, self.head_dim, device=x_tangent.device)
        
        # Source and target node indices
        src, dst = edge_index
        
        # Compute attention scores and perform message passing
        for h in range(self.heads):
            # Compute attention scores
            scores = torch.sum(q[dst, h] * k[src, h], dim=1) / math.sqrt(self.head_dim)
            attention = F.softmax(scores, dim=0)
            
            # For each target node, aggregate messages from its neighbors
            target_nodes = dst.unique()
            for node in target_nodes:
                # Find neighbors (source nodes connected to this target)
                mask = (dst == node)
                if not mask.any():
                    continue
                    
                neighbor_indices = src[mask]
                neighbor_attn = attention[mask]
                
                # Weighted sum of neighbor values
                weighted_values = v[neighbor_indices, h] * neighbor_attn.unsqueeze(1)
                node_message = weighted_values.sum(dim=0)
                
                # Store aggregated message
                messages[node, h] = node_message
        
        # Combine attention heads
        combined_messages = messages.view(num_nodes, self.heads * self.head_dim)
        return self.combine_heads(combined_messages)
    
    def _compute_simple_messages(self, x_tangent: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute messages using a simple message function.
        
        Args:
            x_tangent (torch.Tensor): Node features in tangent space [num_nodes, in_dim]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            
        Returns:
            torch.Tensor: Computed messages [num_nodes, out_dim]
        """
        num_nodes = x_tangent.size(0)
        
        # Initialize messages tensor
        messages = torch.zeros(num_nodes, self.out_dim, device=x_tangent.device)
        
        # Source and target node indices
        src, dst = edge_index
        
        # For each target node, aggregate messages from its neighbors
        target_nodes = dst.unique()
        for node in target_nodes:
            # Find neighbors (source nodes connected to this target)
            mask = (dst == node)
            if not mask.any():
                continue
                
            neighbor_indices = src[mask]
            
            # Concatenate target node features with each neighbor's features
            neighbor_features = x_tangent[neighbor_indices]
            node_features_expanded = x_tangent[node].expand(neighbor_indices.size(0), -1)
            combined_features = torch.cat([node_features_expanded, neighbor_features], dim=1)
            
            # Compute messages
            neighbor_messages = self.message_func(combined_features)
            
            # Mean aggregation
            node_message = neighbor_messages.mean(dim=0)
            
            # Store aggregated message
            messages[node] = node_message
        
        return messages
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for hyperbolic message passing.
        
        Args:
            x (torch.Tensor): Node features in hyperbolic space [num_nodes, in_dim]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            
        Returns:
            torch.Tensor: Updated node features in hyperbolic space [num_nodes, out_dim]
        """
        # Map from hyperbolic to tangent space
        x_tangent = self.hyp.logmap0(x)
        
        # Compute messages using either attention or simple message function
        if self.use_attention:
            messages = self._compute_attention_messages(x_tangent, edge_index)
        else:
            messages = self._compute_simple_messages(x_tangent, edge_index)
        
        # Update node representations with skip connection
        combined = torch.cat([x_tangent, messages], dim=1)
        updated_tangent = self.update_func(combined)
        
        # Skip connection in tangent space (with learnable scaling)
        updated_tangent = updated_tangent + self.skip_scale * x_tangent
        
        # Map back to hyperbolic space
        updated_hyp = self.hyp.expmap0(updated_tangent)
        
        # Ensure points stay within the Poincaré ball
        updated_hyp = self.hyp.proj(updated_hyp)
        
        return updated_hyp


class HyperbolicGNN(BaseEncoder):
    """
    Hyperbolic Graph Neural Network encoder.
    
    This encoder transforms the citation network into embeddings in hyperbolic space,
    which is particularly well-suited for capturing the hierarchical structure of
    citation networks.
    
    Features:
    - Uses hyperbolic message passing for better hierarchy preservation
    - Supports multiple hyperbolic models (Poincaré, Lorentz)
    - Includes attention mechanisms for improved representation learning
    - Implements curvature as a trainable parameter (optional)
    """
    
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int,
                 embed_dim: int,
                 curvature: float = 1.0,
                 train_curvature: bool = False,
                 num_layers: int = 2,
                 use_attention: bool = True,
                 attention_heads: int = 4,
                 dropout: float = 0.1,
                 model: str = 'poincare',
                 residual: bool = True):
        """
        Initialize the hyperbolic GNN encoder.
        
        Args:
            node_dim (int): Dimensionality of node features
            edge_dim (int): Dimensionality of edge features
            embed_dim (int): Dimensionality of output embeddings
            curvature (float, optional): Curvature of hyperbolic space. Defaults to 1.0.
            train_curvature (bool, optional): Whether to learn the curvature parameter. Defaults to False.
            num_layers (int, optional): Number of message passing layers. Defaults to 2.
            use_attention (bool, optional): Whether to use attention. Defaults to True.
            attention_heads (int, optional): Number of attention heads. Defaults to 4.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            model (str, optional): Hyperbolic model ('poincare' or 'lorentz'). Defaults to 'poincare'.
            residual (bool, optional): Whether to use residual connections. Defaults to True.
        """
        super().__init__(node_dim, edge_dim, embed_dim)
        
        self.initial_curvature = curvature
        self.train_curvature = train_curvature
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.model = model
        self.residual = residual
        
        # Initialize curvature parameter
        if train_curvature:
            # Initialize with log(c) for numerical stability and ensure c > 0
            self.log_curvature = nn.Parameter(torch.tensor([math.log(curvature)]))
            print(f"Created trainable log_curvature parameter: {self.log_curvature.data}")
        else:
            self.register_buffer('curvature', torch.tensor([curvature]))
        
        # Initial projection from Euclidean to hyperbolic space
        self.feature_projector = nn.Sequential(
            nn.Linear(node_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        self.e2h = EuclideanToHyperbolic(embed_dim, embed_dim, 
                                         self.get_curvature().item())
        
        # Hyperbolic message passing layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                HyperbolicMessagePassing(
                    in_dim=embed_dim,
                    out_dim=embed_dim,
                    curvature=self.get_curvature().item(),
                    use_attention=use_attention,
                    heads=attention_heads,
                    dropout=dropout,
                    model=model
                )
            )
            
        # Layer normalization in tangent space
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Final hyperbolic transformation
        self.final_transform = HyperbolicLinear(
            embed_dim, embed_dim, 
            curvature=self.get_curvature().item()
        )
    
    def get_curvature(self) -> torch.Tensor:
        """
        Get the curvature parameter.
        
        Returns:
            torch.Tensor: Curvature value
        """
        if self.train_curvature:
            # Ensure curvature is positive by using exponential
            return torch.exp(self.log_curvature)
        else:
            return self.curvature
    
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
        
        # Project node features to embedding space
        x = self.feature_projector(x)
        
        # Get current curvature value
        curvature_value = self.get_curvature().item()
        
        # Create a hyperbolic tangent space with the current curvature
        # This ensures we use the up-to-date curvature value
        hyp = HyperbolicTangentSpace(curvature_value).to(x.device)
        
        # Map to hyperbolic space
        # Use the hyperbolic tangent space created with current curvature
        x_hyp = self.e2h(x)
        
        # Apply hyperbolic message passing layers
        for i, layer in enumerate(self.layers):
            # Update layer's curvature to the current value
            if hasattr(layer, 'hyp'):
                layer.hyp.c.data = torch.tensor([curvature_value], device=layer.hyp.c.device)
                
            x_next = layer(x_hyp, edge_index)
            
            # Apply residual connection in hyperbolic space if enabled
            if self.residual and i > 0:  # Skip first layer for residual
                # Get hyperbolic tangent space from the layer
                layer_hyp = layer.hyp
                
                # Map both to tangent space
                x_tangent = layer_hyp.logmap0(x_hyp)
                x_next_tangent = layer_hyp.logmap0(x_next)
                
                # Residual connection in tangent space
                combined_tangent = x_next_tangent + 0.3 * x_tangent
                
                # Apply layer normalization in tangent space
                normalized_tangent = self.layer_norm(combined_tangent)
                
                # Map back to hyperbolic space
                x_hyp = layer_hyp.expmap0(normalized_tangent)
                x_hyp = layer_hyp.proj(x_hyp)  # Ensure points stay in the Poincaré ball
            else:
                x_hyp = x_next
        
        # Update final transform curvature
        if hasattr(self.final_transform, 'hyp'):
            self.final_transform.hyp.c.data = torch.tensor([curvature_value], device=self.final_transform.hyp.c.device)
        
        # Apply final transformation
        x_hyp = self.final_transform(x_hyp)
        
        return x_hyp
    
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
        
        # Create a new GraphData object for the subgraph
        subgraph = GraphData(
            x=subgraph_x,
            edge_index=subgraph_edge_index
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
        edge_timestamps = None
        
        # Try standardized field name first
        if hasattr(graph, 'edge_timestamps') and graph.edge_timestamps is not None:
            edge_timestamps = graph.edge_timestamps
        # Fall back to edge_attr dictionary
        elif isinstance(graph.edge_attr, dict) and 'time' in graph.edge_attr:
            edge_timestamps = graph.edge_attr['time']
        # Fall back to legacy field name
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
            str: The name of the distance metric
        """
        return 'hyperbolic'
    
    def compute_distance(self, 
                        x1: torch.Tensor, 
                        x2: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic distance between embeddings.
        
        Args:
            x1 (torch.Tensor): First set of embeddings in hyperbolic space
            x2 (torch.Tensor): Second set of embeddings in hyperbolic space
                
        Returns:
            torch.Tensor: Pairwise distances between embeddings
        """
        # Create hyperbolic tangent space with current curvature
        hyp = HyperbolicTangentSpace(self.get_curvature().item())
        # Move to the same device as the input tensors
        hyp = hyp.to(x1.device)
        
        n1 = x1.size(0)
        n2 = x2.size(0)
        
        # Initialize distance matrix on the same device
        dist_matrix = torch.zeros(n1, n2, device=x1.device)
        
        # Compute pairwise distances
        for i in range(n1):
            for j in range(n2):
                # Compute the distance using hyperbolic distance formula
                # Ensure all tensors are on the same device
                dist_matrix[i, j] = hyp.distance(x1[i].unsqueeze(0), x2[j].unsqueeze(0)).item()
        
        return dist_matrix
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration parameters.
        
        Returns:
            Dict[str, Any]: Dictionary containing configuration parameters
        """
        return {
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'embed_dim': self.embed_dim,
            'curvature': self.get_curvature().item(),
            'train_curvature': self.train_curvature,
            'num_layers': self.num_layers,
            'use_attention': self.use_attention,
            'attention_heads': self.attention_heads,
            'dropout': self.dropout,
            'model': self.model,
            'residual': self.residual,
            'embedding_metric': self.get_embedding_metric(),
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'HyperbolicGNN':
        """
        Create an encoder instance from configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
                
        Returns:
            HyperbolicGNN: An instance of the hyperbolic GNN encoder
        """
        # Remove non-constructor parameters
        if 'embedding_metric' in config:
            del config['embedding_metric']
            
        return cls(**config)


# Additional class for hyperbolic GNN with multi-scale support
class MultiScaleHyperbolicGNN(HyperbolicGNN):
    """
    Multi-scale Hyperbolic GNN that captures hierarchical structures at different levels.
    
    This encoder extends the HyperbolicGNN with:
    1. Multi-scale processing using different curvatures
    2. Scale mixing via learnable weights
    3. Support for hierarchical embeddings with different levels of detail
    """
    
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int,
                 embed_dim: int,
                 num_scales: int = 3,
                 min_curvature: float = 0.1,
                 max_curvature: float = 2.0,
                 train_curvature: bool = True,
                 num_layers: int = 2,
                 use_attention: bool = True,
                 attention_heads: int = 4,
                 dropout: float = 0.1,
                 model: str = 'poincare',
                 residual: bool = True,
                 scale_weight_type: str = 'learnable'):
        """
        Initialize the multi-scale hyperbolic GNN encoder.
        
        Args:
            node_dim (int): Dimensionality of node features
            edge_dim (int): Dimensionality of edge features
            embed_dim (int): Dimensionality of output embeddings
            num_scales (int, optional): Number of scales to use. Defaults to 3.
            min_curvature (float, optional): Minimum curvature value. Defaults to 0.1.
            max_curvature (float, optional): Maximum curvature value. Defaults to 2.0.
            train_curvature (bool, optional): Whether to learn curvature parameters. Defaults to True.
            num_layers (int, optional): Number of message passing layers. Defaults to 2.
            use_attention (bool, optional): Whether to use attention. Defaults to True.
            attention_heads (int, optional): Number of attention heads. Defaults to 4.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            model (str, optional): Hyperbolic model ('poincare' or 'lorentz'). Defaults to 'poincare'.
            residual (bool, optional): Whether to use residual connections. Defaults to True.
            scale_weight_type (str, optional): How to weight different scales ('learnable', 'equal', 'adaptive'). 
                                               Defaults to 'learnable'.
        """
        # First set our custom attributes before calling parent constructor
        self.num_scales = num_scales
        self.min_curvature = min_curvature
        self.max_curvature = max_curvature
        self.scale_weight_type = scale_weight_type
        
        # Calculate mid curvature for parent constructor
        mid_curvature = (min_curvature + max_curvature) / 2
        
        # Call parent with middle curvature value
        super().__init__(node_dim, edge_dim, embed_dim, 
                         curvature=mid_curvature, 
                         train_curvature=False,  # We'll handle curvature differently
                         num_layers=num_layers,
                         use_attention=use_attention,
                         attention_heads=attention_heads,
                         dropout=dropout,
                         model=model,
                         residual=residual)
        
        # Override the e2h module with correct curvature value
        self.e2h = EuclideanToHyperbolic(embed_dim, embed_dim, mid_curvature)
        
        # Now we can override these attributes
        self.train_curvature = train_curvature
        
        # Calculate embed_dim_per_scale ensuring it's divisible
        self.embed_dim_per_scale = embed_dim // num_scales
        self.actual_combined_dim = self.embed_dim_per_scale * num_scales
        if self.actual_combined_dim != embed_dim:
            print(f"Warning: embed_dim ({embed_dim}) is not divisible by num_scales ({num_scales}). "
                  f"Using embed_dim_per_scale = {self.embed_dim_per_scale}, "
                  f"actual_combined_dim = {self.actual_combined_dim}")
        
        # Create multiple hyperbolic GNNs with different curvatures
        self.scale_gnns = nn.ModuleList()
        
        for i in range(num_scales):
            # Calculate curvature for this scale
            if num_scales > 1:
                curvature = min_curvature + (max_curvature - min_curvature) * i / (num_scales - 1)
            else:
                curvature = mid_curvature
                
            # Create GNN for this scale
            gnn = HyperbolicGNN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                embed_dim=self.embed_dim_per_scale,  # Use the calculated per-scale dimension
                curvature=curvature,
                train_curvature=train_curvature,
                num_layers=num_layers,
                use_attention=use_attention,
                attention_heads=attention_heads,
                dropout=dropout,
                model=model,
                residual=residual
            )
            
            self.scale_gnns.append(gnn)
        
        # Scale mixing weights
        if scale_weight_type == 'learnable':
            # Learn weights for combining scales
            self.scale_weights = nn.Parameter(torch.ones(num_scales))
        elif scale_weight_type == 'equal':
            # Equal weights, not learnable
            self.register_buffer('scale_weights', torch.ones(num_scales) / num_scales)
        elif scale_weight_type == 'adaptive':
            # Use MLP to determine weights based on graph properties
            self.scale_weight_mlp = nn.Sequential(
                nn.Linear(self.actual_combined_dim, 64),
                nn.SiLU(),
                nn.Linear(64, num_scales),
                nn.Softmax(dim=-1)
            )
        else:
            raise ValueError(f"Unknown scale weight type: {scale_weight_type}")
        
        # Projection for combining scales in Euclidean space
        # Use actual_combined_dim instead of embed_dim to match the concatenated tensor dimensions
        self.scale_combiner = nn.Sequential(
            nn.Linear(self.actual_combined_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        
        # Projection back to hyperbolic space
        self.combined_e2h = EuclideanToHyperbolic(embed_dim, embed_dim, mid_curvature)
    
    def get_curvature(self) -> torch.Tensor:
        """
        Get the curvature parameter.
        
        For multi-scale GNN, we use the middle curvature value.
        
        Returns:
            torch.Tensor: Curvature value
        """
        # Use the middle curvature value
        mid_curvature = (self.min_curvature + self.max_curvature) / 2
        # Create a new tensor for each call to ensure proper device placement
        return torch.tensor([mid_curvature])
    
    def forward(self, graph: GraphData) -> torch.Tensor:
        """
        Transform graph data into multi-scale node embeddings.
        
        Args:
            graph (GraphData): Graph data object containing the citation network information
                
        Returns:
            torch.Tensor: Node embeddings of shape [num_nodes, embed_dim]
        """
        device = graph.x.device
        
        # Get embeddings at each scale
        scale_embeddings = []
        for gnn in self.scale_gnns:
            # Ensure GNN is on the same device as the graph
            gnn = gnn.to(device)
            
            # Each GNN produces embeddings in its own hyperbolic space
            emb = gnn(graph)
            
            # Convert to Euclidean space for concatenation
            # Ensure curvature parameter is on the same device as embeddings
            hyp = HyperbolicTangentSpace(gnn.get_curvature().item())
            hyp = hyp.to(emb.device)
            emb_euclidean = hyp.logmap0(emb)
            
            scale_embeddings.append(emb_euclidean)
        
        # Concatenate scale embeddings
        combined_euclidean = torch.cat(scale_embeddings, dim=1)
        
        # Determine scale weights
        if self.scale_weight_type == 'adaptive':
            # Compute adaptive weights based on graph properties
            mean_embedding = combined_euclidean.mean(dim=0, keepdim=True)
            scale_weights = self.scale_weight_mlp(mean_embedding)
            
            # Apply weights to each scale
            weighted_parts = []
            for i in range(self.num_scales):
                start_idx = i * self.embed_dim_per_scale
                end_idx = (i + 1) * self.embed_dim_per_scale
                part = combined_euclidean[:, start_idx:end_idx] * scale_weights[0, i]
                weighted_parts.append(part)
                
            # Recombine weighted parts
            weighted_combined = torch.cat(weighted_parts, dim=1)
            
        elif self.scale_weight_type == 'learnable':
            # Apply softmax to ensure weights sum to 1
            normalized_weights = F.softmax(self.scale_weights, dim=0)
            
            # Apply weights to each scale
            weighted_parts = []
            for i in range(self.num_scales):
                start_idx = i * self.embed_dim_per_scale
                end_idx = (i + 1) * self.embed_dim_per_scale
                # Move weights to same device as embeddings
                weight = normalized_weights[i].to(device)
                part = combined_euclidean[:, start_idx:end_idx] * weight
                weighted_parts.append(part)
                
            # Recombine weighted parts
            weighted_combined = torch.cat(weighted_parts, dim=1)
            
        else:  # 'equal' weights
            weighted_combined = combined_euclidean
            
        # Apply final transformation
        transformed = self.scale_combiner(weighted_combined)
        
        # Convert back to hyperbolic space with proper device handling
        curvature = self.get_curvature().item()
        hyp = HyperbolicTangentSpace(curvature).to(device)
        final_hyperbolic = self.combined_e2h(transformed)
        
        return final_hyperbolic
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration parameters.
        
        Returns:
            Dict[str, Any]: Dictionary containing configuration parameters
        """
        config = super().get_config()
        config.update({
            'num_scales': self.num_scales,
            'min_curvature': self.min_curvature,
            'max_curvature': self.max_curvature,
            'scale_weight_type': self.scale_weight_type,
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MultiScaleHyperbolicGNN':
        """
        Create an encoder instance from configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
                
        Returns:
            MultiScaleHyperbolicGNN: An instance of the multi-scale hyperbolic GNN encoder
        """
        # Remove non-constructor parameters
        if 'embedding_metric' in config:
            del config['embedding_metric']
        if 'curvature' in config:
            del config['curvature']
            
        return cls(**config) 