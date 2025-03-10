import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import threading
from typing import Dict, Any, Optional, Tuple, List, Union

from .base import BasePredictor
from src.data.datasets import GraphData


class AdaptiveInputLayer(nn.Module):
    """Layer that adapts embeddings of varying dimensions to a target dimension.
    
    This layer is useful when working with different encoders that produce
    embeddings of different dimensions. It creates adaptation layers on-demand
    to handle varying input dimensions.
    """
    
    def __init__(self, target_dim: int, cache_size: int = 10):
        """Initialize the adaptive input layer.
        
        Args:
            target_dim (int): Target dimension for adapted embeddings
            cache_size (int): Maximum number of shapes to cache
        """
        super().__init__()
        self.target_dim = target_dim
        self.adaptation_layers = nn.ModuleDict()
        
        # Add shape cache to prevent redundant layer creation
        self.shape_cache = {}  # Maps shape tuple to str(dim)
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Add thread lock for thread safety when modifying the cache
        self.cache_lock = threading.RLock()
        
    def add_adaptation(self, source_dim: int) -> None:
        """Add an adaptation layer for a new source dimension.
        
        Args:
            source_dim (int): Source dimension to adapt from
        """
        # Thread-safe access to the adaptation layers
        with self.cache_lock:
            if str(source_dim) not in self.adaptation_layers:
                try:
                    # Create a new linear layer for this dimension
                    self.adaptation_layers[str(source_dim)] = nn.Linear(source_dim, self.target_dim)
                    
                    # Move the new layer to the same device as the module
                    if hasattr(self, 'device_param'):
                        device = self.device_param.device
                        self.adaptation_layers[str(source_dim)] = self.adaptation_layers[str(source_dim)].to(device)
                except Exception as e:
                    # If creating the layer fails, ensure we don't leave partial state
                    if str(source_dim) in self.adaptation_layers:
                        del self.adaptation_layers[str(source_dim)]
                    raise ValueError(f"Failed to create adaptation layer for dimension {source_dim}: {str(e)}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dimension adaptation to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [..., input_dim]
            
        Returns:
            torch.Tensor: Adapted tensor of shape [..., target_dim]
        """
        # Get input shape as a tuple for caching
        input_shape = tuple(x.shape)
        device = x.device
        
        # Store a reference parameter for device tracking
        if not hasattr(self, 'device_param'):
            # Create a dummy parameter to track the device
            self.device_param = nn.Parameter(torch.zeros(1, device=device))
        else:
            # Ensure the device_param is on the same device as the input
            self.device_param = self.device_param.to(device)
        
        # Thread-safe access to the cache
        with self.cache_lock:
            # Check if we've seen this exact shape before
            if input_shape in self.shape_cache:
                # Cache hit - use the cached dimension
                input_dim_str = self.shape_cache[input_shape]
                self.cache_hits += 1
            else:
                # Cache miss - extract the dimension and update cache
                input_dim = x.size(-1)
                input_dim_str = str(input_dim)
                
                # Add to cache, potentially removing oldest entry if cache is full
                if len(self.shape_cache) >= self.cache_size:
                    # Remove oldest entry (first key in dictionary)
                    oldest_shape = next(iter(self.shape_cache))
                    del self.shape_cache[oldest_shape]
                    
                self.shape_cache[input_shape] = input_dim_str
                self.cache_misses += 1
                
                # Add adaptation layer if necessary
                self.add_adaptation(input_dim)
            
            # Get the appropriate adaptation layer and ensure it's on the right device
            adaptation_layer = self.adaptation_layers[input_dim_str].to(device)
            
        # Apply the adaptation layer (outside of the lock)
        return adaptation_layer(x)


class TemporalPositionalEncoding(nn.Module):
    """Module for encoding temporal information into embeddings.
    
    This module provides positional encodings based on time differences between papers,
    which is crucial for models to understand the temporal patterns in citation networks.
    It supports both learned and sinusoidal encodings.
    """
    
    def __init__(self, 
                 embed_dim: int, 
                 max_time_diff: float = 10.0, 
                 num_time_bins: int = 100,
                 learned: bool = False):
        """Initialize temporal positional encoding.
        
        Args:
            embed_dim (int): Dimensionality of embeddings
            max_time_diff (float): Maximum time difference to encode
            num_time_bins (int): Number of discrete time bins
            learned (bool): Whether to use learned or fixed encodings
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_time_diff = max_time_diff
        self.num_time_bins = num_time_bins
        self.learned = learned
        
        if learned:
            # Learned positional embeddings
            self.time_embeddings = nn.Parameter(torch.zeros(num_time_bins, embed_dim))
            nn.init.normal_(self.time_embeddings, mean=0, std=0.02)
        else:
            # Sinusoidal positional embeddings (initialized once and fixed)
            pe = torch.zeros(num_time_bins, embed_dim)
            position = torch.arange(0, num_time_bins, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            
            # Even indices get sin, odd indices get cos
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            # Register as buffer (not parameter)
            self.register_buffer('pe', pe)
            
        # Time binning function (maps continuous time to discrete bins)
        time_bin_values = torch.linspace(0, max_time_diff, num_time_bins)
        self.register_buffer('time_bin_values', time_bin_values)
        
    def forward(self, time_diff: torch.Tensor) -> torch.Tensor:
        """Get positional encodings for time differences.
        
        Args:
            time_diff (torch.Tensor): Time differences [batch_size] or [batch_size, 1]
            
        Returns:
            torch.Tensor: Temporal encodings [batch_size, embed_dim]
        """
        # Ensure time_diff is the right shape
        if time_diff.dim() == 2 and time_diff.size(1) == 1:
            time_diff = time_diff.squeeze(1)
            
        # Clamp values to max_time_diff
        time_diff = torch.clamp(time_diff, 0, self.max_time_diff)
        
        # Find closest bin for each time difference
        bin_indices = torch.bucketize(time_diff, self.time_bin_values)
        # Clamp to valid range
        bin_indices = torch.clamp(bin_indices, 0, self.num_time_bins - 1)
        
        # Return either learned or fixed encodings
        if self.learned:
            return self.time_embeddings[bin_indices]
        else:
            return self.pe[bin_indices]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with scaled dot-product attention.
    
    This implementation follows the architecture described in "Attention Is All You Need"
    with additional features for attention visualization and masking.
    """
    
    def __init__(self, 
                embed_dim: int, 
                num_heads: int, 
                dropout: float = 0.1,
                cache_attn: bool = False):
        """Initialize multi-head attention.
        
        Args:
            embed_dim (int): Dimensionality of embeddings
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
            cache_attn (bool): Whether to cache attention weights for visualization
        """
        super().__init__()
        
        # Verify that embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.cache_attn = cache_attn
        
        # Linear projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5
        
        # Cache for attention weights (for visualization)
        self.last_attn_weights = None
        
    def forward(self, query, key, value, mask=None):
        """Forward pass for multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, embed_dim]
            key: Key tensor [batch_size, seq_len_k, embed_dim]
            value: Value tensor [batch_size, seq_len_v, embed_dim]
            mask: Optional mask [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Attention output [batch_size, seq_len_q, embed_dim]
                - Attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        device = query.device
        
        # Ensure all tensors are on the same device
        key = key.to(device)
        value = value.to(device)
        if mask is not None:
            mask = mask.to(device)
        
        # Project query, key, and value
        query = self.query_proj(query)  # [batch_size, seq_len_q, embed_dim]
        key = self.key_proj(key)        # [batch_size, seq_len_k, embed_dim]
        value = self.value_proj(value)  # [batch_size, seq_len_v, embed_dim]
        
        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Cache attention weights if enabled
        if self.cache_attn:
            self.last_attn_weights = attn_weights.detach()
        
        # Calculate attention output
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project back to original dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.output_proj(attn_output)
        
        return output, attn_weights
    
    def get_attention_weights(self):
        """Get the last computed attention weights (for visualization).
        
        Returns:
            torch.Tensor: Attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        return self.last_attn_weights


class CitationTypeQueryGenerator(nn.Module):
    """Generates specialized query vectors for different citation types.
    
    This module creates learned query vectors that can capture different types
    of citation relationships (e.g., background, methodology, comparison, etc.).
    """
    
    def __init__(self, 
                embed_dim: int, 
                num_citation_types: int = 3):
        """Initialize citation type query generator.
        
        Args:
            embed_dim (int): Dimensionality of embeddings
            num_citation_types (int): Number of citation types to model
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_citation_types = num_citation_types
        
        # Learn query vectors for different citation types
        self.citation_queries = nn.Parameter(torch.Tensor(num_citation_types, embed_dim))
        # Initialize with normal distribution
        nn.init.normal_(self.citation_queries, mean=0, std=0.02)
        
        # Attention for mixing citation type queries
        self.query_attention = nn.Sequential(
            nn.Linear(embed_dim * 2, num_citation_types),
            nn.Softmax(dim=1)
        )
        
    def forward(self, 
               src_emb: torch.Tensor, 
               dst_emb: torch.Tensor) -> torch.Tensor:
        """Generate query vectors based on citation types.
        
        Args:
            src_emb (torch.Tensor): Source embeddings [batch_size, embed_dim]
            dst_emb (torch.Tensor): Destination embeddings [batch_size, embed_dim]
            
        Returns:
            torch.Tensor: Citation-type-aware query vectors [batch_size, embed_dim]
        """
        batch_size = src_emb.size(0)
        
        # Compute attention weights for citation types
        # using both source and destination embeddings
        combined = torch.cat([src_emb, dst_emb], dim=1)
        type_weights = self.query_attention(combined)  # [batch_size, num_citation_types]
        
        # Expand citation queries for batch processing
        queries = self.citation_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_types, embed_dim]
        
        # Apply weights to citation queries
        weighted_queries = type_weights.unsqueeze(2) * queries  # [batch_size, num_types, embed_dim]
        
        # Sum weighted queries to get the final query
        final_query = weighted_queries.sum(dim=1)  # [batch_size, embed_dim]
        
        return final_query


class EnhancedAttentionPredictor(nn.Module):
    """
    Enhanced attention-based predictor for link prediction in citation networks.
    
    This predictor uses multi-head attention to score pairs of nodes for link prediction.
    It supports multiple layers for deeper architectures.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None
    ):
        """
        Initialize the enhanced attention predictor.
        
        Args:
            input_dim (int): Dimension of input node embeddings
            hidden_dim (int): Dimension of hidden layers
            num_layers (int): Number of layers in the predictor
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
            edge_dim (Optional[int]): Edge feature dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Calculate the dimension of the concatenated representation
        # attn_output + src_embeds + dst_embeds = 3 * input_dim
        combined_dim = 3 * input_dim
        
        # Add edge feature dimension if used
        if edge_dim is not None and edge_dim > 0:
            combined_dim += input_dim  # We project edge features to input_dim
        
        # Multi-layer scoring network
        scoring_layers = []
        for i in range(num_layers):
            in_dim = combined_dim if i == 0 else hidden_dim
            out_dim = 1 if i == num_layers - 1 else hidden_dim
            
            scoring_layers.append(nn.Linear(in_dim, out_dim))
            
            # Add activation and dropout for all but the last layer
            if i < num_layers - 1:
                scoring_layers.append(nn.LayerNorm(out_dim))
                scoring_layers.append(nn.ReLU())
                scoring_layers.append(nn.Dropout(dropout))
        
        self.scoring_network = nn.Sequential(*scoring_layers)
        
        # Edge feature integration
        self.use_edge_features = edge_dim is not None and edge_dim > 0
        if self.use_edge_features:
            self.edge_proj = nn.Linear(edge_dim, input_dim)
            self.edge_attention = nn.Linear(input_dim * 2 + input_dim, 1)
    
    def forward(
        self,
        src_embeds: torch.Tensor,
        dst_embeds: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to compute scores for pairs of nodes.
        
        Args:
            src_embeds (torch.Tensor): Source node embeddings [batch_size, embed_dim]
            dst_embeds (torch.Tensor): Destination node embeddings [batch_size, embed_dim]
            edge_attr (Optional[torch.Tensor]): Edge attributes [batch_size, edge_dim]
            
        Returns:
            torch.Tensor: Predicted edge scores [batch_size, 1]
        """
        # Compute attention-weighted features
        batch_size = src_embeds.size(0)
        
        # Combine source and destination for attention (using the destination as query)
        key_value = src_embeds.unsqueeze(1)  # [batch_size, 1, embed_dim]
        query = dst_embeds.unsqueeze(1)      # [batch_size, 1, embed_dim]
        
        # Apply attention
        attn_output, _ = self.attention(query, key_value, key_value)
        attn_output = attn_output.squeeze(1)  # [batch_size, embed_dim]
        
        # Incorporate edge features if available
        if self.use_edge_features and edge_attr is not None:
            edge_feats = self.edge_proj(edge_attr)
            combined = torch.cat([attn_output, src_embeds, dst_embeds, edge_feats], dim=1)
        else:
            combined = torch.cat([attn_output, src_embeds, dst_embeds], dim=1)
        
        # Process through multi-layer scoring network
        pair_representation = combined
        scores = self.scoring_network(pair_representation)
        
        return scores.view(-1, 1)


# Alias the Enhanced predictor to maintain backward compatibility
AttentionPredictor = EnhancedAttentionPredictor 