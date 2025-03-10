import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import math
import threading

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
            
            # Get the appropriate adaptation layer
            adaptation_layer = self.adaptation_layers[input_dim_str]
            
        # Apply the adaptation layer (outside of the lock)
        return adaptation_layer(x)


class TemporalEncodingLayer(nn.Module):
    """Advanced temporal encoding layer with multiple encoding strategies.
    
    This layer transforms time differences into rich temporal feature representations
    using various encoding strategies:
    1. Linear encoding: Simple linear transformation of time
    2. Log-scaled encoding: Better for varying time scales
    3. Periodic encoding: Captures seasonal/cyclical patterns
    4. Relative encoding: Better represents publication date differences
    
    These encoding strategies can be combined to capture complex temporal dynamics
    in citation networks.
    """
    
    def __init__(self, 
                 output_dim: int, 
                 num_modes: int = 4,
                 use_log_scale: bool = True,
                 use_periodic: bool = True,
                 periodic_base: float = 2.0,
                 num_periodic_features: int = 4,
                 max_time_scale: float = 10.0,
                 min_time_scale: float = 0.1,
                 temporal_dropout: float = 0.1,
                 **kwargs):
        """Initialize the temporal encoding layer.
        
        Args:
            output_dim (int): Dimensionality of the output temporal features
            num_modes (int): Number of encoding modes/strategies to combine
            use_log_scale (bool): Whether to use log-scaled time encoding
            use_periodic (bool): Whether to use periodic time encoding
            periodic_base (float): Base for periodic encoding frequencies
            num_periodic_features (int): Number of periodic features to use
            max_time_scale (float): Maximum time scale for adaptive scaling
            min_time_scale (float): Minimum time scale for adaptive scaling
            temporal_dropout (float): Dropout rate for temporal features
            **kwargs: Additional parameters
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.num_modes = num_modes
        self.use_log_scale = use_log_scale
        self.use_periodic = use_periodic
        self.periodic_base = periodic_base
        self.num_periodic_features = num_periodic_features
        self.max_time_scale = max_time_scale
        self.min_time_scale = min_time_scale
        self.temporal_dropout = temporal_dropout
        
        # Determine input dimension based on features enabled
        self.input_dim = 1  # Linear time always included
        if use_log_scale:
            self.input_dim += 1
        if use_periodic:
            self.input_dim += 2 * num_periodic_features
        
        # Create feature projection layers for each mode
        self.mode_projections = nn.ModuleList([
            nn.Linear(self.input_dim, output_dim // num_modes)
            for _ in range(num_modes)
        ])
        
        # For learned time scaling
        self.time_scale = nn.Parameter(torch.FloatTensor([1.0]))
        
        # Create final projection and normalization
        self.final_projection = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(temporal_dropout)
        
    def encode_linear(self, time_diff: torch.Tensor) -> torch.Tensor:
        """Simple linear encoding of time difference.
        
        Args:
            time_diff (torch.Tensor): Time difference tensor
            
        Returns:
            torch.Tensor: Linear time encoding
        """
        return time_diff.unsqueeze(-1)
    
    def encode_log_scale(self, time_diff: torch.Tensor) -> torch.Tensor:
        """Log-scaled encoding of time difference.
        
        Better captures varying time scales by using logarithmic transformation.
        
        Args:
            time_diff (torch.Tensor): Time difference tensor
            
        Returns:
            torch.Tensor: Log-scaled time encoding
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        # Use sign(t) * log(1 + |t|) to handle negative times
        sign = torch.sign(time_diff)
        log_time = sign * torch.log(1 + torch.abs(time_diff) + epsilon)
        return log_time.unsqueeze(-1)
    
    def encode_periodic(self, time_diff: torch.Tensor) -> torch.Tensor:
        """Periodic encoding of time difference.
        
        Creates sinusoidal features at different frequencies to capture
        cyclical patterns in citation behavior.
        
        Args:
            time_diff (torch.Tensor): Time difference tensor
            
        Returns:
            torch.Tensor: Periodic time encoding [batch_size, 2*num_periodic_features]
        """
        device = time_diff.device
        batch_size = time_diff.size(0)
        
        # Calculate frequencies at different scales
        # frequencies = self.periodic_base ** torch.arange(0, self.num_periodic_features, device=device)
        frequencies = torch.tensor(
            [self.periodic_base ** i for i in range(self.num_periodic_features)],
            device=device
        )
        
        # Create time features [batch_size, 1] * [num_features] -> [batch_size, num_features]
        time_diff_expanded = time_diff.unsqueeze(-1)
        time_features = time_diff_expanded * frequencies
        
        # Create sinusoidal features [batch_size, 2*num_features]
        sin_features = torch.sin(time_features)
        cos_features = torch.cos(time_features)
        
        # Concatenate sin and cos features
        periodic_features = torch.cat([sin_features, cos_features], dim=-1)
        return periodic_features
    
    def encode_time(self, time_diff: torch.Tensor) -> torch.Tensor:
        """Encode time difference using multiple strategies.
        
        Args:
            time_diff (torch.Tensor): Time difference tensor [batch_size]
            
        Returns:
            torch.Tensor: Encoded time features [batch_size, output_dim]
        """
        # Apply learned time scaling
        scaled_time = time_diff * self.time_scale
        
        # Create list to collect features
        time_features = []
        
        # Always include linear encoding
        time_features.append(self.encode_linear(scaled_time))
        
        # Add log-scaled encoding if enabled
        if self.use_log_scale:
            time_features.append(self.encode_log_scale(scaled_time))
            
        # Add periodic encoding if enabled
        if self.use_periodic:
            time_features.append(self.encode_periodic(scaled_time))
        
        # Concatenate all features
        all_features = torch.cat(time_features, dim=-1)
        
        # Process through different modes and combine
        mode_outputs = []
        for projection in self.mode_projections:
            mode_outputs.append(projection(all_features))
            
        # Combine mode outputs
        combined = torch.cat(mode_outputs, dim=-1)
        
        # Apply final projection and normalization
        encoded = self.final_projection(combined)
        normalized = self.layer_norm(encoded)
        
        # Apply dropout
        return self.dropout(normalized)
    
    def forward(self, time_diff: torch.Tensor) -> torch.Tensor:
        """Forward pass - encode time difference.
        
        Args:
            time_diff (torch.Tensor): Time difference tensor [batch_size]
            
        Returns:
            torch.Tensor: Encoded time features [batch_size, output_dim]
        """
        return self.encode_time(time_diff)

class TemporalPredictor(BasePredictor):
    """A predictor specialized for temporal citation prediction.
    
    TemporalPredictor explicitly incorporates temporal information into citation prediction,
    making it aware of publication time and ensuring temporal constraints (e.g., papers
    can only cite papers published before them). The enhanced implementation includes:
    
    1. Advanced temporal encoding with multiple strategies
    2. Recency bias modeling with domain-specific parameterization
    3. Citation velocity awareness
    4. Configurable prediction horizons
    5. Domain-time interaction modeling
    6. Robust handling of temporal sparsity
    """
    
    def __init__(self, 
                 embed_dim: int,
                 time_encoding_dim: int = 32,
                 hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.2,
                 use_batch_norm: bool = True,
                 # Temporal encoding parameters
                 num_encoding_modes: int = 4,
                 use_log_scale: bool = True,
                 use_periodic: bool = True,
                 # Recency bias parameters
                 recency_bias: bool = True,
                 domain_specific_recency: bool = False,
                 num_domains: int = 5,
                 learn_recency_params: bool = True,
                 default_decay_factor: float = 0.05,
                 # Citation velocity parameters
                 use_citation_velocity: bool = False,
                 velocity_encoding_dim: int = 8,
                 # Temporal prediction parameters
                 confidence_estimation: bool = False,
                 # Device and threading parameters 
                 cache_size: int = 10,
                 **kwargs):
        """Initialize the enhanced temporal predictor.
        
        Args:
            embed_dim (int): Dimensionality of node embeddings
            time_encoding_dim (int): Dimensionality of time encoding
            hidden_dims (List[int]): Dimensions of hidden layers
            dropout (float): Dropout rate
            use_batch_norm (bool): Whether to use batch normalization
            
            # Temporal encoding parameters
            num_encoding_modes (int): Number of encoding modes
            use_log_scale (bool): Whether to use log-scaled encoding
            use_periodic (bool): Whether to use periodic encoding
            
            # Recency bias parameters
            recency_bias (bool): Whether to use recency bias modeling
            domain_specific_recency (bool): Whether to learn domain-specific recency factors
            num_domains (int): Number of research domains for domain-specific parameters
            learn_recency_params (bool): Whether to learn recency parameters or use fixed values
            default_decay_factor (float): Default decay factor when not learning
            
            # Citation velocity parameters
            use_citation_velocity (bool): Whether to use citation velocity features
            velocity_encoding_dim (int): Dimensionality of velocity encoding
            
            # Temporal prediction parameters
            confidence_estimation (bool): Whether to estimate prediction confidence
            
            # Device and threading parameters
            cache_size (int): Size of cache for dimension handling
            
            **kwargs: Additional parameters for the base class
        """
        super().__init__(embed_dim=embed_dim, **kwargs)
        
        # Store basic configuration
        self.time_encoding_dim = time_encoding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Store temporal encoding parameters
        self.num_encoding_modes = num_encoding_modes
        self.use_log_scale = use_log_scale
        self.use_periodic = use_periodic
        
        # Store recency bias parameters
        self.recency_bias = recency_bias
        self.domain_specific_recency = domain_specific_recency
        self.num_domains = num_domains
        self.learn_recency_params = learn_recency_params
        self.default_decay_factor = default_decay_factor
        
        # Store citation velocity parameters
        self.use_citation_velocity = use_citation_velocity
        self.velocity_encoding_dim = velocity_encoding_dim
        
        # Store temporal prediction parameters
        self.confidence_estimation = confidence_estimation
        
        # Cache configuration
        self.cache_size = cache_size
        
        # Create advanced temporal encoding layer
        self.temporal_encoder = TemporalEncodingLayer(
            output_dim=time_encoding_dim,
            num_modes=num_encoding_modes,
            use_log_scale=use_log_scale,
            use_periodic=use_periodic,
            temporal_dropout=dropout,
        )
        
        # Create adaptive input layers for handling varying embedding dimensions
        self.adaptive_src_layer = AdaptiveInputLayer(embed_dim, cache_size=cache_size)
        self.adaptive_dst_layer = AdaptiveInputLayer(embed_dim, cache_size=cache_size)
        
        # Define recency bias parameters if enabled
        if recency_bias:
            if domain_specific_recency:
                # Domain-specific recency factors
                self.domain_recency_factors = nn.Parameter(
                    torch.FloatTensor([default_decay_factor] * num_domains),
                    requires_grad=learn_recency_params
                )
                # Domain embedding layer
                self.domain_embedding = nn.Embedding(num_domains, time_encoding_dim)
            else:
                # Global recency factor
                self.recency_factor = nn.Parameter(
                    torch.FloatTensor([default_decay_factor]),
                    requires_grad=learn_recency_params
                )
        
        # Citation velocity components if enabled
        if use_citation_velocity:
            self.velocity_encoder = nn.Sequential(
                nn.Linear(1, velocity_encoding_dim),
                nn.ReLU(),
                nn.Linear(velocity_encoding_dim, velocity_encoding_dim),
                nn.Tanh()
            )
        
        # Calculate input dimension for the MLP
        input_dim = 2 * embed_dim + time_encoding_dim
        if use_citation_velocity:
            input_dim += velocity_encoding_dim
        if domain_specific_recency:
            input_dim += time_encoding_dim  # For domain embeddings
        
        # Configure prediction confidence estimation if enabled
        if confidence_estimation:
            # Output will have two values: prediction and confidence
            output_dim = 2
        else:
            output_dim = 1
        
        # Define MLP components
        layers = []
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(nn.SiLU())  # SiLU activation (Swish) for better gradients
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, output_dim))
        if not confidence_estimation:
            # Remove the sigmoid layer since we're using BCEWithLogitsLoss in training
            # layers.append(nn.Sigmoid())  # Sigmoid for single-value output
            pass
        
        # Combine all layers
        self.mlp = nn.Sequential(*layers)
        
        # Create cache for domain identification
        self.domain_cache = {}
        
    def get_domain_id(self, node_id: int, existing_graph: GraphData) -> int:
        """Identify the research domain of a paper.
        
        This is a placeholder implementation. In a real-world scenario,
        this would use paper metadata to determine the domain.
        
        Args:
            node_id (int): Node ID of the paper
            existing_graph (GraphData): Graph data containing paper information
            
        Returns:
            int: Domain ID (0 to num_domains-1)
        """
        # Check if we've already identified this node's domain
        if node_id in self.domain_cache:
            return self.domain_cache[node_id]
        
        # In a real implementation, this would use paper metadata
        # For now, we'll use a simple hash function based on node_id
        if hasattr(existing_graph, 'node_features') and existing_graph.node_features is not None:
            # If graph has node features, use them to determine domain
            # This is just a placeholder - in reality, you'd use actual paper categories
            features = existing_graph.node_features[node_id]
            # Simple approach: use max feature dimension as domain
            domain_id = torch.argmax(features).item() % self.num_domains
        else:
            # Fallback to node_id based determination
            domain_id = node_id % self.num_domains
        
        # Cache the result
        self.domain_cache[node_id] = domain_id
        return domain_id
    
    def get_recency_factor(self, 
                         domain_id: Optional[int] = None) -> torch.Tensor:
        """Get the appropriate recency factor based on domain.
        
        Args:
            domain_id (Optional[int]): Domain ID if domain-specific recency is enabled
            
        Returns:
            torch.Tensor: Recency factor(s) to use
        """
        if not self.recency_bias:
            # If recency bias is disabled, return the default value
            return torch.tensor([self.default_decay_factor], 
                               device=self.get_device())
        
        if self.domain_specific_recency and domain_id is not None:
            # Return domain-specific factor
            return self.domain_recency_factors[domain_id]
        else:
            # Return global factor
            return self.recency_factor
    
    def get_device(self) -> torch.device:
        """Helper method to get the device of the model parameters.
        
        Returns:
            torch.device: Device of the model
        """
        return next(self.parameters()).device
    
    def apply_recency_bias(self, 
                          scores: torch.Tensor, 
                          time_diff: torch.Tensor,
                          domains: Optional[torch.Tensor] = None,
                          is_logit: bool = False) -> torch.Tensor:
        """Apply recency bias to prediction scores.
        
        Args:
            scores (torch.Tensor): Raw prediction scores [batch_size], can be probabilities or logits
            time_diff (torch.Tensor): Time differences [batch_size]
            domains (Optional[torch.Tensor]): Domain IDs for domain-specific recency [batch_size]
            is_logit (bool): Whether the scores are logits (True) or probabilities (False)
            
        Returns:
            torch.Tensor: Scores adjusted with recency bias [batch_size], returned in the same format as input
        """
        if not self.recency_bias:
            return scores
        
        # Convert to probabilities if input is logits
        if is_logit:
            probs = torch.sigmoid(scores)
        else:
            probs = scores
            
        # Handle domain-specific recency if enabled
        if self.domain_specific_recency and domains is not None:
            # Get recency factors for each domain in the batch
            batch_factors = self.domain_recency_factors[domains]
            
            # Apply domain-specific decay
            recency_adjustment = torch.exp(-batch_factors.unsqueeze(1) * torch.abs(time_diff.unsqueeze(1)))
        else:
            # Apply global recency decay
            recency_adjustment = torch.exp(-self.recency_factor * torch.abs(time_diff))
        
        # Adjust scores based on recency (newer papers get more weight)
        adjusted_probs = probs * recency_adjustment.squeeze()
        
        # Convert back to logits if input was logits
        if is_logit:
            # Ensure values are in proper range to avoid numerical issues
            adjusted_probs = torch.clamp(adjusted_probs, 1e-7, 1 - 1e-7)
            return torch.log(adjusted_probs / (1 - adjusted_probs))
        else:
            return adjusted_probs
    
    def encode_time_difference(self, time_diff: torch.Tensor) -> torch.Tensor:
        """Encode time difference between papers using advanced encoding mechanisms.
        
        Args:
            time_diff (torch.Tensor): Time difference between papers [batch_size]
                
        Returns:
            torch.Tensor: Encoded time difference [batch_size, time_encoding_dim]
        """
        # Use the advanced temporal encoder
        return self.temporal_encoder(time_diff)
        
    def encode_citation_velocity(self, 
                               velocity: torch.Tensor) -> torch.Tensor:
        """Encode citation velocity (rate of citation accumulation).
        
        Args:
            velocity (torch.Tensor): Citation velocity [batch_size]
                
        Returns:
            torch.Tensor: Encoded velocity [batch_size, velocity_encoding_dim]
        """
        if not self.use_citation_velocity:
            raise ValueError("Citation velocity encoding is disabled in this model")
            
        # Normalize and reshape velocity for encoding
        normalized_velocity = velocity.unsqueeze(1)
        
        # Apply velocity encoder
        return self.velocity_encoder(normalized_velocity)
        
    def forward(self, 
                src_embeddings: torch.Tensor, 
                dst_embeddings: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                citation_velocity: Optional[torch.Tensor] = None,
                domain_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict citation likelihood using temporal information.
        
        Args:
            src_embeddings (torch.Tensor): Embeddings of source nodes (citing papers)
                [batch_size, embed_dim]
            dst_embeddings (torch.Tensor): Embeddings of destination nodes (cited papers)
                [batch_size, embed_dim]
            edge_attr (Optional[torch.Tensor]): Edge attributes, must contain time information
            citation_velocity (Optional[torch.Tensor]): Citation velocity data if available
            domain_ids (Optional[torch.Tensor]): Domain IDs for papers if available
                
        Returns:
            torch.Tensor: Predicted citation likelihood scores [batch_size]
                Or if confidence_estimation is True: [batch_size, 2] with
                prediction and confidence scores
        """
        batch_size = src_embeddings.size(0)

        # Ensure all tensors are on the same device
        if dst_embeddings.device != device:
            dst_embeddings = dst_embeddings.to(device)
        if edge_attr is not None and edge_attr.device != device:
            edge_attr = edge_attr.to(device)
        if citation_velocity is not None and citation_velocity.device != device:
            citation_velocity = citation_velocity.to(device)
        if domain_ids is not None and domain_ids.device != device:
            domain_ids = domain_ids.to(device)
            
        # Ensure adaptive layers are on the correct device
        if hasattr(self, 'adaptive_src_layer'):
            for name, param in self.adaptive_src_layer.named_parameters():
                if param.device != device:
                    param.data = param.data.to(device)
                    
        if hasattr(self, 'adaptive_dst_layer'):
            for name, param in self.adaptive_dst_layer.named_parameters():
                if param.device != device:
                    param.data = param.data.to(device)
                    
        # Ensure MLP is on the correct device  
        for name, param in self.mlp.named_parameters():
            if param.device != device:
                param.data = param.data.to(device)
        device = src_embeddings.device
        
        # Apply adaptive input layers for dimension compatibility
        src_embeddings = self.adaptive_src_layer(src_embeddings)
        dst_embeddings = self.adaptive_dst_layer(dst_embeddings)
        
        # Extract time information from edge attributes
        if edge_attr is None:
            # If no time information, use a default (1.0) - assume valid by default
            # Changed from 0 to 1.0 to avoid zeroing out predictions
            time_diff = torch.ones(batch_size, device=device)
        else:
            # Assuming the first dimension of edge_attr is time difference
            time_diff = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr
        
        # Encode time difference
        time_encoding = self.encode_time_difference(time_diff)
        
        # Prepare feature list for concatenation
        features = [src_embeddings, dst_embeddings, time_encoding]
        
        # Add citation velocity if enabled and provided
        if self.use_citation_velocity and citation_velocity is not None:
            velocity_encoding = self.encode_citation_velocity(citation_velocity)
            features.append(velocity_encoding)
        
        # Add domain embeddings if domain-specific recency is enabled
        if self.domain_specific_recency and domain_ids is not None:
            domain_embeddings = self.domain_embedding(domain_ids)
            features.append(domain_embeddings)
        
        # Concatenate all features
        combined = torch.cat(features, dim=1)
        
        # Apply MLP to predict
        output = self.mlp(combined)
        
        # Handle confidence estimation if enabled
        if self.confidence_estimation:
            # Split output into prediction and confidence
            prediction = output[:, 0]  # Keep as logits
            confidence = torch.sigmoid(output[:, 1])  # Convert confidence to probability
            
            # Combine into a single tensor [batch_size, 2]
            result = torch.stack([prediction, confidence], dim=1)
        else:
            # Keep raw prediction as logits
            result = output
        
        # BUGFIX: Apply temporal constraint properly (can't cite future papers)
        # Create a mask that zeros out invalid citations (time_diff < 0)
        # Only apply this for data with temporal information
        if edge_attr is not None:
            temporal_mask = (time_diff >= 0).float()
            
            # Apply mask based on the shape of the result
            if result.dim() > 1 and result.size(1) > 1:
                # For multi-dimensional output (e.g., with confidence)
                # Set very low score for invalid citations instead of zeroing
                invalid_indices = (temporal_mask == 0).nonzero(as_tuple=True)
                if invalid_indices[0].size(0) > 0:
                    result[invalid_indices[0], 0] = -10.0  # Very negative logit -> near zero probability
            else:
                # For single-dimensional output, apply similar logic
                invalid_indices = (temporal_mask == 0).nonzero(as_tuple=True)
                if invalid_indices[0].size(0) > 0:
                    result[invalid_indices[0]] = -10.0  # Very negative logit -> near zero probability
        
        # Apply recency bias if enabled (after the temporal constraint)
        if self.recency_bias and edge_attr is not None:
            # Apply recency bias directly on logits
            if result.dim() > 1 and result.size(1) > 1:
                # For multi-dimensional output, only apply to prediction (first column)
                result[:, 0] = self.apply_recency_bias(
                    result[:, 0], time_diff, domain_ids, is_logit=True
                )
            else:
                # For single-dimensional output
                result = self.apply_recency_bias(result, time_diff, domain_ids, is_logit=True)
        
        return result
    
    def predict_batch(self, 
                     node_embeddings: torch.Tensor, 
                     edge_indices: torch.Tensor,
                     edge_attr: Optional[torch.Tensor] = None,
                     citation_velocities: Optional[torch.Tensor] = None,
                     domain_ids: Optional[torch.Tensor] = None,
                     existing_graph: Optional[GraphData] = None) -> torch.Tensor:
        """Predict citations for a batch of edges.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings for all papers
                [num_nodes, embed_dim]
            edge_indices (torch.Tensor): Edge indices to predict
                [2, num_edges] where edge_indices[0] are source nodes and
                edge_indices[1] are destination nodes
            edge_attr (Optional[torch.Tensor]): Edge attributes containing time information
            citation_velocities (Optional[torch.Tensor]): Citation velocities for candidate edges
            domain_ids (Optional[torch.Tensor]): Domain IDs for source nodes
            existing_graph (Optional[GraphData]): Graph data to extract domain info if not provided
                
        Returns:
            torch.Tensor: Predicted citation likelihood scores [num_edges]
                Or if confidence_estimation is True: [num_edges, 2] with
                prediction and confidence scores
        """
        # Get src and dst embeddings for the specified edges
        src_indices = edge_indices[0]
        dst_indices = edge_indices[1]
        
        src_embeddings = node_embeddings[src_indices]
        dst_embeddings = node_embeddings[dst_indices]
        
        # If domain IDs not provided but we need them, try to extract from graph
        has_domain_specific_recency = hasattr(self, 'domain_specific_recency') and self.domain_specific_recency
        has_domain_embedding = hasattr(self, 'domain_embedding')
        
        if domain_ids is None and (has_domain_specific_recency or has_domain_embedding):
            if existing_graph is not None:
                # Extract domain IDs for source nodes
                batch_domain_ids = torch.tensor(
                    [self.get_domain_id(idx.item(), existing_graph) for idx in src_indices],
                    device=src_embeddings.device
                )
            else:
                # Use default domain (0) if graph not provided
                batch_domain_ids = torch.zeros(src_indices.size(0), 
                                              dtype=torch.long, 
                                              device=src_embeddings.device)
        else:
            # Use provided domain IDs
            batch_domain_ids = domain_ids
        
        # Use forward method to predict with all available information
        return self.forward(
            src_embeddings, 
            dst_embeddings, 
            edge_attr,
            citation_velocities,
            batch_domain_ids
        )
    
    def predict_citations(self,
                         node_embeddings: torch.Tensor,
                         existing_graph: GraphData,
                         k: int = 10,
                         **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict new citations for papers in an existing graph.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings for all papers
                [num_nodes, embed_dim]
            existing_graph (GraphData): The existing citation network
            k (int): Number of top predictions to return
            **kwargs: Additional parameters:
                - candidate_edges: Pre-computed candidate edges
                - time_window: Time window for citation velocity calculation
                - min_citations: Minimum citations for velocity calculation
                - max_candidates: Maximum number of candidate edges to consider
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Edge indices of top predicted citations [2, k]
                - Scores for the top predicted citations [k]
        """
        # Get device for tensor operations
        device = node_embeddings.device
        
        # If candidate edges provided in kwargs, use them
        candidate_edges = kwargs.get('candidate_edges', None)
        
        # If no candidate edges provided, generate them
        if candidate_edges is None:
            max_candidates = kwargs.get('max_candidates', 1000)
            candidate_edges = self.get_candidate_edges(existing_graph, max_candidates=max_candidates)
        
        # Safety check: If no candidate edges, return empty results
        if candidate_edges.shape[1] == 0:
            return torch.zeros(2, 0, device=device), torch.zeros(0, device=device)
        
        # Get paper publication times if available
        paper_times = None
        if hasattr(existing_graph, 'node_timestamps') and existing_graph.node_timestamps is not None:
            paper_times = existing_graph.node_timestamps
        elif hasattr(existing_graph, 'paper_times') and existing_graph.paper_times is not None:
            paper_times = existing_graph.paper_times
        
        # Calculate citation velocities if enabled
        citation_velocities = None
        if hasattr(self, 'use_citation_velocity') and self.use_citation_velocity:
            citation_velocities = self._calculate_citation_velocities(
                existing_graph, 
                candidate_edges,
                time_window=kwargs.get('time_window', 1.0),
                min_citations=kwargs.get('min_citations', 5)
            )
        
        # Extract domain IDs if domain-specific recency is enabled
        domain_ids = None
        if hasattr(self, 'domain_specific_recency') and self.domain_specific_recency:
            # Get domain IDs for source nodes
            src_indices = candidate_edges[0]
            domain_ids = torch.tensor(
                [self.get_domain_id(idx.item(), existing_graph) for idx in src_indices],
                device=device
            )
        
        # Process in batches to avoid memory issues
        batch_size = min(1024, candidate_edges.shape[1])  # Ensure batch size is not larger than available candidates
        num_candidates = candidate_edges.shape[1]
        all_scores = []
        
        for i in range(0, num_candidates, batch_size):
            end_idx = min(i+batch_size, num_candidates)
            batch_indices = candidate_edges[:, i:end_idx]
            
            # Extract source and destination indices
            src_indices = batch_indices[0]
            dst_indices = batch_indices[1]
            
            # Calculate publication time differences (src - dst)
            # Negative values mean dst was published after src (invalid citation)
            if paper_times is not None:
                src_times = paper_times[src_indices]
                dst_times = paper_times[dst_indices]
                time_diffs = dst_times - src_times  # Earlier papers are cited by later papers
                batch_edge_attr = time_diffs.unsqueeze(1) if time_diffs.dim() == 1 else time_diffs
            else:
                # If no time information, use default (all zeros)
                batch_edge_attr = torch.zeros(batch_indices.shape[1], 1, device=device)
            
            # Extract batch citation velocities if available
            batch_velocities = None
            if citation_velocities is not None:
                batch_velocities = citation_velocities[i:end_idx]
            
            # Extract batch domain IDs if available
            batch_domain_ids = None
            if domain_ids is not None:
                batch_domain_ids = domain_ids[i:end_idx]
            
            # Predict batch scores
            batch_scores = self.predict_batch(
                node_embeddings, 
                batch_indices, 
                batch_edge_attr,
                batch_velocities,
                batch_domain_ids,
                existing_graph
            )
            
            # If confidence estimation is enabled, use only the prediction scores
            if hasattr(self, 'confidence_estimation') and self.confidence_estimation and batch_scores.dim() > 1:
                batch_scores = batch_scores[:, 0]
                
            all_scores.append(batch_scores)
        
        # Combine batch results
        if len(all_scores) > 1:
            scores = torch.cat(all_scores)
        else:
            scores = all_scores[0] if all_scores else torch.tensor([], device=device)
        
        # Get top k predictions
        if len(scores) > 0 and k < len(scores):
            top_k_indices = torch.topk(scores, k=k).indices
            top_edges = candidate_edges[:, top_k_indices]
            top_scores = scores[top_k_indices]
        else:
            # In case there are fewer candidates than k
            top_edges = candidate_edges
            top_scores = scores
        
        return top_edges, top_scores
        
    def _calculate_citation_velocities(self,
                                     existing_graph: GraphData,
                                     candidate_edges: torch.Tensor,
                                     time_window: float = 1.0,
                                     min_citations: int = 5) -> torch.Tensor:
        """Calculate citation velocities for candidate destination papers.
        
        Citation velocity measures how quickly papers accumulate citations over time.
        
        Args:
            existing_graph (GraphData): The existing citation network
            candidate_edges (torch.Tensor): Candidate edges [2, num_candidates]
            time_window (float): Time window for recent citation calculation
            min_citations (int): Minimum citations required for velocity calculation
            
        Returns:
            torch.Tensor: Citation velocities for destination papers [num_candidates]
        """
        device = next(self.parameters()).device
        
        # Get edge indices and timestamps
        edge_index = existing_graph.edge_index
        
        # Get paper timestamps
        if hasattr(existing_graph, 'node_timestamps') and existing_graph.node_timestamps is not None:
            paper_times = existing_graph.node_timestamps
        elif hasattr(existing_graph, 'paper_times') and existing_graph.paper_times is not None:
            paper_times = existing_graph.paper_times
        else:
            # If no time information, return zeros
            return torch.zeros(candidate_edges.shape[1], device=device)
        
        # Get edge timestamps if available
        if hasattr(existing_graph, 'edge_timestamps') and existing_graph.edge_timestamps is not None:
            edge_times = existing_graph.edge_timestamps
        else:
            # If no edge timestamps, use paper timestamps
            # Assume citation happens at the time of the citing paper
            edge_times = paper_times[edge_index[0]]
        
        # Get current time (max timestamp)
        current_time = paper_times.max()
        
        # Calculate time threshold for recent citations
        time_threshold = current_time - time_window
        
        # Initialize velocities
        velocities = torch.zeros(candidate_edges.shape[1], device=device)
        
        # For each candidate edge
        for i in range(candidate_edges.shape[1]):
            # Get destination paper (the one being cited)
            dst_paper = candidate_edges[1, i].item()
            
            # Find all citations to this paper
            dst_citations = (edge_index[1] == dst_paper)
            
            # If no citations, velocity is 0
            if not dst_citations.any():
                continue
                
            # Get timestamps of these citations
            citation_times = edge_times[dst_citations]
            
            # Count total citations
            total_citations = citation_times.size(0)
            
            # Skip if too few citations
            if total_citations < min_citations:
                continue
                
            # Count recent citations
            recent_citations = (citation_times >= time_threshold).sum().float()
            
            # Calculate velocity (citations per time unit)
            if time_window > 0:
                velocity = recent_citations / time_window
            else:
                velocity = 0.0
                
            # Store velocity
            velocities[i] = velocity
        
        # Normalize velocities to [0, 1] range
        if velocities.max() > 0:
            velocities = velocities / velocities.max()
            
        return velocities
    
    def predict_temporal_citations(self,
                                  node_embeddings: torch.Tensor,
                                  existing_graph: GraphData,
                                  time_threshold: float,
                                  future_window: Optional[float] = None,
                                  k: int = 10,
                                  confidence_threshold: Optional[float] = None,
                                  **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict future citations based on a temporal snapshot.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings for all papers
                [num_nodes, embed_dim]
            existing_graph (GraphData): The complete citation network
            time_threshold (float): The timestamp threshold for the snapshot
            future_window (Optional[float]): If provided, only predict citations
                within the window [time_threshold, time_threshold + future_window]
            k (int): Number of top predictions to return
            confidence_threshold (Optional[float]): If provided and confidence estimation
                is enabled, only return predictions with confidence above this threshold
            **kwargs: Additional parameters:
                - min_citations: Minimum citations for velocity calculation
                - domain_weights: Weights for different research domains
                - stratified_sampling: Whether to use stratified sampling by domain
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Edge indices of top predicted future citations [2, k]
                - Scores for the top predicted future citations [k]
                  If confidence_estimation is True, scores will be [k, 2] with
                  prediction and confidence scores
        """
        device = node_embeddings.device
        
        # Get paper publication times
        if hasattr(existing_graph, 'node_timestamps') and existing_graph.node_timestamps is not None:
            paper_times = existing_graph.node_timestamps
        elif hasattr(existing_graph, 'paper_times') and existing_graph.paper_times is not None:
            paper_times = existing_graph.paper_times
        else:
            raise ValueError("Temporal prediction requires node timestamps")
        
        # Create a temporal snapshot of the graph at time_threshold
        # Only include papers published before or at the threshold
        valid_nodes = paper_times <= time_threshold
        num_nodes = valid_nodes.sum().item()
        
        if num_nodes < 2:
            # Not enough nodes for prediction
            return torch.zeros(2, 0, device=device), torch.zeros(0, device=device)
        
        # Get edges in the snapshot (both papers and the citation must be before threshold)
        edge_index = existing_graph.edge_index
        
        # Check if edge timestamps are available
        if hasattr(existing_graph, 'edge_timestamps') and existing_graph.edge_timestamps is not None:
            edge_times = existing_graph.edge_timestamps
            valid_edges = edge_times <= time_threshold
        else:
            # If no edge timestamps, use paper timestamps
            # Both papers must be published before the threshold
            src_times = paper_times[edge_index[0]]
            dst_times = paper_times[edge_index[1]]
            valid_edges = (src_times <= time_threshold) & (dst_times <= time_threshold)
        
        # Extract snapshot edges
        snapshot_edges = edge_index[:, valid_edges]
        
        # Create a mapping from original node IDs to snapshot node IDs
        node_mapping = torch.zeros(len(paper_times), dtype=torch.long, device=device)
        node_mapping[valid_nodes] = torch.arange(num_nodes, device=device)
        
        # Create snapshot graph
        snapshot_graph = GraphData(
            edge_index=snapshot_edges,
            num_nodes=num_nodes
        )
        
        # Copy relevant attributes to snapshot
        if hasattr(existing_graph, 'node_timestamps'):
            snapshot_graph.node_timestamps = paper_times[valid_nodes]
        if hasattr(existing_graph, 'paper_times'):
            snapshot_graph.paper_times = paper_times[valid_nodes]
        if hasattr(existing_graph, 'node_features') and existing_graph.node_features is not None:
            snapshot_graph.node_features = existing_graph.node_features[valid_nodes]
        
        # Generate candidate edges for future prediction
        # These are edges that don't exist in the snapshot but could form in the future
        candidate_edges = self.get_candidate_edges(snapshot_graph, max_candidates=kwargs.get('max_candidates', 10000))
        
        # Filter candidates based on temporal constraints
        # Source paper must be published before or at threshold
        # Destination paper must be published before source paper
        src_indices = candidate_edges[0]
        dst_indices = candidate_edges[1]
        
        src_times = snapshot_graph.node_timestamps[src_indices]
        dst_times = snapshot_graph.node_timestamps[dst_indices]
        
        # Destination must be published before source (valid citation direction)
        valid_direction = dst_times < src_times
        
        # If future_window is provided, only consider citations within the window
        if future_window is not None:
            # Find edges in the original graph that appear in the future window
            if hasattr(existing_graph, 'edge_timestamps') and existing_graph.edge_timestamps is not None:
                future_edges = (edge_times > time_threshold) & (edge_times <= time_threshold + future_window)
                future_edge_index = edge_index[:, future_edges]
            else:
                # If no edge timestamps, use paper timestamps
                # Source paper must be published after threshold but within window
                src_times_orig = paper_times[edge_index[0]]
                future_edges = (src_times_orig > time_threshold) & (src_times_orig <= time_threshold + future_window)
                future_edge_index = edge_index[:, future_edges]
            
            # Create a set of future edges for fast lookup
            future_edge_set = set()
            for i in range(future_edge_index.shape[1]):
                src = future_edge_index[0, i].item()
                dst = future_edge_index[1, i].item()
                if valid_nodes[src] and valid_nodes[dst]:
                    # Map to snapshot node IDs
                    src_mapped = node_mapping[src].item()
                    dst_mapped = node_mapping[dst].item()
                    future_edge_set.add((src_mapped, dst_mapped))
            
            # Filter candidates to only include those that actually form in the future window
            valid_future = torch.tensor(
                [
                    (src_indices[i].item(), dst_indices[i].item()) in future_edge_set
                    for i in range(candidate_edges.shape[1])
                ],
                device=device
            )
            
            # Combine constraints
            valid_candidates = valid_direction & valid_future
        else:
            # If no future window, just use direction constraint
            valid_candidates = valid_direction
        
        # Filter candidate edges
        filtered_candidates = candidate_edges[:, valid_candidates]
        
        if filtered_candidates.shape[1] == 0:
            # No valid candidates
            return torch.zeros(2, 0, device=device), torch.zeros(0, device=device)
        
        # Calculate time differences for filtered candidates
        src_times = snapshot_graph.node_timestamps[filtered_candidates[0]]
        dst_times = snapshot_graph.node_timestamps[filtered_candidates[1]]
        time_diffs = src_times - dst_times
        
        # Calculate citation velocities if enabled
        if self.use_citation_velocity:
            citation_velocities = self._calculate_citation_velocities(
                snapshot_graph, 
                filtered_candidates,
                time_window=kwargs.get('time_window', 1.0),
                min_citations=kwargs.get('min_citations', 5)
            )
        else:
            citation_velocities = None
        
        # Extract domain IDs if domain-specific recency is enabled
        if self.domain_specific_recency:
            domain_ids = torch.tensor(
                [self.get_domain_id(idx.item(), snapshot_graph) for idx in filtered_candidates[0]],
                device=device
            )
            
            # Apply domain weights if provided
            domain_weights = kwargs.get('domain_weights', None)
            if domain_weights is not None:
                domain_weights_tensor = torch.tensor(domain_weights, device=device)
                domain_weight_factors = domain_weights_tensor[domain_ids]
            else:
                domain_weight_factors = None
        else:
            domain_ids = None
            domain_weight_factors = None
        
        # Process in batches to avoid memory issues
        batch_size = 1024
        num_candidates = filtered_candidates.shape[1]
        all_scores = []
        
        for i in range(0, num_candidates, batch_size):
            batch_indices = filtered_candidates[:, i:min(i+batch_size, num_candidates)]
            
            # Extract batch time differences
            batch_time_diffs = time_diffs[i:min(i+batch_size, num_candidates)]
            batch_edge_attr = batch_time_diffs.unsqueeze(1)
            
            # Extract batch citation velocities if available
            batch_velocities = None
            if citation_velocities is not None:
                batch_velocities = citation_velocities[i:min(i+batch_size, num_candidates)]
            
            # Extract batch domain IDs if available
            batch_domain_ids = None
            if domain_ids is not None:
                batch_domain_ids = domain_ids[i:min(i+batch_size, num_candidates)]
            
            # Predict batch scores
            batch_scores = self.predict_batch(
                node_embeddings[valid_nodes], 
                batch_indices, 
                batch_edge_attr,
                batch_velocities,
                batch_domain_ids,
                snapshot_graph
            )
            
            # Apply domain weights if available
            if domain_weight_factors is not None:
                batch_weights = domain_weight_factors[i:min(i+batch_size, num_candidates)]
                if self.confidence_estimation and batch_scores.dim() > 1:
                    # Only apply weights to prediction, not confidence
                    batch_scores[:, 0] = batch_scores[:, 0] * batch_weights
                else:
                    batch_scores = batch_scores * batch_weights
            
            all_scores.append(batch_scores)
        
        # Combine batch results
        if len(all_scores) > 1:
            scores = torch.cat(all_scores)
        else:
            scores = all_scores[0] if all_scores else torch.tensor([], device=device)
        
        # Filter by confidence threshold if enabled
        if self.confidence_estimation and confidence_threshold is not None:
            confidence_scores = scores[:, 1]
            high_confidence = confidence_scores >= confidence_threshold
            
            filtered_candidates = filtered_candidates[:, high_confidence]
            scores = scores[high_confidence]
            
            if filtered_candidates.shape[1] == 0:
                # No high-confidence predictions
                return torch.zeros(2, 0, device=device), torch.zeros(0, device=device)
        
        # Get top k predictions
        if self.confidence_estimation and scores.dim() > 1:
            # Use prediction scores (first column) for ranking
            prediction_scores = scores[:, 0]
            
            if len(prediction_scores) > 0 and k < len(prediction_scores):
                top_k_indices = torch.topk(prediction_scores, k=k).indices
                top_edges = filtered_candidates[:, top_k_indices]
                top_scores = scores[top_k_indices]  # Keep both prediction and confidence
            else:
                top_edges = filtered_candidates
                top_scores = scores
        else:
            # Standard single-score case
            if len(scores) > 0 and k < len(scores):
                top_k_indices = torch.topk(scores, k=k).indices
                top_edges = filtered_candidates[:, top_k_indices]
                top_scores = scores[top_k_indices]
            else:
                top_edges = filtered_candidates
                top_scores = scores
        
        # Map back to original node IDs if needed
        # This step is optional since we're working with the snapshot node IDs
        
        return top_edges, top_scores
    
    def get_config(self) -> Dict[str, Any]:
        """Get predictor configuration parameters.
        
        Returns:
            Dict[str, Any]: Dictionary containing configuration parameters
        """
        config = {
            'embed_dim': self.embed_dim,
            'time_encoding_dim': self.time_encoding_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'use_batch_norm': self.use_batch_norm,
            
            # Temporal encoding parameters
            'num_encoding_modes': self.num_encoding_modes,
            'use_log_scale': self.use_log_scale,
            'use_periodic': self.use_periodic,
            
            # Recency bias parameters
            'recency_bias': self.recency_bias,
            'domain_specific_recency': self.domain_specific_recency,
            'num_domains': self.num_domains,
            'learn_recency_params': self.learn_recency_params,
            'default_decay_factor': self.default_decay_factor,
            
            # Citation velocity parameters
            'use_citation_velocity': self.use_citation_velocity,
            'velocity_encoding_dim': self.velocity_encoding_dim,
            
            # Temporal prediction parameters
            'confidence_estimation': self.confidence_estimation,
            
            # Device and threading parameters
            'cache_size': self.cache_size,
        }
        
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TemporalPredictor':
        """Create a predictor instance from configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
                
        Returns:
            TemporalPredictor: An instance of the temporal predictor
        """
        return cls(**config)

    def get_candidate_edges(self, 
                           existing_graph: GraphData,
                           max_candidates: Optional[int] = None) -> torch.Tensor:
        """Get candidate edges that don't exist in the current graph.
        
        Helper method to generate candidate edges for prediction.
        This version adds safeguards against infinite loops and more efficient sampling.
        
        Args:
            existing_graph (GraphData): The current citation network
            max_candidates (Optional[int]): Maximum number of candidate edges to return
                
        Returns:
            torch.Tensor: Candidate edge indices [2, num_candidates]
        """
        # Calculate num_nodes from the node features or timestamp fields
        if hasattr(existing_graph, 'x') and existing_graph.x is not None:
            num_nodes = existing_graph.x.shape[0]
        elif hasattr(existing_graph, 'node_timestamps') and existing_graph.node_timestamps is not None:
            num_nodes = len(existing_graph.node_timestamps)
        elif hasattr(existing_graph, 'paper_times') and existing_graph.paper_times is not None:
            num_nodes = len(existing_graph.paper_times)
        elif hasattr(existing_graph, 'num_nodes'):
            num_nodes = existing_graph.num_nodes
        else:
            # Fallback: get max node index from edge_index + 1
            num_nodes = existing_graph.edge_index.max().item() + 1
        
        # Implement safety checks
        if num_nodes <= 1:
            # Not enough nodes to create edges
            return torch.zeros(2, 0, dtype=torch.long, device=self.get_device())
        
        # Set reasonable limits for candidate edges
        if max_candidates is None:
            # Default to a reasonable number based on graph size
            max_candidates = min(num_nodes * 10, 1000)
        else:
            # Ensure max_candidates is not unreasonably large
            max_candidates = min(max_candidates, num_nodes * 100, 10000)
        
        # Get existing edges for efficient lookup
        edge_index = existing_graph.edge_index
        
        # Create efficient lookup for existing edges
        existing_edges = set()
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            existing_edges.add((src, dst))
        
        # Efficiently generate candidate edges using rejection sampling with a limit
        candidates = []
        max_attempts = max_candidates * 10  # Limit attempts to avoid infinite loops
        attempts = 0
        
        # Use device from model parameters
        device = self.get_device()
        
        while len(candidates) < max_candidates and attempts < max_attempts:
            # Sample multiple pairs at once for efficiency
            batch_size = min(1000, max_candidates - len(candidates))
            src_nodes = torch.randint(0, num_nodes, (batch_size,), device=device)
            dst_nodes = torch.randint(0, num_nodes, (batch_size,), device=device)
            
            for i in range(batch_size):
                src = src_nodes[i].item()
                dst = dst_nodes[i].item()
                attempts += 1
                
                # Check if edge doesn't exist and nodes are different
                if src != dst and (src, dst) not in existing_edges:
                    candidates.append((src, dst))
                    existing_edges.add((src, dst))  # Mark as considered
                    
                    # Stop if we have enough candidates
                    if len(candidates) >= max_candidates:
                        break
                        
                # Safety check to avoid infinite loops
                if attempts >= max_attempts:
                    break
        
        # Handle case where we couldn't find enough candidates
        if not candidates:
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        
        # Convert to tensor
        candidate_edges = torch.tensor(candidates, dtype=torch.long, device=device).t()
        return candidate_edges 