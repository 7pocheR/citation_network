import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union
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


class FeatureInteractionLayer(nn.Module):
    """Layer that creates sophisticated interactions between source and destination embeddings.
    
    This layer implements multiple feature interaction techniques:
    1. Element-wise product (Hadamard product)
    2. Bilinear interaction
    3. Difference features
    4. L2 distance
    
    These interactions allow the model to better capture complex citation patterns.
    """
    
    def __init__(self, 
                 embed_dim: int, 
                 interaction_dim: int,
                 feature_types: Optional[List[str]] = None):
        """Initialize the feature interaction layer.
        
        Args:
            embed_dim (int): Dimensionality of input embeddings
            interaction_dim (int): Dimensionality of interaction features
            feature_types (Optional[List[str]]): List of feature types to compute.
                Options: 'projections', 'hadamard', 'bilinear', 'difference', 'l2'.
                If None, all feature types are computed.
        """
        super().__init__()
        self.src_proj = nn.Linear(embed_dim, interaction_dim)
        self.dst_proj = nn.Linear(embed_dim, interaction_dim)
        
        # Set default feature types if not specified
        if feature_types is None:
            feature_types = ['projections', 'hadamard', 'bilinear', 'difference', 'l2']
        self.feature_types = feature_types
        
        # Only create bilinear layer if needed
        if 'bilinear' in self.feature_types:
            self.interaction = nn.Bilinear(interaction_dim, interaction_dim, interaction_dim)
        
        # Calculate output dimension based on selected features
        self.output_dim = 0
        if 'projections' in self.feature_types:
            self.output_dim += 2 * interaction_dim  # src_proj and dst_proj
        if 'hadamard' in self.feature_types:
            self.output_dim += interaction_dim  # hadamard product
        if 'bilinear' in self.feature_types:
            self.output_dim += interaction_dim  # bilinear interaction
        if 'difference' in self.feature_types:
            self.output_dim += interaction_dim  # absolute difference
        if 'l2' in self.feature_types:
            self.output_dim += 1  # L2 distance (scalar)
        
    def forward(self, src_emb: torch.Tensor, dst_emb: torch.Tensor) -> torch.Tensor:
        """Create interaction features between source and destination embeddings.
        
        Args:
            src_emb (torch.Tensor): Source embeddings of shape [batch_size, embed_dim]
            dst_emb (torch.Tensor): Destination embeddings of shape [batch_size, embed_dim]
            
        Returns:
            torch.Tensor: Interaction features of shape [batch_size, output_dim]
        """
        src_proj = self.src_proj(src_emb)
        dst_proj = self.dst_proj(dst_emb)
        
        # Selectively compute features based on configuration
        features = []
        
        # Add projections if requested
        if 'projections' in self.feature_types:
            features.append(src_proj)
            features.append(dst_proj)
        
        # Element-wise product (Hadamard)
        if 'hadamard' in self.feature_types:
            hadamard = src_proj * dst_proj
            features.append(hadamard)
        
        # Bilinear interaction
        if 'bilinear' in self.feature_types:
            bilinear = self.interaction(src_proj, dst_proj)
            features.append(bilinear)
        
        # Difference features
        if 'difference' in self.feature_types:
            diff = torch.abs(src_proj - dst_proj)
            features.append(diff)
        
        # L2 distance as a single feature
        if 'l2' in self.feature_types:
            # Calculate squared L2 distance in a numerically stable way
            # Instead of torch.norm which can have numerical stability issues
            # Add small epsilon to avoid potential sqrt(0) issues
            eps = 1e-8
            squared_diff = (src_proj - dst_proj).pow(2).sum(dim=1, keepdim=True)
            l2_dist = torch.sqrt(squared_diff + eps)
            features.append(l2_dist)
        
        # Concatenate selected interaction types
        return torch.cat(features, dim=1)


class NodeMetadataProcessor(nn.Module):
    """Module to process and incorporate node metadata into predictions.
    
    This processor handles various types of metadata that might be available
    for papers in the citation network, such as topics, keywords, etc.
    """
    
    def __init__(self, 
                 metadata_dims: Dict[str, int],
                 output_dim: int):
        """Initialize the metadata processor.
        
        Args:
            metadata_dims (Dict[str, int]): Dictionary mapping metadata field names to dimensions
            output_dim (int): Output dimension after processing
        """
        super().__init__()
        self.metadata_dims = metadata_dims
        self.output_dim = output_dim
        
        # Create processors for each metadata type
        self.processors = nn.ModuleDict()
        for field_name, dim in metadata_dims.items():
            self.processors[field_name] = nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.SiLU()
            )
            
        # Metadata combiner (if multiple metadata fields)
        if len(metadata_dims) > 1:
            self.combiner = nn.Linear(output_dim * len(metadata_dims), output_dim)
        else:
            self.combiner = nn.Identity()
            
    def forward(self, 
                graph: GraphData, 
                src_indices: torch.Tensor, 
                dst_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """Process metadata for source and destination nodes.
        
        Args:
            graph (GraphData): Graph containing node metadata
            src_indices (torch.Tensor): Source node indices
            dst_indices (torch.Tensor): Destination node indices
            
        Returns:
            Optional[torch.Tensor]: Processed metadata features of shape [batch_size, output_dim]
                or None if no metadata is available
        """
        # Check if any metadata is available
        available_fields = [field for field in self.metadata_dims if hasattr(graph, field)]
        
        if not available_fields:
            return None
            
        # Process each available metadata field
        processed_features = []
        
        for field in available_fields:
            # Get metadata
            metadata = getattr(graph, field)
            
            if metadata is None:
                continue
                
            # Get metadata for source and destination nodes
            src_metadata = metadata[src_indices]
            dst_metadata = metadata[dst_indices]
            
            # Process separately and then combine
            processed_src = self.processors[field](src_metadata)
            processed_dst = self.processors[field](dst_metadata)
            
            # Use element-wise product as interaction
            processed = processed_src * processed_dst
            processed_features.append(processed)
            
        if not processed_features:
            return None
            
        # Combine processed features if multiple metadata fields
        if len(processed_features) > 1:
            combined = torch.cat(processed_features, dim=1)
            return self.combiner(combined)
        else:
            return processed_features[0]


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with normalization options.
    
    This block enhances the standard MLP with residual connections,
    normalization layers, and advanced activation functions.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.2,
                 use_layer_norm: bool = True,
                 use_batch_norm: bool = True,
                 activation: str = 'silu'):
        """Initialize the residual MLP block.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            use_layer_norm (bool, optional): Whether to use layer normalization. Defaults to True.
            use_batch_norm (bool, optional): Whether to use batch normalization. Defaults to True.
            activation (str, optional): Activation function. Defaults to 'silu'.
        """
        super().__init__()
        
        # Input projection (if dimensions don't match)
        self.input_projection = None
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            
        # Main branch
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Apply normalization (both can be used together)
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
            
        # Activation function
        if activation == 'silu':
            layers.append(nn.SiLU())
        elif activation == 'mish':
            layers.append(nn.Mish())
        elif activation == 'relu':
            layers.append(nn.ReLU())
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # Second linear layer
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Optional normalization after second linear
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
            
        # Activation after second linear
        if activation == 'silu':
            layers.append(nn.SiLU())
        elif activation == 'mish':
            layers.append(nn.Mish())
        elif activation == 'relu':
            layers.append(nn.ReLU())
            
        # Dropout
        layers.append(nn.Dropout(dropout))
        
        self.main_branch = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Apply main branch
        main_output = self.main_branch(x)
        
        # Handle residual connection with input projection if needed
        if self.input_projection is not None:
            residual = self.input_projection(x)
        else:
            residual = x
            
        return main_output + residual


class MLPPredictor(BasePredictor):
    """An enhanced predictor that uses a multi-layer perceptron to predict citations.
    
    MLPPredictor uses a sophisticated neural network to learn a mapping from pairs of
    node embeddings to citation likelihood scores. It includes several enhancements:
    1. Adaptive input layers for handling embeddings of different dimensions
    2. Advanced feature interaction techniques
    3. Residual connections with normalization options
    4. Support for node metadata features
    5. Advanced activation functions (SiLU/Mish)
    
    Examples:
        Basic usage:
        ```python
        # Create predictor
        predictor = MLPPredictor(embed_dim=64)
        
        # Predict for a single pair
        src_emb = torch.randn(1, 64)
        dst_emb = torch.randn(1, 64)
        score = predictor(src_emb, dst_emb)
        
        # Predict for a batch of pairs
        src_embs = torch.randn(10, 64)
        dst_embs = torch.randn(10, 64)
        scores = predictor(src_embs, dst_embs)
        
        # Predict using node embeddings and edge indices
        node_embeddings = torch.randn(100, 64)
        edge_indices = torch.tensor([[0, 1, 2], [3, 4, 5]])  # 3 edges: (0,3), (1,4), (2,5)
        scores = predictor.predict_batch(node_embeddings, edge_indices)
        
        # Predict new citations for a graph
        graph_data = GraphData(...)
        top_edges, top_scores = predictor.predict_citations(
            node_embeddings=node_embeddings,
            existing_graph=graph_data,
            k=10
        )
        
        # Predict temporal citations
        top_edges, top_scores = predictor.predict_temporal_citations(
            node_embeddings=node_embeddings,
            existing_graph=graph_data,
            time_threshold=2020.0,
            future_window=1.0,
            k=10
        )
        ```
        
        Advanced usage with metadata:
        ```python
        # Create predictor with metadata support
        predictor = MLPPredictor(
            embed_dim=64,
            hidden_dims=[128, 64, 32],
            dropout=0.3,
            use_batch_norm=True,
            use_layer_norm=True,
            activation='silu',
            metadata_fields={'topics': 10, 'keywords': 15},
            feature_types=['projections', 'hadamard', 'bilinear']
        )
        
        # Create graph with metadata
        graph = GraphData(...)
        graph.topics = torch.randn(100, 10)  # Topic vectors for each node
        graph.keywords = torch.randn(100, 15)  # Keyword vectors for each node
        
        # Predict with metadata
        scores = predictor.predict_batch(node_embeddings, edge_indices, graph_data=graph)
        ```
    """
    
    def __init__(self, 
                 embed_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.2,
                 use_batch_norm: bool = True,
                 use_layer_norm: bool = True,
                 activation: str = 'silu',
                 interaction_dim: int = 64,
                 metadata_fields: Optional[Dict[str, int]] = None,
                 feature_types: Optional[List[str]] = None,
                 **kwargs):
        """Initialize the enhanced MLP-based predictor.
        
        Args:
            embed_dim (int): Base dimensionality of node embeddings
            hidden_dims (List[int]): Dimensions of hidden layers in the MLP
            dropout (float): Dropout rate
            use_batch_norm (bool): Whether to use batch normalization
            use_layer_norm (bool): Whether to use layer normalization
            activation (str): Activation function ('relu', 'silu', or 'mish')
            interaction_dim (int): Dimension for feature interactions
            metadata_fields (Optional[Dict[str, int]]): Dictionary mapping metadata field names to dimensions
            feature_types (Optional[List[str]]): List of feature types to compute.
                Options: 'projections', 'hadamard', 'bilinear', 'difference', 'l2'.
                If None, all feature types are computed.
            **kwargs: Additional parameters for the base class
        """
        super().__init__(embed_dim=embed_dim, **kwargs)
        
        # Store configuration
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.activation = activation
        self.interaction_dim = interaction_dim
        self.metadata_fields = metadata_fields or {}
        self.feature_types = feature_types or ['projections', 'hadamard', 'bilinear', 'difference', 'l2']
        
        # Create adaptive input layer
        self.input_adapter = AdaptiveInputLayer(embed_dim)
        
        # Create feature interaction layer
        self.feature_interaction = FeatureInteractionLayer(
            embed_dim=embed_dim,
            interaction_dim=interaction_dim,
            feature_types=self.feature_types
        )
        
        # Feature dimension calculation based on selected feature types
        self.feature_dim = self.feature_interaction.output_dim
        
        # Add metadata dimension if used
        if self.metadata_fields:
            self.metadata_processor = NodeMetadataProcessor(
                metadata_dims=self.metadata_fields,
                output_dim=interaction_dim
            )
            self.feature_dim += interaction_dim
        else:
            self.metadata_processor = None
        
        # Define MLP with residual blocks
        self.layers = nn.ModuleList()
        
        # Input layer
        input_dim = self.feature_dim
        
        # Hidden layers with residual connections
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(
                ResidualMLPBlock(
                    input_dim if i == 0 else hidden_dims[i-1],
                    hidden_dim,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    use_batch_norm=use_batch_norm,
                    activation=activation
                )
            )
        
        # Output layer (single score)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Dimension validation tracker (for debugging)
        self.dimension_errors = 0
        self.max_errors_to_log = 5
        
    def _validate_dimensions(self, 
                           src_embeddings: torch.Tensor, 
                           dst_embeddings: torch.Tensor) -> Tuple[bool, str]:
        """Validate input dimensions to provide helpful error messages.
        
        Args:
            src_embeddings (torch.Tensor): Source embeddings
            dst_embeddings (torch.Tensor): Destination embeddings
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if src_embeddings.dim() < 2:
            return False, f"Source embeddings must have at least 2 dimensions, got shape {src_embeddings.shape}"
            
        if dst_embeddings.dim() < 2:
            return False, f"Destination embeddings must have at least 2 dimensions, got shape {dst_embeddings.shape}"
            
        if src_embeddings.size(0) != dst_embeddings.size(0):
            return False, f"Batch size mismatch: source batch size {src_embeddings.size(0)} != destination batch size {dst_embeddings.size(0)}"
            
        # Note: we don't validate embedding dimension here since it will be handled by the adapter
        
        return True, ""
        
    def forward(self, 
                src_embeddings: torch.Tensor, 
                dst_embeddings: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                graph_data_or_metadata: Optional[Union[GraphData, Dict[str, Any]]] = None,
                _skip_validation: bool = False) -> torch.Tensor:
        """
        Forward pass for the MLP predictor.
        
        Args:
            src_embeddings (torch.Tensor): Source node embeddings [batch_size, embed_dim]
            dst_embeddings (torch.Tensor): Destination node embeddings [batch_size, embed_dim]
            edge_attr (Optional[torch.Tensor], optional): Edge attributes. Defaults to None.
            graph_data_or_metadata (Optional[Union[GraphData, Dict[str, Any]]], optional): 
                Graph data or metadata. Defaults to None.
            _skip_validation (bool, optional): Whether to skip dimension validation.
                Defaults to False. Only set to True if you are sure the dimensions are correct.
                
        Returns:
            torch.Tensor: Predicted probability of citation [batch_size]
        """
        # Handle small batches (batch size 1) that might cause BatchNorm to fail
        small_batch = src_embeddings.size(0) <= 1 and self.use_batch_norm and self.training
        
        if small_batch:
            # Temporarily switch to eval mode for batch norm layers
            batch_norm_states = {}
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    batch_norm_states[name] = module.training
                    module.eval()  # Switch to eval mode to use running statistics
                        
        try:
            # Validation to prevent common dimension errors
            if not _skip_validation:
                is_valid, error_msg = self._validate_dimensions(src_embeddings, dst_embeddings)
                if not is_valid:
                    self.dimension_errors += 1
                    if self.dimension_errors <= self.max_errors_to_log:
                        raise ValueError(f"Error in MLP forward pass: {error_msg}. Feature shape: {src_embeddings.shape}")
                    else:
                        # Still raise error, but with simplified message to avoid log flooding
                        raise ValueError(f"Dimension mismatch in MLPPredictor (error #{self.dimension_errors})")

            # Get device
            device = src_embeddings.device
            
            # Adapt input dimensions if needed
            src_embeddings = self.input_adapter(src_embeddings)
            dst_embeddings = self.input_adapter(dst_embeddings)
            
            # Generate interaction features
            interaction_features = self.feature_interaction(src_embeddings, dst_embeddings)
            
            # Add metadata features if available
            if self.metadata_processor is not None and graph_data_or_metadata is not None:
                # Extract node indices from the graph if provided
                if isinstance(graph_data_or_metadata, GraphData) and hasattr(graph_data_or_metadata, 'edge_index'):
                    # Assuming first batch involves first entries in the edge_index
                    batch_size = src_embeddings.size(0)
                    src_indices = graph_data_or_metadata.edge_index[0, :batch_size]
                    dst_indices = graph_data_or_metadata.edge_index[1, :batch_size]
                    
                    # Process metadata
                    metadata_features = self.metadata_processor(
                        graph_data_or_metadata, 
                        src_indices, 
                        dst_indices
                    )
                    
                    if metadata_features is not None:
                        interaction_features = torch.cat([interaction_features, metadata_features], dim=1)
                
                # Handle dictionary metadata
                elif isinstance(graph_data_or_metadata, dict):
                    # TBD: Implement dictionary metadata handling
                    pass
            
            # Apply MLP layers
            x = interaction_features
            for layer in self.layers:
                x = layer(x)
            
            # Final output layer
            x = self.output_layer(x)
            
            # Sigmoid to get probability
            x = torch.sigmoid(x)
            
            # Ensure output has correct shape (batch_size,)
            if x.dim() > 1 and x.size(1) == 1:
                x = x.squeeze(1)
                
            return x
        
        except Exception as e:
            # Log the error with detailed shapes
            error_msg = f"Error in MLP forward pass: {str(e)}. Feature shape: {src_embeddings.shape}"
            
            # Avoid error loops
            if "Error in MLP forward pass" in str(e):
                raise e
            
            raise ValueError(error_msg)
        
        finally:
            # Restore batch norm states if we modified them
            if small_batch:
                for name, module in self.named_modules():
                    if isinstance(module, nn.BatchNorm1d) and name in batch_norm_states:
                        module.train(batch_norm_states[name])
    
    def predict_batch(self, 
                     node_embeddings: torch.Tensor, 
                     edge_indices: torch.Tensor,
                     edge_attr: Optional[torch.Tensor] = None,
                     graph_data: Optional[GraphData] = None) -> torch.Tensor:
        """Predict citations for a batch of edges.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings for all papers
                [num_nodes, embed_dim]
            edge_indices (torch.Tensor): Edge indices to predict
                [2, num_edges] where edge_indices[0] are source nodes and
                edge_indices[1] are destination nodes
            edge_attr (Optional[torch.Tensor]): Edge attributes, optionally used to condition the prediction
            graph_data (Optional[GraphData]): Graph data for metadata processing
                
        Returns:
            torch.Tensor: Predicted citation likelihood scores [num_edges]
        """
        # Handle empty edge case
        if edge_indices.shape[1] == 0:
            return torch.empty(0, device=node_embeddings.device)
            
        # Ensure edge_indices is on the same device as node_embeddings
        if edge_indices.device != node_embeddings.device:
            edge_indices = edge_indices.to(node_embeddings.device)
            
        # Get src and dst embeddings for the specified edges
        src_indices = edge_indices[0]
        dst_indices = edge_indices[1]
        
        # Handle case where indices are out of bounds
        if src_indices.max() >= node_embeddings.shape[0] or dst_indices.max() >= node_embeddings.shape[0]:
            raise ValueError(f"Edge indices out of bounds: max index {max(src_indices.max().item(), dst_indices.max().item())} >= {node_embeddings.shape[0]}")
        
        src_embeddings = node_embeddings[src_indices]
        dst_embeddings = node_embeddings[dst_indices]
        
        # Ensure edge_attr is on the same device if provided
        if edge_attr is not None and edge_attr.device != node_embeddings.device:
            edge_attr = edge_attr.to(node_embeddings.device)
        
        # Create a local dictionary with metadata instead of modifying the graph_data object
        # This prevents memory leaks by not storing data in the original graph_data
        metadata_dict = None
        if graph_data is not None:
            # Create a shallow copy of the graph_data object for temporary use
            # We'll only add src_indices and dst_indices to this copy
            from copy import copy
            metadata_dict = {'graph_data': graph_data, 'src_indices': src_indices, 'dst_indices': dst_indices}
        
        # Use forward method to predict, but skip validation to prevent potential recursion
        # The validation will happen in the forward method directly
        return self.forward(src_embeddings, dst_embeddings, edge_attr, metadata_dict, _skip_validation=True)
    
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
            **kwargs: Additional parameters
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Edge indices of top predicted citations [2, k]
                - Scores for the top predicted citations [k]
        """
        # If candidate edges provided in kwargs, use them
        candidate_edges = kwargs.get('candidate_edges', None)
        
        # Handle empty graph case
        if node_embeddings.shape[0] == 0:
            return torch.zeros((2, 0), device=node_embeddings.device, dtype=torch.long), torch.zeros(0, device=node_embeddings.device)
            
        # Handle single-node graph case
        if node_embeddings.shape[0] == 1:
            return torch.zeros((2, 0), device=node_embeddings.device, dtype=torch.long), torch.zeros(0, device=node_embeddings.device)
        
        # If no candidate edges provided, generate them
        if candidate_edges is None:
            candidate_edges = self.get_candidate_edges(existing_graph)
            
        # Handle case where no candidate edges are available
        if candidate_edges.shape[1] == 0:
            return torch.zeros((2, 0), device=node_embeddings.device, dtype=torch.long), torch.zeros(0, device=node_embeddings.device)
        
        # Predict scores for all candidate edges
        scores = self.predict_batch(node_embeddings, candidate_edges, graph_data=existing_graph)
        
        # Get top k predictions
        if k < len(scores):
            top_k_indices = torch.topk(scores, k=k).indices
            top_edges = candidate_edges[:, top_k_indices]
            top_scores = scores[top_k_indices]
        else:
            # In case there are fewer candidates than k
            top_edges = candidate_edges
            top_scores = scores
        
        return top_edges, top_scores
    
    def predict_temporal_citations(self,
                                  node_embeddings: torch.Tensor,
                                  existing_graph: GraphData,
                                  time_threshold: float,
                                  future_window: Optional[float] = None,
                                  k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict future citations based on a temporal snapshot.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings for all papers
                [num_nodes, embed_dim]
            existing_graph (GraphData): The complete citation network
            time_threshold (float): The timestamp threshold for the snapshot
            future_window (Optional[float]): If provided, only predict citations
                within the window [time_threshold, time_threshold + future_window]
            k (int): Number of top predictions to return
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Edge indices of top predicted future citations [2, k]
                - Scores for the top predicted future citations [k]
        """
        device = node_embeddings.device
        
        # Handle empty graph case
        if node_embeddings.shape[0] == 0:
            return torch.zeros((2, 0), device=device, dtype=torch.long), torch.zeros(0, device=device)
            
        # Extract papers that exist at the time threshold
        node_timestamps = existing_graph.node_timestamps  # Use standardized field name
        
        # Handle case where node timestamps are missing
        if node_timestamps is None:
            raise ValueError("Node timestamps are required for temporal prediction")
            
        # Ensure node_timestamps is on the same device as node_embeddings
        if node_timestamps.device != device:
            node_timestamps = node_timestamps.to(device)
            
        # Create mask for valid papers (published before or at time_threshold)
        valid_papers_mask = node_timestamps <= time_threshold
        valid_papers = torch.where(valid_papers_mask)[0]
        
        # Handle case where no valid papers exist
        if len(valid_papers) == 0:
            return torch.zeros((2, 0), device=device, dtype=torch.long), torch.zeros(0, device=device)
            
        # Handle case where only one valid paper exists
        if len(valid_papers) == 1:
            return torch.zeros((2, 0), device=device, dtype=torch.long), torch.zeros(0, device=device)
        
        # Get edge timestamps using standardized field name
        edge_timestamps = existing_graph.edge_timestamps
        if edge_timestamps is None:
            # If no time information available, use all existing edges
            edge_timestamps = torch.zeros(existing_graph.edge_index.shape[1], device=device)
        elif edge_timestamps.device != device:
            # Ensure edge_timestamps is on the same device
            edge_timestamps = edge_timestamps.to(device)
        
        # Ensure edge_index is on the same device
        edge_index = existing_graph.edge_index
        if edge_index.device != device:
            edge_index = edge_index.to(device)
            
        # Extract edges prior to time threshold
        edges_at_threshold_mask = edge_timestamps <= time_threshold
        edges_at_threshold = edge_index[:, edges_at_threshold_mask]
        
        # Create a snapshot graph at time_threshold
        snapshot_graph = GraphData(
            x=existing_graph.x,
            edge_index=edges_at_threshold,
            node_timestamps=node_timestamps,  # Use standardized field name
            edge_timestamps=edge_timestamps[edges_at_threshold_mask],
            snapshot_time=time_threshold
        )
        
        # Copy any metadata fields to the snapshot
        for field in self.metadata_fields.keys():
            if hasattr(existing_graph, field):
                field_value = getattr(existing_graph, field)
                # Ensure field value is on the correct device if it's a tensor
                if isinstance(field_value, torch.Tensor) and field_value.device != device:
                    field_value = field_value.to(device)
                setattr(snapshot_graph, field, field_value)
        
        # Get candidate edges that might form in the future
        candidate_edges = self.get_candidate_edges(snapshot_graph)
        
        # Handle case where no candidate edges are available
        if candidate_edges.shape[1] == 0:
            return torch.zeros((2, 0), device=device, dtype=torch.long), torch.zeros(0, device=device)
        
        # Ensure candidate_edges is on the correct device
        if candidate_edges.device != device:
            candidate_edges = candidate_edges.to(device)
            
        # Extract source and destination indices
        src_indices = candidate_edges[0]
        dst_indices = candidate_edges[1]
        
        # Get publication times for all nodes
        src_times = node_timestamps[src_indices]
        dst_times = node_timestamps[dst_indices]
        
        # Create masks for temporal filtering
        # 1. Both papers must exist at time_threshold
        papers_exist_mask = valid_papers_mask[src_indices] & valid_papers_mask[dst_indices]
        
        # 2. Causal constraint: papers can only cite older papers
        causal_mask = src_times >= dst_times
        
        # 3. Future window constraint (if specified)
        if future_window is not None:
            time_diff = src_times - dst_times
            window_mask = time_diff <= future_window
            # Combine all constraints
            valid_mask = papers_exist_mask & causal_mask & window_mask
        else:
            # Combine without window constraint
            valid_mask = papers_exist_mask & causal_mask
        
        # Filter candidate edges using the combined mask
        filtered_candidate_edges = candidate_edges[:, valid_mask]
        
        # Handle case where no valid candidates remain after filtering
        if filtered_candidate_edges.shape[1] == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty(0, device=device)
        
        # Predict scores for filtered candidate edges
        scores = self.predict_batch(node_embeddings, filtered_candidate_edges, graph_data=snapshot_graph)
        
        # Get top k predictions
        if k < len(scores):
            top_k_indices = torch.topk(scores, k=k).indices
            top_edges = filtered_candidate_edges[:, top_k_indices]
            top_scores = scores[top_k_indices]
        else:
            # In case there are fewer candidates than k
            top_edges = filtered_candidate_edges
            top_scores = scores
        
        return top_edges, top_scores
    
    def get_config(self) -> Dict[str, Any]:
        """Get predictor configuration parameters.
        
        Returns:
            Dict[str, Any]: Dictionary containing configuration parameters
        """
        config = super().get_config()
        config.update({
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm,
            'activation': self.activation,
            'interaction_dim': self.interaction_dim,
            'metadata_fields': self.metadata_fields,
            'feature_types': self.feature_types,
        })
        return config 