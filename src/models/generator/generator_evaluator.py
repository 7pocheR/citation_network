import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import traceback
import math
from typing import Dict, List, Tuple, Optional, Union, Any

from src.data.datasets import GraphData

logger = logging.getLogger(__name__)

def create_temporal_mask(graph: GraphData, time_threshold: float) -> torch.Tensor:
    """
    Create mask where nodes after time_threshold are masked.
    
    Args:
        graph: The graph data
        time_threshold: Nodes with timestamps > time_threshold will be masked
        
    Returns:
        Boolean tensor where True = masked node
    """
    if not hasattr(graph, 'node_timestamps') or graph.node_timestamps is None:
        raise ValueError("Graph does not have node_timestamps attribute")
    
    # Ensure timestamps are in the right shape for comparison
    if graph.node_timestamps.dim() > 1 and graph.node_timestamps.size(1) == 1:
        # Squeeze singleton dimension
        timestamps = graph.node_timestamps.squeeze(1)
    else:
        timestamps = graph.node_timestamps
    
    mask = timestamps > time_threshold
    
    # Ensure at least some nodes are masked
    if mask.sum() == 0:
        logger.warning(f"No nodes masked with threshold {time_threshold}. Using highest 10% of timestamps.")
        sorted_times = torch.sort(timestamps).values
        idx = int(len(sorted_times) * 0.9)
        new_threshold = sorted_times[idx].item()
        mask = timestamps > new_threshold
    
    logger.info(f"Temporal mask created: {mask.sum().item()}/{len(mask)} nodes masked " 
               f"({mask.sum().item()/len(mask)*100:.1f}%)")
    
    return mask


def create_random_mask(graph: GraphData, mask_ratio: float = 0.1) -> torch.Tensor:
    """
    Create random mask with given ratio.
    
    Args:
        graph: The graph data
        mask_ratio: Proportion of nodes to mask (0.0-1.0)
        
    Returns:
        Boolean tensor where True = masked node
    """
    num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.size(0)
    mask = torch.rand(num_nodes, device=graph.x.device) < mask_ratio
    
    # Ensure at least some nodes are masked
    if mask.sum() == 0:
        logger.warning(f"No nodes masked with ratio {mask_ratio}. Forcing one random node to be masked.")
        idx = torch.randint(num_nodes, (1,))
        mask[idx] = True
    
    logger.info(f"Random mask created: {mask.sum().item()}/{len(mask)} nodes masked " 
               f"({mask.sum().item()/len(mask)*100:.1f}%)")
    
    return mask


def extract_subgraph(graph: GraphData, node_indices: torch.Tensor) -> GraphData:
    """
    Extract a subgraph containing only the specified nodes.
    
    Args:
        graph: The original graph
        node_indices: Indices of nodes to include
        
    Returns:
        Subgraph with only the specified nodes
    """
    # Ensure node_indices is flattened to 1D
    if node_indices.dim() > 1:
        node_indices = node_indices.flatten()
    
    # If node_indices is empty, return an empty graph
    if node_indices.numel() == 0:
        return GraphData(
            num_nodes=0,
            x=graph.x.new_zeros((0, graph.x.size(1))),
            edge_index=graph.edge_index.new_zeros((2, 0)),
            edge_attr=graph.edge_attr.new_zeros((0,)) if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None,
            node_timestamps=graph.node_timestamps.new_zeros((0,)) if hasattr(graph, 'node_timestamps') and graph.node_timestamps is not None else None
        )
    
    # Get device
    device = graph.x.device
    node_indices = node_indices.to(device)
    
    # Create a mapping from original indices to new indices
    num_nodes = len(node_indices)
    node_mapping = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    node_mapping[node_indices] = torch.arange(num_nodes, device=device)
    
    # Get edges with both source and target in the subgraph
    edge_index = graph.edge_index
    mask = torch.isin(edge_index[0], node_indices) & torch.isin(edge_index[1], node_indices)
    
    # Filter edge_index
    filtered_edge_index = edge_index[:, mask].clone()
    
    # Remap node indices in edge_index
    filtered_edge_index[0] = node_mapping[filtered_edge_index[0]]
    filtered_edge_index[1] = node_mapping[filtered_edge_index[1]]
    
    # Filter edge attributes if they exist
    filtered_edge_attr = None
    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
        try:
            filtered_edge_attr = graph.edge_attr[mask]
        except Exception as e:
            logger.warning(f"Could not filter edge_attr: {mask}")
            filtered_edge_attr = None
    
    # Node timestamps if they exist
    node_timestamps = None
    if hasattr(graph, 'node_timestamps') and graph.node_timestamps is not None:
        node_timestamps = graph.node_timestamps[node_indices]
    
    # Create the subgraph
    subgraph = GraphData(
        num_nodes=num_nodes,
        x=graph.x[node_indices],
        edge_index=filtered_edge_index,
        edge_attr=filtered_edge_attr,
        node_timestamps=node_timestamps
    )
    
    return subgraph


def create_masked_graph_split(graph: GraphData, mask: torch.Tensor) -> Tuple[GraphData, GraphData, Dict[int, int]]:
    """
    Create masked and unmasked graph objects.
    
    Args:
        graph: The original graph
        mask: Boolean tensor where True = masked node
        
    Returns:
        Tuple of (unmasked_graph, masked_graph_with_features_only, index_mapping)
        where index_mapping maps from masked graph indices to original indices
    """
    # Get indices
    masked_indices = torch.nonzero(mask).squeeze()
    unmasked_indices = torch.nonzero(~mask).squeeze()
    
    # Handle case where only one node is masked/unmasked
    if masked_indices.dim() == 0:
        masked_indices = masked_indices.unsqueeze(0)
    if unmasked_indices.dim() == 0:
        unmasked_indices = unmasked_indices.unsqueeze(0)
    
    # Log the number of masked and unmasked nodes
    logger.info(f"Splitting graph - Masked indices: {masked_indices.shape}, Unmasked indices: {unmasked_indices.shape}")
    
    # Create unmasked graph (for encoder) - keep all connections between unmasked nodes
    unmasked_graph = extract_subgraph(graph, unmasked_indices)
    
    # Create masked graph (features only, no edge info)
    masked_graph = GraphData(
        num_nodes=len(masked_indices),
        x=graph.x[masked_indices],
        edge_index=None,  # No edge information
        node_timestamps=graph.node_timestamps[masked_indices] if hasattr(graph, 'node_timestamps') and graph.node_timestamps is not None else None
    )
    
    # Create mapping from masked graph indices to original indices
    masked_to_original = {i: masked_indices[i].item() for i in range(len(masked_indices))}
    
    return unmasked_graph, masked_graph, masked_to_original


def extract_edges_involving_nodes(edge_index: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
    """
    Extract edges where at least one node is in the given set.
    
    Args:
        edge_index: Edge index tensor [2, num_edges]
        node_indices: Indices of nodes to consider
        
    Returns:
        Masked edge index containing only edges involving the specified nodes
    """
    # Check if either source or target is in node_indices
    source_mask = torch.isin(edge_index[0], node_indices)
    target_mask = torch.isin(edge_index[1], node_indices)
    
    # Combined mask - keep edges where either source or target is in node_indices
    mask = source_mask | target_mask
    
    # Extract relevant edges
    filtered_edges = edge_index[:, mask]
    
    return filtered_edges


def compute_citation_metrics(
    predicted_citations: torch.Tensor, 
    actual_citations: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score for citation prediction.
    
    Args:
        predicted_citations: Predicted citation probabilities [num_masked, num_nodes]
        actual_citations: Actual citation links (binary) [num_masked, num_nodes]
        threshold: Probability threshold for positive prediction
        
    Returns:
        Dictionary of metrics
    """
    # Ensure tensors are on the same device
    device = predicted_citations.device
    actual_citations = actual_citations.to(device)
    
    # Convert predictions to binary
    predicted_binary = (predicted_citations > threshold).float()
    
    # Calculate metrics
    true_positives = (predicted_binary * actual_citations).sum().item()
    false_positives = (predicted_binary * (1 - actual_citations)).sum().item()
    false_negatives = ((1 - predicted_binary) * actual_citations).sum().item()
    
    # Compute precision, recall, F1
    precision = true_positives / max(true_positives + false_positives, 1e-8)
    recall = true_positives / max(true_positives + false_negatives, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    # Compute area under ROC curve
    try:
        from sklearn.metrics import roc_auc_score
        # Flatten tensors
        y_true = actual_citations.cpu().numpy().flatten()
        y_scores = predicted_citations.cpu().numpy().flatten()
        # Only compute if we have both positive and negative examples
        if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
            auroc = roc_auc_score(y_true, y_scores)
        else:
            auroc = 0.0
    except Exception as e:
        logger.warning(f"Error computing ROC AUC: {e}")
        auroc = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc
    }


def create_citation_matrix(
    graph: GraphData, 
    masked_indices: torch.Tensor, 
    all_indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Create a citation matrix for masked nodes.
    
    Args:
        graph: The graph data
        masked_indices: Indices of masked nodes
        all_indices: Indices of all nodes to consider (if None, use all nodes)
        
    Returns:
        Binary matrix [len(masked_indices), len(all_indices)] where 1 indicates a citation
    """
    # Ensure consistent device
    device = graph.x.device
    masked_indices = masked_indices.to(device)
    
    if all_indices is None:
        all_indices = torch.arange(graph.num_nodes, device=device)
    else:
        all_indices = all_indices.to(device)
    
    num_masked = len(masked_indices)
    num_all = len(all_indices)
    
    # Create mapping from original indices to positions in all_indices
    all_mapping = torch.full((graph.num_nodes,), -1, dtype=torch.long, device=device)
    all_mapping[all_indices] = torch.arange(num_all, device=device)
    
    # Create empty citation matrix
    citation_matrix = torch.zeros((num_masked, num_all), device=device)
    
    # Fill in citations
    if graph.edge_index is not None:
        # Convert edge_index to a set of tuples for faster membership testing
        edges_set = set((int(graph.edge_index[0, i]), int(graph.edge_index[1, i])) for i in range(graph.edge_index.shape[1]))
        
        # For each masked node
        for i, masked_idx in enumerate(masked_indices):
            masked_idx_int = int(masked_idx.item())
            
            # For each unmasked node
            for j, unmasked_idx in enumerate(all_indices):
                unmasked_idx_int = int(unmasked_idx.item())
                
                # Check if there is a citation from masked to unmasked
                if (masked_idx_int, unmasked_idx_int) in edges_set:
                    # There's a citation
                    try:
                        if j < citation_matrix.shape[1]:  # Check bounds
                            citation_matrix[i, j] = 1.0
                        else:
                            logger.warning(f"Citation index {j} out of bounds for matrix with {citation_matrix.shape[1]} columns")
                    except Exception as e:
                        logger.error(f"Error setting citation: {e}, i={i}, j={j}, shapes={citation_matrix.shape}")
    
    return citation_matrix


class GeneratorEvaluator:
    """
    Evaluator for the generator component with proper metrics for generative capability.
    
    This evaluator implements several methods for assessing the quality of generated papers:
    1. Reconstruction metrics (for training/validation)
    2. Distribution similarity metrics (comparing real vs. generated feature distributions)
    3. Visualization tools for analysis
    4. Citation pattern evaluation
    
    The evaluator supports both temporal and random masking strategies, with
    progressive difficulty across training epochs.
    """
    
    def __init__(
        self,
        feature_preprocessor=None,
        default_mask_ratio: float = 0.1,
        noise_level: float = 0.1,
        citation_threshold: float = 0.5,
        device=None
    ):
        """
        Initialize the generator evaluator.
        
        Args:
            feature_preprocessor: Optional FeaturePreprocessor instance
            default_mask_ratio: Default ratio of nodes to mask
            noise_level: Noise level for feature perturbation
            citation_threshold: Threshold for binary citation prediction
            device: Device to use for computation
        """
        self.default_mask_ratio = default_mask_ratio
        self.noise_level = noise_level
        self.citation_threshold = citation_threshold
        self.feature_preprocessor = feature_preprocessor
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize history tracking
        self.history = {
            'reconstruction_mse': [],
            'citation_f1': [],
            'mmd_score': [],
            'feature_diversity': [],
            'feature_novelty': []
        }
    
    def evaluate(
        self,
        model: Any,  # IntegratedCitationModel
        graph: GraphData,
        mask_type: str = 'temporal',
        mask_ratio: Optional[float] = None,
        time_threshold: Optional[float] = None,
        noise_level: Optional[float] = None,
        return_reconstructions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate generator performance by masking nodes and attempting to reconstruct them.
        
        Args:
            model: The model containing encoder and generator
            graph: The graph data
            mask_type: Type of masking ('temporal' or 'random')
            mask_ratio: Ratio of nodes to mask (for 'random' type)
            time_threshold: Time threshold for masking (for 'temporal' type)
            noise_level: Level of noise to add to features (0.0-1.0)
            return_reconstructions: Whether to return reconstructed features
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Use default values if not provided
        mask_ratio = mask_ratio or self.default_mask_ratio
        noise_level = noise_level or self.noise_level
        
        # Get device
        device = graph.x.device
        
        # 1. Create mask based on selected strategy
        if mask_type == 'temporal' and time_threshold is not None:
            mask = create_temporal_mask(graph, time_threshold)
        else:
            mask = create_random_mask(graph, mask_ratio)
            
        # Number of nodes being masked
        num_masked = mask.sum().item()
        logger.info(f"{mask_type.capitalize()} mask created: {num_masked}/{graph.num_nodes} nodes masked ({100*num_masked/graph.num_nodes:.1f}%)")
        
        # Get masked indices
        masked_indices = torch.nonzero(mask, as_tuple=True)[0]
        unmasked_indices = torch.nonzero(~mask, as_tuple=True)[0]
        
        # 2. Split graph into masked and unmasked parts
        # IMPORTANT: The function returns (unmasked_graph, masked_graph, masked_to_original)
        unmasked_graph, masked_graph, masked_to_original = create_masked_graph_split(graph, mask)
        
        # Get features directly from the masked graph, which now contains only the masked nodes
        masked_features = masked_graph.x
        
        # Log the shapes for debugging 
        #logger.info(f"Masked graph has {masked_graph.num_nodes} nodes, Unmasked graph has {unmasked_graph.num_nodes} nodes")
        #logger.info(f"Masked features shape: {masked_features.shape}")
        
        # Add noise to masked features for robustness if specified
        if noise_level > 0:
            masked_features_input = masked_features + noise_level * torch.randn_like(masked_features)
        else:
            masked_features_input = masked_features
            
        # Ensure data is on the correct device
        masked_features = masked_features.to(device)
        masked_features_input = masked_features_input.to(device)
        
        # 3. Compute unmasked node embeddings using the unmasked graph
        # The model could be either:
        # - A model with direct encoder method (model.encoder)
        # - A CVAEGenerator that doesn't have an encoder but uses cvae.encode
        try:
            # First approach: Try model with a direct encoder method
            if hasattr(model, 'encoder'):
                unmasked_embeddings = model.encoder(unmasked_graph).to(device)
            # Second approach: Handle CVAEGenerator by getting node embeddings
            elif hasattr(model, 'cvae'):
                # For a CVAEGenerator, we can use the model itself to get embeddings from graph
                #logger.info(f"Using alternate method to compute embeddings for CVAEGenerator")
                # Create condition from graph
                if hasattr(unmasked_graph, 'x'):
                    # Get node features from unmasked graph
                    unmasked_features = unmasked_graph.x.to(device)
                    # Use a simple method to get embeddings - use model's condition adapter
                    if hasattr(model, '_process_conditions'):
                        # Get a standardized condition
                        conditions = {
                            'timestamps': unmasked_graph.timestamp if hasattr(unmasked_graph, 'timestamp') else None,
                            'paper_contexts': None,
                            'full_graph_size': len(unmasked_graph.x)
                        }
                        # Use dummy node embeddings (will be processed by the condition adapter)
                        dummy_node_embeddings = torch.zeros(unmasked_features.size(0), model.embed_dim, device=device)
                        # Calculate embeddings from the model's condition processing
                        unmasked_embeddings = model._process_conditions(conditions, dummy_node_embeddings)
                    else:
                        # Simple fallback - use features as embeddings
                        unmasked_embeddings = unmasked_features
                else:
                    # Fallback if no features in unmasked graph
                    unmasked_embeddings = torch.zeros(len(unmasked_indices), model.embed_dim, device=device)
            else:
                # Last fallback - create synthetic embeddings
                logger.warning("Model doesn't have encoder or cvae - using synthetic embeddings")
                if hasattr(model, 'embed_dim'):
                    embed_dim = model.embed_dim
                else:
                    embed_dim = 64  # Default fallback
                unmasked_embeddings = torch.zeros(len(unmasked_indices), embed_dim, device=device)
        except Exception as e:
            # Fallback for any errors
            logger.error(f"Error computing embeddings: {str(e)}")
            # Create a set of zero embeddings as fallback
            if hasattr(model, 'embed_dim'):
                embed_dim = model.embed_dim
            else:
                embed_dim = 64  # Default fallback
            unmasked_embeddings = torch.zeros(len(unmasked_indices), embed_dim, device=device)
        
        # Calculate number of masked and unmasked nodes
        num_masked = masked_indices.shape[0]
        num_unmasked = unmasked_indices.shape[0]
        
        #logger.info(f"Masked nodes: {num_masked}, Unmasked nodes: {num_unmasked}")
        
        # Create context for each masked node (average of unmasked node embeddings or more sophisticated approach)
        # Use more sophisticated context creation if available - using attention mechanism
        if unmasked_embeddings.shape[0] > 0:
            # Create attention-based context
            # We'll create context embeddings for all masked nodes
            if hasattr(model, 'create_node_context') and callable(getattr(model, 'create_node_context')):
                # Use model's built-in context creation if available
                context_embeddings = model.create_node_context(unmasked_embeddings, num_masked)
            else:
                # Fallback: simple context creation - each masked node gets the average unmasked embedding
                context_embeddings = unmasked_embeddings.mean(0, keepdim=True).expand(num_masked, -1)
            
            # Set dimensions for logging
            context_dim = context_embeddings.shape[1]
        else:
            # If no unmasked nodes, use a zero vector (should be rare)
            embed_dim = model.encoder.embed_dim if hasattr(model.encoder, 'embed_dim') else 128
            context_embeddings = torch.zeros(num_masked, embed_dim, device=device)
            context_dim = embed_dim
        
        # Ensure context has correct embedding dimension by checking model's expected dimension
        expected_dim = None
        # First check if this is a direct generator model (has embed_dim directly)
        if hasattr(model, 'embed_dim'):
            expected_dim = model.embed_dim
            
            # Log the dimensions for debugging
            logger.debug(f"Context embeddings shape: {context_embeddings.shape}, expected dim: {expected_dim}")
            
            # Simple context using mean embedding if no specific context creation
            if model.embed_dim != context_embeddings.shape[1]:
                logger.warning(f"Dimension mismatch: context_embeddings {context_embeddings.shape[1]}, expected {model.embed_dim}")
                # Try to resize the context embeddings using a simple approach
                if context_embeddings.shape[1] > expected_dim:
                    # Truncate if too large
                    context_embeddings = context_embeddings[:, :expected_dim]
                else:
                    # Pad with zeros if too small
                    pad_size = expected_dim - context_embeddings.shape[1]
                    padding = torch.zeros(context_embeddings.shape[0], pad_size, device=device)
                    context_embeddings = torch.cat([context_embeddings, padding], dim=1)
        # Then check if this is a nested generator (model.generator exists and has embed_dim)
        elif hasattr(model, 'generator') and hasattr(model.generator, 'embed_dim'):
            expected_dim = model.generator.embed_dim
            
            # Log the dimensions for debugging
            logger.debug(f"Context embeddings shape: {context_embeddings.shape}, expected dim: {expected_dim}")
            
            # Simple context using mean embedding if no specific context creation
            if model.generator.embed_dim != context_embeddings.shape[1]:
                logger.warning(f"Dimension mismatch: context_embeddings {context_embeddings.shape[1]}, expected {model.generator.embed_dim}")
                # Use the original node embeddings directly for simplicity
                # Get the model's encoding of all graph nodes
                full_embeddings = model.encoder(graph)
                
                # Use as consistent context
                context_embeddings = full_embeddings[masked_indices]
        
        # Create timestamps if available (used for conditioning)
        # Use timestamps from the masked nodes
        if hasattr(masked_graph, 'node_timestamps') and masked_graph.node_timestamps is not None:
            masked_timestamps = masked_graph.node_timestamps.to(device)
        elif hasattr(masked_graph, 'timestamps') and masked_graph.timestamps is not None:
            masked_timestamps = masked_graph.timestamps.to(device)
        else:
            masked_timestamps = torch.ones(num_masked, device=device)
        
        # Create conditions dictionary - ensure all tensors have batch size matching masked_features
        conditions = {
            'timestamps': masked_timestamps,
            'paper_contexts': context_embeddings,
            'full_graph_size': graph.num_nodes  # Include full graph size for proper scaling
        }
        
        # Verify tensor shapes again before continuing
        #logger.info(f"Before forward pass - masked_features: {masked_features.shape}, context_embeddings: {context_embeddings.shape}")
        #logger.info(f"masked_timestamps: {masked_timestamps.shape}, full_graph_size: {graph.num_nodes}")
        
        try:
            # All data should be on the same device at this point
            # Generate reconstructed features
            
            # Important: Check dimensions of all tensors to ensure consistency
            #logger.info(f"Evaluating with shapes: masked_features={masked_features.shape}, context_embeddings={context_embeddings.shape}")
            
            # Ensure feature tensor shapes match
            num_masked_nodes = masked_features_input.size(0)
            if context_embeddings.size(0) != num_masked_nodes:
                logger.warning(f"Shape mismatch: context_embeddings has {context_embeddings.size(0)} nodes but masked_features has {num_masked_nodes}")
                
                # Adjust context embeddings to match masked features
                if context_embeddings.size(0) < num_masked_nodes:
                    repeat = num_masked_nodes // context_embeddings.size(0) + 1
                    context_embeddings = context_embeddings.repeat(repeat, 1)[:num_masked_nodes]
                else:
                    context_embeddings = context_embeddings[:num_masked_nodes]
                
                # Update conditions with fixed context embeddings
                conditions['paper_contexts'] = context_embeddings
            
            # Now attempt the forward pass with the corrected dimensions
            try:
                # Case 1: If model is a composite model with model.generator
                if hasattr(model, 'generator'):
                    reconstructed_features, mu, logvar = model.generator.forward(
                        node_embeddings=context_embeddings,  # This should have same batch size as masked_features
                        conditions=conditions,
                        features=masked_features_input
                    )
                # Case 2: If model is a CVAEGenerator directly
                else:
                    # Make the forward call directly on the model itself
                    reconstructed_features, mu, logvar = model.forward(
                        node_embeddings=context_embeddings,
                        conditions=conditions,
                        features=masked_features_input
                    )
            except Exception as e:
                logger.error(f"Error during generator forward pass: {str(e)}")
                # Create empty placeholder tensors in case of error
                reconstructed_features = torch.zeros_like(masked_features_input)
                mu = torch.zeros(masked_features_input.size(0), getattr(model, 'latent_dim', 64), device=device)
                logvar = torch.zeros_like(mu)
            
            # Ensure reconstructed_features is on the same device as masked_features
            reconstructed_features = reconstructed_features.to(device)
            
            # Create citation matrix for evaluation
            # Get all embeddings from the encoder
            try:
                # Try to get embeddings using model.encoder if available
                if hasattr(model, 'encoder'):
                    all_embeddings = model.encoder(graph).to(device)
                # For CVAEGenerator, try another approach to get embeddings
                elif hasattr(model, '_process_conditions'):
                    # Create dummy embeddings that will be processed by condition adapter
                    dummy_embeddings = torch.zeros(graph.num_nodes, model.embed_dim, device=device)
                    conditions_dict = {
                        'timestamps': graph.timestamp if hasattr(graph, 'timestamp') else None,
                        'paper_contexts': None,
                        'full_graph_size': graph.num_nodes
                    }
                    # Use model's condition processing to get embeddings
                    all_embeddings = model._process_conditions(conditions_dict, dummy_embeddings)
                else:
                    # Fallback if no suitable method available
                    logger.warning("Could not compute embeddings for citation evaluation - using zeros")
                    all_embeddings = torch.zeros(graph.num_nodes, getattr(model, 'embed_dim', 64), device=device)
            except Exception as e:
                logger.error(f"Error computing embeddings for citation evaluation: {str(e)}")
                # Create fallback embeddings
                all_embeddings = torch.zeros(graph.num_nodes, getattr(model, 'embed_dim', 64), device=device)

            # Create the actual citation matrix
            actual_citations = torch.zeros(
                num_masked, 
                graph.num_nodes if len(unmasked_indices) == graph.num_nodes else len(unmasked_indices),
                device=device
            )
            
            logger.debug(f"Creating actual_citations with shape {actual_citations.shape}")
            logger.debug(f"Masked indices shape: {masked_indices.shape}, Unmasked indices shape: {unmasked_indices.shape}")
            
            # If unmasked_indices is too small compared to actual_citations, adjust the size
            if len(unmasked_indices) < actual_citations.shape[1]:
                logger.warning(f"Unmasked indices count ({len(unmasked_indices)}) is smaller than actual_citations columns ({actual_citations.shape[1]}). Adjusting matrix.")
                # Create a correctly sized matrix
                actual_citations = torch.zeros(num_masked, len(unmasked_indices), device=device)
            
            # Fill the actual citations matrix
            if graph.edge_index is not None:
                # Convert edge_index to a set of tuples for faster membership testing
                edges_set = set((int(graph.edge_index[0, i]), int(graph.edge_index[1, i])) for i in range(graph.edge_index.shape[1]))
                
                # For each masked node
                for i, masked_idx in enumerate(masked_indices):
                    masked_idx_int = int(masked_idx.item())
                    
                    # For each unmasked node
                    for j, unmasked_idx in enumerate(unmasked_indices):
                        unmasked_idx_int = int(unmasked_idx.item())
                        
                        # Check if there is a citation from masked to unmasked
                        if (masked_idx_int, unmasked_idx_int) in edges_set:
                            # There's a citation
                            try:
                                if j < actual_citations.shape[1]:  # Check bounds
                                    actual_citations[i, j] = 1.0
                                else:
                                    logger.warning(f"Citation index {j} out of bounds for matrix with {actual_citations.shape[1]} columns")
                            except Exception as e:
                                logger.error(f"Error setting citation: {e}, i={i}, j={j}, shapes={actual_citations.shape}")
            
            logger.debug(f"Sum of actual citations: {actual_citations.sum().item()}")
            
            # Compute feature reconstruction metrics
            with torch.no_grad():
                # Ensure both tensors are on the same device
                device = masked_features.device
                reconstructed_features = reconstructed_features.to(device)
                
                # MSE for feature reconstruction
                feature_mse = F.mse_loss(reconstructed_features, masked_features).item()
                
                # Cosine similarity for features
                cos_sim = F.cosine_similarity(
                    reconstructed_features.flatten(), 
                    masked_features.flatten().to(device),
                    dim=0
                ).item()
                
                # Get embeddings from the correct source
                # For citation prediction, we need embeddings of the correct dimension
                logger.debug(f"Getting embeddings for citation prediction - all_embeddings shape: {all_embeddings.shape}")
                logger.debug(f"reconstructed_features shape: {reconstructed_features.shape}")
                
                try:
                    # Citation prediction metrics
                    if hasattr(model, 'predict_citations_from_features'):
                        logger.debug("Using model.predict_citations_from_features")
                        predicted_citations = model.predict_citations_from_features(
                            reconstructed_features, 
                            all_embeddings, 
                            masked_indices, 
                            unmasked_indices
                        )
                    # Check if this is a composite model with model.generator
                    elif hasattr(model, 'generator') and hasattr(model.generator, 'predict_citations'):
                        logger.debug("Using model.generator.predict_citations")
                        predicted_citations = model.generator.predict_citations(
                            reconstructed_features,
                            all_embeddings,
                            masked_indices,
                            unmasked_indices
                        )
                    # Check if the model itself has predict_citations (direct generator case)
                    elif hasattr(model, 'predict_citations'):
                        logger.debug("Using direct model.predict_citations")
                        predicted_citations = model.predict_citations(
                            reconstructed_features,
                            all_embeddings,
                            masked_indices,
                            unmasked_indices
                        )
                    else:
                        # No citation prediction function available
                        logger.warning("No citation prediction method found - using zeros")
                        predicted_citations = torch.zeros(
                            reconstructed_features.size(0), 
                            len(unmasked_indices), 
                            device=device
                        )
                    
                    # Ensure predicted_citations is on the same device
                    if isinstance(predicted_citations, torch.Tensor):
                        predicted_citations = predicted_citations.to(device)
                        logger.debug(f"Predicted citations shape: {predicted_citations.shape}, actual citations shape: {actual_citations.shape}")
                    else:
                        logger.error(f"Expected tensor for predicted_citations, got {type(predicted_citations)}")
                        predicted_citations = torch.zeros_like(actual_citations)
                    
                    # Compute citation metrics
                    citation_metrics = compute_citation_metrics(
                        predicted_citations, 
                        actual_citations,
                        threshold=self.citation_threshold
                    )
                except Exception as e:
                    logger.error(f"Error during citation prediction: {e}")
                    logger.error(traceback.format_exc())
                    # Create empty citation metrics to allow evaluation to continue
                    citation_metrics = {
                        'f1': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'accuracy': 0.0
                    }
                
            # Combined metrics
            metrics = {
                'feature_mse': feature_mse,
                'feature_cosine_similarity': cos_sim,
                'feature_norm_ratio': torch.norm(reconstructed_features) / torch.norm(masked_features),
                'precision': citation_metrics['precision'],
                'recall': citation_metrics['recall'],
                'f1': citation_metrics['f1'],
                'auroc': citation_metrics.get('auroc', 0.0),
                'citation_precision': citation_metrics['precision'],  # Keep legacy names for compatibility
                'citation_recall': citation_metrics['recall'],
                'citation_f1': citation_metrics['f1'],
                'citation_auroc': citation_metrics.get('auroc', 0.0),
                'num_masked_nodes': num_masked
            }
            
            # Append to history
            self.history.append({
                'metrics': metrics,
                'mask_type': mask_type,
                'mask_ratio': mask_ratio,
                'time_threshold': time_threshold,
                'timestamp': time.time()
            })
            
            # Add any other metrics if needed
            if hasattr(model, 'compute_additional_metrics'):
                additional_metrics = model.compute_additional_metrics(
                    reconstructed_features, 
                    masked_features,
                    predicted_citations,
                    actual_citations
                )
                metrics.update(additional_metrics)
                
            # Return metrics and reconstructions if requested
            if return_reconstructions:
                return {
                    'metrics': metrics,
                    'reconstructed_features': reconstructed_features.detach().cpu(),
                    'masked_features': masked_features.detach().cpu(),
                    'predicted_citations': predicted_citations.detach().cpu(),
                    'actual_citations': actual_citations.detach().cpu(),
                    'masked_indices': masked_indices.cpu()
                }
            else:
                return metrics
                
        except Exception as e:
            logger.error(f"Error during generator evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return empty metrics in case of error
            if return_reconstructions:
                return {
                    'metrics': {'error': str(e)},
                    'reconstructed_features': None,
                    'masked_features': masked_features.detach().cpu(),
                    'predicted_citations': None,
                    'actual_citations': None
                }
            else:
                return {'error': str(e),
                        'feature_mse': 0.0,
                        'citation_precision': 0.0,
                        'citation_recall': 0.0,
                        'citation_f1': 0.0,
                        'citation_auroc': 0.0}
    
    def evaluate_epoch(
        self,
        model: Any,  # IntegratedCitationModel
        graph: GraphData,
        epoch: int,
        time_factor: Optional[float] = None,
        use_temporal: bool = True,
        return_reconstructions: bool = False,
        start_ratio: float = 0.8,  # Starting with 80% papers masked
        end_ratio: float = 0.1     # Ending with 10% papers masked
    ) -> Dict[str, Any]:
        """
        Evaluate for a specific epoch with evolving masks.
        
        Args:
            model: The integrated model
            graph: Full graph data
            epoch: Current epoch (affects time threshold or random seed)
            time_factor: Factor to adjust time threshold by epoch (epochs to reach end threshold)
            use_temporal: Whether to use temporal or random masking
            return_reconstructions: Whether to return the reconstructed features and citations
            start_ratio: Initial proportion of nodes to mask (for both temporal and random)
            end_ratio: Final proportion of nodes to mask after time_factor epochs
            
        Returns:
            Evaluation metrics
        """
        # Set default time factor if not provided (total epochs to transition from start to end mask)
        if time_factor is None:
            time_factor = 10.0  # Default: reach end threshold in 10 epochs
            
        # Calculate linear progress ratio (0.0 to 1.0) based on epoch and time_factor
        linear_progress = min(epoch / time_factor, 1.0)
        
        # Apply non-linear transformation to progress (exponential decay)
        # This makes the unmasked proportion increase rapidly at first, then level off
        # The formula: 1 - exp(-k * linear_progress) / (1 - exp(-k))
        # where k controls the curve shape (higher k = faster initial increase)
        k = 3.0  # Curve shape parameter (adjustable)
        if linear_progress < 1.0:  # Avoid division by zero when progress = 1.0
            non_linear_progress = (1.0 - math.exp(-k * linear_progress)) / (1.0 - math.exp(-k))
        else:
            non_linear_progress = 1.0
            
        # Use the non-linear progress for masking
        progress = non_linear_progress
        
        # Ensure randomness is different each epoch but deterministic for the same epoch
        random_seed = int(42 + epoch)  # Different seed for each epoch
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        if use_temporal and hasattr(graph, 'node_timestamps'):
            # Calculate time threshold based on epoch and progress
            min_time = graph.node_timestamps.min().item()
            max_time = graph.node_timestamps.max().item()
            time_span = max_time - min_time
            
            # Start with low threshold (mask most papers) and gradually increase
            # For progress=0, we want to mask start_ratio of nodes
            # For progress=1, we want to mask end_ratio of nodes
            
            # First, determine what time thresholds correspond to our start_ratio and end_ratio
            # Sort timestamps to determine threshold positions
            sorted_times = torch.sort(graph.node_timestamps).values
            num_nodes = len(sorted_times)
            
            # Find time thresholds that correspond to desired ratios
            start_idx = int(num_nodes * (1 - start_ratio))
            end_idx = int(num_nodes * (1 - end_ratio))
            
            # Get the corresponding time thresholds
            start_threshold = sorted_times[start_idx].item()
            end_threshold = sorted_times[min(end_idx, num_nodes-1)].item()
            
            # Interpolate between start and end thresholds based on non-linear progress
            time_threshold = start_threshold + progress * (end_threshold - start_threshold)
            
            # Calculate expected mask ratio for logging
            expected_mask_ratio = start_ratio + progress * (end_ratio - start_ratio)
            
            # Log details about the temporal mask with non-linear progress information
            logger.info(f"Epoch {epoch}: Temporal masking with threshold={time_threshold:.2f}, "
                      f"linear_progress={linear_progress:.2f}, non_linear_progress={progress:.2f}, "
                      f"expected mask ratio ~{expected_mask_ratio:.2f}")
            
            # Evaluate with temporal mask
            metrics = self.evaluate(
                model, graph, 
                mask_type='temporal',
                time_threshold=time_threshold,
                return_reconstructions=return_reconstructions
            )
        else:
            # Use random masking with epoch-specific seed
            # For random masking, we can either:
            # 1. Keep mask_ratio constant but change the random seed
            # 2. Vary mask_ratio based on progress like in temporal masking
            
            # Option 2: Vary mask_ratio using non-linear progression
            current_mask_ratio = start_ratio + progress * (end_ratio - start_ratio)
            
            # Log details about the random mask with non-linear progress
            logger.info(f"Epoch {epoch}: Random masking with ratio={current_mask_ratio:.2f}, "
                      f"linear_progress={linear_progress:.2f}, non_linear_progress={progress:.2f}, seed={random_seed}")
            
            # Evaluate with random mask
            metrics = self.evaluate(
                model, graph, 
                mask_type='random',
                mask_ratio=current_mask_ratio,
                return_reconstructions=return_reconstructions
            )
        
        # Store metrics with epoch info
        metrics['epoch'] = epoch
        metrics['linear_progress'] = linear_progress
        metrics['non_linear_progress'] = progress
        if use_temporal and 'time_threshold' in locals():
            metrics['time_threshold'] = time_threshold
        else:
            metrics['mask_ratio'] = current_mask_ratio
            
        # Add to history with epoch details
        history_entry = {k: v for k, v in metrics.items() if not isinstance(v, torch.Tensor)}
        history_entry['mask_type'] = 'temporal' if use_temporal else 'random'
        self.history.append(history_entry)
            
        return metrics
    
    def plot_history(self, metrics: List[str] = None, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot evaluation metrics over epochs.
        
        Args:
            metrics: List of metric names to plot (default: feature_mse, feature_cosine_similarity, citation_f1)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.history:
            raise ValueError("No evaluation history available")
            
        if metrics is None:
            metrics = ['feature_mse', 'feature_cosine_similarity', 'citation_f1']
        
        fig, ax = plt.subplots(figsize=figsize)
        epochs = [entry['epoch'] for entry in self.history]
        
        for metric in metrics:
            if metric in self.history[0]:
                values = [entry[metric] for entry in self.history]
                ax.plot(epochs, values, marker='o', label=metric)
                
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title('Generator Evaluation Metrics')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def visualize_feature_reconstruction(
        self,
        original_features: torch.Tensor,
        reconstructed_features: torch.Tensor,
        noisy_features: Optional[torch.Tensor] = None,
        num_samples: int = 5,
        num_dims: int = 10,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Visualize original vs reconstructed features for a few samples.
        
        Args:
            original_features: Original features
            reconstructed_features: Reconstructed features
            noisy_features: Noisy features (input to generator)
            num_samples: Number of samples to show
            num_dims: Number of dimensions to show per sample
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(num_samples, 1, figsize=figsize)
        if num_samples == 1:
            axes = [axes]
        
        # Select random samples if we have more than requested
        if original_features.size(0) > num_samples:
            indices = torch.randperm(original_features.size(0))[:num_samples]
            orig = original_features[indices].cpu().numpy()
            recon = reconstructed_features[indices].cpu().numpy()
            noisy = noisy_features[indices].cpu().numpy() if noisy_features is not None else None
        else:
            orig = original_features.cpu().numpy()
            recon = reconstructed_features.cpu().numpy()
            noisy = noisy_features.cpu().numpy() if noisy_features is not None else None
        
        # For each sample
        for i, ax in enumerate(axes):
            # Get features for this sample
            orig_sample = orig[i, :num_dims]
            recon_sample = recon[i, :num_dims]
            
            # X coordinates
            x = np.arange(len(orig_sample))
            
            # Bar width
            width = 0.35
            
            # Plot original and reconstructed
            ax.bar(x - width/2, orig_sample, width, label='Original')
            ax.bar(x + width/2, recon_sample, width, label='Reconstructed')
            
            # Add noisy if provided
            if noisy is not None:
                noisy_sample = noisy[i, :num_dims]
                ax.plot(x, noisy_sample, 'r--', linewidth=2, label='Noisy Input')
            
            # Add error (MSE)
            mse = np.mean((orig_sample - recon_sample) ** 2)
            ax.set_title(f'Sample {i+1} - MSE: {mse:.4f}')
            
            # Format
            ax.set_xlabel('Feature Dimension')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_pca_comparison(
        self,
        original_features: torch.Tensor,
        reconstructed_features: torch.Tensor,
        n_components: int = 2,
        figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Visualize original and reconstructed features in PCA space.
        
        Args:
            original_features: Original features
            reconstructed_features: Reconstructed features
            n_components: Number of PCA components to use
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        orig = original_features.cpu().numpy()
        recon = reconstructed_features.cpu().numpy()
        
        # Combine for PCA
        combined = np.vstack([orig, recon])
        
        # Fit PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(combined)
        
        # Split back
        n_orig = orig.shape[0]
        orig_pca = transformed[:n_orig]
        recon_pca = transformed[n_orig:]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot
        ax.scatter(orig_pca[:, 0], orig_pca[:, 1], c='blue', label='Original', alpha=0.7)
        ax.scatter(recon_pca[:, 0], recon_pca[:, 1], c='red', label='Reconstructed', alpha=0.7)
        
        # Connect corresponding points with lines
        for i in range(n_orig):
            ax.plot([orig_pca[i, 0], recon_pca[i, 0]], 
                    [orig_pca[i, 1], recon_pca[i, 1]], 
                    'k-', alpha=0.2)
        
        # Add labels and legend
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('PCA Visualization of Original vs Reconstructed Features')
        ax.legend()
        ax.grid(alpha=0.3)
        
        return fig
    
    def evaluate_generation(self,
                           model,
                           graph: GraphData,
                           num_samples: int = 10,
                           temperature: float = 1.0,
                           time_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate true generative capability without using test features.
        
        This method splits the graph by time, uses the past graph to generate
        new papers, and compares their distribution to the future papers.
        
        Args:
            model: The integrated citation model
            graph: The full graph data
            num_samples: Number of papers to generate
            temperature: Sampling temperature
            time_threshold: Time threshold for splitting (if None, uses median)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Move graph to device
        device = self.device
        graph = graph.to(device)
        
        # Split graph by time
        if time_threshold is None and hasattr(graph, 'node_timestamps'):
            time_threshold = graph.node_timestamps.median().item()
        elif time_threshold is None:
            raise ValueError("No time_threshold provided and graph has no timestamps")
        
        # Create temporal split
        past_mask = graph.node_timestamps <= time_threshold
        future_mask = graph.node_timestamps > time_threshold
        
        # Extract indices
        past_indices = torch.nonzero(past_mask).squeeze()
        future_indices = torch.nonzero(future_mask).squeeze()
        
        # Handle edge cases
        if past_indices.dim() == 0:
            past_indices = past_indices.unsqueeze(0)
        if future_indices.dim() == 0:
            future_indices = future_indices.unsqueeze(0)
        
        # Extract past and future graphs
        past_graph = extract_subgraph(graph, past_indices)
        future_graph = extract_subgraph(graph, future_indices)
        
        logger.info(f"Split graph: past={len(past_indices)} nodes, future={len(future_indices)} nodes")
        
        # Generate papers using only the past graph
        with torch.no_grad():
            # Generate papers from the past graph
            generated_features, paper_info = model.generate_future_papers(
                graph=past_graph,
                time_threshold=time_threshold,
                future_window=None,
                num_papers=num_samples,
                temperature=temperature
            )
            
        # Get actual future features for comparison
        future_features = future_graph.x
        
        # Compute distribution-based metrics
        metrics = self.evaluate_distribution_similarity(
            generated_features=generated_features,
            test_features=future_features
        )
        
        # Add generation metadata
        metrics['num_generated'] = generated_features.size(0)
        metrics['num_future'] = future_features.size(0)
        metrics['time_threshold'] = time_threshold
        
        # Add feature statistics
        metrics.update(self.compute_feature_statistics(
            generated_features, future_features
        ))
        
        return metrics
    
    def evaluate_distribution_similarity(self,
                                       generated_features: torch.Tensor,
                                       test_features: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate similarity between generated and real feature distributions.
        
        This computes various distribution similarity metrics to assess how well
        the generated features match the distribution of real features.
        
        Args:
            generated_features: Generated paper features
            test_features: Real paper features from test set
            
        Returns:
            Dictionary of similarity metrics
        """
        # Move to CPU for numpy operations
        generated_np = generated_features.detach().cpu().numpy()
        test_np = test_features.detach().cpu().numpy()
        
        # Compute Maximum Mean Discrepancy (MMD)
        mmd_score = self.compute_mmd(generated_np, test_np)
        
        # Compute feature diversity
        feature_diversity = self.compute_feature_diversity(generated_np)
        
        # Compute feature novelty (distance from test features)
        feature_novelty = self.compute_feature_novelty(generated_np, test_np)
        
        # Compute cosine similarity distribution
        cos_mean, cos_std = self.compute_cosine_similarity_stats(generated_np, test_np)
        
        return {
            'mmd_score': mmd_score,
            'feature_diversity': feature_diversity,
            'feature_novelty': feature_novelty,
            'cosine_mean': cos_mean,
            'cosine_std': cos_std
        }
    
    def compute_mmd(self, generated_features: np.ndarray, test_features: np.ndarray) -> float:
        """
        Compute Maximum Mean Discrepancy between generated and test distributions.
        
        Args:
            generated_features: Generated paper features [num_gen, feature_dim]
            test_features: Test paper features [num_test, feature_dim]
            
        Returns:
            MMD score (lower is better)
        """
        try:
            # Sample if too many features (for computational efficiency)
            max_samples = 1000
            if generated_features.shape[0] > max_samples:
                indices = np.random.choice(generated_features.shape[0], max_samples, replace=False)
                generated_features = generated_features[indices]
            if test_features.shape[0] > max_samples:
                indices = np.random.choice(test_features.shape[0], max_samples, replace=False)
                test_features = test_features[indices]
            
            # Compute MMD with RBF kernel
            def rbf_kernel(x, y, sigma=1.0):
                n_x, n_y = x.shape[0], y.shape[0]
                x_sq = np.sum(x**2, axis=1, keepdims=True)
                y_sq = np.sum(y**2, axis=1, keepdims=True)
                x_y = -2 * np.dot(x, y.T)
                pairwise_sq_dist = x_sq + y_sq.T + x_y
                return np.exp(-pairwise_sq_dist / (2 * sigma**2))
            
            # Get kernel values
            x_kernel = rbf_kernel(generated_features, generated_features)
            y_kernel = rbf_kernel(test_features, test_features)
            xy_kernel = rbf_kernel(generated_features, test_features)
            
            # Compute MMD
            mmd = np.mean(x_kernel) + np.mean(y_kernel) - 2 * np.mean(xy_kernel)
            return float(mmd)
        except Exception as e:
            logger.error(f"Error computing MMD: {e}")
            return float('nan')
    
    def compute_feature_diversity(self, features: np.ndarray) -> float:
        """
        Compute diversity of generated features.
        
        Args:
            features: Feature array [num_features, feature_dim]
            
        Returns:
            Diversity score (higher is more diverse)
        """
        try:
            # If only one feature, return 0 diversity
            if features.shape[0] <= 1:
                return 0.0
            
            # Compute pairwise distances
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(features)
            
            # Average pairwise distance (excluding self-comparisons)
            n = distances.shape[0]
            # Create a mask for the diagonal (excluding self-comparisons)
            mask = ~np.eye(n, dtype=bool)
            # Compute mean distance
            diversity = np.mean(distances[mask])
            
            return float(diversity)
        except Exception as e:
            logger.error(f"Error computing feature diversity: {e}")
            return float('nan')
    
    def compute_feature_novelty(self, generated_features: np.ndarray, test_features: np.ndarray) -> float:
        """
        Compute novelty of generated features compared to test features.
        
        Args:
            generated_features: Generated paper features [num_gen, feature_dim]
            test_features: Test paper features [num_test, feature_dim]
            
        Returns:
            Novelty score (higher is more novel)
        """
        try:
            # Compute minimum distance from each generated feature to any test feature
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(generated_features, test_features)
            
            # Minimum distance for each generated feature
            min_distances = np.min(distances, axis=1)
            
            # Average minimum distance (novelty)
            novelty = np.mean(min_distances)
            
            return float(novelty)
        except Exception as e:
            logger.error(f"Error computing feature novelty: {e}")
            return float('nan')
    
    def compute_cosine_similarity_stats(self, generated_features: np.ndarray, test_features: np.ndarray) -> Tuple[float, float]:
        """
        Compute statistics of cosine similarity between generated and test features.
        
        Args:
            generated_features: Generated paper features [num_gen, feature_dim]
            test_features: Test paper features [num_test, feature_dim]
            
        Returns:
            Tuple of (mean_similarity, std_similarity)
        """
        try:
            # Compute cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(generated_features, test_features)
            
            # Compute mean and std
            mean_sim = np.mean(similarity)
            std_sim = np.std(similarity)
            
            return float(mean_sim), float(std_sim)
        except Exception as e:
            logger.error(f"Error computing cosine similarity stats: {e}")
            return float('nan'), float('nan')
    
    def compute_feature_statistics(self, generated_features: torch.Tensor, test_features: torch.Tensor) -> Dict[str, float]:
        """
        Compute basic statistics about the generated features.
        
        Args:
            generated_features: Generated paper features
            test_features: Real paper features from test set
            
        Returns:
            Dictionary of feature statistics
        """
        # Move to CPU for numpy operations
        generated_np = generated_features.detach().cpu().numpy()
        test_np = test_features.detach().cpu().numpy()
        
        # Basic statistics
        gen_mean = np.mean(generated_np)
        gen_std = np.std(generated_np)
        test_mean = np.mean(test_np)
        test_std = np.std(test_np)
        
        # Mean feature values
        gen_feature_means = np.mean(generated_np, axis=0)
        test_feature_means = np.mean(test_np, axis=0)
        
        # Correlation between mean feature values
        try:
            feature_correlation = np.corrcoef(gen_feature_means, test_feature_means)[0, 1]
        except:
            feature_correlation = float('nan')
        
        return {
            'gen_mean': float(gen_mean),
            'gen_std': float(gen_std),
            'test_mean': float(test_mean),
            'test_std': float(test_std),
            'feature_correlation': float(feature_correlation)
        }
    
    def evaluate_reconstruction(self, reconstructed_features: torch.Tensor, original_features: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate reconstruction quality for model training/validation.
        
        Args:
            reconstructed_features: Reconstructed features [batch_size, feature_dim]
            original_features: Original features [batch_size, feature_dim]
            
        Returns:
            Dictionary of reconstruction metrics
        """
        # Move tensors to CPU for numpy operations
        recon_np = reconstructed_features.detach().cpu().numpy()
        orig_np = original_features.detach().cpu().numpy()
        
        # MSE
        mse = np.mean((recon_np - orig_np) ** 2)
        
        # MAE
        mae = np.mean(np.abs(recon_np - orig_np))
        
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim = np.mean([
            cosine_similarity(recon_np[i:i+1], orig_np[i:i+1])[0, 0]
            for i in range(len(recon_np))
        ])
        
        # Update history
        self.history['reconstruction_mse'].append(float(mse))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'cosine_similarity': float(cos_sim)
        }
    
    def visualize_feature_distributions(self,
                                      generated_features: torch.Tensor,
                                      test_features: torch.Tensor,
                                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Visualize the distribution of generated vs. test features.
        
        Args:
            generated_features: Generated paper features
            test_features: Real paper features from test set
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Move to CPU for numpy operations
        generated_np = generated_features.detach().cpu().numpy()
        test_np = test_features.detach().cpu().numpy()
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Overall distribution
        ax = axes[0, 0]
        ax.hist(generated_np.flatten(), bins=50, alpha=0.5, label='Generated', density=True)
        ax.hist(test_np.flatten(), bins=50, alpha=0.5, label='Real', density=True)
        ax.set_title('Overall Feature Distribution')
        ax.legend()
        
        # 2. Mean feature values comparison
        ax = axes[0, 1]
        gen_means = np.mean(generated_np, axis=0)
        test_means = np.mean(test_np, axis=0)
        
        # Select a subset of features for clarity
        num_features = min(20, len(gen_means))
        indices = np.random.choice(len(gen_means), num_features, replace=False)
        
        ax.scatter(test_means[indices], gen_means[indices], alpha=0.7)
        
        # Add y=x line
        min_val = min(np.min(gen_means), np.min(test_means))
        max_val = max(np.max(gen_means), np.max(test_means))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        ax.set_xlabel('Real Features (mean)')
        ax.set_ylabel('Generated Features (mean)')
        ax.set_title('Mean Feature Value Comparison')
        
        # 3. Feature variance comparison
        ax = axes[1, 0]
        gen_vars = np.var(generated_np, axis=0)
        test_vars = np.var(test_np, axis=0)
        
        ax.scatter(test_vars[indices], gen_vars[indices], alpha=0.7)
        
        # Add y=x line
        min_val = min(np.min(gen_vars), np.min(test_vars))
        max_val = max(np.max(gen_vars), np.max(test_vars))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        ax.set_xlabel('Real Features (variance)')
        ax.set_ylabel('Generated Features (variance)')
        ax.set_title('Feature Variance Comparison')
        
        # 4. PCA visualization
        ax = axes[1, 1]
        try:
            from sklearn.decomposition import PCA
            
            # Combine features for PCA
            combined = np.vstack([generated_np, test_np])
            
            # Apply PCA
            pca = PCA(n_components=2)
            projected = pca.fit_transform(combined)
            
            # Split back into generated and test
            gen_projected = projected[:len(generated_np)]
            test_projected = projected[len(generated_np):]
            
            # Plot
            ax.scatter(test_projected[:, 0], test_projected[:, 1], alpha=0.7, label='Real')
            ax.scatter(gen_projected[:, 0], gen_projected[:, 1], alpha=0.7, label='Generated')
            ax.set_title('PCA Projection')
            ax.legend()
        except Exception as e:
            logger.error(f"Error in PCA visualization: {e}")
            ax.text(0.5, 0.5, "PCA Failed", ha='center', va='center')
        
        fig.tight_layout()
        return fig 