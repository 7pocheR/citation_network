import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from copy import deepcopy

from src.data.datasets import GraphData

logger = logging.getLogger(__name__)

def create_temporal_mask(
    graph: GraphData, 
    time_threshold: float, 
    include_edges: bool = True
) -> torch.Tensor:
    """Create a binary mask for nodes before/after a time threshold.
    
    Args:
        graph (GraphData): The citation graph
        time_threshold (float): Time threshold for masking
        include_edges (bool): Whether to include edges in the mask
        
    Returns:
        torch.Tensor: Binary mask where 1 indicates nodes before threshold
    """
    # Check if graph has timestamp information
    if not hasattr(graph, 'node_timestamps') or graph.node_timestamps is None:
        logger.warning("Graph does not have node_timestamps attribute. Cannot create temporal mask.")
        return torch.ones(graph.x.size(0), dtype=torch.bool, device=graph.x.device)
    
    # Create mask based on timestamps
    timestamps = graph.node_timestamps
    mask = timestamps <= time_threshold
    
    # Create edge mask if requested
    if include_edges and hasattr(graph, 'edge_index'):
        # Get source and target nodes
        src, dst = graph.edge_index
        
        # Create edge mask: both nodes must be before threshold
        src_mask = mask[src]
        dst_mask = mask[dst]
        edge_mask = src_mask & dst_mask
        
        # Store edge mask in graph
        graph.edge_mask = edge_mask
    
    return mask

def create_temporal_snapshot(
    graph: GraphData,
    time_threshold: float
) -> GraphData:
    """
    Create a temporal snapshot of the graph at the given threshold.
    
    This function creates a subgraph containing only nodes and edges
    up to the specified time threshold.
    
    Args:
        graph: The input graph
        time_threshold: The time threshold
        
    Returns:
        A new graph with only nodes and edges up to the threshold
    """
    # Create mask for nodes before time threshold
    mask = create_temporal_mask(graph, time_threshold)
    
    # Get indices of nodes that satisfy the mask
    node_indices = torch.where(mask)[0]
    
    # Create snapshot with filtered nodes
    snapshot = graph.subgraph(node_indices)
    
    # No need to set num_nodes as it's now computed automatically from the data
    # snapshot.num_nodes = mask.sum().item()  # This line is removed
    
    return snapshot

def split_graph_by_time(
    graph: GraphData,
    time_threshold: float,
    future_window: Optional[float] = None
) -> Tuple[GraphData, GraphData]:
    """Split a graph into past and future components based on time.
    
    Args:
        graph (GraphData): The citation graph
        time_threshold (float): Time threshold for splitting
        future_window (Optional[float]): If provided, only include nodes
            within this time window after threshold
        
    Returns:
        Tuple[GraphData, GraphData]: Past graph and future graph
    """
    # Create past graph (snapshot up to time_threshold)
    past_graph = create_temporal_snapshot(graph, time_threshold)
    
    # Create future graph
    future_graph = deepcopy(graph)
    
    # Create mask for future nodes
    if not hasattr(graph, 'node_timestamps') or graph.node_timestamps is None:
        print("Warning: Graph does not have node_timestamps, using node_timestamp if available")
        if hasattr(graph, 'node_timestamp') and graph.node_timestamp is not None:
            graph.node_timestamps = graph.node_timestamp
        else:
            # Create default timestamps as a fallback
            num_nodes = graph.x.size(0)
            timestamps = torch.linspace(0, 10, num_nodes, device=graph.x.device)
            graph.node_timestamps = timestamps
    
    if future_window is not None:
        # Only include nodes within future window
        future_mask = (graph.node_timestamps > time_threshold) & (graph.node_timestamps <= time_threshold + future_window)
    else:
        # Include all future nodes
        future_mask = graph.node_timestamps > time_threshold
    
    # Apply mask to future nodes
    future_graph.x = graph.x[future_mask]
    
    # If available, filter node timestamps
    if hasattr(graph, 'node_timestamps') and graph.node_timestamps is not None:
        future_graph.node_timestamps = graph.node_timestamps[future_mask]
    
    # Filter edge_index for future graph
    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
        try:
            src, dst = graph.edge_index
            
            # Create edge mask: at least one node must be in future
            src_future = future_mask[src]
            dst_future = future_mask[dst]
            edge_mask = src_future | dst_future
            
            # Create node index mapping (old -> new)
            old_to_new = torch.full(
                (future_mask.size(0),), -1, dtype=torch.long, device=future_mask.device
            )
            old_to_new[future_mask] = torch.arange(future_mask.sum(), device=future_mask.device)
            
            # Filter edges using mask
            filtered_edge_index = graph.edge_index[:, edge_mask]
            
            # Set invalid indices for nodes not in future
            src_valid = future_mask[filtered_edge_index[0]]
            dst_valid = future_mask[filtered_edge_index[1]]
            
            # Create a new edge index with only valid edges
            valid_edge_mask = src_valid & dst_valid
            valid_edge_index = filtered_edge_index[:, valid_edge_mask]
            
            # Remap node indices
            remapped_edge_index = old_to_new[valid_edge_index]
            
            # Set new edge_index
            future_graph.edge_index = remapped_edge_index
            
            # If available, filter edge attributes
            edge_attributes = [
                'edge_attr', 'edge_timestamps', 'edge_weight', 'edge_type', 
                'edge_features', 'edge_time'
            ]
            
            for attr in edge_attributes:
                if hasattr(graph, attr) and getattr(graph, attr) is not None:
                    try:
                        # Apply both masks sequentially
                        attr_data = getattr(graph, attr)
                        
                        # Handle edge_attr differently based on its type
                        if attr == 'edge_attr' and isinstance(attr_data, dict):
                            # For dictionary type, filter each value separately
                            filtered_dict = {}
                            for key, value in attr_data.items():
                                if edge_mask.sum() > 0:
                                    filtered_value = value[edge_mask]
                                    if valid_edge_mask.sum() > 0 and filtered_value.size(0) > 0:
                                        filtered_dict[key] = filtered_value[valid_edge_mask]
                            
                            if filtered_dict:  # Only set if not empty
                                setattr(future_graph, attr, filtered_dict)
                        else:
                            # Standard tensor filtering
                            if edge_mask.sum() > 0:
                                filtered_attr = attr_data[edge_mask]
                                if valid_edge_mask.sum() > 0 and filtered_attr.size(0) > 0:
                                    setattr(future_graph, attr, filtered_attr[valid_edge_mask])
                    except Exception as e:
                        print(f"Warning: Could not filter {attr} for future graph: {e}")
        except Exception as e:
            print(f"Warning: Error processing edges for future graph: {e}")
            # Set empty edge_index as fallback
            future_graph.edge_index = torch.zeros((2, 0), dtype=torch.long, device=graph.x.device)
    
    # Update num_nodes
    future_graph.num_nodes = future_mask.sum().item()
    
    return past_graph, future_graph

def create_temporal_training_data(
    graph: GraphData,
    time_threshold: float,
    future_window: Optional[float] = None,
    validation_ratio: float = 0.2
) -> Tuple[GraphData, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create training data for temporal prediction.
    
    This function prepares data for training a model to predict future papers.
    
    Args:
        graph (GraphData): The citation graph
        time_threshold (float): Time threshold for splitting
        future_window (Optional[float]): If provided, only include nodes
            within this time window after threshold
        validation_ratio (float): Ratio of future nodes to use for validation
        
    Returns:
        Tuple[GraphData, torch.Tensor, torch.Tensor, torch.Tensor]:
            - Past graph (input graph)
            - Future node features (ground truth for generation)
            - Training mask for future nodes
            - Validation mask for future nodes
    """
    # Split graph into past and future
    past_graph, future_graph = split_graph_by_time(graph, time_threshold, future_window)
    
    # Get future node features
    future_node_features = future_graph.x
    
    # Create training/validation masks for future nodes
    num_future = future_graph.num_nodes
    indices = torch.randperm(num_future)
    
    # Calculate split point
    val_size = int(num_future * validation_ratio)
    train_size = num_future - val_size
    
    # Create masks
    train_mask = torch.zeros(num_future, dtype=torch.bool, device=future_node_features.device)
    val_mask = torch.zeros(num_future, dtype=torch.bool, device=future_node_features.device)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:]] = True
    
    return past_graph, future_node_features, train_mask, val_mask

def extract_citation_targets(
    graph: GraphData,
    future_graph: GraphData,
    past_mask: torch.Tensor,
    future_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract citation targets from future papers to past papers.
    
    This creates pairs of indices where future papers cite past papers.
    
    Args:
        graph (GraphData): The original graph
        future_graph (GraphData): The future graph
        past_mask (torch.Tensor): Mask for past nodes
        future_mask (torch.Tensor): Mask for future nodes
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Source indices (future papers)
            - Target indices (past papers)
    """
    # Check if graph has edge data
    if not hasattr(graph, 'edge_index') or graph.edge_index is None:
        logger.warning("Graph does not have edge_index. Cannot extract citation targets.")
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
    
    # Extract edges from original graph
    src, dst = graph.edge_index
    
    # Create mapping for past nodes
    past_indices = torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)
    past_indices[past_mask] = torch.arange(past_mask.sum(), device=graph.x.device)
    
    # Create mapping for future nodes
    future_indices = torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)
    future_indices[future_mask] = torch.arange(future_mask.sum(), device=graph.x.device)
    
    # Find edges from future to past
    future_to_past_mask = future_mask[src] & past_mask[dst]
    
    # Extract edges that satisfy the mask
    future_paper_indices = src[future_to_past_mask]
    past_paper_indices = dst[future_to_past_mask]
    
    # Remap indices to their respective spaces
    remapped_future_indices = future_indices[future_paper_indices]
    remapped_past_indices = past_indices[past_paper_indices]
    
    return remapped_future_indices, remapped_past_indices

def create_sliding_windows(
    graph: GraphData,
    window_size: float,
    stride: float,
    min_threshold: Optional[float] = None,
    max_threshold: Optional[float] = None
) -> List[Tuple[float, float]]:
    """Create a series of sliding time windows for temporal prediction.
    
    Args:
        graph (GraphData): The citation graph
        window_size (float): Size of each window
        stride (float): Stride between consecutive windows
        min_threshold (Optional[float]): Minimum time threshold
        max_threshold (Optional[float]): Maximum time threshold
        
    Returns:
        List[Tuple[float, float]]: List of (time_threshold, future_window) pairs
    """
    # Check if graph has timestamp information
    if not hasattr(graph, 'node_timestamps') or graph.node_timestamps is None:
        logger.warning("Graph does not have node_timestamps attribute. Cannot create sliding windows.")
        return [(0.0, window_size)]
    
    # Get timestamp range
    min_time = graph.node_timestamps.min().item()
    max_time = graph.node_timestamps.max().item()
    
    # Apply optional constraints
    if min_threshold is not None:
        min_time = max(min_time, min_threshold)
    
    if max_threshold is not None:
        max_time = min(max_time, max_threshold)
    
    # Create windows
    windows = []
    current_time = min_time
    
    while current_time + window_size <= max_time:
        windows.append((current_time, window_size))
        current_time += stride
    
    return windows

def prepare_temporal_batches(
    graph: GraphData,
    batch_size: int,
    time_windows: List[Tuple[float, float]]
) -> List[Tuple[GraphData, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Prepare batches of temporal training data for multiple time windows.
    
    Args:
        graph (GraphData): The citation graph
        batch_size (int): Batch size for each time window
        time_windows (List[Tuple[float, float]]): List of (time_threshold, future_window) pairs
        
    Returns:
        List[Tuple[GraphData, torch.Tensor, torch.Tensor, torch.Tensor]]:
            List of (past_graph, future_features, train_mask, val_mask) for each window
    """
    batches = []
    
    for time_threshold, future_window in time_windows:
        # Create training data for this window
        data = create_temporal_training_data(
            graph=graph,
            time_threshold=time_threshold,
            future_window=future_window
        )
        
        # Skip if not enough future nodes
        if data[1].size(0) < batch_size:
            continue
        
        batches.append(data)
    
    return batches 