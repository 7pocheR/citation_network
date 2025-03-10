import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
import logging
import random
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Simple Data class to replace PyTorch Geometric's Data class
class GraphData:
    """A graph data container for citation networks with proper edge attributes."""
    def __init__(self, 
                 x=None,                     # Node features 
                 edge_index=None,            # Edge connectivity
                 adjacency_matrix=None,      # Adjacency matrix (sparse format)
                 edge_attr=None,             # Combined edge attributes
                 edge_time=None,             # Edge timestamps (legacy)
                 edge_timestamps=None,       # Edge timestamps (standardized)
                 paper_times=None,           # Publication timestamps (legacy)
                 node_timestamps=None,       # Node timestamps (standardized)
                 snapshot_time=None,         # Overall snapshot timestamp
                 batch_nodes=None,           # Original node indices for edge-based batching
                 original_edge_indices=None, # Original edge indices for edge-based batching
                 **kwargs):
                 
        self.x = x  # node features
        
        # Handle edge connectivity representation
        # We support both edge_index and adjacency_matrix formats
        # and will convert between them as needed
        self._edge_index = None
        self._adjacency_matrix = None
        
        # Set edge_index and adjacency_matrix
        if edge_index is not None:
            self.edge_index = edge_index  # This will call the setter
        elif adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix  # This will call the setter
            
        self.batch_nodes = batch_nodes  # original node indices for edge-based batching
        self.original_edge_indices = original_edge_indices  # original edge indices for edge-based batching
        self.train_index = None
        self.val_index = None
        self.test_index = None
        
        # Standardize timestamp fields
        self._edge_timestamps = edge_timestamps if edge_timestamps is not None else edge_time
        self._node_timestamps = node_timestamps if node_timestamps is not None else paper_times
        
        # Handle edge attributes and timestamps
        if self._edge_timestamps is not None:
            if isinstance(edge_attr, dict):
                edge_attr['time'] = self._edge_timestamps
                self.edge_attr = edge_attr
            else:
                self.edge_attr = {'time': self._edge_timestamps}
                if edge_attr is not None:
                    self.edge_attr['attr'] = edge_attr
        else:
            self.edge_attr = edge_attr
            
        # Store snapshot timestamp
        self.snapshot_time = snapshot_time
        
        # Store any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    # Properties for edge connectivity representation
    @property
    def edge_index(self):
        """Get edge index representation"""
        return self._edge_index
        
    @edge_index.setter
    def edge_index(self, value):
        """Set edge index and update adjacency matrix"""
        if value is not None:
            # Ensure edge_index is always a 2D tensor with shape [2, num_edges]
            if len(value.shape) == 1:
                # Convert 1D tensor to proper 2D format
                # This is likely an error, but we'll reshape it to avoid crashes
                # Assuming the 1D tensor contains flattened [src, dst] pairs
                if value.size(0) % 2 == 0:  # Even number of elements
                    value = value.view(2, -1)
                else:
                    # If odd number of elements, we can't reshape properly
                    # Create a valid empty edge_index instead
                    logger.warning(f"Invalid 1D edge_index with odd length {value.size(0)}. Creating empty edge_index.")
                    value = torch.zeros((2, 0), dtype=torch.long, device=value.device)
            elif len(value.shape) == 2 and value.shape[0] != 2:
                # If it's 2D but in wrong orientation [num_edges, 2], transpose it
                if value.shape[1] == 2:
                    value = value.t()
                else:
                    # If it's in a completely wrong shape, log warning and create empty edge_index
                    logger.warning(f"Invalid edge_index shape {value.shape}. Should be [2, num_edges]. Creating empty edge_index.")
                    value = torch.zeros((2, 0), dtype=torch.long, device=value.device)
                    
        self._edge_index = value
        if value is not None:
            # Update adjacency matrix to match edge_index
            self._adjacency_matrix = self._edge_index_to_adjacency_matrix(value)
            
    @property
    def adjacency_matrix(self):
        """Get adjacency matrix representation"""
        return self._adjacency_matrix
        
    @adjacency_matrix.setter
    def adjacency_matrix(self, value):
        """Set adjacency matrix and update edge_index"""
        self._adjacency_matrix = value
        if value is not None:
            # Update edge_index to match adjacency matrix
            self._edge_index = self._adjacency_matrix_to_edge_index(value)
            
    @property
    def num_nodes(self):
        """Get number of nodes in the graph"""
        if self.x is not None:
            return self.x.size(0)
        elif self._adjacency_matrix is not None:
            return self._adjacency_matrix.size(0)
        elif self._edge_index is not None and len(self._edge_index.size()) > 0:
            return max(self._edge_index.max().item() + 1 if self._edge_index.numel() > 0 else 0, 0)
        return 0
    
    @num_nodes.setter
    def num_nodes(self, value):
        """Set number of nodes in the graph - mainly used for backwards compatibility"""
        # We don't actually store this value directly anymore, but we keep the setter for compatibility
        self._num_nodes_override = value
    
    def _edge_index_to_adjacency_matrix(self, edge_index):
        """Convert edge_index to sparse adjacency matrix"""
        # Edge case: empty graph or no edges
        if edge_index is None or len(edge_index.size()) == 0 or edge_index.numel() == 0:
            # Return empty sparse matrix
            n = self.num_nodes if self.x is not None else 0
            indices = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            values = torch.zeros(0, device=self.device)
            return torch.sparse_coo_tensor(indices, values, (n, n))
            
        # Validate edge_index format
        if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
            logger.warning(f"Invalid edge_index shape in _edge_index_to_adjacency_matrix: {edge_index.shape}")
            # Try to fix the shape if possible
            if len(edge_index.shape) == 2 and edge_index.shape[1] == 2:
                # Transpose from [num_edges, 2] to [2, num_edges]
                edge_index = edge_index.t()
                logger.info(f"Transposed edge_index to shape: {edge_index.shape}")
            elif len(edge_index.shape) == 1 and edge_index.size(0) % 2 == 0:
                # Reshape from 1D to 2D
                edge_index = edge_index.view(2, -1)
                logger.info(f"Reshaped 1D edge_index to shape: {edge_index.shape}")
            else:
                # Cannot fix, return empty adjacency matrix
                n = self.num_nodes if self.x is not None else 0
                indices = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                values = torch.zeros(0, device=self.device)
                return torch.sparse_coo_tensor(indices, values, (n, n))
            
        # Get number of nodes
        n = max(edge_index.max().item() + 1, self.num_nodes) if edge_index.numel() > 0 else self.num_nodes
        
        # Create sparse adjacency matrix
        device = edge_index.device
        
        # Create a COO tensor directly from edge_index
        values = torch.ones(edge_index.size(1), device=device)
        adj_matrix = torch.sparse_coo_tensor(edge_index, values, (n, n))
        
        return adj_matrix
    
    def _adjacency_matrix_to_edge_index(self, adj_matrix):
        """Convert sparse adjacency matrix to edge_index"""
        # Edge case: empty graph
        if adj_matrix is None or adj_matrix.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
            
        # Get device
        device = adj_matrix.device
        
        # For sparse matrix, convert directly
        if adj_matrix.is_sparse:
            indices = adj_matrix._indices()
            # Filter out zero values if needed
            if adj_matrix._values().numel() > 0:
                mask = adj_matrix._values() > 0
                indices = indices[:, mask]
            
            # Ensure indices has shape [2, num_edges]
            if indices.shape[0] != 2:
                if indices.shape[1] == 2:
                    # Transpose if needed
                    indices = indices.t()
                    logger.info(f"Transposed indices from adjacency matrix to shape: {indices.shape}")
                else:
                    logger.warning(f"Invalid indices shape from adjacency matrix: {indices.shape}")
                    # Return empty edge_index if we can't fix it
                    return torch.zeros((2, 0), dtype=torch.long, device=device)
                    
            return indices
        
        # For dense matrix, convert to edge_index
        else:
            # Find non-zero elements
            edge_index = adj_matrix.nonzero().t()
            
            # Ensure edge_index has shape [2, num_edges]
            if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
                edge_index = edge_index.t()
                
            return edge_index
            
    @property
    def device(self):
        """Get device of the graph data"""
        if self.x is not None:
            return self.x.device
        elif self._edge_index is not None:
            return self._edge_index.device
        elif self._adjacency_matrix is not None:
            return self._adjacency_matrix.device
        return torch.device('cpu')
            
    def to(self, device):
        """Move the graph to specified device"""
        if self.x is not None:
            self.x = self.x.to(device)
        if self._edge_index is not None:
            self._edge_index = self._edge_index.to(device)
        if self._adjacency_matrix is not None:
            self._adjacency_matrix = self._adjacency_matrix.to(device)
        if self.edge_attr is not None:
            if isinstance(self.edge_attr, dict):
                for k, v in self.edge_attr.items():
                    if torch.is_tensor(v):
                        self.edge_attr[k] = v.to(device)
            elif torch.is_tensor(self.edge_attr):
                self.edge_attr = self.edge_attr.to(device)
        if self._edge_timestamps is not None and torch.is_tensor(self._edge_timestamps):
            self._edge_timestamps = self._edge_timestamps.to(device)
        if self._node_timestamps is not None and torch.is_tensor(self._node_timestamps):
            self._node_timestamps = self._node_timestamps.to(device)
        if hasattr(self, 'paper_times') and self.paper_times is not None and torch.is_tensor(self.paper_times):
            self.paper_times = self.paper_times.to(device)
        if self.batch_nodes is not None:
            self.batch_nodes = self.batch_nodes.to(device)
        if self.original_edge_indices is not None:
            self.original_edge_indices = self.original_edge_indices.to(device)
            
        # Handle any other tensor attributes that might not be covered above
        for attr_name in dir(self):
            if attr_name.startswith('_') or attr_name in ['x', '_edge_index', '_adjacency_matrix', 
                                                         'edge_attr', '_edge_timestamps', '_node_timestamps',
                                                         'paper_times', 'batch_nodes', 'original_edge_indices']:
                continue
                
            attr = getattr(self, attr_name)
            if torch.is_tensor(attr):
                setattr(self, attr_name, attr.to(device))
            elif isinstance(attr, list) and len(attr) > 0 and torch.is_tensor(attr[0]):
                setattr(self, attr_name, [t.to(device) for t in attr])
                
        return self
    
    # Properties to maintain backward compatibility while standardizing field names
    @property
    def edge_time(self):
        """Legacy accessor for edge timestamps."""
        if hasattr(self, '_edge_timestamps') and self._edge_timestamps is not None:
            return self._edge_timestamps
        if isinstance(self.edge_attr, dict) and 'time' in self.edge_attr:
            return self.edge_attr['time']
        return None
        
    @edge_time.setter
    def edge_time(self, value):
        """Legacy setter for edge timestamps."""
        self._edge_timestamps = value
        if isinstance(self.edge_attr, dict):
            self.edge_attr['time'] = value
        else:
            self.edge_attr = {'time': value}
            
    @property
    def edge_timestamps(self):
        """Standardized accessor for edge timestamps."""
        return self.edge_time
        
    @edge_timestamps.setter
    def edge_timestamps(self, value):
        """Standardized setter for edge timestamps."""
        self.edge_time = value
        
    @property
    def paper_times(self):
        """Legacy accessor for node timestamps."""
        return self._node_timestamps
        
    @paper_times.setter
    def paper_times(self, value):
        """Legacy setter for node timestamps."""
        self._node_timestamps = value
            
    @property
    def node_timestamps(self):
        """Standardized accessor for node timestamps."""
        return self._node_timestamps
        
    @node_timestamps.setter
    def node_timestamps(self, value):
        """Standardized setter for node timestamps."""
        self._node_timestamps = value
    
    def __repr__(self):
        info = []
        for key, value in self.__dict__.items():
            if key == 'metadata' and isinstance(value, dict):
                # For metadata, just show the keys and count of items
                metadata_info = {}
                for meta_key, meta_value in value.items():
                    if isinstance(meta_value, list) and len(meta_value) > 10:
                        metadata_info[meta_key] = f"list with {len(meta_value)} items"
                    elif isinstance(meta_value, dict) and len(meta_value) > 10:
                        metadata_info[meta_key] = f"dict with {len(meta_value)} items"
                    else:
                        metadata_info[meta_key] = type(meta_value).__name__
                info.append(f'metadata={metadata_info}')
            else:
                # For other attributes, show shape if available
                info.append(f'{key}={list(value.shape) if hasattr(value, "shape") else value}')
        
        return f"{self.__class__.__name__}({', '.join(info)})"
    
    @property
    def num_edges(self):
        """Get the number of edges in the graph."""
        if self.edge_index is None:
            return 0
            
        # Edge index should always be a 2D tensor with shape [2, num_edges]
        if len(self.edge_index.shape) == 2:
            return self.edge_index.size(1)
        else:
            # This should not happen with the fixed edge_index setter
            logger.warning(f"Unexpected edge_index shape: {self.edge_index.shape}. Expected 2D tensor.")
            return 0
    
    def __getitem__(self, key):
        """Make the class dict-like to be compatible with the encoder."""
        # Handle legacy and standardized field names
        if key == 'edge_time' or key == 'edge_timestamps':
            return self.edge_timestamps
        if key == 'paper_times' or key == 'node_timestamps':
            return self.node_timestamps
            
        if key == 'edge_attr' and hasattr(self, 'edge_attr') and isinstance(self.edge_attr, dict):
            # For compatibility with models expecting a tensor, concatenate dict values
            attr_list = []
            for attr_tensor in self.edge_attr.values():
                if attr_tensor.dim() == 1:
                    attr_tensor = attr_tensor.unsqueeze(1)
                attr_list.append(attr_tensor)
            return torch.cat(attr_list, dim=1) if attr_list else None
        return getattr(self, key, None)
    
    def get(self, key, default=None):
        """Mimic dict.get() method."""
        # Handle legacy and standardized field names
        if key == 'edge_time' or key == 'edge_timestamps':
            return self.edge_timestamps if self.edge_timestamps is not None else default
        if key == 'paper_times' or key == 'node_timestamps':
            return self.node_timestamps if self.node_timestamps is not None else default
            
        return getattr(self, key, default)
    
    def keys(self):
        """Return all attributes as keys."""
        keys = list(self.__dict__.keys())
        # Ensure standardized field names are included
        if self.edge_timestamps is not None and 'edge_timestamps' not in keys:
            keys.append('edge_timestamps')
        if self.node_timestamps is not None and 'node_timestamps' not in keys:
            keys.append('node_timestamps')
        return keys

    def subgraph(self, subset):
        """
        Extract a subgraph containing only the specified nodes.
        
        Args:
            subset: Boolean mask or list of node indices
            
        Returns:
            A new GraphData object containing the subgraph
        """
        # Ensure subset is a tensor
        if not isinstance(subset, torch.Tensor):
            subset = torch.tensor(subset, dtype=torch.long)
            
        # Get device
        if hasattr(subset, 'device'):
            device = subset.device
        else:
            device = self.x.device if hasattr(self, 'x') and self.x is not None else torch.device('cpu')
            
        # Convert indices to boolean mask if needed
        if subset.dtype == torch.bool:
            # It's already a mask
            subset = subset.to(device)  # Ensure mask is on the correct device
        elif subset.dtype in (torch.int32, torch.int64, torch.long):
            # Convert indices to mask
            subset = subset.to(device)  # Ensure subset is on the right device
            mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=device)
            mask[subset] = True
            subset = mask
        else:
            raise ValueError(f"Unsupported subset dtype: {subset.dtype}")
            
        # Create new graph
        new_graph = self.__class__()
        
        # Copy node features
        if self.x is not None:
            # Ensure x and subset are on the same device
            x = self.x.to(device)
            subset = subset.to(device)
            new_graph.x = x[subset]
            
        # Copy node timestamps
        if self.node_timestamps is not None:
            # Ensure node_timestamps and subset are on the same device
            node_timestamps = self.node_timestamps.to(device)
            subset = subset.to(device)
            new_graph._node_timestamps = node_timestamps[subset]
            
        # Create node mapping for edges
        num_nodes = self.num_nodes
        node_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        # Ensure subset is on the same device as node_idx
        subset = subset.to(device)
        
        # Calculate the number of True values in the subset
        subset_sum = subset.sum().item()
        
        # Create mapping for nodes in the subset
        node_idx[subset] = torch.arange(subset_sum, device=device)
        
        # Handle edge_index: Only keep edges where both endpoints are in the subset
        if self.edge_index is not None:
            # Ensure edge_index is on the same device as subset
            edge_index = self.edge_index.to(device)
            
            # Get source and destination nodes
            src, dst = edge_index
            
            # Create mask for edges where both endpoints are in the subset
            edge_mask = subset[src] & subset[dst]
            
            # Filter edges using the mask
            filtered_edge_index = edge_index[:, edge_mask]
            
            # Remap node indices
            src_filtered = filtered_edge_index[0]
            dst_filtered = filtered_edge_index[1]
            
            # Apply node mapping
            new_src = node_idx[src_filtered]
            new_dst = node_idx[dst_filtered]
            
            # Create new edge index
            new_edge_index = torch.stack([new_src, new_dst], dim=0)
            
            # Set edge index for the new graph
            new_graph.edge_index = new_edge_index
            
            # Handle edge attributes if present
            if isinstance(self.edge_attr, dict):
                new_edge_attr = {}
                for key, value in self.edge_attr.items():
                    if isinstance(value, torch.Tensor) and value.size(0) == self.num_edges:
                        new_edge_attr[key] = value[edge_mask]
                new_graph.edge_attr = new_edge_attr
                
        # Copy paper times if available (legacy support)
        if hasattr(self, 'paper_times') and self.paper_times is not None:
            paper_times = self.paper_times.to(device)
            subset = subset.to(device)
            new_graph.paper_times = paper_times[subset]
            
        # Copy metadata
        if hasattr(self, 'metadata') and self.metadata is not None:
            new_graph.metadata = self.metadata.copy() if isinstance(self.metadata, dict) else self.metadata
            
        # Copy timestamps
        if hasattr(self, 'timestamps') and self.timestamps is not None:
            timestamps = self.timestamps.to(device)
            subset = subset.to(device)
            new_graph.timestamps = timestamps[subset]
        
        return new_graph

    def mask_edges(self, edges_to_mask):
        """
        Create a new graph with specified edges masked/removed from the edge_index.
        This is crucial for proper transductive link prediction, where edges being
        predicted shouldn't be used during message passing.
        
        Args:
            edges_to_mask: Tensor of shape [2, num_edges] containing edges to mask
                           in the same format as edge_index
            
        Returns:
            A new GraphData object with the specified edges masked
        """
        if self.edge_index is None:
            logger.warning("No edges to mask in the graph")
            return self.clone()
            
        # Get device
        device = self.edge_index.device
        
        # Ensure edges_to_mask is on the same device
        edges_to_mask = edges_to_mask.to(device)
        
        # Create a new graph with the same nodes
        new_graph = self.clone()
        
        # Create a set of edges to mask for efficient lookup
        edges_to_mask_set = set()
        for i in range(edges_to_mask.shape[1]):
            src, dst = edges_to_mask[0, i].item(), edges_to_mask[1, i].item()
            edges_to_mask_set.add((src, dst))
        
        # Create a mask identifying edges to keep (those not in edges_to_mask)
        edge_mask = torch.ones(self.edge_index.shape[1], dtype=torch.bool, device=device)
        
        for i in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            if (src, dst) in edges_to_mask_set:
                edge_mask[i] = False
        
        # Apply mask to edge_index
        new_graph.edge_index = self.edge_index[:, edge_mask]
        
        # Apply mask to edge attributes if present
        if isinstance(self.edge_attr, dict):
            new_edge_attr = {}
            for key, value in self.edge_attr.items():
                if isinstance(value, torch.Tensor) and value.size(0) == self.num_edges:
                    new_edge_attr[key] = value[edge_mask]
            new_graph.edge_attr = new_edge_attr
        elif self.edge_attr is not None and hasattr(self.edge_attr, 'shape') and self.edge_attr.shape[0] == self.num_edges:
            new_graph.edge_attr = self.edge_attr[edge_mask]
        
        return new_graph
    
    def clone(self):
        """
        Create a shallow copy of the graph.
        
        Returns:
            A new GraphData object with the same attributes
        """
        new_graph = self.__class__()
        
        # Copy all attributes
        for key, value in self.__dict__.items():
            # Skip private attributes
            if key.startswith('_'):
                continue
                
            # Skip methods and functions
            if callable(value):
                continue
                
            # Copy the attribute
            setattr(new_graph, key, value)
            
        # Copy private attributes needed for proper functioning
        if hasattr(self, '_edge_index'):
            new_graph._edge_index = self._edge_index
        if hasattr(self, '_adjacency_matrix'):
            new_graph._adjacency_matrix = self._adjacency_matrix
        if hasattr(self, '_edge_timestamps'):
            new_graph._edge_timestamps = self._edge_timestamps
        if hasattr(self, '_node_timestamps'):
            new_graph._node_timestamps = self._node_timestamps
            
        return new_graph

class CitationNetworkDataset(Dataset):
    """
    Dataset for citation network data.
    
    Creates a dataset that provides batches of historical graph snapshots
    and features for training the model.
    """
    
    def __init__(self, 
                 node_features: torch.Tensor,
                 edge_indices: torch.Tensor,
                 timestamps: torch.Tensor,
                 paper_years: torch.Tensor = None,
                 sequence_length: int = 5,
                 time_window: int = 365,  # days
                 stride: int = 180,  # days
                 data_loader = None  # The data loader that created this dataset
                 ):
        """
        Initialize the dataset.
        
        Args:
            node_features: Node feature tensor [num_nodes, feature_dim]
            edge_indices: Edge index tensor [2, num_edges]
            timestamps: Edge timestamp tensor [num_edges]
            paper_years: Publication years for each paper [num_nodes]
            sequence_length: Number of snapshots in each sequence
            time_window: Time window for each snapshot in days
            stride: Stride between snapshots in days
            data_loader: The data loader that created this dataset
        """
        self.node_features = node_features
        self.edge_indices = edge_indices
        self.timestamps = timestamps
        self.paper_years = paper_years
        self.sequence_length = sequence_length
        self.time_window = time_window
        self.stride = stride
        self.data_loader = data_loader
        
        # Sort edges by timestamp
        sorted_indices = torch.argsort(timestamps)
        self.sorted_edge_indices = edge_indices[:, sorted_indices]
        self.sorted_timestamps = timestamps[sorted_indices]
        
        # Create snapshots
        self.create_snapshots()
        
        logger.info(f"Created {len(self.snapshot_data)} snapshots")

    def create_snapshots(self):
        """Create graph snapshots based on time windows."""
        # For simplicity in testing, create a fixed number of snapshots
        # by dividing edges evenly
        num_edges = self.edge_indices.shape[1]
        edges_per_snapshot = max(1, num_edges // self.sequence_length)
        
        self.snapshot_data = []
        
        for i in range(self.sequence_length):
            start_idx = i * edges_per_snapshot
            end_idx = min((i + 1) * edges_per_snapshot, num_edges)
            
            # Get edges for this snapshot
            snapshot_edges = self.sorted_edge_indices[:, start_idx:end_idx]
            snapshot_timestamps = self.sorted_timestamps[start_idx:end_idx]
            
            # FIX 1: Select only active nodes in this snapshot
            active_nodes = torch.unique(snapshot_edges)
            snapshot_features = self.node_features[active_nodes]
            
            # Create node mapping for reindexing edges
            node_mapping = {int(node): i for i, node in enumerate(active_nodes)}
            
            # Reindex edges to match filtered nodes
            reindexed_edges = torch.zeros_like(snapshot_edges)
            for j in range(snapshot_edges.shape[1]):
                reindexed_edges[0, j] = node_mapping[int(snapshot_edges[0, j])]
                reindexed_edges[1, j] = node_mapping[int(snapshot_edges[1, j])]
            
            # Get paper years for active nodes if available
            if hasattr(self, 'paper_years') and self.paper_years is not None:
                snapshot_paper_years = self.paper_years[active_nodes]
            else:
                snapshot_paper_years = None
            
            # Create GraphData object with FIX 2: proper edge timestamps
            # Note: We pass edge_time directly to ensure it's stored in edge_attr
            # for compatibility with the TGN model
            snapshot = GraphData(
                x=snapshot_features,
                edge_index=reindexed_edges,
                edge_time=snapshot_timestamps,  # This will be stored in edge_attr['time']
                paper_times=snapshot_paper_years,
                snapshot_time=torch.tensor([self.sorted_timestamps[end_idx-1].item()])  # Use last timestamp as snapshot time
            )
            
            self.snapshot_data.append(snapshot)
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        # For testing, return a small number of samples
        return 10
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - historical_graph: List of graph snapshots
                - paper_features: Features of papers in the current snapshot
                - target_citations: Target citation matrix
                - loader: The data loader that created this dataset (if available)
        """
        # For testing purposes, create random data
        # In a real implementation, you would return actual data
        
        # Create historical graph snapshots
        historical_graph = self.snapshot_data.copy()
        
        # Create random paper features
        num_nodes = self.node_features.shape[0]
        feature_dim = self.node_features.shape[1]
        paper_features = torch.randn(num_nodes, feature_dim)
        
        # Create random target citations
        # In a real implementation, these would be the actual citation links
        target_citations = torch.zeros(num_nodes, num_nodes)
        num_citations = min(20, num_nodes * num_nodes // 10)  # Limit the number of citations
        
        source_nodes = torch.randint(0, num_nodes, (num_citations,))
        target_nodes = torch.randint(0, num_nodes, (num_citations,))
        
        for s, t in zip(source_nodes, target_nodes):
            if s != t:  # Avoid self-citations
                target_citations[s, t] = 1.0
        
        result = {
            'historical_graph': historical_graph,
            'paper_features': paper_features,
            'target_citations': target_citations,
        }
        
        # Add paper years if available
        if self.paper_years is not None:
            # Make sure paper_years is a tensor
            if isinstance(self.paper_years, list):
                result['paper_years'] = torch.tensor(self.paper_years)
            else:
                result['paper_years'] = self.paper_years
        else:
            # Fallback: create random years between 2000 and 2020
            # In a real implementation, these would be actual paper years
            logger.warning("paper_years not provided, using random values")
            result['paper_years'] = torch.randint(2000, 2020, (num_nodes,))

        # Add the data loader if available
        if self.data_loader is not None:
            result['loader'] = self.data_loader
            
            # If the loader has feature info, add topic count directly to the batch
            if hasattr(self.data_loader, 'get_feature_info'):
                feature_info = self.data_loader.get_feature_info()
                if 'topic_count' in feature_info:
                    result['topic_count'] = feature_info['topic_count']
        
        return result
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate function for creating batches.
        
        Args:
            batch: List of samples
            
        Returns:
            Collated batch
        """
        # Combine historical_graph lists from all samples
        combined_batch = {
            'historical_graph': [],
            'paper_features': [],
            'target_citations': [],
            'paper_years': []
        }
        
        for sample in batch:
            # Since GraphData objects can't be batched directly,
            # we'll keep them as a list
            combined_batch['historical_graph'].extend(sample['historical_graph'])
            
            # Combine other tensor data properly
            combined_batch['paper_features'].append(sample['paper_features'])
            combined_batch['target_citations'].append(sample['target_citations'])
            combined_batch['paper_years'].append(sample['paper_years'])
        
        # Stack tensor data
        combined_batch['paper_features'] = torch.stack(combined_batch['paper_features'])
        combined_batch['target_citations'] = torch.stack(combined_batch['target_citations'])
        combined_batch['paper_years'] = torch.stack(combined_batch['paper_years'])
        
        return combined_batch


class TemporalCitationGraphDataset(Dataset):
    """
    Dataset for processing temporal citation graphs for the dynamic GNN encoder.
    Creates temporal graph snapshots for learning node embeddings.
    """
    
    def __init__(self, 
                 node_features: torch.Tensor,
                 edge_index: torch.Tensor,
                 edge_timestamps: torch.Tensor,
                 node_timestamps: Optional[torch.Tensor] = None,
                 paper_times: Optional[torch.Tensor] = None,  # For backward compatibility
                 num_snapshots: int = 10):
        """
        Initialize the temporal citation graph dataset.
        
        Args:
            node_features: Node feature matrix (num_nodes x feature_dim)
            edge_index: Edge index tensor (2 x num_edges)
            edge_timestamps: Edge timestamps (num_edges)
            node_timestamps: Node timestamps (num_nodes), standardized field name
            paper_times: Node timestamps (num_nodes), legacy field name for backward compatibility
            num_snapshots: Number of temporal snapshots to create
        """
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_timestamps = edge_timestamps
        
        # Store node timestamps, with fallback to paper_times for backward compatibility
        if node_timestamps is not None:
            self.node_timestamps = node_timestamps
        elif paper_times is not None:
            self.node_timestamps = paper_times
        else:
            self.node_timestamps = None
            
        self.num_snapshots = num_snapshots
        
        # Sort edges by time
        sorted_edge_idx = torch.argsort(edge_timestamps)
        self.edge_index_sorted = edge_index[:, sorted_edge_idx]
        self.edge_timestamps_sorted = edge_timestamps[sorted_edge_idx]
        
        # Create snapshots
        self.snapshots = self._create_snapshots()
        
    def _create_snapshots(self) -> List[GraphData]:
        """Create temporal snapshots of the citation graph."""
        min_time = self.edge_timestamps_sorted.min().item()
        max_time = self.edge_timestamps_sorted.max().item()
        time_range = max_time - min_time
        
        snapshots = []
        for i in range(self.num_snapshots):
            # Calculate time threshold for this snapshot
            time_threshold = min_time + (i + 1) * time_range / self.num_snapshots
            
            # Select edges before this threshold
            mask = self.edge_timestamps_sorted <= time_threshold
            snapshot_edge_index = self.edge_index_sorted[:, mask]
            snapshot_edge_timestamps = self.edge_timestamps_sorted[mask]
            
            # FIX 1: Select only active nodes in this snapshot
            active_nodes = torch.unique(snapshot_edge_index)
            snapshot_features = self.node_features[active_nodes]
            
            # Create node mapping for reindexing edges
            node_mapping = {int(node): i for i, node in enumerate(active_nodes)}
            
            # Reindex edges to match filtered nodes
            reindexed_edges = torch.zeros((2, snapshot_edge_index.shape[1]), dtype=snapshot_edge_index.dtype)
            for j in range(snapshot_edge_index.shape[1]):
                reindexed_edges[0, j] = node_mapping[int(snapshot_edge_index[0, j])]
                reindexed_edges[1, j] = node_mapping[int(snapshot_edge_index[1, j])]
            
            # Get paper years for active nodes if available
            if hasattr(self, 'node_timestamps') and self.node_timestamps is not None:
                snapshot_node_timestamps = self.node_timestamps[active_nodes]
            else:
                snapshot_node_timestamps = None
            
            # Create GraphData object with proper edge attributes
            # Note: Using standardized field names
            snapshot = GraphData(
                x=snapshot_features,
                edge_index=reindexed_edges,
                edge_timestamps=snapshot_edge_timestamps,
                node_timestamps=snapshot_node_timestamps,
                snapshot_time=torch.tensor([time_threshold])
            )
            
            # Store original node indices for reference
            snapshot.original_node_indices = active_nodes
            
            snapshots.append(snapshot)
        
        return snapshots
    
    def __len__(self) -> int:
        """Get the number of snapshots."""
        return self.num_snapshots
    
    def __getitem__(self, idx: int) -> GraphData:
        """Get a specific snapshot."""
        return self.snapshots[idx] 

def compute_temporal_embeddings(self, 
                              node_features: torch.Tensor, 
                              edge_index: torch.Tensor, 
                              edge_timestamps: torch.Tensor,
                              hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute temporal embeddings using a temporal GNN.
    
    Args:
        node_features: Node feature matrix
        edge_index: Edge index tensor
        edge_timestamps: Edge timestamp tensor
        hidden_state: Optional initial hidden state
        
    Returns:
        Node embeddings
    """
    # Implementation would go here
    pass 