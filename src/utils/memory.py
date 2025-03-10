import torch
import gc
import numpy as np
import logging
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

logger = logging.getLogger(__name__)

def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Get memory statistics for the given device.
    
    Args:
        device: Device to get statistics for (defaults to current device if None)
        
    Returns:
        Dictionary with memory statistics
    """
    if device is None and torch.cuda.is_available():
        device = torch.cuda.current_device()
    
    stats = {}
    
    if torch.cuda.is_available() and (isinstance(device, torch.device) and device.type == 'cuda' or isinstance(device, int)):
        device_idx = device if isinstance(device, int) else device.index
        
        # Get memory stats
        stats["allocated_bytes"] = torch.cuda.memory_allocated(device_idx)
        stats["reserved_bytes"] = torch.cuda.memory_reserved(device_idx)
        stats["allocated_mb"] = stats["allocated_bytes"] / (1024 * 1024)
        stats["reserved_mb"] = stats["reserved_bytes"] / (1024 * 1024)
        
        # Get device properties
        props = torch.cuda.get_device_properties(device_idx)
        stats["total_memory_mb"] = props.total_memory / (1024 * 1024)
        stats["utilization_pct"] = (stats["reserved_bytes"] / props.total_memory) * 100
        
        # Number of cached tensors (use memory_reserved instead of memory_cached)
        stats["cached_tensors"] = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
    
    # CPU memory is relevant for preprocessing and data loading
    import psutil
    vm = psutil.virtual_memory()
    stats["cpu_total_gb"] = vm.total / (1024**3)
    stats["cpu_available_gb"] = vm.available / (1024**3)
    stats["cpu_used_pct"] = vm.percent
    
    return stats

@contextmanager
def autocast_context(enabled: bool = True, dtype: torch.dtype = torch.float16, cache_enabled: bool = True):
    """Context manager for automatic mixed precision.
    
    Args:
        enabled: Whether to enable autocast
        dtype: Data type to use for autocast
        cache_enabled: Whether to enable autocast cache
        
    Yields:
        None
    """
    if enabled and torch.cuda.is_available():
        with torch.cuda.amp.autocast(enabled=enabled, dtype=dtype, cache_enabled=cache_enabled):
            yield
    else:
        yield

@contextmanager
def memory_efficient_context(enabled: bool = True, 
                             enable_autocast: bool = True,
                             enable_checkpoint: bool = False,
                             clear_cache: bool = True):
    """Context manager for memory-efficient computation.
    
    Args:
        enabled: Whether to enable memory optimizations
        enable_autocast: Whether to enable automatic mixed precision
        enable_checkpoint: Whether to enable gradient checkpointing
        clear_cache: Whether to clear CUDA cache before and after
        
    Yields:
        None
    """
    if not enabled:
        yield
        return
    
    # Clear cache before
    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Use mixed precision if enabled
    with autocast_context(enabled=enable_autocast):
        yield
    
    # Clear cache after
    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def pin_memory(tensor: torch.Tensor) -> torch.Tensor:
    """Pin memory for efficient GPU transfer.
    
    Args:
        tensor: Tensor to pin
        
    Returns:
        Pinned tensor
    """
    if not torch.cuda.is_available():
        return tensor
    
    if isinstance(tensor, torch.Tensor) and not tensor.is_pinned():
        return tensor.pin_memory()
    return tensor

def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    """Move optimizer states to the specified device.
    
    Args:
        optimizer: Optimizer to move
        device: Target device
        
    Returns:
        None
    """
    for param in optimizer.state.values():
        # Not all params have state yet
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)

def enable_gradient_checkpointing(model: torch.nn.Module, enabled: bool = True):
    """Enable or disable gradient checkpointing for a model.
    
    Args:
        model: Model to modify
        enabled: Whether to enable checkpointing
        
    Returns:
        None
    """
    if hasattr(model, "gradient_checkpointing_enable") and hasattr(model, "gradient_checkpointing_disable"):
        if enabled:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
    else:
        # Legacy support for older PyTorch versions
        def enable_checkpointing(submodule):
            if hasattr(submodule, "checkpoint_activations"):
                submodule.checkpoint_activations = enabled
            elif hasattr(submodule, "gradient_checkpointing"):
                submodule.gradient_checkpointing = enabled
        
        # Apply recursively to all submodules
        model.apply(enable_checkpointing)
        
        # Set attribute for tracking
        model._uses_gradient_checkpointing = enabled

class GradientAccumulator:
    """Helper class for gradient accumulation.
    
    This class helps manage gradient accumulation for efficient training
    with large batch sizes by simulating larger batches with multiple
    forward and backward passes before updating model weights.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, steps: int = 1, scaler=None):
        """Initialize the accumulator.
        
        Args:
            optimizer: Optimizer to use
            steps: Number of steps to accumulate
            scaler: Optional GradScaler for mixed precision training
        """
        self.optimizer = optimizer
        self.steps = max(1, steps)
        self.current_step = 0
        self.scaler = scaler
        self._loss_value = 0.0
        self._should_optimize = False
    
    def backward(self, loss: torch.Tensor, retain_graph: bool = False):
        """Backward pass with gradient accumulation.
        
        Args:
            loss: Loss tensor
            retain_graph: Whether to retain graph
            
        Returns:
            None
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward(retain_graph=retain_graph)
        else:
            scaled_loss.backward(retain_graph=retain_graph)
        
        # Track loss value
        self._loss_value += loss.detach().item()
        
        # Increment step counter
        self.current_step += 1
        
        # Check if we should optimize
        self._should_optimize = (self.current_step % self.steps == 0)
        
        return self._should_optimize
    
    def step(self):
        """Perform optimization step.
        
        Returns:
            Loss value
        """
        if not self._should_optimize:
            return None
        
        # Clip gradients
        if hasattr(self.optimizer, "clip_grad_norm"):
            self.optimizer.clip_grad_norm()
        
        # Perform optimization step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Reset optimizer
        self.optimizer.zero_grad()
        
        # Calculate average loss
        avg_loss = self._loss_value / self.steps
        
        # Reset accumulation
        self._loss_value = 0.0
        self._should_optimize = False
        
        return avg_loss
    
    def zero_grad(self):
        """Reset gradients.
        
        Returns:
            None
        """
        self.optimizer.zero_grad()
        self.current_step = 0
        self._loss_value = 0.0
        self._should_optimize = False

class MemoryEfficientLinear(torch.nn.Module):
    """Memory-efficient linear layer.
    
    This layer processes the input in chunks to reduce memory usage for
    large matrices, useful for very large embedding layers or attention.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, chunk_size: int = 1024):
        """Initialize the layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use bias
            chunk_size: Size of chunks for processing
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chunk_size = chunk_size
        
        # Create weight and bias
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features)) if bias else None
        
        # Initialize parameters
        torch.nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        batch_size = input.size(0)
        
        # Process in one go if smaller than chunk size
        if batch_size <= self.chunk_size:
            return torch.nn.functional.linear(input, self.weight, self.bias)
        
        # Process in chunks
        outputs = []
        for i in range(0, batch_size, self.chunk_size):
            chunk = input[i:i+self.chunk_size]
            outputs.append(torch.nn.functional.linear(chunk, self.weight, self.bias))
        
        return torch.cat(outputs, dim=0)

class MixedPrecisionTraining:
    """Helper class for mixed precision training.
    
    This class wraps the components needed for mixed precision training,
    including the GradScaler and autocast context.
    """
    
    def __init__(self, enabled: bool = True, 
                dtype: torch.dtype = torch.float16,
                init_scale: float = 2.**16, 
                growth_factor: float = 2.0,
                backoff_factor: float = 0.5,
                growth_interval: int = 2000):
        """Initialize mixed precision training.
        
        Args:
            enabled: Whether to enable mixed precision
            dtype: Data type to use for autocast
            init_scale: Initial scale factor for GradScaler
            growth_factor: Growth factor for GradScaler
            backoff_factor: Backoff factor for GradScaler
            growth_interval: Growth interval for GradScaler
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        
        # Create scaler
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.enabled,
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval
        ) if self.enabled else None
    
    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision.
        
        Yields:
            None
        """
        with autocast_context(enabled=self.enabled, dtype=self.dtype):
            yield
    
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training.
        
        Args:
            loss: Loss tensor
            
        Returns:
            Scaled loss tensor
        """
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer: torch.optim.Optimizer):
        """Perform optimization step with loss scaling.
        
        Args:
            optimizer: Optimizer to step
            
        Returns:
            None
        """
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def get_scale(self) -> float:
        """Get current scale factor.
        
        Returns:
            Current scale factor
        """
        if self.enabled:
            return self.scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary.
        
        Returns:
            State dictionary
        """
        if self.enabled:
            return self.scaler.state_dict()
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary.
        
        Args:
            state_dict: State dictionary
            
        Returns:
            None
        """
        if self.enabled:
            self.scaler.load_state_dict(state_dict)

def compute_memory_requirement(num_nodes: int, 
                             num_edges: int, 
                             feat_dim: int, 
                             embed_dim: int, 
                             dtype: torch.dtype = torch.float32,
                             include_model: bool = True,
                             include_optimizer: bool = True) -> Dict[str, float]:
    """Estimate memory requirements for a graph model.
    
    Args:
        num_nodes: Number of nodes in the graph
        num_edges: Number of edges in the graph
        feat_dim: Feature dimension
        embed_dim: Embedding dimension
        dtype: Data type for model parameters
        include_model: Whether to include model parameters in the estimate
        include_optimizer: Whether to include optimizer states in the estimate
        
    Returns:
        Dictionary with memory requirement estimates
    """
    # Get bytes per element based on dtype
    if dtype == torch.float32:
        bytes_per_elem = 4
    elif dtype == torch.float16:
        bytes_per_elem = 2
    elif dtype == torch.int64:
        bytes_per_elem = 8
    elif dtype == torch.int32:
        bytes_per_elem = 4
    else:
        bytes_per_elem = 4  # Default
    
    # Compute graph memory requirements
    node_features_mb = (num_nodes * feat_dim * bytes_per_elem) / (1024 * 1024)
    edge_index_mb = (2 * num_edges * 8) / (1024 * 1024)  # 8 bytes for int64 indices
    edge_attr_mb = (num_edges * 1 * bytes_per_elem) / (1024 * 1024)  # Assuming 1 feature per edge
    
    # Estimate memory for node embeddings
    embeddings_mb = (num_nodes * embed_dim * bytes_per_elem) / (1024 * 1024)
    
    # Estimate model parameter size (very rough estimate)
    model_params_mb = 0
    if include_model:
        # Encoder parameters (estimate as 3x embedding size for simplicity)
        model_params_mb += (3 * embed_dim * embed_dim * bytes_per_elem) / (1024 * 1024)
        # Linear layers for feature projection
        model_params_mb += (feat_dim * embed_dim * bytes_per_elem) / (1024 * 1024)
        # Predictor parameters
        model_params_mb += (3 * embed_dim * embed_dim * bytes_per_elem) / (1024 * 1024)
    
    # Optimizer states (Adam needs 2 additional states per parameter)
    optimizer_mb = 0
    if include_optimizer:
        optimizer_mb = model_params_mb * 2
    
    # Activation memory (rough estimate, depends on batch size)
    batch_size = min(512, num_nodes)  # Default batch size, capped at 512
    activations_mb = (batch_size * embed_dim * bytes_per_elem * 10) / (1024 * 1024)  # Estimate as 10x the embedding size
    
    # Compute total memory requirement
    total_mb = node_features_mb + edge_index_mb + edge_attr_mb + embeddings_mb + model_params_mb + optimizer_mb + activations_mb
    total_gb = total_mb / 1024
    
    # Return memory requirements
    result = {
        "node_features_mb": node_features_mb,
        "edge_index_mb": edge_index_mb,
        "edge_attr_mb": edge_attr_mb,
        "embeddings_mb": embeddings_mb,
        "model_params_mb": model_params_mb,
        "optimizer_mb": optimizer_mb,
        "activations_mb": activations_mb,
        "total_mb": total_mb,
        "total_gb": total_gb
    }
    
    return result

# Overloaded version that accepts a GraphData object
def compute_memory_requirement_from_graph(graph_data, 
                                        embed_dim: int, 
                                        batch_size: int = 128,
                                        dtype: torch.dtype = torch.float32,
                                        include_model: bool = True,
                                        include_optimizer: bool = True) -> float:
    """Estimate memory requirements for a graph model using a GraphData object.
    
    Args:
        graph_data: GraphData object
        embed_dim: Embedding dimension
        batch_size: Batch size for training
        dtype: Data type for model parameters
        include_model: Whether to include model parameters in the estimate
        include_optimizer: Whether to include optimizer states in the estimate
        
    Returns:
        Estimated memory requirement in GB
    """
    num_nodes = graph_data.num_nodes
    num_edges = graph_data.num_edges
    feat_dim = graph_data.x.size(1)
    
    # Use the original function
    result = compute_memory_requirement(
        num_nodes=num_nodes,
        num_edges=num_edges,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        dtype=dtype,
        include_model=include_model,
        include_optimizer=include_optimizer
    )
    
    return result["total_gb"]

def sparse_tensor_to_dict(sparse_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Convert a sparse tensor to a dictionary for efficient storage.
    
    Args:
        sparse_tensor: Sparse tensor to convert
        
    Returns:
        Dictionary representation of the sparse tensor
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("Input tensor is not sparse")
    
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    size = sparse_tensor.size()
    
    return {
        "indices": indices,
        "values": values,
        "size": size
    }

def dict_to_sparse_tensor(sparse_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Convert a dictionary back to a sparse tensor.
    
    Args:
        sparse_dict: Dictionary representation of a sparse tensor
        
    Returns:
        Sparse tensor
    """
    indices = sparse_dict["indices"]
    values = sparse_dict["values"]
    size = sparse_dict["size"]
    
    return torch.sparse.FloatTensor(indices, values, size)

def get_device_memory_info(device: Optional[torch.device] = None) -> Tuple[float, float]:
    """Get memory usage on the specified device.
    
    Args:
        device: Device to get memory usage for (defaults to current device)
        
    Returns:
        Tuple of (used memory in GB, total memory in GB)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    if device is None:
        device = torch.cuda.current_device()
    
    # Get total memory in bytes
    total_memory = torch.cuda.get_device_properties(device).total_memory
    
    # Get free memory in bytes
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    
    # Convert to GB
    total_memory_gb = total_memory / (1024**3)
    used_memory_gb = allocated_memory / (1024**3)
    
    return used_memory_gb, total_memory_gb

def calculate_optimal_batch_size(num_nodes: int, 
                                embed_dim: int, 
                                max_memory_gb: float,
                                safety_factor: float = 0.5,
                                dtype: torch.dtype = torch.float32) -> int:
    """Calculate the optimal batch size based on available memory.
    
    Args:
        num_nodes: Number of nodes in the graph
        embed_dim: Embedding dimension
        max_memory_gb: Maximum memory to use in GB
        safety_factor: Safety factor to avoid OOM errors
        dtype: Data type for model parameters
        
    Returns:
        Optimal batch size
    """
    available_memory_gb = max_memory_gb * safety_factor
    
    # Get bytes per element based on dtype
    if dtype == torch.float32:
        bytes_per_elem = 4
    elif dtype == torch.float16:
        bytes_per_elem = 2
    else:
        bytes_per_elem = 4  # Default
    
    # Calculate memory per edge
    # Each edge requires embeddings for source and destination nodes
    # plus additional memory for gradients and temporary activations
    bytes_per_edge = 2 * embed_dim * bytes_per_elem * 3  # 3x for gradients and activations
    
    # Convert available memory to bytes
    available_memory_bytes = available_memory_gb * (1024**3)
    
    # Calculate optimal batch size
    optimal_batch_size = int(available_memory_bytes / bytes_per_edge)
    
    # Ensure batch size is at least 1
    optimal_batch_size = max(1, optimal_batch_size)
    
    return optimal_batch_size 