import os
import torch
import torch.distributed as dist
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

logger = logging.getLogger(__name__)

def setup_distributed_environment(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None
) -> Tuple[int, int]:
    """Setup distributed training environment.
    
    Args:
        backend: Distributed backend to use (nccl for GPU, gloo for CPU)
        init_method: URL to use for process group initialization
        world_size: Number of processes to use (defaults to WORLD_SIZE env var)
        rank: Rank of this process (defaults to RANK env var)
        
    Returns:
        Tuple of (world_size, rank)
    """
    if not dist.is_available():
        logger.warning("Distributed package not available, defaulting to single process mode")
        return 1, 0
    
    # Get world size and rank from environment variables if not provided
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    
    # Choose backend
    if backend == "nccl" and not torch.cuda.is_available():
        logger.warning("NCCL backend requested but CUDA not available, falling back to gloo")
        backend = "gloo"
    
    # Set device if using NCCL
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # Generate init method if not provided
    if init_method is None:
        # Default to env:// if MASTER_ADDR and MASTER_PORT are set
        if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
            init_method = "env://"
        else:
            # Use file-based init on local file system
            init_method = f"file://{os.path.expanduser('~')}/.torch_distributed_init"
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    
    logger.info(f"Distributed environment initialized: rank={rank}, world_size={world_size}, backend={backend}")
    
    return world_size, rank

def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_rank() -> int:
    """Get rank of current process.
    
    Returns:
        Rank of current process (0 if not distributed)
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size() -> int:
    """Get number of processes.
    
    Returns:
        Number of processes (1 if not distributed)
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def is_main_process() -> bool:
    """Check if this is the main process (rank 0).
    
    Returns:
        True if this is the main process
    """
    return get_rank() == 0

def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """Perform all-reduce operation on a tensor.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation (sum, product, min, max)
        
    Returns:
        Reduced tensor
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    
    # Create a copy to avoid modifying the input
    result = tensor.clone()
    
    # Choose reduction operation
    if op == "sum":
        dist_op = dist.ReduceOp.SUM
    elif op == "product":
        dist_op = dist.ReduceOp.PRODUCT
    elif op == "min":
        dist_op = dist.ReduceOp.MIN
    elif op == "max":
        dist_op = dist.ReduceOp.MAX
    else:
        raise ValueError(f"Unsupported reduction operation: {op}")
    
    # Perform all-reduce
    dist.all_reduce(result, op=dist_op)
    
    return result

def all_gather(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of gathered tensors
    """
    if not dist.is_available() or not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    
    # Create list of tensors to hold results
    result = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Perform all-gather
    dist.all_gather(result, tensor)
    
    return result

def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source rank to all processes.
    
    Args:
        tensor: Tensor to broadcast
        src: Source rank
        
    Returns:
        Broadcasted tensor
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    
    dist.broadcast(tensor, src=src)
    
    return tensor

def sync_params(model: torch.nn.Module):
    """Synchronize model parameters across processes.
    
    Args:
        model: Model to synchronize
        
    Returns:
        None
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

class DistributedSampler:
    """Sampler for distributed training.
    
    This sampler ensures that each process gets a unique subset of data.
    It's similar to PyTorch's DistributedSampler but specialized for our
    edge sampling in graph data.
    """
    
    def __init__(self, 
                 num_edges: int, 
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0):
        """Initialize the sampler.
        
        Args:
            num_edges: Total number of edges
            num_replicas: Number of processes (defaults to world size)
            rank: Rank of this process (defaults to current rank)
            shuffle: Whether to shuffle indices
            seed: Random seed
        """
        if num_replicas is None:
            num_replicas = get_world_size()
        
        if rank is None:
            rank = get_rank()
        
        self.num_edges = num_edges
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        
        # Calculate number of edges per process
        self.num_samples = self.num_edges // self.num_replicas
        if self.num_edges % self.num_replicas != 0:
            self.num_samples += 1
        
        # Calculate total size
        self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self):
        """Get edge indices for this process.
        
        Returns:
            Iterator over edge indices
        """
        # Deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_edges, generator=g).tolist()
        else:
            indices = list(range(self.num_edges))
        
        # Add extra samples to make it evenly divisible
        indices = indices + indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        
        # Subsample based on rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)
    
    def __len__(self):
        """Get number of samples for this process.
        
        Returns:
            Number of samples
        """
        return self.num_samples
    
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling.
        
        Args:
            epoch: Current epoch
        """
        self.epoch = epoch

class DistributedTraining:
    """Helper class for distributed training.
    
    This class provides utilities for distributed training, including
    data sharding, gradient synchronization, and checkpoint management.
    """
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 device_ids: Optional[List[int]] = None,
                 output_device: Optional[int] = None,
                 find_unused_parameters: bool = False):
        """Initialize distributed training.
        
        Args:
            model: Model to distribute
            device_ids: List of device IDs to use
            output_device: Device to output results to
            find_unused_parameters: Whether to find unused parameters
        """
        self.model = model
        self.is_distributed = dist.is_available() and dist.is_initialized()
        
        # Use all available devices if not specified
        if device_ids is None and torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
        
        self.device_ids = device_ids
        self.output_device = output_device
        
        # Wrap model with DistributedDataParallel if distributed
        if self.is_distributed and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{get_rank() % torch.cuda.device_count()}")
            model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=find_unused_parameters
            )
            self.is_ddp = True
        # Use DataParallel if not distributed but multiple GPUs available
        elif torch.cuda.is_available() and len(self.device_ids) > 1:
            self.device = torch.device(f"cuda:{self.device_ids[0]}")
            model.to(self.device)
            self.model = torch.nn.DataParallel(
                model,
                device_ids=self.device_ids,
                output_device=self.output_device
            )
            self.is_ddp = False
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(self.device)
            self.is_ddp = False
    
    def create_sampler(self, num_edges: int, shuffle: bool = True) -> DistributedSampler:
        """Create a distributed sampler for edge indices.
        
        Args:
            num_edges: Total number of edges
            shuffle: Whether to shuffle indices
            
        Returns:
            Distributed sampler
        """
        if self.is_distributed:
            return DistributedSampler(
                num_edges=num_edges,
                shuffle=shuffle
            )
        return None
    
    def save_checkpoint(self, state: Dict[str, Any], filename: str, is_best: bool = False):
        """Save checkpoint in a distributed-aware manner.
        
        Args:
            state: State dictionary
            filename: Filename to save to
            is_best: Whether this is the best model so far
            
        Returns:
            None
        """
        # Only save from the main process
        if is_main_process():
            torch.save(state, filename)
            if is_best:
                import shutil
                shutil.copyfile(filename, filename.replace(".pt", "_best.pt"))
    
    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        """Load checkpoint in a distributed-aware manner.
        
        Args:
            filename: Filename to load from
            
        Returns:
            State dictionary
        """
        # Load on main process and broadcast
        map_location = {"cuda:0": f"cuda:{get_rank() % torch.cuda.device_count()}"} if self.is_distributed else self.device
        state_dict = torch.load(filename, map_location=map_location)
        
        if self.is_distributed:
            # If loaded from DDP model but not using DDP now
            if not self.is_ddp and list(state_dict["model_state_dict"].keys())[0].startswith("module."):
                state_dict["model_state_dict"] = {k[7:]: v for k, v in state_dict["model_state_dict"].items()}
            # If not loaded from DDP model but using DDP now
            elif self.is_ddp and not list(state_dict["model_state_dict"].keys())[0].startswith("module."):
                state_dict["model_state_dict"] = {f"module.{k}": v for k, v in state_dict["model_state_dict"].items()}
        
        return state_dict
    
    def prepare_batch(self, batch: Any) -> Any:
        """Move batch to the correct device.
        
        Args:
            batch: Batch data
            
        Returns:
            Batch on the correct device
        """
        def _to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self.device)
            elif isinstance(obj, list):
                return [_to_device(x) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(_to_device(x) for x in obj)
            elif isinstance(obj, dict):
                return {k: _to_device(v) for k, v in obj.items()}
            return obj
        
        return _to_device(batch)
    
    def all_reduce_scalar(self, value: float, op: str = "sum") -> float:
        """All-reduce a scalar value across processes.
        
        Args:
            value: Scalar value
            op: Reduction operation
            
        Returns:
            Reduced value
        """
        if not self.is_distributed:
            return value
        
        tensor = torch.tensor([value], device=self.device)
        dist.all_reduce(tensor, op=getattr(dist.ReduceOp, op.upper()))
        return tensor.item()
    
    def all_reduce_dict(self, data: Dict[str, float], op: str = "sum") -> Dict[str, float]:
        """All-reduce a dictionary of scalars across processes.
        
        Args:
            data: Dictionary of scalar values
            op: Reduction operation
            
        Returns:
            Reduced dictionary
        """
        if not self.is_distributed:
            return data
        
        # Convert dict to tensor
        names = sorted(data.keys())
        values = torch.tensor([data[name] for name in names], device=self.device)
        
        # All-reduce
        dist.all_reduce(values, op=getattr(dist.ReduceOp, op.upper()))
        
        # Convert back to dict
        return {name: values[i].item() for i, name in enumerate(names)}
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()

def launch_distributed_workers(
    training_script: str,
    world_size: int,
    num_gpus_per_machine: int,
    machine_rank: int = 0,
    dist_url: str = "auto",
    args: List[str] = None
):
    """Launch distributed worker processes.
    
    Args:
        training_script: Path to training script
        world_size: Total number of processes
        num_gpus_per_machine: Number of GPUs per machine
        machine_rank: Rank of this machine
        dist_url: URL for distributed initialization
        args: Additional arguments for the training script
        
    Returns:
        None
    """
    import subprocess
    import sys
    
    if args is None:
        args = []
    
    # Auto dist_url
    if dist_url == "auto":
        import random
        import socket
        ip = socket.gethostbyname(socket.gethostname())
        port = random.randint(10000, 20000)
        dist_url = f"tcp://{ip}:{port}"
    
    # Launch workers
    current_env = os.environ.copy()
    for local_rank in range(num_gpus_per_machine):
        global_rank = machine_rank * num_gpus_per_machine + local_rank
        
        # Skip if more workers than needed
        if global_rank >= world_size:
            break
        
        # Set environment variables
        current_env["RANK"] = str(global_rank)
        current_env["LOCAL_RANK"] = str(local_rank)
        current_env["WORLD_SIZE"] = str(world_size)
        current_env["MASTER_ADDR"] = dist_url.split("://")[1].split(":")[0]
        current_env["MASTER_PORT"] = dist_url.split(":")[-1]
        
        # Construct command
        cmd = [sys.executable, training_script] + args
        
        # Launch process
        process = subprocess.Popen(cmd, env=current_env)
        
        # Only print output for the first worker
        if local_rank == 0:
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        
        # Only create 1 worker if debug
        if os.environ.get("DEBUG", "0") == "1":
            break
    
    # Wait for all processes to finish
    for local_rank in range(num_gpus_per_machine):
        global_rank = machine_rank * num_gpus_per_machine + local_rank
        if global_rank >= world_size:
            break
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd) 