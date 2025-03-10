import os
import time
import json
import torch
import numpy as np
import psutil
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)

@dataclass
class ProfilingResult:
    """Container for profiling results"""
    name: str
    time_ms: float
    memory_mb: float
    compute_utilization: float = 0.0
    throughput: float = 0.0
    batch_size: int = 0
    additional_metrics: Dict[str, float] = field(default_factory=dict)

@contextmanager
def profile_section(name: str, profile_memory: bool = True, profile_time: bool = True):
    """Context manager for profiling a section of code.
    
    Args:
        name: Name of the section
        profile_memory: Whether to profile memory usage
        profile_time: Whether to profile execution time
        
    Yields:
        None
    """
    start_time = time.time()
    if profile_memory and torch.cuda.is_available():
        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
    
    yield
    
    if profile_time:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        logger.info(f"PROFILE {name}: {elapsed_ms:.2f} ms")
    
    if profile_memory and torch.cuda.is_available():
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        memory_diff = end_memory - start_memory
        logger.info(f"PROFILE {name} memory: {memory_diff:.2f} MB (allocated: {end_memory:.2f} MB)")


class TrainingProfiler:
    """Comprehensive profiler for training pipelines.
    
    This class provides tools for profiling model training, including memory
    usage, execution time, GPU utilization, and throughput. It can be used to
    identify bottlenecks and optimize training performance.
    """
    
    def __init__(self, 
                 trainer, 
                 graph_data, 
                 log_dir: str = "./profile_logs",
                 profile_memory: bool = True,
                 profile_compute: bool = True,
                 profile_throughput: bool = True):
        """Initialize the profiler.
        
        Args:
            trainer: The trainer to profile
            graph_data: The graph data to use for profiling
            log_dir: Directory to save profiling results
            profile_memory: Whether to profile memory usage
            profile_compute: Whether to profile compute utilization
            profile_throughput: Whether to profile throughput
        """
        self.trainer = trainer
        self.graph_data = graph_data
        self.log_dir = log_dir
        self.profile_memory = profile_memory and torch.cuda.is_available()
        self.profile_compute = profile_compute and torch.cuda.is_available()
        self.profile_throughput = profile_throughput
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize results storage
        self.memory_stats = {}
        self.compute_stats = {}
        self.time_stats = {}
        self.throughput_stats = {}
        self.layer_stats = {}
        
        # Detailed profiling results
        self.profiling_results = []
    
    def _log_gpu_memory(self) -> Dict[str, float]:
        """Log current GPU memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        if not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0, "cached_mb": 0}
        
        memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB
        memory_cached = torch.cuda.memory_cached() / (1024 * 1024)        # MB
        
        return {
            "allocated_mb": memory_allocated,
            "reserved_mb": memory_reserved,
            "cached_mb": memory_cached
        }
    
    def _log_cpu_memory(self) -> Dict[str, float]:
        """Log current CPU memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),     # Resident Set Size
            "vms_mb": memory_info.vms / (1024 * 1024),     # Virtual Memory Size
            "percent": process.memory_percent()
        }
    
    def profile_model_memory(self, batch_size: int = 32) -> Dict[str, float]:
        """Profile memory usage of the model.
        
        Args:
            batch_size: Batch size to use for profiling
            
        Returns:
            Dictionary with memory statistics
        """
        if not self.profile_memory:
            return {}
        
        # Clear CUDA cache to get accurate measurements
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get initial memory usage
        baseline_memory = self._log_gpu_memory()
        
        # Generate a batch
        pos_edges, neg_edges = self._generate_batch(batch_size)
        
        # Memory after batch generation
        batch_memory = self._log_gpu_memory()
        
        # Forward pass
        with torch.no_grad():
            with profile_section("Forward pass (memory profiling)", profile_time=False):
                self.trainer.encoder.eval()
                self.trainer.predictor.eval()
                
                # Move graph to device
                graph_data = self.trainer._move_to_device(self.graph_data)
                
                # Encode
                node_embeddings = self.trainer.encoder(graph_data)
                
                # Predict
                src_idx = pos_edges[0]
                dst_idx = pos_edges[1]
                src_embeddings = node_embeddings[src_idx]
                dst_embeddings = node_embeddings[dst_idx]
                pos_scores = self.trainer.predictor(src_embeddings, dst_embeddings)
        
        # Memory after forward pass
        forward_memory = self._log_gpu_memory()
        
        # Backward pass (with grad)
        self.trainer.encoder.train()
        self.trainer.predictor.train()
        
        with profile_section("Backward pass (memory profiling)", profile_time=False):
            # Move graph to device
            graph_data = self.trainer._move_to_device(self.graph_data)
            
            # Encode
            node_embeddings = self.trainer.encoder(graph_data)
            
            # Predict
            src_idx = pos_edges[0]
            dst_idx = pos_edges[1]
            src_embeddings = node_embeddings[src_idx]
            dst_embeddings = node_embeddings[dst_idx]
            pos_scores = self.trainer.predictor(src_embeddings, dst_embeddings)
            
            # Compute loss (simple binary cross entropy)
            loss = torch.mean((1 - pos_scores) ** 2)
            
            # Backward pass
            loss.backward()
        
        # Memory after backward pass
        backward_memory = self._log_gpu_memory()
        
        # Calculate deltas
        batch_delta = {f"batch_{k}": v - baseline_memory[k] for k, v in batch_memory.items()}
        forward_delta = {f"forward_{k}": v - batch_memory[k] for k, v in forward_memory.items()}
        backward_delta = {f"backward_{k}": v - forward_memory[k] for k, v in backward_memory.items()}
        total_delta = {f"total_{k}": v - baseline_memory[k] for k, v in backward_memory.items()}
        
        # Combine results
        results = {
            **baseline_memory,
            **batch_delta,
            **forward_delta,
            **backward_delta,
            **total_delta,
            "batch_size": batch_size
        }
        
        self.memory_stats[f"batch_{batch_size}"] = results
        return results
    
    def profile_forward_pass(self, batch_size: int = 32) -> Dict[str, float]:
        """Profile forward pass performance.
        
        Args:
            batch_size: Batch size to use for profiling
            
        Returns:
            Dictionary with performance statistics
        """
        # Generate a batch
        pos_edges, neg_edges = self._generate_batch(batch_size)
        
        # Move graph to device
        graph_data = self.trainer._move_to_device(self.graph_data)
        
        # Warmup
        self.trainer.encoder.eval()
        self.trainer.predictor.eval()
        with torch.no_grad():
            for _ in range(3):  # Warmup iterations
                node_embeddings = self.trainer.encoder(graph_data)
                src_idx = pos_edges[0]
                dst_idx = pos_edges[1]
                src_embeddings = node_embeddings[src_idx]
                dst_embeddings = node_embeddings[dst_idx]
                pos_scores = self.trainer.predictor(src_embeddings, dst_embeddings)
        
        # Measure time
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Actual profiling
        with torch.no_grad():
            for _ in range(10):  # Multiple iterations for stable measurement
                node_embeddings = self.trainer.encoder(graph_data)
                src_idx = pos_edges[0]
                dst_idx = pos_edges[1]
                src_embeddings = node_embeddings[src_idx]
                dst_embeddings = node_embeddings[dst_idx]
                pos_scores = self.trainer.predictor(src_embeddings, dst_embeddings)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate statistics
        elapsed_ms = (end_time - start_time) * 1000 / 10  # Average time per iteration
        throughput = batch_size / (elapsed_ms / 1000)  # Samples per second
        
        result = ProfilingResult(
            name=f"forward_pass_batch_{batch_size}",
            time_ms=elapsed_ms,
            memory_mb=0,  # Not measured here
            throughput=throughput,
            batch_size=batch_size
        )
        
        self.profiling_results.append(result)
        logger.info(f"Forward pass (batch={batch_size}): {elapsed_ms:.2f} ms, {throughput:.2f} samples/sec")
        
        return {
            "time_ms": elapsed_ms,
            "throughput": throughput,
            "batch_size": batch_size
        }
    
    def profile_backward_pass(self, batch_size: int = 32) -> Dict[str, float]:
        """Profile backward pass performance.
        
        Args:
            batch_size: Batch size to use for profiling
            
        Returns:
            Dictionary with performance statistics
        """
        # Generate a batch
        pos_edges, neg_edges = self._generate_batch(batch_size)
        
        # Move graph to device
        graph_data = self.trainer._move_to_device(self.graph_data)
        
        # Switch to train mode
        self.trainer.encoder.train()
        self.trainer.predictor.train()
        
        # Warmup
        for _ in range(3):  # Warmup iterations
            # Zero gradients
            self.trainer.optimizer.zero_grad()
            
            # Forward pass
            node_embeddings = self.trainer.encoder(graph_data)
            src_idx = pos_edges[0]
            dst_idx = pos_edges[1]
            src_embeddings = node_embeddings[src_idx]
            dst_embeddings = node_embeddings[dst_idx]
            pos_scores = self.trainer.predictor(src_embeddings, dst_embeddings)
            
            # Compute loss
            loss = torch.mean((1 - pos_scores) ** 2)
            
            # Backward pass
            loss.backward()
        
        # Measure time
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Actual profiling
        for _ in range(10):  # Multiple iterations for stable measurement
            # Zero gradients
            self.trainer.optimizer.zero_grad()
            
            # Forward pass
            node_embeddings = self.trainer.encoder(graph_data)
            src_idx = pos_edges[0]
            dst_idx = pos_edges[1]
            src_embeddings = node_embeddings[src_idx]
            dst_embeddings = node_embeddings[dst_idx]
            pos_scores = self.trainer.predictor(src_embeddings, dst_embeddings)
            
            # Compute loss
            loss = torch.mean((1 - pos_scores) ** 2)
            
            # Backward pass
            loss.backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate statistics
        elapsed_ms = (end_time - start_time) * 1000 / 10  # Average time per iteration
        throughput = batch_size / (elapsed_ms / 1000)  # Samples per second
        
        result = ProfilingResult(
            name=f"backward_pass_batch_{batch_size}",
            time_ms=elapsed_ms,
            memory_mb=0,  # Not measured here
            throughput=throughput,
            batch_size=batch_size
        )
        
        self.profiling_results.append(result)
        logger.info(f"Backward pass (batch={batch_size}): {elapsed_ms:.2f} ms, {throughput:.2f} samples/sec")
        
        return {
            "time_ms": elapsed_ms,
            "throughput": throughput,
            "batch_size": batch_size
        }
    
    def profile_full_training(self, batch_sizes: List[int] = None) -> Dict[str, Dict[str, float]]:
        """Profile full training process with different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to profile
            
        Returns:
            Dictionary with profiling results for each batch size
        """
        if batch_sizes is None:
            batch_sizes = [16, 32, 64, 128, 256]
        
        results = {}
        for batch_size in batch_sizes:
            logger.info(f"Profiling batch size: {batch_size}")
            
            # Profile memory
            if self.profile_memory:
                memory_stats = self.profile_model_memory(batch_size)
            else:
                memory_stats = {}
            
            # Profile forward pass
            forward_stats = self.profile_forward_pass(batch_size)
            
            # Profile backward pass
            backward_stats = self.profile_backward_pass(batch_size)
            
            # Combine results
            combined_stats = {
                "batch_size": batch_size,
                **memory_stats,
                "forward_time_ms": forward_stats["time_ms"],
                "forward_throughput": forward_stats["throughput"],
                "backward_time_ms": backward_stats["time_ms"],
                "backward_throughput": backward_stats["throughput"],
                "total_time_ms": forward_stats["time_ms"] + backward_stats["time_ms"],
                "total_throughput": batch_size / ((forward_stats["time_ms"] + backward_stats["time_ms"]) / 1000)
            }
            
            results[f"batch_{batch_size}"] = combined_stats
        
        # Save results
        self._save_profiling_results(results, "batch_size_scaling.json")
        
        return results
    
    def profile_component_breakdown(self) -> Dict[str, float]:
        """Profile time breakdown of different components in the training pipeline.
        
        Returns:
            Dictionary with time breakdown by component
        """
        # Generate a batch
        batch_size = 64  # Use a moderate batch size
        pos_edges, neg_edges = self._generate_batch(batch_size)
        
        # Move graph to device
        graph_data = self.trainer._move_to_device(self.graph_data)
        
        # Component timings
        timings = {}
        
        # Data preparation timing
        start_time = time.time()
        for _ in range(10):
            self._generate_batch(batch_size)
        end_time = time.time()
        timings["data_preparation"] = (end_time - start_time) * 100  # ms
        
        # Encoding timing
        self.trainer.encoder.eval()
        with torch.no_grad():
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for _ in range(10):
                node_embeddings = self.trainer.encoder(graph_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
        timings["encoding"] = (end_time - start_time) * 100  # ms
        
        # Prediction timing
        with torch.no_grad():
            # Get embeddings first
            node_embeddings = self.trainer.encoder(graph_data)
            src_idx = pos_edges[0]
            dst_idx = pos_edges[1]
            src_embeddings = node_embeddings[src_idx]
            dst_embeddings = node_embeddings[dst_idx]
            
            # Time just the prediction
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for _ in range(10):
                pos_scores = self.trainer.predictor(src_embeddings, dst_embeddings)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
        timings["prediction"] = (end_time - start_time) * 100  # ms
        
        # Loss computation timing
        with torch.no_grad():
            # Get scores first
            node_embeddings = self.trainer.encoder(graph_data)
            src_idx = pos_edges[0]
            dst_idx = pos_edges[1]
            src_embeddings = node_embeddings[src_idx]
            dst_embeddings = node_embeddings[dst_idx]
            pos_scores = self.trainer.predictor(src_embeddings, dst_embeddings)
            
            # Time just the loss computation
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for _ in range(10):
                loss = torch.mean((1 - pos_scores) ** 2)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
        timings["loss_computation"] = (end_time - start_time) * 100  # ms
        
        # Backward pass timing
        self.trainer.encoder.train()
        self.trainer.predictor.train()
        
        # Setup
        node_embeddings = self.trainer.encoder(graph_data)
        src_idx = pos_edges[0]
        dst_idx = pos_edges[1]
        src_embeddings = node_embeddings[src_idx]
        dst_embeddings = node_embeddings[dst_idx]
        pos_scores = self.trainer.predictor(src_embeddings, dst_embeddings)
        loss = torch.mean((1 - pos_scores) ** 2)
        
        # Time just the backward pass
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for _ in range(10):
            self.trainer.optimizer.zero_grad()
            loss.backward(retain_graph=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        timings["backward_pass"] = (end_time - start_time) * 100  # ms
        
        # Optimizer step timing
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for _ in range(10):
            self.trainer.optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        timings["optimizer_step"] = (end_time - start_time) * 100  # ms
        
        # Calculate percentages
        total_time = sum(timings.values())
        percentages = {k: (v / total_time) * 100 for k, v in timings.items()}
        
        # Combine results
        results = {
            "times_ms": timings,
            "percentages": percentages,
            "total_time_ms": total_time,
            "batch_size": batch_size
        }
        
        # Save results
        self._save_profiling_results(results, "component_breakdown.json")
        
        return results
    
    def profile_layer_activations(self) -> Dict[str, Dict[str, float]]:
        """Profile layer activations in the model.
        
        Returns:
            Dictionary with activation statistics by layer
        """
        # Generate a batch
        batch_size = 64  # Use a moderate batch size
        pos_edges, neg_edges = self._generate_batch(batch_size)
        
        # Move graph to device
        graph_data = self.trainer._move_to_device(self.graph_data)
        
        # Register forward hooks
        activation_stats = {}
        hooks = []
        
        def hook_fn(name):
            def _hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Calculate activation statistics
                    activation_stats[name] = {
                        "mean": output.mean().item(),
                        "std": output.std().item(),
                        "min": output.min().item(),
                        "max": output.max().item(),
                        "shape": list(output.shape),
                        "size_mb": output.element_size() * output.nelement() / (1024 * 1024)
                    }
            return _hook
        
        # Register hooks for encoder
        for name, module in self.trainer.encoder.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                hooks.append(module.register_forward_hook(hook_fn(f"encoder.{name}")))
        
        # Register hooks for predictor
        for name, module in self.trainer.predictor.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                hooks.append(module.register_forward_hook(hook_fn(f"predictor.{name}")))
        
        # Forward pass to collect activation statistics
        with torch.no_grad():
            self.trainer.encoder.eval()
            self.trainer.predictor.eval()
            
            # Encode
            node_embeddings = self.trainer.encoder(graph_data)
            
            # Predict
            src_idx = pos_edges[0]
            dst_idx = pos_edges[1]
            src_embeddings = node_embeddings[src_idx]
            dst_embeddings = node_embeddings[dst_idx]
            pos_scores = self.trainer.predictor(src_embeddings, dst_embeddings)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Save results
        self._save_profiling_results(activation_stats, "layer_activations.json")
        
        return activation_stats
    
    def visualize_bottlenecks(self, output_path: str = None):
        """Create visualizations of the bottlenecks.
        
        Args:
            output_path: Path to save the visualization
        """
        if not output_path:
            output_path = os.path.join(self.log_dir, "bottleneck_analysis.png")
        
        # Check if we have component breakdown
        if not hasattr(self, "_component_breakdown") or self._component_breakdown is None:
            self._component_breakdown = self.profile_component_breakdown()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot component breakdown
        labels = list(self._component_breakdown["percentages"].keys())
        sizes = list(self._component_breakdown["percentages"].values())
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')
        ax1.set_title('Time Breakdown by Component')
        
        # Plot batch size scaling
        if hasattr(self, "profiling_results") and self.profiling_results:
            batch_sizes = []
            forward_times = []
            backward_times = []
            
            for result in self.profiling_results:
                if result.name.startswith("forward_pass"):
                    batch_sizes.append(result.batch_size)
                    forward_times.append(result.time_ms)
                elif result.name.startswith("backward_pass"):
                    backward_times.append(result.time_ms)
            
            ax2.plot(batch_sizes, forward_times, 'o-', label='Forward Pass')
            ax2.plot(batch_sizes, backward_times, 'o-', label='Backward Pass')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Time (ms)')
            ax2.set_title('Scaling with Batch Size')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Bottleneck visualization saved to {output_path}")
    
    def _generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch for profiling.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of positive and negative edge tensors
        """
        if hasattr(self.trainer, '_generate_batches'):
            batches = self.trainer._generate_batches(self.graph_data, batch_size)
            if batches:
                pos_edges, neg_edges = batches[0]
                return pos_edges, neg_edges
        
        # Fallback: manual batch generation
        edge_index = self.graph_data.edge_index
        num_edges = edge_index.size(1)
        
        # Sample positive edges
        perm = torch.randperm(num_edges)[:batch_size]
        pos_edges = edge_index[:, perm]
        
        # Generate negative edges
        num_nodes = self.graph_data.x.size(0)
        neg_edges = torch.randint(0, num_nodes, (2, batch_size), dtype=torch.long)
        
        return pos_edges, neg_edges
    
    def _save_profiling_results(self, results: Dict, filename: str):
        """Save profiling results to a JSON file.
        
        Args:
            results: Results to save
            filename: Name of the file
        """
        # Convert any non-serializable objects
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            return obj
        
        # Convert all values
        serialized_results = json.loads(
            json.dumps(results, default=convert_for_json)
        )
        
        # Save to file
        file_path = os.path.join(self.log_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(serialized_results, f, indent=2)
        
        logger.info(f"Profiling results saved to {file_path}") 