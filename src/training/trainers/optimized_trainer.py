import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from tqdm import tqdm
import gc
import psutil

from src.training.trainers.citation_prediction_trainer import CitationPredictionTrainer, TrainingConfig
from src.utils.memory import (
    memory_efficient_context, enable_gradient_checkpointing, 
    GradientAccumulator, MixedPrecisionTraining, get_memory_stats,
    calculate_optimal_batch_size
)
from src.utils.profiling import TrainingProfiler, profile_section
from src.utils.distributed import (
    setup_distributed_environment, get_rank, get_world_size, 
    is_main_process, DistributedTraining
)

logger = logging.getLogger(__name__)

@dataclass
class OptimizedTrainingConfig(TrainingConfig):
    """Enhanced configuration for the OptimizedTrainer.
    
    This extends the base TrainingConfig with additional parameters for
    large-scale training optimizations.
    """
    # Memory optimization parameters
    use_mixed_precision: bool = True  # Use mixed precision training
    mixed_precision_dtype: str = "float16"  # Data type for mixed precision
    use_gradient_checkpointing: bool = False  # Use gradient checkpointing
    memory_optimization_level: int = 1  # Level of memory optimization (0-3)
    
    # Computational optimization parameters
    use_jit: bool = False  # Use JIT compilation
    optimize_memory_for_16bit: bool = False  # Special optimizations for 16-bit training
    
    # Data management
    use_streaming_data: bool = False  # Use streaming data loading
    prefetch_factor: int = 2  # Prefetch factor for data loading
    num_workers: int = 4  # Number of workers for data loading
    pin_memory: bool = True  # Pin memory for faster data transfer
    
    # Distributed training
    distributed_training: bool = False  # Use distributed training
    distributed_backend: str = "nccl"  # Backend for distributed training
    find_unused_parameters: bool = False  # Find unused parameters in DDP
    
    # Batching and scaling
    auto_scale_batch_size: bool = False  # Automatically scale batch size
    max_memory_usage_gb: float = 0.8  # Maximum memory usage as fraction of available
    activation_checkpointing: bool = False  # Checkpoint activations
    
    # Profiling
    profile_training: bool = False  # Profile training performance
    profile_memory: bool = True  # Profile memory usage
    profile_compute: bool = True  # Profile compute utilization
    profile_log_dir: str = "./profile_logs"  # Directory for profiling logs
    
    # Sparse operations
    use_sparse_operations: bool = False  # Use sparse operations
    sparse_attention: bool = False  # Use sparse attention
    
    # Checkpointing and recovery
    checkpoint_frequency_minutes: int = 30  # Checkpoint every N minutes
    recover_from_oom: bool = True  # Try to recover from OOM errors
    
    # Advanced optimization
    kernel_fusion: bool = False  # Fuse kernels for faster execution
    optimize_for: str = "memory"  # Optimize for "memory" or "speed"


class OptimizedTrainer(CitationPredictionTrainer):
    """Optimized trainer for large-scale citation prediction models.
    
    This trainer extends CitationPredictionTrainer with large-scale training
    capabilities, including memory optimizations, distributed training, and
    performance profiling.
    """
    
    def __init__(self, config: OptimizedTrainingConfig):
        """Initialize the optimized trainer with the given configuration.
        
        Args:
            config: Configuration for the trainer
        """
        super().__init__(config)
        self.config = config
        
        # Initialize memory optimization components
        self._setup_memory_optimizations()
        
        # Initialize distributed training
        self._setup_distributed_training()
        
        # Setup profiling
        self._setup_profiling()
        
        # Handle batching and scaling
        self._setup_batching()
        
        # Initialize sparse operations
        self._setup_sparse_operations()
        
        # Initialize checkpointing and recovery
        self._setup_checkpointing()
        
        logger.info("OptimizedTrainer initialized with memory optimizations enabled")
    
    def _setup_memory_optimizations(self):
        """Setup memory optimization features.
        
        This includes mixed precision training, gradient checkpointing,
        and other memory-saving techniques.
        """
        # Initialize mixed precision training
        if self.config.use_mixed_precision and torch.cuda.is_available():
            dtype = torch.float16
            if self.config.mixed_precision_dtype == "bfloat16" and hasattr(torch, "bfloat16"):
                dtype = torch.bfloat16
            
            self.mixed_precision = MixedPrecisionTraining(
                enabled=True, 
                dtype=dtype
            )
            logger.info(f"Mixed precision training enabled with {self.config.mixed_precision_dtype}")
        else:
            self.mixed_precision = MixedPrecisionTraining(enabled=False)
            
        # Gradient accumulation wrapper
        if self.config.gradient_accumulation_steps > 1:
            self.grad_accumulator = GradientAccumulator(
                optimizer=None,  # Will set after optimizer initialization
                steps=self.config.gradient_accumulation_steps,
                scaler=self.mixed_precision.scaler if self.config.use_mixed_precision else None
            )
            logger.info(f"Gradient accumulation enabled with {self.config.gradient_accumulation_steps} steps")
        else:
            self.grad_accumulator = None
            
        # Apply different memory optimization levels
        if self.config.memory_optimization_level > 0:
            # Level 1+: Basic optimizations
            torch.backends.cudnn.benchmark = True
            
            # Level 2+: More aggressive optimizations
            if self.config.memory_optimization_level >= 2:
                # Reset peak memory stats
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                # Free unused memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Level 3: Most aggressive optimizations
            if self.config.memory_optimization_level >= 3:
                # Force garbage collection more frequently
                gc.set_threshold(100, 5, 5)
            
            logger.info(f"Memory optimization level {self.config.memory_optimization_level} enabled")
    
    def _setup_distributed_training(self):
        """Setup distributed training if enabled."""
        self.is_distributed = False
        self.distributed_handler = None
        
        if self.config.distributed_training:
            try:
                # Initialize distributed environment
                world_size, rank = setup_distributed_environment(
                    backend=self.config.distributed_backend
                )
                
                self.is_distributed = world_size > 1
                self.world_size = world_size
                self.rank = rank
                
                logger.info(f"Distributed training initialized with world_size={world_size}, rank={rank}")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed training: {e}")
                self.is_distributed = False
    
    def _setup_profiling(self):
        """Setup performance profiling if enabled."""
        self.profiler = None
        if self.config.profile_training:
            self.profile_dir = os.path.join(
                self.config.profile_log_dir,
                f"profile_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(self.profile_dir, exist_ok=True)
            
            logger.info(f"Training profiling enabled, results will be saved to {self.profile_dir}")
    
    def _setup_batching(self):
        """Setup batching and scaling strategies."""
        self.optimal_batch_size = self.config.batch_size
        
        # Automatically determine optimal batch size if requested
        if self.config.auto_scale_batch_size and torch.cuda.is_available():
            try:
                # Get GPU memory information
                device = torch.cuda.current_device()
                gpu_props = torch.cuda.get_device_properties(device)
                total_memory_gb = gpu_props.total_memory / (1024**3)
                
                # Calculate optimal batch size based on available memory
                estimated_optimal_batch_size = calculate_optimal_batch_size(
                    num_nodes=10000,  # Will be updated with actual graph size during training
                    embed_dim=self.config.embed_dim,
                    max_memory_gb=total_memory_gb * self.config.max_memory_usage_gb,
                    dtype=torch.float16 if self.config.use_mixed_precision else torch.float32
                )
                
                self.optimal_batch_size = min(max(estimated_optimal_batch_size, 1), 1024)
                logger.info(f"Auto-scaled batch size: {self.optimal_batch_size} (from original {self.config.batch_size})")
                
                # Update configuration
                self.config.batch_size = self.optimal_batch_size
            except Exception as e:
                logger.warning(f"Failed to auto-scale batch size: {e}")
    
    def _setup_sparse_operations(self):
        """Setup sparse operation optimizations if enabled."""
        self.use_sparse_ops = self.config.use_sparse_operations
        
        if self.use_sparse_ops:
            logger.info("Sparse operations enabled for large adjacency matrices")
    
    def _setup_checkpointing(self):
        """Setup checkpointing and recovery mechanisms."""
        self.last_checkpoint_time = time.time()
        self.checkpoint_frequency_seconds = self.config.checkpoint_frequency_minutes * 60
    
    def initialize_models(self, graph_data):
        """Initialize encoder and predictor models with optimizations.
        
        Args:
            graph_data: Graph data to use for initialization
            
        Returns:
            None
        """
        # First use the parent class to initialize models
        super().initialize_models(graph_data)
        
        # Apply additional optimizations after models are created
        
        # Enable gradient checkpointing if requested (significant memory savings at slight compute cost)
        if self.config.use_gradient_checkpointing:
            enable_gradient_checkpointing(self.encoder, enabled=True)
            logger.info("Gradient checkpointing enabled for encoder")
        
        # Apply JIT compilation if requested
        if self.config.use_jit and torch.cuda.is_available():
            try:
                # Try to JIT compile the models
                device = torch.device(self.config.device)
                dummy_x = torch.randn(graph_data.num_nodes, graph_data.x.size(1), device=device)
                dummy_edge_index = torch.randint(0, graph_data.num_nodes, (2, min(100, graph_data.num_edges)), device=device)
                
                # Create scripted model for encoder
                if hasattr(self.encoder, "is_scriptable") and self.encoder.is_scriptable:
                    self.encoder = torch.jit.script(self.encoder)
                    logger.info("JIT compilation applied to encoder")
                    
                # Create scripted model for predictor
                if hasattr(self.predictor, "is_scriptable") and self.predictor.is_scriptable:
                    self.predictor = torch.jit.script(self.predictor)
                    logger.info("JIT compilation applied to predictor")
            except Exception as e:
                logger.warning(f"Failed to apply JIT compilation: {e}")
        
        # Setup activation checkpointing if enabled
        if self.config.activation_checkpointing:
            def enable_activation_checkpointing(module):
                if isinstance(module, nn.Sequential):
                    module._activation_checkpointing = True
                    
            self.encoder.apply(enable_activation_checkpointing)
            self.predictor.apply(enable_activation_checkpointing)
            logger.info("Activation checkpointing enabled")
        
        # Move models to device
        self.encoder = self.encoder.to(self.device)
        self.predictor = self.predictor.to(self.device)
        
        # Setup distributed training wrappers if enabled
        if self.is_distributed:
            # Create distributed handler
            self.distributed_handler = DistributedTraining(
                model=nn.ModuleList([self.encoder, self.predictor]),
                find_unused_parameters=self.config.find_unused_parameters
            )
            
            # Update models with distributed versions
            self.encoder = self.distributed_handler.model[0]
            self.predictor = self.distributed_handler.model[1]
            
            logger.info("Models wrapped with distributed training support")
    
    def _initialize_optimizer(self):
        """Initialize optimizer with optimizations.
        
        Returns:
            None
        """
        # First use the parent class to initialize the optimizer
        super()._initialize_optimizer()
        
        # Apply additional optimizations after optimizer is created
        
        # Update gradient accumulator with optimizer
        if self.grad_accumulator is not None:
            self.grad_accumulator.optimizer = self.optimizer
        
        # Apply memory optimizations for optimizer states
        if torch.cuda.is_available() and self.config.memory_optimization_level >= 2:
            # Try to use parameter fused optimizers if available
            if self.config.optimizer_type.lower() == "adam":
                try:
                    from torch.optim.adamw import AdamW
                    # Replace optimizer with fused version if available
                    if hasattr(torch.optim, 'FusedAdamW'):
                        params = [p for p in self.optimizer.param_groups[0]['params'] if p.requires_grad]
                        self.optimizer = torch.optim.FusedAdamW(
                            params,
                            lr=self.config.learning_rate,
                            weight_decay=self.config.weight_decay
                        )
                        logger.info("Using FusedAdamW optimizer for better performance")
                except ImportError:
                    pass
        
        # Apply memory optimization for 16-bit training
        if self.config.optimize_memory_for_16bit and self.config.use_mixed_precision:
            for param_group in self.optimizer.param_groups:
                param_group['foreach'] = True
                param_group['fused'] = True
            logger.info("Optimizer states optimized for 16-bit training")
    
    def _apply_sparse_optimizations(self, graph_data):
        """Apply sparse operation optimizations to graph data.
        
        Args:
            graph_data: Graph data to optimize
            
        Returns:
            Optimized graph data
        """
        if not self.use_sparse_ops:
            return graph_data
        
        # Convert dense adjacency to sparse if needed
        if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None:
            num_nodes = graph_data.x.size(0)
            num_edges = graph_data.edge_index.size(1)
            
            # Only convert to sparse if we have a large, sparse graph
            sparsity = 1.0 - (num_edges / (num_nodes * num_nodes))
            if sparsity > 0.9 and num_nodes > 1000:
                # Keep original edge_index but create sparse adjacency
                sparse_adj = torch.sparse.FloatTensor(
                    graph_data.edge_index, 
                    torch.ones(num_edges, device=graph_data.edge_index.device),
                    torch.Size([num_nodes, num_nodes])
                )
                graph_data.sparse_adj = sparse_adj
                logger.info(f"Converted adjacency matrix to sparse format (sparsity: {sparsity:.2f})")
        
        return graph_data
    
    def _initialize_dataloader(self, graph_data, batch_size=None):
        """Initialize optimized data loading.
        
        Args:
            graph_data: Graph data to load
            batch_size: Batch size to use
            
        Returns:
            DataLoader
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Apply sparse optimizations to graph data
        graph_data = self._apply_sparse_optimizations(graph_data)
        
        # For the optimized trainer, we'll implement a more sophisticated
        # dataloader that can handle streaming large graphs if needed
        if self.config.use_streaming_data:
            # Implement streaming data loading
            # This is just a placeholder - actual implementation would depend on data format
            logger.info("Using streaming data loading for large graphs")
            # For now, fall back to standard loading
        
        # If using distributed training, create a distributed sampler
        if self.is_distributed:
            # Create distributed sampler for edges
            sampler = self.distributed_handler.create_sampler(
                num_edges=graph_data.edge_index.size(1),
                shuffle=True
            )
            
            # Create batches with distributed sampler
            # For now, we'll still use the parent's implementation
            # but with a note that this should be enhanced for distributed
        
        # For standard loading, use parent class implementation
        return graph_data
    
    def train(self, train_graph, val_graph=None, test_graph=None):
        """Train the model with large-scale optimizations.
        
        Args:
            train_graph: Training graph data
            val_graph: Validation graph data
            test_graph: Test graph data
            
        Returns:
            Dictionary with training metrics
        """
        # Start profiling if enabled
        if self.config.profile_training:
            # Create profiler after graph data is available
            self.profiler = TrainingProfiler(
                trainer=self,
                graph_data=train_graph,
                log_dir=self.profile_dir,
                profile_memory=self.config.profile_memory,
                profile_compute=self.config.profile_compute
            )
            
            # Profile model before training to identify any issues
            if self.profiler is not None:
                logger.info("Profiling model memory usage before training...")
                memory_stats = self.profiler.profile_model_memory(batch_size=self.config.batch_size)
                logger.info(f"Initial memory profile: {memory_stats.get('total_mb', 0):.2f} MB")
        
        try:
            # Initialize models, optimizer, etc.
            self.initialize_models(train_graph)
            self._initialize_optimizer()
            
            # Update grad accumulator now that optimizer is initialized
            if self.grad_accumulator is not None:
                self.grad_accumulator.optimizer = self.optimizer
            
            # Measure graph size for auto-scaling if needed
            if self.config.auto_scale_batch_size:
                actual_num_nodes = train_graph.x.size(0)
                actual_embed_dim = self.config.embed_dim
                
                # Recalculate optimal batch size with actual graph size
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    gpu_props = torch.cuda.get_device_properties(device)
                    total_memory_gb = gpu_props.total_memory / (1024**3)
                    
                    updated_batch_size = calculate_optimal_batch_size(
                        num_nodes=actual_num_nodes,
                        embed_dim=actual_embed_dim,
                        max_memory_gb=total_memory_gb * self.config.max_memory_usage_gb,
                        dtype=torch.float16 if self.config.use_mixed_precision else torch.float32
                    )
                    
                    # Update batch size if needed
                    if updated_batch_size != self.config.batch_size:
                        logger.info(f"Adjusted batch size from {self.config.batch_size} to {updated_batch_size} based on actual graph size")
                        self.config.batch_size = updated_batch_size
            
            # Training loop
            best_val_metric = float('inf') if self.config.early_stopping_metric.lower() == 'loss' else 0.0
            patience_counter = 0
            
            # Create progress tracking
            progress_bar = tqdm(range(self.config.num_epochs), desc="Training")
            
            # Main training loop
            for epoch in range(self.config.num_epochs):
                # Set epoch for distributed samplers
                if self.is_distributed and hasattr(self, 'train_sampler') and self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                
                # Train for one epoch
                train_metrics = self.train_epoch(train_graph)
                
                # Evaluate on validation set
                val_metrics = {}
                if val_graph is not None and (epoch + 1) % self.config.evaluation_interval == 0:
                    val_metrics = self.validate(val_graph)
                
                # Combine metrics
                metrics = {**train_metrics, **val_metrics}
                
                # Update progress bar with metrics
                progress_bar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))})
                progress_bar.update(1)
                
                # Log metrics
                if (epoch + 1) % self.config.log_interval == 0 or epoch == 0:
                    metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
                    logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - {metric_str}")
                
                # Save checkpoint
                current_time = time.time()
                time_since_last_checkpoint = current_time - self.last_checkpoint_time
                
                if (epoch + 1) % self.config.checkpoint_interval == 0 or time_since_last_checkpoint >= self.checkpoint_frequency_seconds:
                    checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                    self.save_checkpoint(checkpoint_path, epoch, metrics)
                    self.last_checkpoint_time = current_time
                
                # Early stopping check
                current_metric = metrics.get(self.config.early_stopping_metric, None)
                if current_metric is not None:
                    improved = (self.config.early_stopping_metric.lower() == 'loss' and current_metric < best_val_metric) or \
                              (self.config.early_stopping_metric.lower() != 'loss' and current_metric > best_val_metric)
                    
                    if improved:
                        best_val_metric = current_metric
                        patience_counter = 0
                        
                        # Save best model if requested
                        if self.config.save_best_only:
                            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
                            self.save_checkpoint(best_path, epoch, metrics)
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.early_stopping_patience:
                            logger.info(f"Early stopping triggered at epoch {epoch+1}")
                            break
            
            # Final evaluation on test set
            test_metrics = {}
            if test_graph is not None:
                logger.info("Evaluating on test set...")
                test_metrics = self.validate(test_graph, prefix="test")
                metrics.update(test_metrics)
                
                # Log test metrics
                test_metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items() if isinstance(v, (int, float))])
                logger.info(f"Test metrics: {test_metric_str}")
            
            # Generate final profiling report if enabled
            if self.profiler is not None:
                logger.info("Generating final performance profile...")
                self.profiler.visualize_bottlenecks()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            if self.config.recover_from_oom and "CUDA out of memory" in str(e):
                # Attempt recovery from OOM
                self._recover_from_oom()
                # Restart training with reduced batch size
                self.config.batch_size = max(1, self.config.batch_size // 2)
                logger.info(f"Attempting to continue with reduced batch size: {self.config.batch_size}")
                return self.train(train_graph, val_graph, test_graph)
            raise
    
    def train_epoch(self, graph_data):
        """Train for one epoch with memory optimizations.
        
        Args:
            graph_data: Graph data to train on
            
        Returns:
            Dictionary with training metrics
        """
        # Set models to train mode
        self.encoder.train()
        self.predictor.train()
        
        # Generate batches
        batches = self._generate_batches(graph_data, self.config.batch_size)
        
        total_loss = 0.0
        total_samples = 0
        batch_times = []
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(batches, desc="Training batches")
        
        # Training loop
        for batch_idx, (pos_edges, neg_edges) in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # Move batch to device
            pos_edges = self._move_to_device(pos_edges)
            neg_edges = self._move_to_device(neg_edges)
            graph_data_device = self._move_to_device(graph_data)
            
            # Clear gradients
            if self.grad_accumulator is None:
                self.optimizer.zero_grad()
            
            # Forward and backward pass with memory optimizations
            batch_loss = 0.0
            
            # Use memory efficient context for forward/backward
            with memory_efficient_context(
                enabled=self.config.memory_optimization_level > 0,
                enable_autocast=self.config.use_mixed_precision,
                enable_checkpoint=self.config.use_gradient_checkpointing
            ):
                # If using mixed precision, use autocast
                if self.config.use_mixed_precision:
                    with self.mixed_precision.autocast():
                        # Forward pass
                        node_embeddings = self.encoder(graph_data_device)
                        
                        # Get embeddings for source and destination nodes
                        pos_src_idx, pos_dst_idx = pos_edges
                        neg_src_idx, neg_dst_idx = neg_edges
                        
                        pos_src_embeddings = node_embeddings[pos_src_idx]
                        pos_dst_embeddings = node_embeddings[pos_dst_idx]
                        neg_src_embeddings = node_embeddings[neg_src_idx]
                        neg_dst_embeddings = node_embeddings[neg_dst_idx]
                        
                        # Predict scores
                        pos_scores = self.predictor(pos_src_embeddings, pos_dst_embeddings)
                        neg_scores = self.predictor(neg_src_embeddings, neg_dst_embeddings)
                        
                        # Compute loss
                        batch_loss = self.compute_loss(pos_scores, neg_scores)
                    
                    # Backward pass with mixed precision
                    if self.grad_accumulator is not None:
                        # Use gradient accumulation
                        should_step = self.grad_accumulator.backward(batch_loss)
                        if should_step:
                            # Step with gradient clipping
                            if self.config.max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(
                                    self.encoder.parameters(), 
                                    self.config.max_grad_norm
                                )
                                nn.utils.clip_grad_norm_(
                                    self.predictor.parameters(), 
                                    self.config.max_grad_norm
                                )
                            # Optimizer step with mixed precision
                            self.mixed_precision.step(self.optimizer)
                    else:
                        # Standard backward pass with mixed precision
                        self.mixed_precision.scale(batch_loss).backward()
                        
                        # Step with gradient clipping
                        if self.config.max_grad_norm is not None:
                            nn.utils.clip_grad_norm_(
                                self.encoder.parameters(), 
                                self.config.max_grad_norm
                            )
                            nn.utils.clip_grad_norm_(
                                self.predictor.parameters(), 
                                self.config.max_grad_norm
                            )
                        # Optimizer step with mixed precision
                        self.mixed_precision.step(self.optimizer)
                else:
                    # Standard forward pass without mixed precision
                    node_embeddings = self.encoder(graph_data_device)
                    
                    # Get embeddings for source and destination nodes
                    pos_src_idx, pos_dst_idx = pos_edges
                    neg_src_idx, neg_dst_idx = neg_edges
                    
                    pos_src_embeddings = node_embeddings[pos_src_idx]
                    pos_dst_embeddings = node_embeddings[pos_dst_idx]
                    neg_src_embeddings = node_embeddings[neg_src_idx]
                    neg_dst_embeddings = node_embeddings[neg_dst_idx]
                    
                    # Predict scores
                    pos_scores = self.predictor(pos_src_embeddings, pos_dst_embeddings)
                    neg_scores = self.predictor(neg_src_embeddings, neg_dst_embeddings)
                    
                    # Compute loss
                    batch_loss = self.compute_loss(pos_scores, neg_scores)
                    
                    # Standard backward pass
                    if self.grad_accumulator is not None:
                        # Use gradient accumulation
                        should_step = self.grad_accumulator.backward(batch_loss)
                        if should_step:
                            # Step with gradient clipping
                            if self.config.max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(
                                    self.encoder.parameters(), 
                                    self.config.max_grad_norm
                                )
                                nn.utils.clip_grad_norm_(
                                    self.predictor.parameters(), 
                                    self.config.max_grad_norm
                                )
                            # Standard optimizer step
                            self.optimizer.step()
                    else:
                        # Standard backward pass
                        batch_loss.backward()
                        
                        # Step with gradient clipping
                        if self.config.max_grad_norm is not None:
                            nn.utils.clip_grad_norm_(
                                self.encoder.parameters(), 
                                self.config.max_grad_norm
                            )
                            nn.utils.clip_grad_norm_(
                                self.predictor.parameters(), 
                                self.config.max_grad_norm
                            )
                        # Standard optimizer step
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            
            # Update scheduler if exists
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Batch statistics
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            
            batch_size = len(pos_src_idx)
            total_samples += batch_size
            total_loss += batch_loss.item() * batch_size
            
            # Update progress bar
            batch_loss_value = batch_loss.item()
            avg_batch_time = sum(batch_times[-10:]) / min(len(batch_times), 10)
            samples_per_sec = batch_size / avg_batch_time if avg_batch_time > 0 else 0
            
            progress_bar.set_postfix({
                'loss': f"{batch_loss_value:.4f}",
                'samples/sec': f"{samples_per_sec:.1f}"
            })
            
            # Clear cache for large graphs
            if self.config.memory_optimization_level >= 2 and batch_idx % 10 == 0:
                # Only clear cache periodically to avoid overhead
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Calculate throughput
        avg_time_per_batch = sum(batch_times) / len(batch_times) if batch_times else 0
        throughput = total_samples / sum(batch_times) if sum(batch_times) > 0 else 0
        
        metrics = {
            'train_loss': avg_loss,
            'train_samples': total_samples,
            'train_throughput': throughput,
            'train_avg_batch_time': avg_time_per_batch
        }
        
        return metrics
    
    def validate(self, graph_data, prefix="val"):
        """Validate the model with memory optimizations.
        
        Args:
            graph_data: Graph data to validate on
            prefix: Prefix for metric names
            
        Returns:
            Dictionary with validation metrics
        """
        # Set models to eval mode
        self.encoder.eval()
        self.predictor.eval()
        
        total_loss = 0.0
        all_pos_scores = []
        all_neg_scores = []
        total_samples = 0
        
        # Generate batches
        batches = self._generate_batches(graph_data, self.config.batch_size)
        
        # Validation loop
        with torch.no_grad():
            for pos_edges, neg_edges in batches:
                # Move batch to device
                pos_edges = self._move_to_device(pos_edges)
                neg_edges = self._move_to_device(neg_edges)
                graph_data_device = self._move_to_device(graph_data)
                
                # Use memory efficient context for large graphs
                with memory_efficient_context(
                    enabled=self.config.memory_optimization_level > 0,
                    enable_autocast=self.config.use_mixed_precision,
                    enable_checkpoint=False  # No need for checkpointing in validation
                ):
                    # If using mixed precision, use autocast
                    if self.config.use_mixed_precision:
                        with self.mixed_precision.autocast():
                            # Forward pass
                            node_embeddings = self.encoder(graph_data_device)
                            
                            # Get embeddings for source and destination nodes
                            pos_src_idx, pos_dst_idx = pos_edges
                            neg_src_idx, neg_dst_idx = neg_edges
                            
                            pos_src_embeddings = node_embeddings[pos_src_idx]
                            pos_dst_embeddings = node_embeddings[pos_dst_idx]
                            neg_src_embeddings = node_embeddings[neg_src_idx]
                            neg_dst_embeddings = node_embeddings[neg_dst_idx]
                            
                            # Predict scores
                            pos_scores = self.predictor(pos_src_embeddings, pos_dst_embeddings)
                            neg_scores = self.predictor(neg_src_embeddings, neg_dst_embeddings)
                            
                            # Compute loss
                            batch_loss = self.compute_loss(pos_scores, neg_scores)
                    else:
                        # Standard forward pass without mixed precision
                        node_embeddings = self.encoder(graph_data_device)
                        
                        # Get embeddings for source and destination nodes
                        pos_src_idx, pos_dst_idx = pos_edges
                        neg_src_idx, neg_dst_idx = neg_edges
                        
                        pos_src_embeddings = node_embeddings[pos_src_idx]
                        pos_dst_embeddings = node_embeddings[pos_dst_idx]
                        neg_src_embeddings = node_embeddings[neg_src_idx]
                        neg_dst_embeddings = node_embeddings[neg_dst_idx]
                        
                        # Predict scores
                        pos_scores = self.predictor(pos_src_embeddings, pos_dst_embeddings)
                        neg_scores = self.predictor(neg_src_embeddings, neg_dst_embeddings)
                        
                        # Compute loss
                        batch_loss = self.compute_loss(pos_scores, neg_scores)
                
                # Collect statistics
                batch_size = len(pos_src_idx)
                total_samples += batch_size
                total_loss += batch_loss.item() * batch_size
                
                # Collect scores for metrics
                all_pos_scores.append(pos_scores.cpu())
                all_neg_scores.append(neg_scores.cpu())
                
                # Clear cache for large graphs
                if self.config.memory_optimization_level >= 2:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Calculate evaluation metrics
        metrics = {}
        metrics[f'{prefix}_loss'] = avg_loss
        
        if all_pos_scores and all_neg_scores:
            # Concatenate all scores
            all_pos_scores = torch.cat(all_pos_scores, dim=0).numpy()
            all_neg_scores = torch.cat(all_neg_scores, dim=0).numpy()
            
            # Calculate AUC and AP
            from sklearn.metrics import roc_auc_score, average_precision_score
            
            y_true = np.concatenate([np.ones(len(all_pos_scores)), np.zeros(len(all_neg_scores))])
            y_score = np.concatenate([all_pos_scores, all_neg_scores])
            
            metrics[f'{prefix}_auc'] = roc_auc_score(y_true, y_score)
            metrics[f'{prefix}_ap'] = average_precision_score(y_true, y_score)
        
        return metrics
    
    def save_checkpoint(self, path, epoch, metrics=None):
        """Save model checkpoint with all optimization info.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Dictionary of metrics
            
        Returns:
            None
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create state dictionary
        state = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        # Add scheduler state if it exists
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Add mixed precision state if using
        if self.config.use_mixed_precision:
            state['mixed_precision_state_dict'] = self.mixed_precision.state_dict()
        
        # Add metrics if provided
        if metrics is not None:
            state['metrics'] = metrics
        
        # Save in a distributed-aware manner
        if self.is_distributed:
            self.distributed_handler.save_checkpoint(state, path)
        else:
            torch.save(state, path)
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint with all optimization info.
        
        Args:
            path: Path to load checkpoint from
            
        Returns:
            Dictionary with checkpoint information
        """
        # Load in a distributed-aware manner
        if self.is_distributed:
            state = self.distributed_handler.load_checkpoint(path)
        else:
            state = torch.load(path, map_location=self.device)
        
        # Load encoder and predictor states
        self.encoder.load_state_dict(state['encoder_state_dict'])
        self.predictor.load_state_dict(state['predictor_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        # Load mixed precision state
        if 'mixed_precision_state_dict' in state and self.config.use_mixed_precision:
            self.mixed_precision.load_state_dict(state['mixed_precision_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path} (epoch {state.get('epoch', 'unknown')})")
        
        return state
    
    def _recover_from_oom(self):
        """Attempt to recover from out-of-memory errors."""
        # Free cached memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Reset models
        self.encoder = None
        self.predictor = None
        self.optimizer = None
        self.scheduler = None
        
        # Increase memory optimization level
        self.config.memory_optimization_level = min(3, self.config.memory_optimization_level + 1)
        
        # Enable gradient checkpointing if not already
        if not self.config.use_gradient_checkpointing:
            self.config.use_gradient_checkpointing = True
            
        # Enable mixed precision if not already
        if not self.config.use_mixed_precision:
            self.config.use_mixed_precision = True
            
        logger.info("Attempted recovery from OOM error with enhanced memory optimizations")
    
    def compute_loss(self, pos_scores, neg_scores):
        """Compute prediction loss with support for different loss functions.
        
        Args:
            pos_scores: Scores for positive edges
            neg_scores: Scores for negative edges
            
        Returns:
            Loss tensor
        """
        # Default implementation - can be extended for different loss functions
        # Basic binary cross entropy
        pos_loss = -torch.log(pos_scores + 1e-6).mean()
        neg_loss = -torch.log(1 - neg_scores + 1e-6).mean()
        loss = pos_loss + neg_loss
        
        return loss 