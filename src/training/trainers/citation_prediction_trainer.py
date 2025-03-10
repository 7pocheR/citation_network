import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from tqdm import tqdm
import random
import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
import importlib

from src.data.dataset import GraphData
from src.models.encoder.base import BaseEncoder
from src.models.predictors.base import BasePredictor
from src.training.metrics import CitationPredictionMetrics

# Implementation Progress Log
'''
Step 1: Define configuration dataclass
- Created TrainingConfig with model parameters, training parameters, and optimization parameters
- Added fields for device selection, logging, and checkpointing

Step 2: Create CitationPredictionTrainer class skeleton
- Added initialization method with configuration and setup methods
- Created placeholders for key methods: train, train_epoch, validate

Step 3: Implement model initialization
- Added dynamic model loading based on configuration
- Implemented methods to get encoder and predictor parameters from graph data
- Added optimization setup with support for different optimizers and schedulers
- Added device management utilities for handling tensors across CPU/GPU

Decision points:
- For encoder initialization: 
  * Option 1: Use direct imports for known encoder types ✓
  * Option 2: Use dynamic imports for flexibility with any encoder type (implemented)
  * Chose Option 2 for maximum flexibility

- For predictor initialization:
  * Option 1: Assume embed_dim is the only required parameter ✗
  * Option 2: Get information about required parameters from encoder output ✓
  * Chose Option 2 for better compatibility with different encoder-predictor pairs

Step 4: Implement training loop with mini-batch support
- Added methods for generating batches from graph data
- Implemented training loop with progress tracking
- Added support for memory-efficient training
- Implemented gradient accumulation for large batch training

Decision points:
- For batch generation:
  * Option 1: Full-graph batching (using all edges at once) ✗
  * Option 2: Mini-batch generation with edge sampling ✓
  * Chose Option 2 for better scalability with large graphs

- For negative sampling:
  * Option 1: Random negative sampling throughout the graph ✗
  * Option 2: Constrained negative sampling respecting timestamps ✓
  * Chose Option 2 for more realistic negative examples respecting temporal causality
'''

@dataclass
class TrainingConfig:
    """Configuration for the CitationPredictionTrainer.
    
    This class uses Python's dataclasses to provide a clean and type-hinted
    configuration system for the training pipeline.
    """
    # Model configuration
    encoder_type: str = "TGNEncoder"  # Type of encoder to use
    predictor_type: str = "MLPPredictor"  # Type of predictor to use
    embed_dim: int = 128  # Embedding dimension
    encoder_params: Dict[str, Any] = field(default_factory=dict)  # Additional encoder parameters
    predictor_params: Dict[str, Any] = field(default_factory=dict)  # Additional predictor parameters
    
    # Training parameters
    batch_size: int = 32  # Batch size for training
    num_epochs: int = 100  # Number of epochs to train for
    early_stopping_patience: int = 10  # Patience for early stopping
    evaluation_interval: int = 1  # Evaluate every N epochs
    memory_efficient: bool = False  # Use memory-efficient training
    gradient_accumulation_steps: int = 1  # Steps for gradient accumulation
    
    # Optimization parameters
    learning_rate: float = 0.001  # Learning rate
    weight_decay: float = 1e-5  # Weight decay for regularization
    optimizer_type: str = "Adam"  # Type of optimizer to use
    scheduler_type: Optional[str] = None  # Type of learning rate scheduler
    scheduler_params: Dict[str, Any] = field(default_factory=dict)  # Parameters for scheduler
    max_grad_norm: Optional[float] = 1.0  # Maximum gradient norm for clipping
    
    # Data parameters
    train_ratio: float = 0.8  # Ratio of data to use for training
    val_ratio: float = 0.1  # Ratio of data to use for validation
    test_ratio: float = 0.1  # Ratio of data to use for testing
    temporal_split: bool = True  # Use temporal splitting for data
    neg_sampling_ratio: float = 1.0  # Ratio of negative to positive samples
    
    # Device and environment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Device to use
    seed: int = 42  # Random seed for reproducibility
    
    # Logging and checkpointing
    log_dir: str = "./logs"  # Directory for logs
    checkpoint_dir: str = "./checkpoints"  # Directory for checkpoints
    log_interval: int = 10  # Log every N batches
    checkpoint_interval: int = 1  # Save checkpoint every N epochs
    save_best_only: bool = True  # Only save the best model


class CitationPredictionTrainer:
    """Trainer for citation prediction models.
    
    This trainer is designed to be modular and component-independent,
    working with any encoder and predictor that implement the appropriate
    base interfaces. It handles training, validation, checkpointing, and
    provides utilities for data management and optimization.
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer with the given configuration.
        
        Args:
            config: Configuration for the trainer
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Create directories
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Setup logger
        self._setup_logger()
        
        # Set random seeds for reproducibility
        self._set_seed(config.seed)
        
        # Initialize metrics tracker
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_ap': [],
        }
        
        # Best validation metrics for early stopping and checkpointing
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        self.best_epoch = 0
        
        # Track training state
        self.current_epoch = 0
        self.early_stopping_counter = 0
        
        # Components will be initialized later
        self.encoder = None
        self.predictor = None
        self.optimizer = None
        self.scheduler = None
    
    def _setup_logger(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.config.log_dir, f"training_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"CitationPredictionTrainer initialized with config: {self.config}")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility.
        
        Args:
            seed: Random seed to use
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Using the default GPU device
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.logger.info(f"Random seed set to {seed}")
    
    def _move_to_device(self, obj: Any) -> Any:
        """Move a tensor or collection of tensors to the configured device.
        
        Args:
            obj: Tensor, list of tensors, dict of tensors, or GraphData
            
        Returns:
            The object moved to the device
        """
        if obj is None:
            return None
        elif isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        elif isinstance(obj, list):
            return [self._move_to_device(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._move_to_device(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, GraphData):
            # Use GraphData's to method
            return obj.to(self.device)
        return obj
    
    def _get_encoder_class(self) -> type:
        """Dynamically load the encoder class based on configuration.
        
        Returns:
            The encoder class
        """
        encoder_type = self.config.encoder_type
        try:
            # First try direct import from encoder directory
            module_path = f"src.models.encoder.{encoder_type.lower()}"
            module = importlib.import_module(module_path)
            encoder_class = getattr(module, encoder_type)
        except (ImportError, AttributeError):
            # Try to find it in other modules
            modules_to_check = [
                f"src.models.encoder.{encoder_type.lower()}",
                "src.models.encoder",
                "src.models.encoder.tgn",
                "src.models.encoder.hyperbolic",
                "src.models.encoder.hyperbolic_gnn"
            ]
            
            for module_path in modules_to_check:
                try:
                    module = importlib.import_module(module_path)
                    encoder_class = getattr(module, encoder_type)
                    break
                except (ImportError, AttributeError):
                    continue
            else:
                # If we've tried all modules and still can't find it, raise error
                raise ValueError(f"Could not find encoder class {encoder_type}")
        
        self.logger.info(f"Found encoder class: {encoder_class.__name__}")
        return encoder_class
    
    def _get_predictor_class(self) -> type:
        """Dynamically load the predictor class based on configuration.
        
        Returns:
            The predictor class
        """
        predictor_type = self.config.predictor_type
        try:
            # First try direct import from predictors directory
            module_path = f"src.models.predictors.{predictor_type.lower()}"
            module = importlib.import_module(module_path)
            predictor_class = getattr(module, predictor_type)
        except (ImportError, AttributeError):
            # Try to find it with adjusted name (e.g., mlp_predictor -> MLPPredictor)
            try:
                module_path = f"src.models.predictors.{predictor_type.lower()}"
                module = importlib.import_module(module_path)
                
                # Try to match class name even if case is different
                for attr_name in dir(module):
                    if attr_name.lower() == predictor_type.lower():
                        predictor_class = getattr(module, attr_name)
                        break
                else:
                    raise AttributeError(f"Could not find predictor class {predictor_type} in module {module_path}")
            except (ImportError, AttributeError):
                # Try to find it in other modules
                modules_to_check = [
                    "src.models.predictors",
                    "src.models.predictors.mlp_predictor",
                    "src.models.predictors.distance_predictor",
                    "src.models.predictors.attention_predictor",
                    "src.models.predictors.temporal_predictor"
                ]
                
                for module_path in modules_to_check:
                    try:
                        module = importlib.import_module(module_path)
                        predictor_class = getattr(module, predictor_type)
                        break
                    except (ImportError, AttributeError):
                        continue
                else:
                    # If we've tried all modules and still can't find it, raise error
                    raise ValueError(f"Could not find predictor class {predictor_type}")
        
        self.logger.info(f"Found predictor class: {predictor_class.__name__}")
        return predictor_class
    
    def _get_encoder_params(self, graph_data: GraphData) -> Dict[str, Any]:
        """Get parameters for encoder initialization from graph data.
        
        Args:
            graph_data: Graph data to extract parameters from
            
        Returns:
            Dict of encoder initialization parameters
        """
        params = {}
        
        # Get node feature dimension
        if hasattr(graph_data, 'x') and graph_data.x is not None:
            params['node_dim'] = graph_data.x.size(1)
        else:
            # Default to embed_dim if no node features
            params['node_dim'] = self.config.embed_dim
        
        # Get edge feature dimension
        if hasattr(graph_data, 'edge_attr') and isinstance(graph_data.edge_attr, torch.Tensor):
            params['edge_dim'] = graph_data.edge_attr.size(1)
        else:
            # Default to 0 if no edge features
            params['edge_dim'] = 0
        
        # Get number of nodes
        if hasattr(graph_data, 'x') and graph_data.x is not None:
            params['num_nodes'] = graph_data.x.size(0)
        elif hasattr(graph_data, 'node_timestamps') and graph_data.node_timestamps is not None:
            params['num_nodes'] = len(graph_data.node_timestamps)
        elif hasattr(graph_data, 'paper_times') and graph_data.paper_times is not None:
            params['num_nodes'] = len(graph_data.paper_times)
        elif hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None:
            params['num_nodes'] = graph_data.edge_index.max().item() + 1
        
        # Always include embed_dim from config
        params['embed_dim'] = self.config.embed_dim
        
        # Special handling for TGNEncoder
        if self.config.encoder_type == "TGNEncoder" or self.config.encoder_type == "TemporalGraphNetwork":
            # TGNEncoder needs these parameters
            params.update({
                'memory_dim': self.config.embed_dim,  # Start with same as embed_dim
                'time_dim': self.config.embed_dim // 2,  # Half of embed_dim
                'num_gnn_layers': 2,  # Default to 2 layers
                'aggregator_type': 'mean',  # Default aggregator
            })
        
        # Add any user-provided encoder params from config (overrides defaults)
        params.update(self.config.encoder_params)
        
        self.logger.info(f"Encoder parameters: {params}")
        return params
    
    def _create_dummy_forward(self, graph_data: GraphData) -> torch.Tensor:
        """Perform a dummy forward pass with the encoder to get output shape.
        
        Args:
            graph_data: Graph data to use for the dummy forward pass
            
        Returns:
            Dummy output from encoder
        """
        # Create a small version of the graph for the dummy pass
        if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None:
            # Get a small number of nodes
            max_node_idx = graph_data.edge_index.max().item()
            num_nodes = min(max_node_idx + 1, 10)  # Use at most 10 nodes
            
            # Get a small number of edges
            num_edges = min(graph_data.edge_index.size(1), 20)  # Use at most 20 edges
            
            # Create a small graph
            small_edge_index = graph_data.edge_index[:, :num_edges]
            if hasattr(graph_data, 'x') and graph_data.x is not None:
                small_x = graph_data.x[:num_nodes]
            else:
                small_x = None
                
            small_graph = GraphData(
                x=small_x,
                edge_index=small_edge_index,
            )
            
            # Add timestamps if available
            if hasattr(graph_data, 'node_timestamps') and graph_data.node_timestamps is not None:
                small_graph.node_timestamps = graph_data.node_timestamps[:num_nodes]
            if hasattr(graph_data, 'edge_timestamps') and graph_data.edge_timestamps is not None:
                small_graph.edge_timestamps = graph_data.edge_timestamps[:num_edges]
                
            # Move to device
            small_graph = self._move_to_device(small_graph)
            
            # Run dummy forward pass
            with torch.no_grad():
                output = self.encoder(small_graph)
                
            # If output is a list or tuple, use first element
            if isinstance(output, (list, tuple)):
                output = output[0]
                
            # If output has 3 dimensions, take first batch
            if output.dim() == 3:
                output = output[0]
                
            return output
        else:
            # If no edge_index, just return a dummy tensor with the right shape
            return torch.zeros((1, self.config.embed_dim), device=self.device)
    
    def _get_predictor_params(self, dummy_output: torch.Tensor) -> Dict[str, Any]:
        """Get parameters for predictor initialization from encoder output.
        
        Args:
            dummy_output: Output from a dummy encoder forward pass
            
        Returns:
            Dict of predictor initialization parameters
        """
        params = {}
        
        # Get embedding dimension from encoder output
        if dummy_output.dim() >= 2:
            params['embed_dim'] = dummy_output.size(-1)
        else:
            # Default to config embed_dim
            params['embed_dim'] = self.config.embed_dim
        
        # Specific parameters for different predictor types
        if self.config.predictor_type == "MLPPredictor":
            params.update({
                'hidden_dims': [2 * params['embed_dim'], params['embed_dim']],
                'dropout': 0.1,
            })
        elif self.config.predictor_type == "AttentionPredictor":
            params.update({
                'num_heads': 4,
                'dropout': 0.1,
                'use_time_encoding': True,
            })
        elif self.config.predictor_type == "TemporalPredictor":
            params.update({
                'temporal_encoding_dim': params['embed_dim'] // 2,
                'use_recency_bias': True,
            })
        
        # Add any user-provided predictor params from config (overrides defaults)
        params.update(self.config.predictor_params)
        
        self.logger.info(f"Predictor parameters: {params}")
        return params
    
    def initialize_models(self, graph_data: GraphData):
        """Initialize encoder and predictor models.
        
        Args:
            graph_data: Graph data to extract parameters from
        """
        # Skip initialization if models are already initialized
        if self.encoder is not None and self.predictor is not None:
            self.logger.info("Models already initialized, skipping initialization")
            return
            
        try:
            # Get encoder class and parameters
            encoder_class = self._get_encoder_class()
            encoder_params = self._get_encoder_params(graph_data)
            
            # Add additional parameters from config
            encoder_params.update(self.config.encoder_params)
            
            # Add embed_dim parameter
            encoder_params['embed_dim'] = self.config.embed_dim
            
            # Initialize encoder
            self.logger.info(f"Encoder parameters: {encoder_params}")
            self.encoder = encoder_class(**encoder_params)
            
            # Move encoder to device
            self.encoder = self.encoder.to(self.device)
            
            # Get predictor class and parameters
            predictor_class = self._get_predictor_class()
            predictor_params = self._get_predictor_params(self._create_dummy_forward(graph_data))
            
            # Add additional parameters from config
            predictor_params.update(self.config.predictor_params)
            
            # Initialize predictor
            self.logger.info(f"Predictor parameters: {predictor_params}")
            self.predictor = predictor_class(**predictor_params)
            
            # Move predictor to device
            self.predictor = self.predictor.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
    
    def _validate_dimensions(self, encoder_output: torch.Tensor, predictor_input_dim: int):
        """Validate that encoder output dimensions match predictor input dimensions.
        
        Args:
            encoder_output: Output from encoder
            predictor_input_dim: Expected input dimension for predictor
            
        Raises:
            ValueError: If dimensions don't match and can't be adapted
        """
        output_dim = encoder_output.size(-1)
        
        if output_dim != predictor_input_dim:
            # Check if predictor has adaptive input handling
            if hasattr(self.predictor, 'add_adaptation') or hasattr(self.predictor, 'adaptive_input'):
                self.logger.info(f"Predictor has adaptive input handling. Will adapt {output_dim} to {predictor_input_dim}.")
            else:
                self.logger.warning(
                    f"Encoder output dimension {output_dim} doesn't match predictor input dimension {predictor_input_dim}. "
                    f"This may cause errors if the predictor doesn't handle dimension adaptation internally."
                )
    
    def _initialize_optimizer(self):
        """Initialize the optimizer based on configuration."""
        params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        
        optimizer_params = {
            'lr': self.config.learning_rate,
            'weight_decay': self.config.weight_decay
        }
        
        if self.config.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(params, **optimizer_params)
        elif self.config.optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(params, **optimizer_params)
        elif self.config.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(params, momentum=0.9, **optimizer_params)
        elif self.config.optimizer_type == 'RMSprop':
            self.optimizer = optim.RMSprop(params, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.config.optimizer_type}")
        
        self.logger.info(f"Initialized optimizer: {self.config.optimizer_type}")
    
    def _initialize_scheduler(self):
        """Initialize the learning rate scheduler based on configuration."""
        if self.config.scheduler_type == 'StepLR':
            default_params = {'step_size': 10, 'gamma': 0.1}
            params = {**default_params, **self.config.scheduler_params}
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **params)
        elif self.config.scheduler_type == 'ExponentialLR':
            default_params = {'gamma': 0.95}
            params = {**default_params, **self.config.scheduler_params}
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, **params)
        elif self.config.scheduler_type == 'ReduceLROnPlateau':
            default_params = {'mode': 'min', 'factor': 0.1, 'patience': 10}
            params = {**default_params, **self.config.scheduler_params}
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **params)
        elif self.config.scheduler_type == 'CosineAnnealingLR':
            default_params = {'T_max': 10}
            params = {**default_params, **self.config.scheduler_params}
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **params)
        else:
            raise ValueError(f"Unsupported scheduler type: {self.config.scheduler_type}")
        
        self.logger.info(f"Initialized scheduler: {self.config.scheduler_type}")
    
    def train(self, train_graph: GraphData, val_graph: Optional[GraphData] = None):
        """Train the model on the given graph data.
        
        Args:
            train_graph: Graph data to train on
            val_graph: Optional graph data to validate on
        """
        # Initialize models if not already done
        if self.encoder is None or self.predictor is None:
            self.initialize_models(train_graph)
        
        # Move data to device
        train_graph = self._move_to_device(train_graph)
        if val_graph is not None:
            val_graph = self._move_to_device(val_graph)
        
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            self.encoder.train()
            self.predictor.train()
            train_loss = self.train_epoch(train_graph)
            
            # Log training progress
            self.metrics['train_loss'].append(train_loss)
            if epoch % self.config.log_interval == 0:
                self.logger.info(f"Epoch {epoch}/{self.config.num_epochs}, Train Loss: {train_loss:.4f}")
            
            # Validation
            if val_graph is not None and epoch % self.config.evaluation_interval == 0:
                self.encoder.eval()
                self.predictor.eval()
                val_loss, val_auc, val_ap = self.validate(val_graph)
                
                # Log validation metrics
                self.metrics['val_loss'].append(val_loss)
                self.metrics['val_auc'].append(val_auc)
                self.metrics['val_ap'].append(val_ap)
                
                self.logger.info(
                    f"Epoch {epoch}/{self.config.num_epochs}, "
                    f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}"
                )
                
                # Check for early stopping and save model
                self._check_early_stopping(val_loss, epoch)
                
                # Update learning rate scheduler if using ReduceLROnPlateau
                if (self.scheduler is not None and 
                    isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                    self.scheduler.step(val_loss)
            
            # Update learning rate scheduler if not ReduceLROnPlateau
            elif (self.scheduler is not None and 
                  not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                self.scheduler.step()
            
            # Save checkpoint if needed
            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
            
            # Check if we should stop early
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model if available
        if self.best_epoch > 0:
            self._load_checkpoint(f"best_model.pt")
            self.logger.info(f"Loaded best model from epoch {self.best_epoch}")
        
        self.logger.info("Training completed")
        return self.metrics
    
    def _check_early_stopping(self, val_loss: float, epoch: int):
        """Check if we should stop early based on validation loss.
        
        Args:
            val_loss: Validation loss
            epoch: Current epoch
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.early_stopping_counter = 0
            
            # Save best model
            if self.config.save_best_only:
                self._save_checkpoint(epoch, filename="best_model.pt")
        else:
            self.early_stopping_counter += 1
    
    def _save_checkpoint(self, epoch: int, filename: Optional[str] = None):
        """Save a checkpoint of the model.
        
        Args:
            epoch: Current epoch
            filename: Optional filename for the checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_auc': self.best_val_auc,
            'best_epoch': self.best_epoch,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _load_checkpoint(self, filename: str):
        """Load a checkpoint.
        
        Args:
            filename: Filename of the checkpoint to load
        """
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint {checkpoint_path} does not exist")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if it exists
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load other attributes
        self.metrics = checkpoint['metrics']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        self.best_epoch = checkpoint['best_epoch']
        self.current_epoch = checkpoint['epoch'] + 1
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def _generate_batches(self, 
                         graph_data: GraphData, 
                         batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate batches of positive and negative edges for training.
        
        Args:
            graph_data: Graph data to generate batches from
            batch_size: Number of edges per batch
            
        Returns:
            List of tuples (pos_edges, neg_edges) for each batch
        """
        edge_index = graph_data.edge_index
        num_edges = edge_index.size(1)
        
        # Shuffle edge indices
        perm = torch.randperm(num_edges, device=self.device)
        edge_index_shuffled = edge_index[:, perm]
        
        # Determine number of batches
        num_batches = (num_edges + batch_size - 1) // batch_size
        
        batches = []
        for i in range(num_batches):
            # Get batch of positive edges
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_edges)
            pos_edges = edge_index_shuffled[:, start_idx:end_idx]
            
            # Generate negative edges for this batch
            neg_edges = self._generate_negative_edges(graph_data, pos_edges.size(1))
            
            batches.append((pos_edges, neg_edges))
        
        return batches
    
    def _generate_negative_edges(self, 
                               graph_data: GraphData, 
                               num_samples: int) -> torch.Tensor:
        """Generate negative edges that don't exist in the graph.
        
        Args:
            graph_data: Graph data containing existing edges
            num_samples: Number of negative edges to generate
            
        Returns:
            Tensor of negative edge indices [2, num_samples]
        """
        # Calculate num_nodes from graph data
        if hasattr(graph_data, 'x') and graph_data.x is not None:
            num_nodes = graph_data.x.size(0)
        elif hasattr(graph_data, 'node_timestamps') and graph_data.node_timestamps is not None:
            num_nodes = len(graph_data.node_timestamps)
        elif hasattr(graph_data, 'paper_times') and graph_data.paper_times is not None:
            num_nodes = len(graph_data.paper_times)
        else:
            num_nodes = graph_data.edge_index.max().item() + 1

        # Create a mask for existing edges
        edge_index = graph_data.edge_index
        
        # Convert edge_index to a set of tuples for fast lookup
        existing_edges = set()
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            existing_edges.add((src, dst))
        
        # Check if we have temporal information
        has_temporal = False
        if hasattr(graph_data, 'node_timestamps') and graph_data.node_timestamps is not None:
            node_timestamps = graph_data.node_timestamps
            has_temporal = True
        elif hasattr(graph_data, 'paper_times') and graph_data.paper_times is not None:
            node_timestamps = graph_data.paper_times
            has_temporal = True
        
        # Generate negative edges
        neg_edges = torch.zeros((2, num_samples), dtype=torch.long, device=self.device)
        
        sampled = 0
        max_attempts = num_samples * 10  # Avoid infinite loop
        attempts = 0
        
        while sampled < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Sample random source and destination nodes
            src = torch.randint(0, num_nodes, (1,), device=self.device).item()
            dst = torch.randint(0, num_nodes, (1,), device=self.device).item()
            
            # Skip self-loops and existing edges
            if src == dst or (src, dst) in existing_edges:
                continue
            
            # If using temporal constraints, ensure temporal causality
            if has_temporal:
                # Skip if destination was published after source (can't cite future papers)
                if node_timestamps[dst] > node_timestamps[src]:
                    continue
            
            # Add to negative edges
            neg_edges[0, sampled] = src
            neg_edges[1, sampled] = dst
            sampled += 1
        
        # If we couldn't sample enough edges, just duplicate the ones we have
        if sampled < num_samples:
            self.logger.warning(
                f"Could only sample {sampled}/{num_samples} negative edges. "
                f"Duplicating edges to reach requested count."
            )
            
            # Duplicate existing samples to fill the tensor
            repeat_factor = (num_samples + sampled - 1) // sampled  # Ceiling division
            neg_edges_valid = neg_edges[:, :sampled]
            neg_edges_repeated = neg_edges_valid.repeat(1, repeat_factor)[:, :num_samples]
            neg_edges = neg_edges_repeated
        
        return neg_edges
    
    def train_epoch(self, graph_data: GraphData) -> float:
        """Train for a single epoch.
        
        Args:
            graph_data: Graph data to train on
            
        Returns:
            float: Average loss for the epoch
        """
        # Initialize variables to track epoch statistics
        total_loss = 0.0
        num_batches = 0
        
        # Generate batches
        batches = self._generate_batches(graph_data, self.config.batch_size)
        
        # Training loop with progress bar
        progress_bar = tqdm(batches, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (pos_edges, neg_edges) in enumerate(progress_bar):
            # Process batch differently based on memory efficiency mode
            if self.config.memory_efficient:
                batch_loss = self._train_batch_memory_efficient(graph_data, pos_edges, neg_edges)
            else:
                batch_loss = self._train_batch(graph_data, pos_edges, neg_edges)
            
            # Update statistics
            total_loss += batch_loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})
        
        # Return average loss
        return total_loss / max(1, num_batches)
    
    def _train_batch(self, 
                   graph_data: GraphData, 
                   pos_edges: torch.Tensor, 
                   neg_edges: torch.Tensor) -> float:
        """Train on a single batch.
        
        Args:
            graph_data: Graph data to train on
            pos_edges: Positive edges [2, batch_size]
            neg_edges: Negative edges [2, batch_size]
            
        Returns:
            float: Loss for this batch
        """
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Forward pass through encoder
        node_embeddings = self.encoder(graph_data)
        
        # Handle different output formats from encoder
        if isinstance(node_embeddings, tuple) or isinstance(node_embeddings, list):
            node_embeddings = node_embeddings[0]
        
        # If node_embeddings has 3 dimensions (TGN returns [batch, nodes, features]), take first batch
        if node_embeddings.dim() == 3:
            node_embeddings = node_embeddings[0]
        
        # Get embeddings for source and destination nodes
        pos_src_emb = node_embeddings[pos_edges[0]]
        pos_dst_emb = node_embeddings[pos_edges[1]]
        neg_src_emb = node_embeddings[neg_edges[0]]
        neg_dst_emb = node_embeddings[neg_edges[1]]
        
        # Forward pass through predictor
        pos_scores = self.predictor(pos_src_emb, pos_dst_emb)
        neg_scores = self.predictor(neg_src_emb, neg_dst_emb)
        
        # Prepare labels
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)
        
        # Compute loss
        loss_fn = torch.nn.BCEWithLogitsLoss()
        pos_loss = loss_fn(pos_scores, pos_labels)
        neg_loss = loss_fn(neg_scores, neg_labels)
        loss = pos_loss + neg_loss
        
        # Backward pass
        loss.backward()
        
        # Clip gradients if configured
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.predictor.parameters()),
                self.config.max_grad_norm
            )
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()
    
    def _train_batch_memory_efficient(self, 
                                    graph_data: GraphData, 
                                    pos_edges: torch.Tensor, 
                                    neg_edges: torch.Tensor) -> float:
        """Train on a single batch with memory-efficient processing.
        
        This version uses gradient accumulation and smaller sub-batches
        to reduce memory usage during training.
        
        Args:
            graph_data: Graph data to train on
            pos_edges: Positive edges [2, batch_size]
            neg_edges: Negative edges [2, batch_size]
            
        Returns:
            float: Loss for this batch
        """
        # Determine sub-batch size (divide batch size by gradient accumulation steps)
        sub_batch_size = max(1, pos_edges.size(1) // self.config.gradient_accumulation_steps)
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Process all nodes once to get embeddings (avoid recalculating)
        with torch.set_grad_enabled(True):
            node_embeddings = self.encoder(graph_data)
            
            # Handle different output formats from encoder
            if isinstance(node_embeddings, tuple) or isinstance(node_embeddings, list):
                node_embeddings = node_embeddings[0]
            
            # If node_embeddings has 3 dimensions, take first batch
            if node_embeddings.dim() == 3:
                node_embeddings = node_embeddings[0]
        
        total_loss = 0.0
        num_sub_batches = 0
        
        # Process sub-batches for gradient accumulation
        for i in range(0, pos_edges.size(1), sub_batch_size):
            # Get sub-batch of positive edges
            end_idx = min(i + sub_batch_size, pos_edges.size(1))
            pos_edges_sub = pos_edges[:, i:end_idx]
            
            # Get corresponding sub-batch of negative edges
            neg_edges_sub = neg_edges[:, i:end_idx]
            
            # Get embeddings for source and destination nodes
            pos_src_emb = node_embeddings[pos_edges_sub[0]]
            pos_dst_emb = node_embeddings[pos_edges_sub[1]]
            neg_src_emb = node_embeddings[neg_edges_sub[0]]
            neg_dst_emb = node_embeddings[neg_edges_sub[1]]
            
            # Forward pass through predictor
            pos_scores = self.predictor(pos_src_emb, pos_dst_emb)
            neg_scores = self.predictor(neg_src_emb, neg_dst_emb)
            
            # Prepare labels
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)
            
            # Compute loss
            loss_fn = torch.nn.BCEWithLogitsLoss()
            pos_loss = loss_fn(pos_scores, pos_labels)
            neg_loss = loss_fn(neg_scores, neg_labels)
            sub_loss = (pos_loss + neg_loss) / self.config.gradient_accumulation_steps
            
            # Backward pass
            sub_loss.backward()
            
            total_loss += sub_loss.item() * self.config.gradient_accumulation_steps
            num_sub_batches += 1
            
            # Free memory
            del pos_src_emb, pos_dst_emb, neg_src_emb, neg_dst_emb
            del pos_scores, neg_scores, pos_labels, neg_labels
            del pos_loss, neg_loss, sub_loss
            torch.cuda.empty_cache()
        
        # Clip gradients if configured
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.predictor.parameters()),
                self.config.max_grad_norm
            )
        
        # Update parameters
        self.optimizer.step()
        
        # Free memory
        del node_embeddings
        torch.cuda.empty_cache()
        
        return total_loss / max(1, num_sub_batches)
    
    def validate(self, graph_data: GraphData) -> Tuple[float, float, float]:
        """Validate the model on the given graph data.
        
        Args:
            graph_data: Graph data to validate on
            
        Returns:
            Tuple[float, float, float]: Validation loss, AUC, and AP
        """
        # Ensure model is in evaluation mode
        self.encoder.eval()
        self.predictor.eval()
        
        # Prepare for metrics calculation
        all_pos_scores = []
        all_neg_scores = []
        
        # Process graph data in a memory-efficient way by sampling batches
        with torch.no_grad():
            # Generate validation batches
            val_batches = self._generate_batches(graph_data, self.config.batch_size)
            
            # Embed nodes once (avoid recomputing for each batch)
            node_embeddings = self.encoder(graph_data)
            
            # Handle different output formats from encoder
            if isinstance(node_embeddings, tuple) or isinstance(node_embeddings, list):
                node_embeddings = node_embeddings[0]
            
            # If node_embeddings has 3 dimensions (TGN returns [batch, nodes, features]), take first batch
            if node_embeddings.dim() == 3:
                node_embeddings = node_embeddings[0]
            
            # Process each batch
            for pos_edges, neg_edges in val_batches:
                # Get embeddings for positive edges
                pos_src_emb = node_embeddings[pos_edges[0]]
                pos_dst_emb = node_embeddings[pos_edges[1]]
                
                # Get embeddings for negative edges
                neg_src_emb = node_embeddings[neg_edges[0]]
                neg_dst_emb = node_embeddings[neg_edges[1]]
                
                # Predict scores
                pos_scores = self.predictor(pos_src_emb, pos_dst_emb)
                neg_scores = self.predictor(neg_src_emb, neg_dst_emb)
                
                # Store scores for metrics calculation
                all_pos_scores.append(pos_scores.cpu())
                all_neg_scores.append(neg_scores.cpu())
            
            # Concatenate all scores
            all_pos_scores = torch.cat(all_pos_scores)
            all_neg_scores = torch.cat(all_neg_scores)
            
            # Calculate loss
            pos_labels = torch.ones_like(all_pos_scores)
            neg_labels = torch.zeros_like(all_neg_scores)
            
            loss_fn = torch.nn.BCEWithLogitsLoss()
            pos_loss = loss_fn(all_pos_scores, pos_labels)
            neg_loss = loss_fn(all_neg_scores, neg_labels)
            val_loss = (pos_loss + neg_loss).item()
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_pos_scores, all_neg_scores)
            
            return val_loss, metrics['auc'], metrics['ap']
    
    def _calculate_metrics(self, 
                         pos_scores: torch.Tensor, 
                         neg_scores: torch.Tensor) -> Dict[str, float]:
        """Calculate evaluation metrics for link prediction.
        
        Args:
            pos_scores: Scores for positive edges
            neg_scores: Scores for negative edges
            
        Returns:
            Dictionary containing metrics (AUC, AP, etc.)
        """
        # Convert to numpy
        pos_scores_np = pos_scores.detach().cpu().numpy()
        neg_scores_np = neg_scores.detach().cpu().numpy()
        
        # Combine scores and labels
        scores = np.concatenate([pos_scores_np, neg_scores_np])
        labels = np.concatenate([np.ones_like(pos_scores_np), np.zeros_like(neg_scores_np)])
        
        # Calculate AUC
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            self.logger.warning("Could not calculate AUC (possibly only one class present)")
            auc = 0.5  # Default value for random classifier
        
        # Calculate AP
        try:
            ap = average_precision_score(labels, scores)
        except ValueError:
            self.logger.warning("Could not calculate AP (possibly only one class present)")
            ap = sum(labels) / len(labels)  # Default value based on class proportion
        
        # Return metrics
        return {
            'auc': auc,
            'ap': ap,
        }
    
    def _split_graph_data(self, 
                        graph_data: GraphData, 
                        val_ratio: float = 0.1, 
                        test_ratio: float = 0.1, 
                        temporal_split: bool = True) -> Tuple[GraphData, GraphData, GraphData]:
        """Split graph data into train, validation, and test sets.
        
        Args:
            graph_data: Graph data to split
            val_ratio: Ratio of edges for validation
            test_ratio: Ratio of edges for testing
            temporal_split: Whether to use temporal splitting
            
        Returns:
            Tuple of (train_graph, val_graph, test_graph)
        """
        # Clone graph data for each split to avoid modifying original
        train_graph = self._clone_graph_without_edges(graph_data)
        val_graph = self._clone_graph_without_edges(graph_data)
        test_graph = self._clone_graph_without_edges(graph_data)
        
        # Get edge indices and attributes
        edge_index = graph_data.edge_index
        
        if temporal_split and hasattr(graph_data, 'edge_timestamps') and graph_data.edge_timestamps is not None:
            # Sort edges by timestamp for temporal splitting
            timestamps = graph_data.edge_timestamps
            sorted_indices = torch.argsort(timestamps)
            edge_index = edge_index[:, sorted_indices]
            timestamps = timestamps[sorted_indices]
            
            # Calculate split points
            num_edges = edge_index.size(1)
            train_size = int(num_edges * (1 - val_ratio - test_ratio))
            val_size = int(num_edges * val_ratio)
            
            # Split edges
            train_edges = edge_index[:, :train_size]
            val_edges = edge_index[:, train_size:train_size+val_size]
            test_edges = edge_index[:, train_size+val_size:]
            
            # Split timestamps
            train_timestamps = timestamps[:train_size]
            val_timestamps = timestamps[train_size:train_size+val_size]
            test_timestamps = timestamps[train_size+val_size:]
            
            # Assign edges to each graph
            train_graph.edge_index = train_edges
            val_graph.edge_index = val_edges
            test_graph.edge_index = test_edges
            
            # Assign timestamps to each graph
            train_graph.edge_timestamps = train_timestamps
            val_graph.edge_timestamps = val_timestamps
            test_graph.edge_timestamps = test_timestamps
        else:
            # Random splitting
            num_edges = edge_index.size(1)
            perm = torch.randperm(num_edges)
            edge_index = edge_index[:, perm]
            
            # Calculate split points
            train_size = int(num_edges * (1 - val_ratio - test_ratio))
            val_size = int(num_edges * val_ratio)
            
            # Split edges
            train_edges = edge_index[:, :train_size]
            val_edges = edge_index[:, train_size:train_size+val_size]
            test_edges = edge_index[:, train_size+val_size:]
            
            # Assign edges to each graph
            train_graph.edge_index = train_edges
            val_graph.edge_index = val_edges
            test_graph.edge_index = test_edges
            
            # Split edge attributes if available
            if hasattr(graph_data, 'edge_timestamps') and graph_data.edge_timestamps is not None:
                timestamps = graph_data.edge_timestamps[perm]
                train_graph.edge_timestamps = timestamps[:train_size]
                val_graph.edge_timestamps = timestamps[train_size:train_size+val_size]
                test_graph.edge_timestamps = timestamps[train_size+val_size:]
        
        return train_graph, val_graph, test_graph
    
    def _clone_graph_without_edges(self, graph_data: GraphData) -> GraphData:
        """Clone a graph data object without edges.
        
        Creates a copy of the graph with all node-related attributes but no edges.
        
        Args:
            graph_data: Graph data to clone
            
        Returns:
            New GraphData object with same node-related attributes
        """
        # Collect node-related attributes
        attrs = {}
        
        # Copy x if available
        if hasattr(graph_data, 'x') and graph_data.x is not None:
            attrs['x'] = graph_data.x.clone()
        
        # Copy node timestamps if available
        if hasattr(graph_data, 'node_timestamps') and graph_data.node_timestamps is not None:
            attrs['node_timestamps'] = graph_data.node_timestamps.clone()
        elif hasattr(graph_data, 'paper_times') and graph_data.paper_times is not None:
            attrs['paper_times'] = graph_data.paper_times.clone()
        
        # Copy any other node-related attributes
        for key, value in vars(graph_data).items():
            if key not in ['x', 'edge_index', 'edge_attr', 'edge_timestamps', 'node_timestamps', 'paper_times']:
                if isinstance(value, torch.Tensor):
                    attrs[key] = value.clone()
                else:
                    attrs[key] = value
        
        # Create new graph with empty edge index
        new_graph = GraphData(**attrs)
        
        return new_graph
    
    def predict(self, 
              graph_data: GraphData, 
              k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict top-k most likely citation links for a graph.
        
        Args:
            graph_data: Graph data to predict on
            k: Number of top predictions to return
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted edge indices [2, k] and scores [k]
        """
        # Ensure model is in evaluation mode
        self.encoder.eval()
        self.predictor.eval()
        
        # Move data to device
        graph_data = self._move_to_device(graph_data)
        
        with torch.no_grad():
            # Encode graph
            node_embeddings = self.encoder(graph_data)
            
            # Handle different output formats from encoder
            if isinstance(node_embeddings, tuple) or isinstance(node_embeddings, list):
                node_embeddings = node_embeddings[0]
            
            # If node_embeddings has 3 dimensions, take first batch
            if node_embeddings.dim() == 3:
                node_embeddings = node_embeddings[0]
            
            # Use predictor's predict_citations method if available
            if hasattr(self.predictor, 'predict_citations'):
                edge_indices, scores = self.predictor.predict_citations(
                    node_embeddings=node_embeddings,
                    existing_graph=graph_data,
                    k=k
                )
                return edge_indices, scores
            
            # Otherwise, implement prediction ourselves
            # Generate candidate edges
            candidate_edges = self._generate_candidate_edges(graph_data, max_candidates=10000)
            
            # Get embeddings for candidates
            src_emb = node_embeddings[candidate_edges[0]]
            dst_emb = node_embeddings[candidate_edges[1]]
            
            # Predict scores
            scores = self.predictor(src_emb, dst_emb)
            
            # Get top-k predictions
            if k > scores.size(0):
                k = scores.size(0)
                self.logger.warning(f"Requested {k} predictions but only {scores.size(0)} candidates available")
                
            _, indices = torch.topk(scores, k)
            top_edges = candidate_edges[:, indices]
            top_scores = scores[indices]
            
            return top_edges, top_scores
    
    def _generate_candidate_edges(self, 
                                graph_data: GraphData, 
                                max_candidates: int = 10000) -> torch.Tensor:
        """Generate candidate edges for prediction.
        
        Args:
            graph_data: Graph data
            max_candidates: Maximum number of candidates to generate
            
        Returns:
            Tensor of candidate edge indices [2, num_candidates]
        """
        # Try to use the predictor's method first
        if hasattr(self.predictor, 'get_candidate_edges'):
            return self.predictor.get_candidate_edges(graph_data, max_candidates)
        
        # Otherwise, implement ourselves
        # Get number of nodes
        if hasattr(graph_data, 'x') and graph_data.x is not None:
            num_nodes = graph_data.x.size(0)
        elif hasattr(graph_data, 'node_timestamps') and graph_data.node_timestamps is not None:
            num_nodes = len(graph_data.node_timestamps)
        elif hasattr(graph_data, 'paper_times') and graph_data.paper_times is not None:
            num_nodes = len(graph_data.paper_times)
        else:
            num_nodes = graph_data.edge_index.max().item() + 1
            
        # Create a mask for existing edges
        edge_index = graph_data.edge_index
        
        # Convert edge_index to a set of tuples for fast lookup
        existing_edges = set()
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            existing_edges.add((src, dst))
        
        # Check if we have temporal information
        has_temporal = False
        if hasattr(graph_data, 'node_timestamps') and graph_data.node_timestamps is not None:
            node_timestamps = graph_data.node_timestamps
            has_temporal = True
        elif hasattr(graph_data, 'paper_times') and graph_data.paper_times is not None:
            node_timestamps = graph_data.paper_times
            has_temporal = True
            
        # Generate candidate edges
        candidates = []
        max_attempts = max_candidates * 10  # Avoid infinite loop
        attempts = 0
        
        while len(candidates) < max_candidates and attempts < max_attempts:
            attempts += 1
            
            # Sample random source and destination nodes
            src = torch.randint(0, num_nodes, (1,), device=self.device).item()
            dst = torch.randint(0, num_nodes, (1,), device=self.device).item()
            
            # Skip self-loops and existing edges
            if src == dst or (src, dst) in existing_edges:
                continue
                
            # If using temporal constraints, ensure temporal causality
            if has_temporal:
                # Skip if destination was published after source (can't cite future papers)
                if node_timestamps[dst] > node_timestamps[src]:
                    continue
                    
            # Add to candidates
            candidates.append((src, dst))
            existing_edges.add((src, dst))  # Avoid duplicates
            
        # If we couldn't generate enough candidates, use what we have
        if len(candidates) < max_candidates:
            self.logger.warning(
                f"Could only generate {len(candidates)}/{max_candidates} candidate edges"
            )
            
        # Convert to tensor
        candidate_edges = torch.tensor(candidates, dtype=torch.long, device=self.device).t()
        return candidate_edges 