#!/usr/bin/env python
"""
Integrated Citation Network Model Training Script

This script provides a training pipeline for the IntegratedCitationModel, which
combines HyperbolicEncoder, AttentionPredictor, and CVAEGenerator into a
unified model that can perform both link prediction and paper generation tasks.

The script now supports a new training mode (scheme 4) which directly trains
the encoder and generator, bypassing the encoder+predictor phase.

Usage:
    python train_integrated_model.py --dataset data/test_dataset.json
                                     --embed_dim 128
                                     --batch_size 64
                                     --epochs 30
                                     --training_scheme 4  # Direct encoder+generator training
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import copy
import traceback
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import pickle

# Configure file logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_debug.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to path if needed
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Add data_utils directory to path if needed
data_utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_utils')
if data_utils_path not in sys.path:
    sys.path.append(data_utils_path)

# Import model and data utilities
from src.data.dataset import GraphData
from data_utils.citation_data_loading import (
    load_graph_data,
    create_train_val_test_split,
    load_embedding_dictionaries
)
from src.data.temporal_data_utils import (
    create_temporal_mask,
    create_temporal_snapshot,
    split_graph_by_time,
    create_temporal_training_data
)
from src.models.integrated_citation_model import IntegratedCitationModel
from src.models.generator.evaluation import (
    evaluate_generation,
    GenerationEvaluator,
    TemporalGenerationEvaluator
)
from src.models.encoder.hyperbolic_encoder import HyperbolicEncoder
from src.models.predictors.attention_predictor import EnhancedAttentionPredictor
# Import our new generator and evaluator implementations
from src.models.generator.cvae_generator_wrapper import CVAEGenerator
from src.models.generator.generator_evaluator import GeneratorEvaluator
from src.evaluation.temporal_evaluation import TemporalEvaluator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train integrated citation model')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset file')
    parser.add_argument('--embedding_dict', type=str, default=None,
                        help='Path to embedding dictionaries file')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save output')
    
    # Shared model arguments (for backward compatibility)
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Shared embedding dimension (if not using module-specific dims)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Shared hidden dimension (if not using module-specific dims)')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Shared latent dimension (if not using module-specific dims)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--curvature', type=float, default=1.0,
                        help='Hyperbolic curvature parameter')
    parser.add_argument('--kl_weight', type=float, default=0.1,
                        help='Weight for KL divergence loss')
    parser.add_argument('--use_hierarchical', action='store_true',
                        help='Use hierarchical encoding for documents')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Module-specific architecture arguments
    # Encoder
    parser.add_argument('--encoder_hidden_dim', type=int, default=None,
                        help='Hidden dimension specifically for encoder (uses --hidden_dim if not set)')
    parser.add_argument('--encoder_embed_dim', type=int, default=None,
                        help='Output embedding dimension for encoder (uses --embed_dim if not set)')
    parser.add_argument('--num_encoder_layers', type=int, default=2,
                        help='Number of layers in encoder')
    
    # Predictor
    parser.add_argument('--predictor_hidden_dim', type=int, default=None,
                        help='Hidden dimension specifically for predictor (uses --hidden_dim if not set)')
    parser.add_argument('--predictor_heads', type=int, default=None,
                        help='Number of attention heads in predictor (uses --num_heads if not set)')
    parser.add_argument('--num_predictor_layers', type=int, default=2,
                        help='Number of layers in predictor')
    
    # Generator
    parser.add_argument('--generator_hidden_dim', type=int, default=256,
                        help='Hidden dimension for generator')
    parser.add_argument('--generator_latent_dim', type=int, default=128,
                        help='Latent dimension for generator CVAE')
    parser.add_argument('--num_generator_layers', type=int, default=2,
                        help='Number of layers in generator')
    parser.add_argument('--n_pca_components', type=float, default=0.95,
                        help='Number of PCA components to use (or variance to retain)')
    parser.add_argument('--beta_warmup_steps', type=int, default=1000,
                        help='Steps for KL annealing')
    parser.add_argument('--standardize_features', action='store_true',
                        help='Whether to standardize features before PCA')
    
    # Projection matrix approach arguments
    parser.add_argument('--projection_pooling', type=str, default='combined',
                        choices=['mean', 'max', 'weighted', 'combined'],
                        help='Pooling method for projection matrix approach')
    parser.add_argument('--projection_layers', type=int, default=2,
                        help='Number of layers in the projection network')
    parser.add_argument('--timestamp_embed_dim', type=int, default=16,
                        help='Dimension for timestamp embeddings')
    parser.add_argument('--use_layer_norm', action='store_true',
                        help='Use layer normalization in projection networks')
    parser.add_argument('--global_embed_multiplier', type=int, default=12,
                        help='Multiplier for global embedding dimension (default: 12x condition_dim)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--phase_1_lr', type=float, default=0.001,
                        help='Learning rate for phase 1 (link prediction)')
    parser.add_argument('--phase_2_lr', type=float, default=0.001,
                        help='Learning rate for phase 2 (generation)')
    parser.add_argument('--encoder_lr_ratio', type=float, default=0.1,
                        help='Ratio of encoder learning rate to generator learning rate in phase 2 (for scheme 3)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test ratio')
    parser.add_argument('--temporal_split', action='store_true',
                        help='Use temporal split instead of random')
    parser.add_argument('--link_pred_weight', type=float, default=1.0,
                        help='Weight for link prediction loss')
    parser.add_argument('--generation_weight', type=float, default=1.0,
                        help='Weight for generation loss')
    parser.add_argument('--time_split_ratio', type=float, default=0.8,
                        help='Ratio of data to use for temporal split')
    parser.add_argument('--future_window', type=float, default=None,
                        help='Future time window (years) for temporal split')
    parser.add_argument('--num_generated_papers', type=int, default=10,
                        help='Number of papers to generate in evaluation')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')
    
    # Training scheme
    parser.add_argument('--training_scheme', type=int, default=3, choices=[1, 2, 3, 4],
                        help='Training scheme: 1=joint, 2=alternating, 3=sequential, 4=direct encoder+generator')
    parser.add_argument('--phase_1_epochs', type=int, default=20,
                        help='Number of epochs for phase 1 (link prediction) in sequential training')
    parser.add_argument('--phase_2_epochs', type=int, default=20,
                        help='Number of epochs for phase 2 (generation) in sequential training')
    parser.add_argument('--initial_epochs', type=int, default=20,
                        help='DEPRECATED: Use phase_1_epochs and phase_2_epochs instead')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--evaluation_interval', type=int, default=5,
                        help='Epoch interval for detailed generator evaluation')
    parser.add_argument('--mask_ratio', type=float, default=0.1,
                        help='Proportion of nodes to mask for generator evaluation')
    
    # New masking strategy parameters
    parser.add_argument('--use_temporal_mask', action='store_true',
                        help='Use temporal masking instead of random masking for generator evaluation')
    parser.add_argument('--mask_start_ratio', type=float, default=0.8,
                        help='Starting proportion of nodes to mask in the first epoch (gradually decreases)')
    parser.add_argument('--mask_end_ratio', type=float, default=0.1,
                        help='Final proportion of nodes to mask in the last epoch')
    
    # Add temporal evaluation arguments
    temporal_eval_group = parser.add_argument_group('Temporal Evaluation')
    temporal_eval_group.add_argument('--temporal_eval', action='store_true',
                        help='Enable temporal hold-out evaluation')
    temporal_eval_group.add_argument('--temporal_threshold', type=float, default=None,
                        help='Time threshold for temporal split (default: median timestamp)')
    temporal_eval_group.add_argument('--eval_num_papers', type=int, default=20,
                        help='Number of papers to generate for evaluation')
    temporal_eval_group.add_argument('--eval_temperature', type=float, default=1.0,
                        help='Temperature for generation during evaluation')
    temporal_eval_group.add_argument('--temporal_eval_freq', type=int, default=5,
                        help='Frequency (in epochs) to run temporal evaluation')
    
    # Add a new argument in the parse_args function
    parser.add_argument('--disable_reconstruction', action='store_true',
                        help='Disable reconstruction loss in generation (focus on citation patterns only)')
    
    args = parser.parse_args()
    
    # Set module-specific dimensions if not provided
    if args.encoder_hidden_dim is None:
        args.encoder_hidden_dim = args.hidden_dim
    if args.encoder_embed_dim is None:
        args.encoder_embed_dim = args.embed_dim
    if args.predictor_hidden_dim is None:
        args.predictor_hidden_dim = args.hidden_dim
    if args.predictor_heads is None:
        args.predictor_heads = args.num_heads
    if args.generator_hidden_dim is None:
        args.generator_hidden_dim = args.hidden_dim
    if args.generator_latent_dim is None:
        args.generator_latent_dim = args.latent_dim
        
    return args


def load_and_prepare_data(args):
    """Load and prepare data for training.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple containing:
        - Full graph data
        - Dictionary of data splits and temporal information
    """
    # Load the graph data using citation_data_loading utilities
    try:
        graph_data = load_graph_data(
        file_path=args.dataset,
        embedding_dict_path=args.embedding_dict
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None
    
    # Move to the specified device
    device = torch.device(args.device)
    graph_data = graph_data.to(device)
    
    logger.info(f"Loaded graph with {graph_data.x.size(0)} nodes and {graph_data.edge_index.size(1)} edges")
    
    # Check if node timestamps are available
    if not hasattr(graph_data, 'node_timestamps') or graph_data.node_timestamps is None:
        # Use node_timestamp if available (from citation_data_loading.py)
        if hasattr(graph_data, 'node_timestamp') and graph_data.node_timestamp is not None:
            graph_data.node_timestamps = graph_data.node_timestamp
        else:
            logger.warning("No timestamp information found. Using synthetic timestamps.")
            # Create synthetic timestamps (publication years)
            num_nodes = graph_data.x.size(0)
            # Span 10 years
            start_year = 2010
            timestamps = torch.linspace(start_year, start_year + 10, num_nodes)
            graph_data.node_timestamps = timestamps.to(device)
    
    # Create train/val/test splits for link prediction
    train_graph, val_graph, test_graph = create_train_val_test_split(
        graph_data=graph_data,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        temporal_split=args.temporal_split
    )
    
    # Create positive and negative edge indices for training
    train_pos_edge_index, train_neg_edge_index = create_edge_indices(train_graph)
    val_pos_edge_index, val_neg_edge_index = create_edge_indices(val_graph)
    test_pos_edge_index, test_neg_edge_index = create_edge_indices(test_graph)
    
    # Calculate time threshold for generation task (past/future split)
    # This is based on the time_split_ratio parameter
    time_split_data = {}
    
    if hasattr(graph_data, 'node_timestamps') and graph_data.node_timestamps is not None:
        time_values = graph_data.node_timestamps.cpu().numpy()
        time_values.sort()
        threshold_idx = int(len(time_values) * args.time_split_ratio)
        time_threshold = time_values[threshold_idx]
        
        # Calculate future window if not specified
        if args.future_window is None:
            future_window = time_values[-1] - time_threshold
        else:
            future_window = args.future_window
            
        logger.info(f"Time threshold: {time_threshold}, Future window: {future_window}")
        
        # Split graph into past and future parts
        past_graph, future_graph = split_graph_by_time(
            graph=graph_data,
            time_threshold=time_threshold,
            future_window=future_window
        )
        
        logger.info(f"Past graph: {past_graph.x.size(0)} nodes, Future graph: {future_graph.x.size(0)} nodes")
        
        # Store future features and timestamps for generator training
        future_features = future_graph.x if hasattr(future_graph, 'x') else None
        future_timestamps = future_graph.node_timestamps if hasattr(future_graph, 'node_timestamps') else None
        
        # Add to time split data
        time_split_data.update({
            'past_graph': past_graph,
            'future_graph': future_graph,
            'future_features': future_features,
            'future_timestamps': future_timestamps,
            'time_threshold': time_threshold,
            'future_window': future_window
        })
    else:
        logger.warning("No timestamps available for temporal generation task.")
        time_split_data.update({
            'past_graph': graph_data,
            'future_graph': None,
            'future_features': None,
            'future_timestamps': None,
            'time_threshold': None,
            'future_window': None
        })
    
    # Add edge indices to time split data
    time_split_data.update({
        'train_edges': (train_pos_edge_index, train_neg_edge_index),
        'val_edges': (val_pos_edge_index, val_neg_edge_index),
        'test_edges': (test_pos_edge_index, test_neg_edge_index)
    })
    
    return graph_data, time_split_data


def create_edge_indices(graph: GraphData):
    """
    Create positive and negative edge indices for link prediction training.
    
    Args:
        graph: Graph data
        
    Returns:
        Tuple containing positive and negative edge indices
    """
    # Use edge_index as positive edges
    pos_edge_index = graph.edge_index
    
    # Create negative edges by sampling random node pairs that don't have an edge
    num_nodes = graph.x.size(0)
    num_neg_samples = pos_edge_index.size(1)
    
    # Creating a set of existing edges for fast lookup
    existing_edges = set()
    for i in range(pos_edge_index.size(1)):
        src, dst = pos_edge_index[0, i].item(), pos_edge_index[1, i].item()
        existing_edges.add((src, dst))
    
    # Sample negative edges
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        # Random node pair
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        
        # Avoid self-loops and existing edges
        if src != dst and (src, dst) not in existing_edges:
            neg_edges.append([src, dst])
            existing_edges.add((src, dst))  # Add to existing to avoid duplicates
    
    # Convert to tensor
    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long, device=graph.x.device).t()
    
    return pos_edge_index, neg_edge_index


def initialize_model(args, graph_data):
    """Initialize the integrated citation model.
    
    Args:
        args: Command-line arguments
        graph_data: Graph data
        
    Returns:
        IntegratedCitationModel: The initialized model
    """
    # Get dimensionality of node features
    node_feature_dim = graph_data.x.size(1)
    num_nodes = graph_data.num_nodes
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Initialize encoder
    encoder = HyperbolicEncoder(
        node_dim=node_feature_dim,
        hidden_dim=args.encoder_hidden_dim,
        embed_dim=args.encoder_embed_dim,
        num_layers=args.num_encoder_layers,
        curvature=args.curvature,
        dropout=args.dropout
    )
    
    # Initialize predictor
    predictor = EnhancedAttentionPredictor(
        input_dim=args.encoder_embed_dim,
        hidden_dim=args.predictor_hidden_dim,
        num_layers=args.num_predictor_layers,
        num_heads=args.predictor_heads,
        dropout=args.dropout,
        edge_dim=None  # No edge features for now
    )
    
    # Initialize generator - use our new CVAEGenerator with projection matrix approach
    generator = CVAEGenerator(
        embed_dim=args.encoder_embed_dim,  # Match encoder output dim
        node_feature_dim=node_feature_dim,
        condition_dim=args.encoder_embed_dim,  # Match encoder output dim
        latent_dim=args.generator_latent_dim,
        hidden_dims=[args.generator_hidden_dim, args.generator_hidden_dim],
        n_pca_components=args.n_pca_components,
        kl_weight=args.kl_weight,
        beta_warmup_steps=args.beta_warmup_steps,
        dropout=args.dropout,
        standardize_features=args.standardize_features,
        # Projection matrix approach parameters
        projection_pooling=args.projection_pooling,
        projection_layers=args.projection_layers,
        timestamp_embed_dim=args.timestamp_embed_dim,
        use_layer_norm=args.use_layer_norm,
        global_embed_multiplier=args.global_embed_multiplier
    )
    
    # Log special note about the projection matrix approach
    logger.info("Using projection matrix approach for conditioning:")
    logger.info("  - Creates fixed-dimensional condition vectors independent of graph size")
    logger.info("  - Handles batch size mismatches between node embeddings and generated papers")
    logger.info("  - Uses global graph pooling operations (mean, max, weighted) for robust conditioning")
    logger.info("  - Projects pooled representations to condition dimension via learnable projections")
    
    # Initialize integrated model
    model = IntegratedCitationModel(
        encoder=encoder,
        predictor=predictor,
        generator=generator,
        embed_dim=args.encoder_embed_dim,  # Use encoder's output dimension
        device=device
    )
    
    # Move model to device
    model = model.to(device)
    
    # Log model initialization
    logger.info(f"Initializing integrated model with {args.encoder_embed_dim} embedding dim, "
                f"{args.encoder_hidden_dim}/{args.predictor_hidden_dim}/{args.generator_hidden_dim} hidden dims, "
                f"{args.generator_latent_dim} latent dim")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    predictor_params = sum(p.numel() for p in model.predictor.parameters())
    generator_params = sum(p.numel() for p in model.generator.parameters())
    
    # Log parameter counts
    logger.info("=" * 50)
    logger.info("MODEL ARCHITECTURE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Encoder parameters: {encoder_params:,} ({encoder_params/total_params:.1%})")
    logger.info(f"Predictor parameters: {predictor_params:,} ({predictor_params/total_params:.1%})")
    logger.info(f"Generator parameters: {generator_params:,} ({generator_params/total_params:.1%})")
    logger.info("=" * 50)
    
    # Detailed parameter breakdown by module and layer
    logger.info("\nDETAILED PARAMETER BREAKDOWN")
    logger.info("-" * 30)
    
    # Function to log parameters by layer
    def log_module_parameters(module, module_name, indent=0):
        if module is None:
            logger.info(f"{' ' * indent}{module_name}: None (not yet initialized)")
            return
            
        total_module_params = sum(p.numel() for p in module.parameters() if p is not None)
        prefix = ' ' * indent
        logger.info(f"{prefix}{module_name}: {total_module_params:,} parameters")
        
        # Special logging for CVAEGenerator which has lazy initialization
        if module.__class__.__name__ == 'CVAEGenerator':
            logger.info(f"{prefix}  - feature_preprocessor: initialized={module.feature_preprocessor.fitted}")
            if hasattr(module, 'cvae') and module.cvae is not None:
                cvae_params = sum(p.numel() for p in module.cvae.parameters())
                logger.info(f"{prefix}  - cvae: {cvae_params:,} parameters ({cvae_params/total_module_params:.1%} of module)")
            else:
                logger.info(f"{prefix}  - cvae: Not yet initialized (lazy initialization)")
            
            logger.info(f"{prefix}  - Note: Full parameters will be known after first forward pass")
            return
            
        # Special detailed logging for EnhancedCVAE which has the large output layers
        if module.__class__.__name__ == 'EnhancedCVAE' and module is not None:
            if hasattr(module, 'decoder') and module.decoder is not None:
                decoder_params = sum(p.numel() for p in module.decoder.parameters())
                logger.info(f"{prefix}  - decoder: {decoder_params:,} parameters ({decoder_params/total_module_params:.1%} of module)")
                
            if hasattr(module, 'encoder') and module.encoder is not None:
                encoder_params = sum(p.numel() for p in module.encoder.parameters())
                logger.info(f"{prefix}  - encoder: {encoder_params:,} parameters ({encoder_params/total_module_params:.1%} of module)")
        
        # Recursively log child modules
        for name, child in module.named_children():
            if child is None:
                logger.info(f"{prefix}  - {name}: None (not initialized)")
                continue
                
            child_params = sum(p.numel() for p in child.parameters() if p is not None)
            if child_params > 0:  # Only log modules with parameters
                percent = child_params / total_module_params * 100 if total_module_params > 0 else 0
                logger.info(f"{prefix}  - {name}: {child_params:,} parameters ({percent:.1f}% of module)")
                
                # Go one level deeper for important modules
                if child_params > 10000:  # Only expand modules with significant parameters
                    for subname, subchild in child.named_children():
                        if subchild is None:
                            continue
                        subchild_params = sum(p.numel() for p in subchild.parameters() if p is not None)
                        if subchild_params > 0:
                            subpercent = subchild_params / child_params * 100
                            logger.info(f"{prefix}    - {subname}: {subchild_params:,} parameters ({subpercent:.1f}% of parent)")
    
        # Log detailed parameters for each major component
        logger.info("\nENCODER COMPONENT")
        logger.info("-" * 30)
        log_module_parameters(model.encoder, "Encoder")
        
        logger.info("\nPREDICTOR COMPONENT")
        logger.info("-" * 30)
        log_module_parameters(model.predictor, "Predictor")
        
        logger.info("\nGENERATOR COMPONENT")
        logger.info("-" * 30)
        log_module_parameters(model.generator, "Generator")
        
        # Special note about lazy initialization
        logger.info("\nNOTE ON GENERATOR INITIALIZATION")
        logger.info("-" * 30)
        logger.info("The CVAEGenerator uses lazy initialization and will be fully initialized")
        logger.info("during the first forward pass when actual node features are provided.")
        logger.info("Parameter counts will be updated then.")
        logger.info("=" * 50)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.phase_1_lr,
        weight_decay=args.weight_decay
    )
    
    # Load checkpoint if provided
    if args.load_checkpoint:
        logger.info(f"Loading model from checkpoint: {args.load_checkpoint}")
        model.load(args.load_checkpoint)
    
    return model, optimizer


def train_epoch(
    model: IntegratedCitationModel,
    graph_data: GraphData,
    train_pos_edge_index: torch.Tensor,
    train_neg_edge_index: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    past_graph: Optional[GraphData] = None,
    future_features: Optional[torch.Tensor] = None,
    future_timestamps: Optional[torch.Tensor] = None,
    task_weights: Dict[str, float] = {'link_prediction': 1.0, 'generation': 1.0}
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        graph_data: The input graph
        train_pos_edge_index: Positive edge indices
        train_neg_edge_index: Negative edge indices
        optimizer: The optimizer
        past_graph: Past graph for generation task
        future_features: Features of future papers
        future_timestamps: Timestamps of future papers
        task_weights: Weights for each task
    
    Returns:
        Dict[str, float]: Dictionary of loss values for this epoch
    """
    # Set model to training mode
    model.train()
    
    # Perform one training step
    try:
        train_metrics = model.train_step(
        graph=graph_data,
        pos_edge_index=train_pos_edge_index,
        neg_edge_index=train_neg_edge_index,
        past_graph=past_graph,
        future_features=future_features,
        future_timestamps=future_timestamps,
        optimizer=optimizer,
        task_weights=task_weights
        )
        logger.info(f"Training metrics: {train_metrics}")
        
        # Fix for infinite loss - ensure combined loss is valid
        train_loss = train_metrics.get('combined', 0.0)
        if not torch.isfinite(torch.tensor(train_loss)) or train_loss > 1e6:
            logger.warning(f"Non-finite or extremely large train loss detected: {train_loss}. Using component losses instead.")
            # Calculate combined loss from individual components with proper scaling
            link_pred_loss = train_metrics.get('link_prediction', 0.0)
            gen_loss = train_metrics.get('generation', 0.0)
            
            # Only include losses that are finite
            valid_losses = []
            if torch.isfinite(torch.tensor(link_pred_loss)) and link_pred_loss < 1e6:
                valid_losses.append(link_pred_loss * task_weights.get('link_prediction', 1.0))
            if torch.isfinite(torch.tensor(gen_loss)) and gen_loss < 1e6:
                valid_losses.append(gen_loss * task_weights.get('generation', 1.0))
            
            if valid_losses:
                train_loss = sum(valid_losses) / len(valid_losses)
            else:
                train_loss = 1.0  # Fallback to a reasonable default
            
            # Update metrics with corrected combined loss
            train_metrics['combined'] = train_loss
            logger.info(f"Corrected train loss: {train_loss}")
        
        return train_metrics
    except Exception as e:
        logger.error(f"Error in train_epoch: {e}")
        logger.error(traceback.format_exc())
        # Return default values to avoid breaking the training loop
        return {'combined': float('inf'), 'link_prediction': float('inf'), 'generation': float('inf')}


def convert_to_serializable(obj):
    """
    Convert any non-serializable objects (like tensors or numpy arrays) to serializable types.
    
    Args:
        obj: Any Python object
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    else:
        # For other types, convert to string
        try:
            return str(obj)
        except:
            return "UNSERIALIZABLE_OBJECT"

def get_citation_loss(metrics):
    """
    Extract the citation loss from metrics. This is important for Phase 2
    where we focus more on generating papers with realistic citation patterns.
    
    Args:
        metrics: Dictionary of metrics from training/validation step
        
    Returns:
        citation_loss: The citation loss value as a float
    """
    # Check if we have citation loss in the metrics
    citation_loss = 0.0
    
    if 'citation_loss' in metrics:
        citation_loss = metrics['citation_loss']
    elif 'gen_citation_loss' in metrics:
        citation_loss = metrics['gen_citation_loss']
    
    # Convert to float if tensor
    if isinstance(citation_loss, torch.Tensor):
        citation_loss = citation_loss.item()
        
    return citation_loss


def train_model(
    model: IntegratedCitationModel,
    graph_data: GraphData,
    train_pos_edge_index: torch.Tensor,
    train_neg_edge_index: torch.Tensor,
    val_pos_edge_index: torch.Tensor,
    val_neg_edge_index: torch.Tensor,
    past_graph: Optional[GraphData] = None,
    future_features: Optional[torch.Tensor] = None,
    future_timestamps: Optional[torch.Tensor] = None,
    epochs: int = 50,
    phase_1_lr: float = 0.001,
    phase_2_lr: float = 0.001,
    weight_decay: float = 1e-5,
    early_stopping: bool = False,
    patience: int = 5,
    task_weights: Dict[str, float] = {'link_prediction': 1.0, 'generation': 1.0},
    training_scheme: int = 3,
    initial_epochs: int = 20,
    phase_1_epochs: int = 20,
    phase_2_epochs: int = 20,
    evaluation_interval: int = 5,
    mask_ratio: float = 0.1,
    args = None,
    device_manager = None,
    # Temporal evaluation parameters
    temporal_eval: bool = False,
    temporal_eval_freq: int = 5,
    time_threshold: Optional[float] = None,
    future_window: Optional[float] = None,
    eval_num_papers: int = 20,
    eval_temperature: float = 1.0
) -> Tuple[IntegratedCitationModel, Dict[str, Any]]:
    """
    Train the integrated model.
    
    Args:
        model: The model to train
        graph_data: The graph data
        train_pos_edge_index: Positive edge indices for training
        train_neg_edge_index: Negative edge indices for training
        val_pos_edge_index: Positive edge indices for validation
        val_neg_edge_index: Negative edge indices for validation
        past_graph: Past graph for generation
        future_features: Features of future papers
        future_timestamps: Timestamps of future papers
        epochs: Number of epochs
        phase_1_lr: Learning rate for phase 1 (link prediction)
        phase_2_lr: Learning rate for phase 2 (generation)
        weight_decay: Weight decay
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        task_weights: Weights for each task
        training_scheme: Training scheme (1-4)
        initial_epochs: Number of epochs for initial training (deprecated)
        phase_1_epochs: Number of epochs for phase 1 (link prediction)
        phase_2_epochs: Number of epochs for phase 2 (generation)
        evaluation_interval: Epoch interval for detailed generator evaluation
        mask_ratio: Proportion of nodes to mask for generator evaluation
        args: Command-line arguments
        device_manager: Device manager for device handling
        temporal_eval: Whether to perform temporal evaluation
        temporal_eval_freq: Frequency for temporal evaluation
        time_threshold: Time threshold for temporal split
        future_window: Time window for future papers
        eval_num_papers: Number of papers to generate for evaluation
        eval_temperature: Temperature for generation during evaluation
        
    Returns:
        Tuple of (trained model, training history)
    """
    device = next(model.parameters()).device
    logger.info(f"Training model on device: {device}")
    
    # Handle backward compatibility for initial_epochs
    if phase_1_epochs == 20 and initial_epochs != 20:  # Default value for phase_1_epochs and non-default for initial_epochs
        logger.warning("Using 'initial_epochs' for phase 1 (backward compatibility). Consider using 'phase_1_epochs' instead.")
        phase_1_epochs = initial_epochs
    
    # Initialize optimizer and scheduler based on training scheme
    if training_scheme == 1:
        # Link prediction only - optimize encoder and predictor
        optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + 
            list(model.predictor.parameters()),
            lr=phase_1_lr, 
            weight_decay=weight_decay
        )
    elif training_scheme == 2:
        # Generation only - optimize encoder and generator
        optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + 
            list(model.generator.parameters()),
            lr=phase_2_lr, 
            weight_decay=weight_decay
        )
    else:
        # Initial phase - optimize all parameters
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=phase_1_lr,
            weight_decay=weight_decay
        )
        
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=patience//2, factor=0.5, verbose=True
    )
    
    # Track best model
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    early_stop_counter = 0
    
    # Track best model based on temporal evaluation
    best_temporal_score = 0 if temporal_eval else None
    best_temporal_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'link_pred_loss': [],
        'val_link_pred_loss': [],
        'gen_loss': [],
        'val_gen_loss': [],
        'citation_loss': [],  # Add citation loss tracking
        'val_citation_loss': [],  # Add validation citation loss tracking
        'learning_rate': [],
        'temporal_eval': [] if temporal_eval else None  # Add temporal evaluation tracking
    }
    
    # Initialize temporal evaluator if needed
    if temporal_eval:
        logger.info("Initializing temporal evaluator...")
        
        # Set default time threshold if not provided
        if time_threshold is None:
            if hasattr(graph_data, 'node_timestamps') and graph_data.node_timestamps is not None:
                time_threshold = torch.median(graph_data.node_timestamps).item()
                logger.info(f"Using median timestamp as threshold: {time_threshold}")
            else:
                time_threshold = 0.5
                logger.warning(f"No timestamps found, using default threshold: {time_threshold}")
                
        # Perform initial temporal evaluation
        logger.info("Performing initial temporal evaluation...")
        initial_results = perform_temporal_evaluation(
                            model=model,
            graph_data=graph_data,
            args=argparse.Namespace(
                temporal_threshold=time_threshold,
                future_window=future_window,
                eval_num_papers=eval_num_papers,
                eval_temperature=eval_temperature
            ),
            device_manager=device_manager,
            epoch=0
        )
        
        # Add to history
        history['temporal_eval'].append({
            'epoch': 0,
            'results': initial_results
        })
        
        # Initialize best temporal score
        best_temporal_score = initial_results['overall_score']
        best_temporal_model_state = copy.deepcopy(model.state_dict())
    
    # Implement training based on scheme
    if training_scheme == 3:  # Sequential training scheme
        logger.info(f"Using training scheme {training_scheme}: Sequential training (Phase 1: Link Prediction, Phase 2: Generation)")
        
        # Phase 1: Train encoder and predictor for link prediction
        logger.info(f"Phase 1: Training encoder and predictor for {phase_1_epochs} epochs...")
        
        # Configure Phase 1 training
        phase1_optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.predictor.parameters()),
            lr=phase_1_lr,
            weight_decay=weight_decay
            )
            
        phase1_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            phase1_optimizer, 'min', patience=patience//2, factor=0.5, verbose=True
        )
        
        phase1_task_weights = {'link_prediction': 1.0, 'generation': 0.0}
        
        # Track best model for Phase 1
        phase1_best_val_loss = float('inf')
        phase1_best_model_state = None
        phase1_early_stop_counter = 0
            
                # Phase 1 training loop
        for epoch in range(1, phase_1_epochs + 1):
            # Train for one epoch
            model.train()
            train_metrics = model.train_step(
                graph=graph_data,
                pos_edge_index=train_pos_edge_index,
                neg_edge_index=train_neg_edge_index,
                        optimizer=phase1_optimizer,
                        task_weights=phase1_task_weights
                    )
            train_loss = train_metrics.get('combined', float('inf'))
            
            # Validate
            model.eval()
            with torch.no_grad():
                val_metrics = model.validation_step(
                    graph=graph_data,
                    val_pos_edge_index=val_pos_edge_index,
                    val_neg_edge_index=val_neg_edge_index,
                        task_weights=phase1_task_weights
                    )
            val_loss = val_metrics.get('combined', float('inf'))
            
            # Update scheduler
            phase1_scheduler.step(val_loss)
            current_lr = phase1_optimizer.param_groups[0]['lr']
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['link_pred_loss'].append(train_metrics.get('link_prediction', 0))
            history['val_link_pred_loss'].append(val_metrics.get('link_prediction', 0))
            history['learning_rate'].append(current_lr)
            
                    # Log progress
            logger.info(f"Phase 1 - Epoch {epoch}/{phase_1_epochs} - "
                      f"Train loss: {train_loss:.4f}, "
                      f"Val loss: {val_loss:.4f}, "
                      f"LR: {current_lr:.6f}")
            
            # Save best model
            if val_loss < phase1_best_val_loss and torch.isfinite(torch.tensor(val_loss)):
                phase1_best_val_loss = val_loss
                phase1_best_model_state = copy.deepcopy(model.state_dict())
                phase1_early_stop_counter = 0
                logger.info(f"New best model in Phase 1! Val loss: {phase1_best_val_loss:.4f}")
            else:
                phase1_early_stop_counter += 1
            
            # Early stopping
            if early_stopping and phase1_early_stop_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs in Phase 1")
                break
                        
            # Temporal evaluation is skipped in Phase 1 (focusing only on link prediction)
            # The model's generation capabilities aren't expected to be meaningful until Phase 2
            """
            # Temporal evaluation if enabled
            if temporal_eval and epoch % temporal_eval_freq == 0:
                logger.info(f"Performing temporal evaluation at Phase 1 epoch {epoch}...")
                temporal_results = perform_temporal_evaluation(
                    model=model,
                    graph_data=graph_data,
                    args=argparse.Namespace(
                        temporal_threshold=time_threshold,
                        future_window=future_window,
                        eval_num_papers=eval_num_papers,
                        eval_temperature=eval_temperature
                    ),
                    device_manager=device_manager,
                    epoch=epoch
                )
                
                # Add to history
                history['temporal_eval'].append({
                    'epoch': epoch,
                    'phase': 1,
                    'results': temporal_results
                })
            """
        
        # Load best model from Phase 1
        if phase1_best_model_state is not None:
            logger.info(f"Loading best model from Phase 1 with validation loss: {phase1_best_val_loss:.4f}")
            model.load_state_dict(phase1_best_model_state)
        
        # Phase 2: Train encoder, predictor, and generator together
        logger.info(f"Phase 2: Training encoder and generator for {phase_2_epochs} epochs...")
        logger.info(f"Freezing predictor module during Phase 2 to focus on generation capabilities")
        
        # Add debugging info about the model state
        print("\n======== DEBUG INFO ========")
        print(f"Model architecture: {model.__class__.__name__}")
        print(f"Encoder: {model.encoder.__class__.__name__} (requires_grad: {any(p.requires_grad for p in model.encoder.parameters())})")
        print(f"Predictor: {model.predictor.__class__.__name__} (requires_grad: {any(p.requires_grad for p in model.predictor.parameters())})")
        print(f"Generator: {model.generator.__class__.__name__} (requires_grad: {any(p.requires_grad for p in model.generator.parameters())})")
        print(f"Past graph: {past_graph is not None}")
        print(f"Future features: {future_features is not None}")
        print(f"Future features shape: {future_features.shape if future_features is not None else 'N/A'}")
        print(f"Task weights: {task_weights}")
        print("===========================\n")
        
        # Freeze predictor parameters
        for param in model.predictor.parameters():
            param.requires_grad = False
            
        # Configure Phase 2 training - only include encoder and generator parameters
        phase2_optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.generator.parameters()),
            lr=phase_2_lr,
                weight_decay=weight_decay
                )
        
        phase2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            phase2_optimizer, 'min', patience=patience//2, factor=0.5, verbose=True
        )
        
        # Use only generation loss for Phase 2, ignoring link prediction
        phase2_task_weights = {'link_prediction': 0.0, 'generation': 1.0}
        logger.info("Phase 2 will use only generation loss (link_prediction weight: 0.0, generation weight: 1.0)")
        
        # Function to extract citation loss from metrics
        def get_citation_loss(metrics):
            citation_loss = 0.0
            if 'gen_metrics' in metrics:
                citation_loss = metrics['gen_metrics'].get('citation_loss', 0.0)
            elif 'citation_loss' in metrics:
                citation_loss = metrics['citation_loss']
                
            if isinstance(citation_loss, torch.Tensor):
                citation_loss = citation_loss.item()
            return citation_loss
        
        # Add citation loss tracking to history
        history['citation_loss'] = []
        history['val_citation_loss'] = []
        
        # Track best model for Phase 2
        phase2_best_val_loss = float('inf')
        phase2_best_model_state = None
        phase2_early_stop_counter = 0
        
        # Phase 2 training loop
        for epoch in range(1, phase_2_epochs + 1):
            # Add debug output
            print(f"\n--- Phase 2 - Epoch {epoch}/{phase_2_epochs} - Starting training ---")
            
            # Train for one epoch
            model.train()
            try:
                print("Calling model.train_step...")
                train_metrics = model.train_step(
                    graph=graph_data,
                    pos_edge_index=train_pos_edge_index,
                    neg_edge_index=train_neg_edge_index,
                    past_graph=past_graph,
                    future_features=future_features,
                    future_timestamps=future_timestamps,
                    optimizer=phase2_optimizer,
                    task_weights=phase2_task_weights
                )
                print(f"Train step completed. Metrics: {train_metrics}")
                train_loss = train_metrics.get('combined', float('inf'))
            except Exception as e:
                print(f"Error in train_step: {str(e)}")
                import traceback
                print(traceback.format_exc())
                train_loss = float('inf')
                train_metrics = {}
            
            # Validate
            model.eval()
            try:
                print("Starting validation...")
                with torch.no_grad():
                    val_metrics = model.validation_step(
                        graph=graph_data,
                        val_pos_edge_index=val_pos_edge_index,
                        val_neg_edge_index=val_neg_edge_index,
                        val_past_graph=past_graph,
                        val_future_features=future_features,
                        val_future_timestamps=future_timestamps,
                        task_weights=phase2_task_weights
                    )
                print(f"Validation completed. Metrics: {val_metrics}")
                val_loss = val_metrics.get('combined', float('inf'))
            except Exception as e:
                print(f"Error in validation_step: {str(e)}")
                import traceback
                print(traceback.format_exc())
                val_loss = float('inf')
                val_metrics = {}
            
            # Update scheduler
            phase2_scheduler.step(val_loss)
            current_lr = phase2_optimizer.param_groups[0]['lr']
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            if 'link_prediction' in train_metrics:
                history['link_pred_loss'].append(train_metrics['link_prediction'])
                history['val_link_pred_loss'].append(val_metrics.get('link_prediction', 0))
                
            if 'generation' in train_metrics:
                history['gen_loss'].append(train_metrics['generation'])
                history['val_gen_loss'].append(val_metrics.get('generation', 0))
                
            # Extract and log citation loss explicitly
            train_citation_loss = get_citation_loss(train_metrics)
            val_citation_loss = get_citation_loss(val_metrics)
            history['citation_loss'].append(train_citation_loss)
            history['val_citation_loss'].append(val_citation_loss)
                
            history['learning_rate'].append(current_lr)
            
            # Log progress with citation loss
            logger.info(f"Phase 2 - Epoch {epoch}/{phase_2_epochs} - "
                      f"Train loss: {train_loss:.4f}, "
                      f"Val loss: {val_loss:.4f}, "
                      f"Citation loss: {train_citation_loss:.4f}, "
                      f"Val Citation loss: {val_citation_loss:.4f}, "
                      f"LR: {current_lr:.6f}")
            
            # Perform detailed generation evaluation at regular intervals
            if epoch % evaluation_interval == 0 or epoch == phase_2_epochs:
                logger.info(f"Performing detailed generation evaluation at Phase 2 epoch {epoch}...")
                
                try:
                    # Create evaluation subset
                    if not 'eval_future_features' in locals():
                        # Use a subset of future features for evaluation
                        num_eval_papers = min(100, len(future_features))
                        eval_indices = torch.randperm(len(future_features))[:num_eval_papers]
                        eval_future_features = future_features[eval_indices]
                        if future_timestamps is not None:
                            eval_future_timestamps = future_timestamps[eval_indices]
                        else:
                            eval_future_timestamps = None
                    
                    # Generate papers
                    model.eval()
                    with torch.no_grad():
                        # Get node embeddings
                        node_embeddings = model.encoder(past_graph)
                        
                        # Generate features
                        generated_features = model.generator.generate(
                            node_embeddings=node_embeddings,
                            num_samples=len(eval_future_features),
                            temperature=eval_temperature
                        )
                        
                        # Compute citation probabilities
                        real_citations = model._predict_citations_for_new_paper(
                            node_embeddings,
                            eval_future_features
                        )
                        
                        generated_citations = model._predict_citations_for_new_paper(
                            node_embeddings,
                            generated_features
                        )
                        
                        # Calculate citation metrics
                        citation_metrics = calculate_citation_metrics(
                            real_citations,
                            generated_citations
                        )
                        
                        # Compute feature metrics
                        feature_diversity = compute_feature_diversity(generated_features)
                        feature_novelty = compute_feature_novelty(generated_features, past_graph.x)
                        
                        # Combine all metrics
                        generation_metrics = {
                            'feature_diversity': feature_diversity,
                            'feature_novelty': feature_novelty,
                            **citation_metrics
                        }
                        
                        # Log metrics
                        logger.info("Generation evaluation metrics:")
                        for key, value in generation_metrics.items():
                            logger.info(f"  - {key}: {value:.4f}")
                        
                        # Add to history
                        if 'generation_metrics' not in history:
                            history['generation_metrics'] = []
                        
                        history['generation_metrics'].append({
                            'epoch': epoch,
                            'phase': 2,
                            'metrics': generation_metrics
                        })
                        
                        # Save generated examples for visualization
                        if args and hasattr(args, 'output_dir') and args.output_dir:
                            # Save a small sample of generations
                            save_sample_size = min(10, len(generated_features))
                            save_sample = {
                                'epoch': epoch,
                                'generated_features': generated_features[:save_sample_size].cpu().numpy(),
                                'real_features': eval_future_features[:save_sample_size].cpu().numpy(),
                                'generated_citations': generated_citations[:save_sample_size].cpu().numpy(),
                                'real_citations': real_citations[:save_sample_size].cpu().numpy()
                            }
                            
                            sample_path = os.path.join(args.output_dir, f"generation_samples_epoch_{epoch}.pkl")
                            with open(sample_path, 'wb') as f:
                                pickle.dump(save_sample, f)
                            logger.info(f"Saved generation samples to {sample_path}")
                            
                except Exception as e:
                    logger.error(f"Error in generation evaluation: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Save best model
            if val_loss < phase2_best_val_loss and torch.isfinite(torch.tensor(val_loss)):
                phase2_best_val_loss = val_loss
                phase2_best_model_state = copy.deepcopy(model.state_dict())
                phase2_early_stop_counter = 0
                logger.info(f"New best model in Phase 2! Val loss: {phase2_best_val_loss:.4f}")
            else:
                phase2_early_stop_counter += 1
            
            # Early stopping
            if early_stopping and phase2_early_stop_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs in Phase 2")
                break
                
            # Temporal evaluation if enabled
            if temporal_eval and epoch % temporal_eval_freq == 0:
                logger.info(f"Performing temporal evaluation at Phase 2 epoch {epoch}...")
                temporal_results = perform_temporal_evaluation(
                    model=model,
                    graph_data=graph_data,
                    args=argparse.Namespace(
                        temporal_threshold=time_threshold,
                        future_window=future_window,
                        eval_num_papers=eval_num_papers,
                        eval_temperature=eval_temperature
                    ),
                    device_manager=device_manager,
                    epoch=epoch
                )
                
                # Add to history
                history['temporal_eval'].append({
                    'epoch': epoch,
                    'phase': 2,
                    'results': temporal_results
                })
                
                # Save best model based on temporal evaluation score
                if temporal_results['overall_score'] > best_temporal_score:
                    logger.info(f"New best model based on temporal evaluation (score: {temporal_results['overall_score']:.4f})")
                    best_temporal_score = temporal_results['overall_score']
                    best_temporal_model_state = copy.deepcopy(model.state_dict())
                    
                    # Save model checkpoint if output directory is specified
                    if args and hasattr(args, 'output_dir') and args.output_dir:
                        temporal_checkpoint_path = os.path.join(args.output_dir, f"best_model_temporal.pt")
                        torch.save({
                            'epoch': epoch,
                            'phase': 2,
                            'model_state_dict': best_temporal_model_state,
                            'temporal_score': best_temporal_score,
                            'args': vars(args) if args else {}
                        }, temporal_checkpoint_path)
                        logger.info(f"Saved best temporal model to {temporal_checkpoint_path}")
        
        # Determine which model to load as the final model
        if phase2_best_model_state is not None:
            logger.info(f"Loading best model from Phase 2 with validation loss: {phase2_best_val_loss:.4f}")
            model.load_state_dict(phase2_best_model_state)
            best_val_loss = phase2_best_val_loss
            best_model_state = phase2_best_model_state
            best_epoch = phase_2_epochs
        else:
            logger.warning("No best model state was saved in Phase 2. Using best model from Phase 1.")
            model.load_state_dict(phase1_best_model_state)
            best_val_loss = phase1_best_val_loss
            best_model_state = phase1_best_model_state
            best_epoch = phase_1_epochs
    
    else:
        # Training loop for other training schemes
        logger.info("Starting training...")
        
        # Function to train for one epoch
        def train_one_epoch(model, optimizer):
            model.train()
            
            # Train step
            logger.info("Starting training step...")
            try:
                train_metrics = model.train_step(
                                graph=graph_data,
                    pos_edge_index=train_pos_edge_index,
                    neg_edge_index=train_neg_edge_index,
                    past_graph=past_graph,
                    future_features=future_features,
                    future_timestamps=future_timestamps,
                    optimizer=optimizer,
                    task_weights=task_weights
                )
                logger.info(f"Training metrics: {train_metrics}")
                
                # Fix for infinite loss - ensure combined loss is valid
                train_loss = train_metrics.get('combined', 0.0)
                if not torch.isfinite(torch.tensor(train_loss)) or train_loss > 1e6:
                    logger.warning(f"Non-finite or extremely large train loss detected: {train_loss}. Using component losses instead.")
                    # Calculate combined loss from individual components with proper scaling
                    link_pred_loss = train_metrics.get('link_prediction', 0.0)
                    gen_loss = train_metrics.get('generation', 0.0)
                    
                    # Only include losses that are finite
                    valid_losses = []
                    if torch.isfinite(torch.tensor(link_pred_loss)) and link_pred_loss < 1e6:
                        valid_losses.append(link_pred_loss * task_weights.get('link_prediction', 1.0))
                    if torch.isfinite(torch.tensor(gen_loss)) and gen_loss < 1e6:
                        valid_losses.append(gen_loss * task_weights.get('generation', 1.0))
                    
                    if valid_losses:
                        train_loss = sum(valid_losses) / len(valid_losses)
                    else:
                        train_loss = 1.0  # Fallback to a reasonable default
                    
                    # Update metrics with corrected combined loss
                    train_metrics['combined'] = train_loss
                    logger.info(f"Corrected train loss: {train_loss}")
                
                return train_metrics
            except Exception as e:
                logger.error(f"Error during training step: {str(e)}")
                logger.error(traceback.format_exc())
                return {'loss': 1.0}  # Use a reasonable default instead of inf
        
        # Function to validate
        def validate(model):
            model.eval()
            
            logger.info("Starting validation step...")
            try:
                with torch.no_grad():
                    val_metrics = model.validation_step(
                        graph=graph_data,
                        val_pos_edge_index=val_pos_edge_index,
                        val_neg_edge_index=val_neg_edge_index,
                        val_past_graph=past_graph,
                        val_future_features=future_features,
                        val_future_timestamps=future_timestamps,
                        task_weights=task_weights
                    )
                logger.info(f"Validation metrics: {val_metrics}")
                
                # Fix for infinite loss - ensure combined loss is valid
                val_loss = val_metrics.get('combined', 0.0)
                if not torch.isfinite(torch.tensor(val_loss)) or val_loss > 1e6:
                    logger.warning(f"Non-finite or extremely large validation loss detected: {val_loss}. Using component losses instead.")
                    # Calculate combined loss from individual components with proper scaling
                    link_pred_loss = val_metrics.get('link_prediction', 0.0)
                    gen_loss = val_metrics.get('generation', 0.0)
                    
                    # Only include losses that are finite
                    valid_losses = []
                    if torch.isfinite(torch.tensor(link_pred_loss)) and link_pred_loss < 1e6:
                        valid_losses.append(link_pred_loss * task_weights.get('link_prediction', 1.0))
                    if torch.isfinite(torch.tensor(gen_loss)) and gen_loss < 1e6:
                        valid_losses.append(gen_loss * task_weights.get('generation', 1.0))
                    
                    if valid_losses:
                        val_loss = sum(valid_losses) / len(valid_losses)
                    else:
                        val_loss = 1.0  # Fallback to a reasonable default
                    
                    # Update metrics with corrected combined loss
                    val_metrics['combined'] = val_loss
                    logger.info(f"Corrected validation loss: {val_loss}")
                
                return val_metrics
            except Exception as e:
                logger.error(f"Error during validation step: {str(e)}")
                logger.error(traceback.format_exc())
                return {'loss': 1.0}  # Use a reasonable default instead of inf
        
        # Training loop
        for epoch in range(1, epochs + 1):
            # Train for one epoch
            train_metrics = train_one_epoch(model, optimizer)
            train_loss = train_metrics.get('combined', float('inf'))
            
            # Validate
            val_metrics = validate(model)
            val_loss = val_metrics.get('combined', float('inf'))
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(current_lr)
            
            if 'link_prediction' in train_metrics:
                history['link_pred_loss'].append(train_metrics['link_prediction'])
                history['val_link_pred_loss'].append(val_metrics.get('link_prediction', 0))
                
            if 'generation' in train_metrics:
                history['gen_loss'].append(train_metrics['generation'])
                history['val_gen_loss'].append(val_metrics.get('generation', 0))
            
            # Log progress with corrected loss values
            logger.info(f"Epoch {epoch}/{epochs} - "
                      f"Train loss: {train_loss:.4f}, "
                      f"Val loss: {val_loss:.4f}, "
                      f"LR: {current_lr:.6f}")
            
            # Check if this is the best model
            if val_loss < best_val_loss and torch.isfinite(torch.tensor(val_loss)):
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                early_stop_counter = 0
                logger.info(f"New best model! Val loss: {best_val_loss:.4f}")
            else:
                early_stop_counter += 1
            
            # Perform temporal evaluation if enabled
            if temporal_eval and epoch % temporal_eval_freq == 0:
                logger.info(f"Performing temporal evaluation at epoch {epoch}...")
                temporal_results = perform_temporal_evaluation(
                    model=model,
                    graph_data=graph_data,
                    args=argparse.Namespace(
                        temporal_threshold=time_threshold,
                        future_window=future_window,
                        eval_num_papers=eval_num_papers,
                        eval_temperature=eval_temperature
                    ),
                    device_manager=device_manager,
                    epoch=epoch
                )
                
                # Add to history
                history['temporal_eval'].append({
                    'epoch': epoch,
                    'results': temporal_results
                })
                
                # Save best model based on temporal evaluation score
                if temporal_results['overall_score'] > best_temporal_score:
                    logger.info(f"New best model based on temporal evaluation (score: {temporal_results['overall_score']:.4f})")
                    best_temporal_score = temporal_results['overall_score']
                    best_temporal_model_state = copy.deepcopy(model.state_dict())
                    
                    # Save model checkpoint if output directory is specified
                    if args and hasattr(args, 'output_dir') and args.output_dir:
                        temporal_checkpoint_path = os.path.join(args.output_dir, f"best_model_temporal.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': best_temporal_model_state,
                            'temporal_score': best_temporal_score,
                            'args': vars(args) if args else {}
                        }, temporal_checkpoint_path)
                        logger.info(f"Saved best temporal model to {temporal_checkpoint_path}")
            
            # Check for early stopping
            if early_stopping and early_stop_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
                        
    # Load best model
    logger.info(f"Training complete. Loading best model from epoch {best_epoch}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        logger.warning("No best model state was saved (possibly due to NaN/Inf losses). Using current model state.")
    
    # Perform final temporal evaluation if enabled
    if temporal_eval:
        logger.info("Performing final temporal evaluation...")
        final_temporal_results = perform_temporal_evaluation(
            model=model,
            graph_data=graph_data,
            args=argparse.Namespace(
                temporal_threshold=time_threshold,
                future_window=future_window,
                eval_num_papers=eval_num_papers,
                eval_temperature=eval_temperature
            ),
            device_manager=device_manager
        )
        
        # Add to history
        history['final_temporal_eval'] = final_temporal_results
        
        # Save detailed results
        if args and hasattr(args, 'output_dir') and args.output_dir:
            temporal_results_path = os.path.join(args.output_dir, "temporal_evaluation_results.json")
            with open(temporal_results_path, 'w') as f:
                # Convert numpy/torch types to Python native types
                serializable_results = convert_to_serializable(final_temporal_results)
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Saved temporal evaluation results to {temporal_results_path}")
            
        # Load best temporal model if it performed better
        if best_temporal_score > final_temporal_results['overall_score']:
            logger.info(f"Loading best temporal model (score: {best_temporal_score:.4f})")
            model.load_state_dict(best_temporal_model_state)
    
    return model, history


def test_link_prediction(model, test_graph, test_pos_edge_index, test_neg_edge_index):
    """Test link prediction performance.
    
    Args:
        model: The trained model
        test_graph: Test graph
        test_pos_edge_index: Positive edge indices for testing
        test_neg_edge_index: Negative edge indices for testing
        
    Returns:
        Dict[str, float]: Dictionary of test metrics
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Compute metrics
    with torch.no_grad():
        test_metrics = model.compute_link_prediction_metrics(
            graph=test_graph,
            pos_edge_index=test_pos_edge_index,
            neg_edge_index=test_neg_edge_index
        )
    
    # Log results
    logger.info(f"Link Prediction Test Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return test_metrics


def test_generation(model, past_graph, future_graph, args):
    """Test paper generation capabilities.
    
    Args:
        model: The trained model
        past_graph: Past graph for generation
        future_graph: Future graph for validation
        args: Command-line arguments
        
    Returns:
        Tuple of (generation metrics, generated papers)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    try:
    # Generate papers
        logger.info(f"Generating {args.num_generated_papers} papers...")
        
        # Get time threshold from past graph
        if hasattr(past_graph, 'node_timestamps') and past_graph.node_timestamps is not None:
            time_threshold = past_graph.node_timestamps.max().item()
        else:
            time_threshold = 0.0
            
        # Set default future window if not provided
        future_window = args.future_window
        if future_window is None:
            # Default: 10% of time span
            if hasattr(past_graph, 'node_timestamps') and past_graph.node_timestamps is not None:
                time_span = past_graph.node_timestamps.max().item() - past_graph.node_timestamps.min().item()
                future_window = time_span * 0.1
            else:
                future_window = 1.0  # Default value
        
        logger.info(f"Using time_threshold={time_threshold}, future_window={future_window}")
        
        # Instead of using the complex generate_future_papers, let's use our simpler generate_single_paper
        # that we have already verified works correctly
        generated_features_list = []
        citation_scores_list = []
        
        # Generate papers one by one
        for i in range(args.num_generated_papers):
            # Calculate time for this paper (evenly distribute through future window)
            paper_time = time_threshold + (i + 1) * (future_window / args.num_generated_papers)
            
            logger.info(f"Generating paper {i+1}/{args.num_generated_papers} at time {paper_time}")
            
            # Generate a single paper
            try:
                paper_features, citation_scores = model.generate_single_paper(
                    past_graph,
                    time=paper_time,
                    temperature=0.8
                )
                
                # Add to lists
                generated_features_list.append(paper_features)
                citation_scores_list.append(citation_scores)
                
                logger.info(f"Successfully generated paper {i+1} with features shape {paper_features.shape}")
                
            except Exception as e:
                logger.error(f"Error generating paper {i+1}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Stack features if we have any
        if generated_features_list:
            all_generated_features = torch.cat(generated_features_list, dim=0)
            all_citation_scores = torch.cat(citation_scores_list, dim=0)
            
            # Create metadata for generated papers
            generated_papers = []
            for i in range(len(generated_features_list)):
                # Get top citation indices
                top_k = min(5, all_citation_scores.size(1))
                top_indices = torch.topk(all_citation_scores[i], k=top_k).indices.cpu().tolist()
                
                paper_metadata = {
                    'id': f'generated_{i}',
                    'time': time_threshold + (i + 1) * (future_window / args.num_generated_papers),
                    'top_citations': top_indices
                }
                generated_papers.append(paper_metadata)
                
            # Evaluate generation quality
            metrics = {
                'num_generated': len(generated_papers),
                'avg_citations': sum(len(p['top_citations']) for p in generated_papers) / len(generated_papers) if generated_papers else 0,
            }
            
            # Add feature similarity if future_graph is available
            if future_graph is not None and hasattr(future_graph, 'x'):
                future_features = future_graph.x
                # Calculate average cosine similarity with future papers
                gen_features_norm = F.normalize(all_generated_features, p=2, dim=1)
                future_features_norm = F.normalize(future_features, p=2, dim=1)
                
                # Compute pairwise similarities
                similarities = torch.mm(gen_features_norm, future_features_norm.t())
                
                # Get max similarity for each generated paper
                max_similarities, _ = torch.max(similarities, dim=1)
                
                # Calculate average
                metrics['avg_similarity'] = max_similarities.mean().item()
                
            # Log results
            logger.info(f"Generation Test Results:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
                
            return metrics, generated_papers
        else:
            logger.warning("No papers were successfully generated")
            return {'num_generated': 0}, []
            
    except Exception as e:
        logger.error(f"Generation test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': str(e)}, []


def visualize_generation_metrics(history, output_dir):
    """
    Visualize generation metrics across training.
    
    Args:
        history: Training history dictionary
        output_dir: Output directory for plots
    """
    if 'generation_metrics' not in history or not history['generation_metrics']:
        logger.info("No generation metrics to visualize")
        return
    
    # Extract metrics
    epochs = [entry['epoch'] for entry in history['generation_metrics']]
    phases = [entry['phase'] for entry in history['generation_metrics']]
    
    # Create figure for citation metrics
    plt.figure(figsize=(12, 8))
    
    # Plot citation metrics
    citation_metrics = [
        'citation_precision', 'citation_recall', 'citation_f1', 
        'citation_count_similarity'
    ]
    
    for i, metric in enumerate(citation_metrics):
        values = [entry['metrics'][metric] for entry in history['generation_metrics']]
        plt.subplot(2, 2, i+1)
        plt.plot(epochs, values, 'o-', label=f'Phase {phases[0]}')
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'citation_metrics.png'))
    plt.close()
    
    # Create figure for feature metrics
    plt.figure(figsize=(12, 5))
    
    # Plot feature metrics
    feature_metrics = ['feature_diversity', 'feature_novelty']
    
    for i, metric in enumerate(feature_metrics):
        values = [entry['metrics'][metric] for entry in history['generation_metrics']]
        plt.subplot(1, 2, i+1)
        plt.plot(epochs, values, 'o-', label=f'Phase {phases[0]}')
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_metrics.png'))
    plt.close()
    
    # Plot combined metrics over time
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    
    for metric in citation_metrics + feature_metrics:
        values = [entry['metrics'][metric] for entry in history['generation_metrics']]
        # Normalize to [0, 1] for better comparison
        if values:
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                values = [(v - min_val) / (max_val - min_val) for v in values]
        plt.plot(epochs, values, 'o-', label=metric.replace('_', ' ').title())
    
    plt.title('Normalized Generation Metrics Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_generation_metrics.png'))
    plt.close()

def visualize_training_history(history, output_dir):
    """
    Visualize training history with plots.
    
    Args:
        history: Training history dictionary
        output_dir: Output directory for plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 8))
    
    # Plot combined loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.axvline(x=len(history['link_pred_loss']), color='r', linestyle='--', label='Phase 1 End')
    plt.title('Combined Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot link prediction loss
    plt.subplot(2, 2, 2)
    plt.plot(history['link_pred_loss'], label='Train')
    plt.plot(history['val_link_pred_loss'], label='Validation')
    plt.title('Link Prediction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot generation loss if available
    if 'gen_loss' in history and history['gen_loss']:
        plt.subplot(2, 2, 3)
        plt.plot(history['gen_loss'], label='Train')
        plt.plot(history['val_gen_loss'], label='Validation')
        plt.title('Generation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot citation loss if available
    if 'citation_loss' in history and history['citation_loss']:
        plt.subplot(2, 2, 4)
        plt.plot(history['citation_loss'], label='Train')
        plt.plot(history['val_citation_loss'], label='Validation')
        plt.title('Citation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    
    # Plot learning rate
    plt.figure(figsize=(10, 5))
    plt.plot(history['learning_rate'])
    plt.axvline(x=len(history['link_pred_loss']), color='r', linestyle='--', label='Phase 1 End')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
    plt.close()
    
    # Visualize generation metrics if available
    if 'generation_metrics' in history and history['generation_metrics']:
        visualize_generation_metrics(history, output_dir)
    
    # Visualize temporal evaluation metrics if available
    if 'temporal_eval' in history and history['temporal_eval']:
        # Visualize temporal evaluation here if needed
        pass


def save_results(test_metrics, generation_metrics, generated_papers, history, args):
    """
    Save evaluation results and model history.
    
    Args:
        test_metrics: Link prediction metrics on test set
        generation_metrics: Paper generation metrics
        generated_papers: List of generated paper data
        history: Training history
        args: Command-line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save test metrics
    test_results = {
        'link_prediction': test_metrics,
        'generation': generation_metrics,
        'args': vars(args)
    }
    
    # Save link prediction results
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        # Convert numpy/torch types to Python types
        converted_results = convert_to_serializable(test_results)
        json.dump(converted_results, f, indent=2)
    
    # Save generation results (with more detail)
    if generated_papers:
        # Save full generated papers for later analysis
        with open(os.path.join(args.output_dir, 'generated_papers.pkl'), 'wb') as f:
            pickle.dump(generated_papers, f)
        
        # Also save a JSON-friendly version with key metrics
        json_papers = []
        for i, paper in enumerate(generated_papers[:10]):  # Limit to 10 papers for JSON
            json_paper = {
                'id': i,
                'features': paper.get('features_norm', 0.0),
                'num_citations': int(paper.get('num_citations', 0)),
                'citation_score': float(paper.get('citation_score', 0.0))
            }
            json_papers.append(json_paper)
            
        with open(os.path.join(args.output_dir, 'generated_papers_summary.json'), 'w') as f:
            json.dump(json_papers, f, indent=2)
    
    # Visualize training history
    visualize_training_history(history, args.output_dir)
    
    # Save final generation metrics
    if 'generation_metrics' in history and history['generation_metrics']:
        final_metrics = history['generation_metrics'][-1]['metrics']
        
        # Log detailed generation metrics
        logger.info("\nFinal Generation Metrics:")
        logger.info("========================")
        logger.info(f"Feature Diversity: {final_metrics.get('feature_diversity', 0.0):.4f}")
        logger.info(f"Feature Novelty: {final_metrics.get('feature_novelty', 0.0):.4f}")
        logger.info(f"Citation Precision: {final_metrics.get('citation_precision', 0.0):.4f}")
        logger.info(f"Citation Recall: {final_metrics.get('citation_recall', 0.0):.4f}")
        logger.info(f"Citation F1: {final_metrics.get('citation_f1', 0.0):.4f}")
        logger.info(f"Citation Count Similarity: {final_metrics.get('citation_count_similarity', 0.0):.4f}")
        
        # Save metrics to a separate file
        with open(os.path.join(args.output_dir, 'generation_metrics.json'), 'w') as f:
            json.dump(convert_to_serializable(final_metrics), f, indent=2)
    
    logger.info(f"Results saved to {args.output_dir}")


def plot_training_curves(history, plot_path):
    """
    Plot training curves from history data.
    
    Args:
        history: Dictionary containing training history
        plot_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    # Create a new figure
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    if 'train_loss' in history and 'val_loss' in history:
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    # Plot link prediction loss if available
    if 'link_pred_loss' in history and 'val_link_pred_loss' in history:
        plt.subplot(2, 2, 2)
        plt.plot(history['link_pred_loss'], label='Link Pred Train')
        plt.plot(history['val_link_pred_loss'], label='Link Pred Val')
        plt.title('Link Prediction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    # Plot generation loss
    if 'gen_loss' in history and 'val_gen_loss' in history:
        plt.subplot(2, 2, 3)
        plt.plot(history['gen_loss'], label='Generation Train')
        plt.plot(history['val_gen_loss'], label='Generation Val')
        plt.title('Generation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
    # Plot citation loss
    if 'citation_loss' in history and 'val_citation_loss' in history:
        plt.subplot(2, 2, 4)
        plt.plot(history['citation_loss'], label='Citation Train')
        plt.plot(history['val_citation_loss'], label='Citation Val')
        plt.title('Citation Loss (Phase 2 Focus)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(plot_path)
    
    # Create a second figure for temporal evaluation metrics
    if 'temporal_eval' in history and history['temporal_eval']:
        plt.figure(figsize=(15, 10))
        
        # Extract metrics
        epochs = []
        phases = []
        feature_sim = []
        citation_acc = []
        topic_corr = []
        temporal_coh = []
        overall_score = []
        
        for eval_result in history['temporal_eval']:
            epochs.append(eval_result.get('epoch', 0))
            phases.append(eval_result.get('phase', 0))
            
            results = eval_result.get('results', {})
            
            # Extract metrics with default values
            feature_sim.append(results.get('feature_similarity', 0))
            citation_acc.append(results.get('citation_accuracy', 0))
            topic_corr.append(results.get('topic_correlation', 0))
            temporal_coh.append(results.get('temporal_coherence', 0))
            overall_score.append(results.get('overall_score', 0))
        
        # Create markers and colors based on phase
        markers = ['o' if p == 1 else 's' for p in phases]
        colors = ['blue' if p == 1 else 'green' for p in phases]
        
        # Plot metrics
        plt.subplot(2, 2, 1)
        plt.scatter(epochs, feature_sim, marker='o')
        plt.plot(epochs, feature_sim)
        plt.title('Feature Similarity')
        plt.xlabel('Epoch')
        plt.ylabel('Similarity')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.scatter(epochs, citation_acc, marker='o')
        plt.plot(epochs, citation_acc)
        plt.title('Citation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.scatter(epochs, topic_corr, marker='o')
        plt.plot(epochs, topic_corr)
        plt.title('Topic Correlation')
        plt.xlabel('Epoch')
        plt.ylabel('Correlation')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.scatter(epochs, overall_score, marker='o')
        plt.plot(epochs, overall_score)
        plt.title('Overall Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        temporal_plot_path = plot_path.replace('.png', '_temporal.png')
        plt.savefig(temporal_plot_path)
    plt.close()


def add_arguments(parser):
    # Add temporal evaluation arguments
    temporal_eval_group = parser.add_argument_group('Temporal Evaluation')
    temporal_eval_group.add_argument('--temporal_eval', action='store_true',
                        help='Enable temporal hold-out evaluation')
    temporal_eval_group.add_argument('--temporal_threshold', type=float, default=None,
                        help='Time threshold for temporal split (default: median timestamp)')
    temporal_eval_group.add_argument('--eval_num_papers', type=int, default=20,
                        help='Number of papers to generate for evaluation')
    temporal_eval_group.add_argument('--eval_temperature', type=float, default=1.0,
                        help='Temperature for generation during evaluation')
    temporal_eval_group.add_argument('--temporal_eval_freq', type=int, default=5,
                        help='Frequency (in epochs) to run temporal evaluation')


def perform_temporal_evaluation(model, graph_data, args, device_manager, epoch=None):
    """
    Perform temporal hold-out evaluation on the model.
    
    Args:
        model: The model to evaluate
        graph_data: The complete graph data
        args: Command line arguments
        device_manager: Device manager for handling devices
        epoch: Current epoch number (for logging)
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info(f"Running temporal hold-out evaluation{f' (epoch {epoch})' if epoch is not None else ''}...")
    
    # Initialize evaluator
    device = device_manager.device if device_manager else 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = TemporalEvaluator(device=device)
    
    # Set time threshold if not provided
    time_threshold = args.temporal_threshold
    if time_threshold is None:
        # Use median timestamp
        if hasattr(graph_data, 'node_timestamps') and graph_data.node_timestamps is not None:
            time_threshold = torch.median(graph_data.node_timestamps).item()
            logger.info(f"Using median timestamp as threshold: {time_threshold}")
        else:
            time_threshold = 0.5
            logger.warning(f"No timestamps found, using default threshold: {time_threshold}")
    
    # Run evaluation
    results = evaluator.evaluate_model(
        model=model,
        graph=graph_data,
        time_threshold=time_threshold,
        future_window=args.future_window,
        num_papers=args.eval_num_papers,
        temperature=args.eval_temperature
    )
    
    # Log summary
    logger.info("Temporal Evaluation Summary:")
    summary = {
        'feature_similarity': 1.0 - results['feature_distribution']['mse'],
        'citation_accuracy': results['citation_patterns']['citation_accuracy'],
        'topic_correlation': results['topic_evolution']['topic_rank_correlation'],
        'temporal_coherence': results['temporal_coherence']['time_correlation'],
        'overall_score': results['overall_score']
    }
    
    for metric, value in summary.items():
        logger.info(f"  {metric}: {value:.4f}")
        
    return results


def calculate_citation_metrics(real_citations, generated_citations, threshold=0.5):
    """
    Calculate metrics comparing real vs generated citation patterns.
    
    Args:
        real_citations: Real citation probabilities [num_papers, num_nodes]
        generated_citations: Generated citation probabilities [num_papers, num_nodes]
        threshold: Probability threshold for converting to binary predictions
        
    Returns:
        Dictionary of citation metrics
    """
    # Convert to numpy if tensors
    if isinstance(real_citations, torch.Tensor):
        real_citations = real_citations.detach().cpu().numpy()
    if isinstance(generated_citations, torch.Tensor):
        generated_citations = generated_citations.detach().cpu().numpy()
    
    # Calculate binary predictions
    real_binary = (real_citations >= threshold).astype(float)
    gen_binary = (generated_citations >= threshold).astype(float)
    
    # Calculate metrics (using matrix operations for efficiency)
    
    # Citation precision: What percentage of predicted citations are correct
    # For each paper, calculate precision and then average across papers
    paper_precisions = []
    for i in range(real_binary.shape[0]):
        if np.sum(gen_binary[i]) > 0:
            precision = np.sum(real_binary[i] * gen_binary[i]) / np.sum(gen_binary[i])
            paper_precisions.append(precision)
    
    citation_precision = np.mean(paper_precisions) if paper_precisions else 0.0
    
    # Citation recall: What percentage of real citations are captured
    # For each paper, calculate recall and then average across papers
    paper_recalls = []
    for i in range(real_binary.shape[0]):
        if np.sum(real_binary[i]) > 0:
            recall = np.sum(real_binary[i] * gen_binary[i]) / np.sum(real_binary[i])
            paper_recalls.append(recall)
    
    citation_recall = np.mean(paper_recalls) if paper_recalls else 0.0
    
    # Citation F1: Harmonic mean of precision and recall
    if citation_precision + citation_recall > 0:
        citation_f1 = 2 * (citation_precision * citation_recall) / (citation_precision + citation_recall)
    else:
        citation_f1 = 0.0
    
    # Citation distribution similarity: Compare citation count distributions
    real_citation_counts = np.sum(real_binary, axis=1)
    gen_citation_counts = np.sum(gen_binary, axis=1)
    
    # Calculate metrics of citation count distribution
    real_mean = np.mean(real_citation_counts)
    gen_mean = np.mean(gen_citation_counts)
    
    # Mean absolute error in citation counts
    citation_count_mae = np.mean(np.abs(real_citation_counts - gen_citation_counts))
    
    # Citation count distribution similarity (higher is better)
    citation_count_similarity = 1.0 / (1.0 + np.abs(real_mean - gen_mean))
    
    return {
        'citation_precision': citation_precision,
        'citation_recall': citation_recall,
        'citation_f1': citation_f1,
        'citation_count_similarity': citation_count_similarity,
        'citation_count_mae': citation_count_mae
    }

def compute_feature_diversity(features):
    """
    Compute diversity of generated features.
    
    Args:
        features: Generated feature vectors [num_papers, feature_dim]
        
    Returns:
        Diversity score (higher means more diverse features)
    """
    # Convert to numpy if tensor
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    
    # If there's only one feature, return 0 diversity
    if features.shape[0] <= 1:
        return 0.0
    
    # Calculate feature diversity as average pairwise distance
    num_features = features.shape[0]
    total_distance = 0.0
    
    # Compute pairwise distances efficiently
    # Normalize features first
    normalized_features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix (dot product of normalized features)
    similarity_matrix = np.matmul(normalized_features, normalized_features.T)
    
    # Clamp to [-1, 1] to handle numerical issues
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
    
    # Convert to distances
    distances = 1.0 - similarity_matrix
    
    # Zero out the diagonal
    np.fill_diagonal(distances, 0.0)
    
    # Compute average distance
    total_distance = np.sum(distances)
    num_pairs = num_features * (num_features - 1)
    
    return total_distance / num_pairs

def compute_feature_novelty(generated_features, existing_features):
    """
    Compute novelty of generated features compared to existing ones.
    
    Args:
        generated_features: Generated feature vectors [num_papers, feature_dim]
        existing_features: Existing feature vectors [num_existing, feature_dim]
        
    Returns:
        Novelty score (higher means more novel features)
    """
    # Convert to numpy if tensors
    if isinstance(generated_features, torch.Tensor):
        generated_features = generated_features.detach().cpu().numpy()
    if isinstance(existing_features, torch.Tensor):
        existing_features = existing_features.detach().cpu().numpy()
    
    # If there are no existing features, return maximum novelty
    if existing_features.shape[0] == 0:
        return 1.0
    
    # Normalize features
    normalized_generated = generated_features / (np.linalg.norm(generated_features, axis=1, keepdims=True) + 1e-8)
    normalized_existing = existing_features / (np.linalg.norm(existing_features, axis=1, keepdims=True) + 1e-8)
    
    # For each generated feature, find distance to closest existing feature
    novelty_scores = []
    
    for i in range(normalized_generated.shape[0]):
        # Compute similarities to all existing features
        similarities = np.dot(normalized_generated[i], normalized_existing.T)
        
        # Find maximum similarity (closest existing feature)
        max_similarity = np.max(similarities)
        
        # Convert to novelty (1 - similarity)
        novelty = 1.0 - max_similarity
        novelty_scores.append(novelty)
    
    # Return average novelty
    return np.mean(novelty_scores)


def main():
    """Main function to run the training process."""
    # Write a debug message to a file to verify the script is running
    with open("debug_started.txt", "w") as f:
        f.write("Training script started!\n")
        
    # Parse arguments
    args = parse_args()
    
    # Initialize device manager
    device_manager = None  # Initialize to None first
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Set up device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Create a simple DeviceManager for handling tensor movement
    class DeviceManager:
        def __init__(self, device):
            self.device = device
            
        def to_device(self, tensor):
            if tensor is not None and isinstance(tensor, torch.Tensor):
                return tensor.to(self.device)
            return tensor
            
    device_manager = DeviceManager(device)  # Now initialize device_manager
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data
    graph_data, time_split = load_and_prepare_data(args)
    if graph_data is None or time_split is None:
        logger.error("Failed to load data")
        return
    
    # Initialize model
    model, optimizer = initialize_model(args, graph_data)
    
    # Extract necessary data for training
    train_pos_edge_index, train_neg_edge_index = time_split.get('train_edges', (None, None))
    val_pos_edge_index, val_neg_edge_index = time_split.get('val_edges', (None, None))
    past_graph = time_split.get('past_graph', None)
    future_features = time_split.get('future_features', None)
    future_timestamps = time_split.get('future_timestamps', None)
    
    # Log training scheme details
    if args.training_scheme == 1:
        logger.info(f"Using training scheme 1: Joint training (link prediction)")
        logger.info(f"Training for {args.epochs} epochs with learning rate {args.phase_1_lr}")
    elif args.training_scheme == 2:
        logger.info(f"Using training scheme 2: Joint training (generation)")
        logger.info(f"Training for {args.epochs} epochs with learning rate {args.phase_2_lr}")
    elif args.training_scheme == 3:
        logger.info(f"Using training scheme 3: Sequential training")
        logger.info(f"Phase 1 (link prediction): {args.phase_1_epochs} epochs with learning rate {args.phase_1_lr}")
        logger.info(f"Phase 2 (generation): {args.phase_2_epochs} epochs with learning rate {args.phase_2_lr}")
    elif args.training_scheme == 4:
        logger.info(f"Using training scheme 4: Direct encoder+generator training")
        logger.info(f"Training for {args.epochs} epochs with learning rate {args.phase_1_lr}")
    
    # Train the model
    model, history = train_model(
        model=model,
        graph_data=graph_data,
        train_pos_edge_index=train_pos_edge_index,
        train_neg_edge_index=train_neg_edge_index,
        val_pos_edge_index=val_pos_edge_index,
        val_neg_edge_index=val_neg_edge_index,
        past_graph=past_graph,
        future_features=future_features,
        future_timestamps=future_timestamps,
        epochs=args.epochs,
        phase_1_lr=args.phase_1_lr,
        phase_2_lr=args.phase_2_lr,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        patience=args.patience,
        task_weights={'link_prediction': args.link_pred_weight, 'generation': args.generation_weight},
        training_scheme=args.training_scheme,
        initial_epochs=args.initial_epochs,
        phase_1_epochs=args.phase_1_epochs,
        phase_2_epochs=args.phase_2_epochs,
        mask_ratio=args.mask_ratio,
        args=args,
        device_manager=device_manager,
        # Temporal evaluation parameters
        temporal_eval=args.temporal_eval if hasattr(args, 'temporal_eval') else False,
        temporal_eval_freq=args.temporal_eval_freq if hasattr(args, 'temporal_eval_freq') else 5,
        time_threshold=args.temporal_threshold if hasattr(args, 'temporal_threshold') else None,
        future_window=args.future_window if hasattr(args, 'future_window') else None,
        eval_num_papers=args.eval_num_papers if hasattr(args, 'eval_num_papers') else 20,
        eval_temperature=args.eval_temperature if hasattr(args, 'eval_temperature') else 1.0
    )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.checkpoint_dir, f"integrated_model_{timestamp}.pt")
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")
    
    # Save history
    history_path = os.path.join(args.output_dir, f"training_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_history = {}
        for key, values in history.items():
            if isinstance(values, list) and len(values) > 0 and isinstance(values[0], np.ndarray):
                serializable_history[key] = [v.tolist() for v in values]
            else:
                serializable_history[key] = values
        
        json.dump(serializable_history, f, indent=2)
    
    logger.info(f"Training history saved to {history_path}")
    
    # Plot training curves
    plot_path = os.path.join(args.output_dir, f"training_curves_{timestamp}.png")
    plot_training_curves(history, plot_path)
    logger.info(f"Training curves saved to {plot_path}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    try:
        logger.info("Starting training script...")
        main() 
        logger.info("Training script completed successfully.")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1) 