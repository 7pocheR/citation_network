#!/usr/bin/env python
"""
Autoregressive Citation Model Training Script

This script trains an autoregressive model for citation prediction in two phases:
1. Link prediction training
2. Autoregressive generation training

Author: AI Assistant
"""
import os
import argparse
import logging
import torch
import numpy as np
import random
import time
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# Add the parent directory to the path to allow importing local modules
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import data loading utilities
from data_utils.citation_data_loading import load_graph_data, create_train_val_test_split

# Import model components
from src.data.dataset import GraphData
from src.models.autoregressive_citation_model import AutoregressiveCitationModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autoregressive_model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train an autoregressive citation model")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="data/full_dataset.json", 
                        help="Path to the dataset JSON file")
    parser.add_argument("--embedding_dict_path", type=str, default="data/embedding_dictionaries.pkl", 
                        help="Path to embedding dictionaries")
    parser.add_argument("--output_dir", type=str, default="output" + time.strftime("%Y%m%d_%H%M%S"), 
                        help="Directory to save model checkpoints and results")
    parser.add_argument("--temporal_split", action="store_true", 
                        help="Use temporal splitting for train/val/test")
    parser.add_argument("--val_ratio", type=float, default=0.1, 
                        help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.1, 
                        help="Ratio of test data")
    parser.add_argument("--resample_interval", type=int, default=20, 
                        help="Number of epochs between dataset resamplings")
    
    # Model parameters
    parser.add_argument("--embed_dim", type=int, default=128, 
                        help="Dimension of node embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256, 
                        help="Dimension of hidden layers")
    parser.add_argument("--num_encoder_layers", type=int, default=3, 
                        help="Number of layers in the encoder")
    parser.add_argument("--num_predictor_layers", type=int, default=2, 
                        help="Number of layers in the predictor")
    parser.add_argument("--num_heads", type=int, default=4, 
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.2, 
                        help="Dropout rate")
    parser.add_argument("--use_hierarchical", action="store_true", 
                        help="Use hierarchical encoding")
    parser.add_argument("--curvature", type=float, default=1.0, 
                        help="Hyperbolic curvature parameter")
    parser.add_argument("--ordering_strategy", type=str, default="citation", 
                        choices=["citation", "temporal"], 
                        help="Strategy for ordering autoregressive predictions")
    parser.add_argument("--temperature", type=float, default=1.0, 
                        help="Temperature for controlling prediction randomness")
    parser.add_argument("--reveal_ratio", type=float, default=0.1, 
                        help="Proportion of links to reveal during training")
    
    # Autoregressive refinement parameters
    parser.add_argument("--num_iterations", type=int, default=12,
                        help="Number of iterations for autoregressive refinement")
    parser.add_argument("--citation_threshold", type=float, default=0.9,
                        help="Probability threshold for adding citations in refinement")
    
    # Training parameters
    parser.add_argument("--phase_1_epochs", type=int, default=300, 
                        help="Maximum epochs for Phase 1 (link prediction)")
    parser.add_argument("--phase_2_epochs", type=int, default= 200, 
                        help="Maximum epochs for Phase 2 (autoregressive generation)")
    parser.add_argument("--lr_phase_1", type=float, default=0.001, 
                        help="Learning rate for Phase 1")
    parser.add_argument("--lr_phase_2", type=float, default=0.001, 
                        help="Learning rate for Phase 2")
    parser.add_argument("--weight_decay", type=float, default=0.0001, 
                        help="L2 regularization strength")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for early stopping")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for training")
    parser.add_argument("--max_edges_per_batch", type=int, default=10000,
                        help="Maximum number of edges (positive + negative) to use per batch for link prediction")
    
    # Temporal parameters for Phase 2
    parser.add_argument("--t1", type=float, default=2020, 
                        help="Lower bound of time threshold range for random selection")
    parser.add_argument("--t2", type=float, default=2023.8, 
                        help="Upper bound of time threshold range for random selection")
    parser.add_argument("--delta_t", type=float, default=0.3, 
                        help="Size of time window for finding future papers")
    parser.add_argument("--num_future_papers", type=int, default=50,
                        help="Number of future papers to sample per epoch")
    
    # Remove progression_schedule since we're now using random selection
    # Keep the flag in args parser for backward compatibility but mark as deprecated
    parser.add_argument("--progression_schedule", type=str, default="random", 
                        choices=["random", "linear", "exponential", "sigmoid"], 
                        help="DEPRECATED: Now using random threshold selection")
    
    # Device parameters
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (cuda or cpu, default: auto-detect)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    # Testing and evaluation
    parser.add_argument("--skip_phase_1", action="store_true", 
                        help="Skip Phase 1 training (load from checkpoint)")
    parser.add_argument("--skip_phase_2", action="store_true", 
                        help="Skip Phase 2 training (load from checkpoint)")
    parser.add_argument("--phase_1_checkpoint", type=str, default=None, 
                        help="Path to Phase 1 checkpoint to load")
    parser.add_argument("--phase_2_checkpoint", type=str, default=None, 
                        help="Path to Phase 2 checkpoint to load")
    parser.add_argument("--test_only", action="store_true", 
                        help="Only run testing, no training")
    parser.add_argument("--dynamic_test_threshold", action="store_true", 
                        help="Dynamically adjust the time threshold during testing to ensure sufficient nodes")
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def load_and_prepare_data(args):
    """
    Load and prepare data for training.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (graph_data, train_graph, val_graph, test_graph)
    """
    logger.info(f"Loading graph data from {args.data_path}")
    
    # Load graph data using the citation_data_loading utility
    graph_data = load_graph_data(
        file_path=args.data_path,
        embedding_dict_path=args.embedding_dict_path,
    )
    
    logger.info(f"Loaded graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
    
    # Create train/validation/test splits
    train_graph, val_graph, test_graph = create_train_val_test_split(
        graph_data=graph_data,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        temporal_split=args.temporal_split
    )
    
    logger.info(f"Created splits - Train: {train_graph.num_nodes} nodes, {train_graph.num_edges} edges")
    logger.info(f"Validation: {val_graph.num_nodes} nodes, {val_graph.num_edges} edges")
    logger.info(f"Test: {test_graph.num_nodes} nodes, {test_graph.num_edges} edges")
    
    return graph_data, train_graph, val_graph, test_graph

def create_edge_indices(graph, split='train'):
    """
    Create positive and negative edge indices for link prediction.
    
    Args:
        graph: GraphData object
        split: 'train', 'val', or 'test'
        
    Returns:
        tuple: (pos_edge_index, neg_edge_index)
    """
    # Get original edge index
    all_edge_index = graph.edge_index
    
    # Check if graph has edges
    if all_edge_index is None or all_edge_index.shape[1] == 0:
        logger.warning(f"No edges found in graph. Creating empty edge indices for {split}.")
        # Return empty tensors with correct shape
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long, device=graph.device)
        return empty_edge_index, empty_edge_index
    
    # Get the mask for the current split
    if split == 'train':
        node_mask = graph.train_index
    elif split == 'val':
        node_mask = graph.val_index
    elif split == 'test':
        node_mask = graph.test_index
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
    
    if node_mask is None:
        logger.warning(f"No {split} mask found in graph. Using all edges.")
        pos_edge_index = all_edge_index
    else:
        # Filter positive edges based on source node mask
        mask_indices = []
        for i in range(all_edge_index.shape[1]):
            src_node = all_edge_index[0, i].item()
            if node_mask[src_node]:
                mask_indices.append(i)
        
        # Create positive edge index for this split
        if len(mask_indices) > 0:
            pos_edge_index = all_edge_index[:, mask_indices]
        else:
            logger.warning(f"No edges found for {split} after applying mask. Creating empty edge indices.")
            empty_edge_index = torch.zeros((2, 0), dtype=torch.long, device=all_edge_index.device)
            return empty_edge_index, empty_edge_index
    
    # Generate negative edges (edges that don't exist in the graph)
    num_nodes = graph.num_nodes
    
    # Safety check - if graph has too few nodes, we might not be able to generate negative edges
    if num_nodes < 2:
        logger.warning(f"Graph has only {num_nodes} nodes, cannot generate negative edges.")
        # Return empty tensor for negative edges
        empty_neg_edge_index = torch.zeros((2, 0), dtype=torch.long, device=pos_edge_index.device)
        logger.info(f"Created {split} edge indices - Positive: {pos_edge_index.shape[1]}, Negative: 0")
        return pos_edge_index, empty_neg_edge_index
    
    neg_edges = []
    
    # Get existing edges as a set for quick lookup
    existing_edges = set()
    for i in range(all_edge_index.shape[1]):
        src, dst = all_edge_index[0, i].item(), all_edge_index[1, i].item()
        existing_edges.add((src, dst))
    
    # Calculate max possible negative edges
    max_possible_neg_edges = num_nodes * (num_nodes - 1) - len(existing_edges)
    
    # Limit negative edges to either positive edge count or max possible
    num_neg_edges = min(pos_edge_index.shape[1], max_possible_neg_edges)
    
    # Check if we can generate enough negative edges
    if num_neg_edges <= 0:
        logger.warning(f"Cannot generate any negative edges for {split}.")
        empty_neg_edge_index = torch.zeros((2, 0), dtype=torch.long, device=pos_edge_index.device)
        logger.info(f"Created {split} edge indices - Positive: {pos_edge_index.shape[1]}, Negative: 0")
        return pos_edge_index, empty_neg_edge_index
    
    # Get the eligible source nodes (based on mask) for sampling
    if node_mask is not None:
        eligible_src_nodes = torch.nonzero(node_mask).squeeze(-1).tolist()
        if not eligible_src_nodes:
            logger.warning(f"No eligible source nodes for {split}. Cannot generate negative edges.")
            empty_neg_edge_index = torch.zeros((2, 0), dtype=torch.long, device=pos_edge_index.device)
            return pos_edge_index, empty_neg_edge_index
    else:
        eligible_src_nodes = list(range(num_nodes))
    
    # Generate random negative edges with a limit on attempts
    max_attempts = num_neg_edges * 10  # Limit attempts to avoid infinite loops
    attempt_count = 0
    
    while len(neg_edges) < num_neg_edges and attempt_count < max_attempts:
        # Sample source node from eligible nodes (based on mask)
        src = random.choice(eligible_src_nodes)
        # Sample any destination node
        dst = random.randint(0, num_nodes - 1)
        attempt_count += 1
        
        if src != dst and (src, dst) not in existing_edges:
            neg_edges.append([src, dst])
            existing_edges.add((src, dst))  # Add to prevent duplicates
    
    # Check if we found enough negative edges
    if len(neg_edges) == 0:
        logger.warning(f"Failed to generate negative edges for {split} after {attempt_count} attempts.")
        empty_neg_edge_index = torch.zeros((2, 0), dtype=torch.long, device=pos_edge_index.device)
        logger.info(f"Created {split} edge indices - Positive: {pos_edge_index.shape[1]}, Negative: 0")
        return pos_edge_index, empty_neg_edge_index
    
    # Convert to tensor
    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long, device=pos_edge_index.device).t()
    
    logger.info(f"Created {split} edge indices - Positive: {pos_edge_index.shape[1]}, Negative: {neg_edge_index.shape[1]}")
    
    return pos_edge_index, neg_edge_index

def rebalance_edge_indices(pos_edge_index, neg_edge_index, max_edges=None):
    """
    Rebalance positive and negative edge indices to have the same number of edges.
    Optionally limit the total number of edges to prevent memory issues.
    
    Args:
        pos_edge_index: Positive edge indices tensor [2, num_pos_edges]
        neg_edge_index: Negative edge indices tensor [2, num_neg_edges]
        max_edges: Maximum number of positive and negative edges combined
        
    Returns:
        tuple: (rebalanced_pos_edge_index, rebalanced_neg_edge_index)
    """
    num_pos = pos_edge_index.shape[1]
    num_neg = neg_edge_index.shape[1]
    
    # Check if we need rebalancing
    if num_pos == num_neg and (max_edges is None or num_pos + num_neg <= max_edges):
        return pos_edge_index, neg_edge_index
    
    device = pos_edge_index.device
    
    # Compute the target number of edges per class
    if max_edges is not None:
        # Limit total edges and balance classes
        target_per_class = min(num_pos, num_neg, max_edges // 2)
    else:
        # Just balance classes
        target_per_class = min(num_pos, num_neg)
    
    # Sample positive edges if needed
    if num_pos > target_per_class:
        perm = torch.randperm(num_pos, device=device)[:target_per_class]
        pos_edge_index = pos_edge_index[:, perm]
    
    # Sample negative edges if needed
    if num_neg > target_per_class:
        perm = torch.randperm(num_neg, device=device)[:target_per_class]
        neg_edge_index = neg_edge_index[:, perm]
    
    return pos_edge_index, neg_edge_index

def initialize_model(args, graph_data):
    """
    Initialize the autoregressive citation model.
    
    Args:
        args: Command line arguments
        graph_data: GraphData object containing the graph
        
    Returns:
        AutoregressiveCitationModel: Initialized model
    """
    # Set device
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get number of nodes and feature dimension
    num_nodes = graph_data.num_nodes
    node_feature_dim = graph_data.x.shape[1] if graph_data.x is not None else None
    
    # Initialize model
    model = AutoregressiveCitationModel(
        num_nodes=num_nodes,
        node_feature_dim=node_feature_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        use_hierarchical=args.use_hierarchical,
        curvature=args.curvature,
        dropout=args.dropout,
        num_encoder_layers=args.num_encoder_layers,
        num_predictor_layers=args.num_predictor_layers,
        ordering_strategy=args.ordering_strategy,
        temperature=args.temperature,
        reveal_ratio=args.reveal_ratio,
        device=device
    )
    
    # Move model to device
    model = model.to(device)
    
    logger.info(f"Initialized model with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Model is on device: {device}")
    
    return model

def update_time_threshold(args, epoch, phase_2_epochs, previous_threshold=None):
    """
    Select a random time threshold for training.
    
    Args:
        args: Command line arguments
        epoch: Current epoch
        phase_2_epochs: Total epochs for Phase 2
        previous_threshold: Previous threshold value (not used anymore, kept for compatibility)
        
    Returns:
        tuple: (current_threshold, delta_t) where delta_t is the fixed time window size
    """
    # Randomly select a threshold between T1 and T2
    current_threshold = args.t1 + (args.t2 - args.t1) * torch.rand(1).item()
    
    # Use the fixed delta_t from arguments
    delta_t = args.delta_t
    
    return current_threshold, delta_t

def train_phase_1(args, model, train_graph, val_graph, device):
    """
    Phase 1: Train the model for link prediction.
    
    Args:
        args: Command line arguments
        model: AutoregressiveCitationModel
        train_graph: GraphData object for training
        val_graph: GraphData object for validation
        device: Device to use
        
    Returns:
        model: Trained model
        dict: Training history
    """
    logger.info("Starting Phase 1: Link Prediction Training")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr_phase_1,
        weight_decay=args.weight_decay
    )
    
    # Create edge indices for training and validation
    train_pos_edge_index, train_neg_edge_index = create_edge_indices(train_graph, 'train')
    val_pos_edge_index, val_neg_edge_index = create_edge_indices(val_graph, 'val')
    
    # Rebalance edge indices to have equal number of positive and negative examples
    # This helps with stable training and prevents class imbalance issues
    logger.info("Rebalancing edge indices for training and validation")
    train_pos_edge_index, train_neg_edge_index = rebalance_edge_indices(
        train_pos_edge_index, train_neg_edge_index, max_edges=args.max_edges_per_batch if hasattr(args, 'max_edges_per_batch') else None
    )
    val_pos_edge_index, val_neg_edge_index = rebalance_edge_indices(
        val_pos_edge_index, val_neg_edge_index, max_edges=args.max_edges_per_batch if hasattr(args, 'max_edges_per_batch') else None
    )
    
    logger.info(f"After rebalancing - Train edges: {train_pos_edge_index.shape[1]} positive, {train_neg_edge_index.shape[1]} negative")
    logger.info(f"After rebalancing - Val edges: {val_pos_edge_index.shape[1]} positive, {val_neg_edge_index.shape[1]} negative")
    
    # Move data to device
    train_graph = train_graph.to(device)
    val_graph = val_graph.to(device)
    train_pos_edge_index = train_pos_edge_index.to(device)
    train_neg_edge_index = train_neg_edge_index.to(device)
    val_pos_edge_index = val_pos_edge_index.to(device)
    val_neg_edge_index = val_neg_edge_index.to(device)
    
    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_ap': []
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.phase_1_epochs):
        start_time = time.time()
        
        # Training step
        model.train()
        train_loss_dict = model.train_step(
            graph=train_graph,
            pos_edge_index=train_pos_edge_index,
            neg_edge_index=train_neg_edge_index,
            optimizer=optimizer,
            task_weights={'link_prediction': 1.0, 'autoregressive': 0.0}  # Only link prediction in Phase 1
        )
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_metrics = model.validation_step(
                graph=val_graph,
                val_pos_edge_index=val_pos_edge_index,
                val_neg_edge_index=val_neg_edge_index,
                task_weights={'link_prediction': 1.0, 'autoregressive': 0.0}
            )
        
        # Extract metrics
        train_loss = train_loss_dict['total_loss']
        val_loss = val_metrics['total_loss']
        val_auc = val_metrics['link_prediction_auc']
        val_ap = val_metrics['link_prediction_ap']
        
        # Save metrics to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_ap'].append(val_ap)
        
        # Log metrics
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{args.phase_1_epochs} - "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val AUC: {val_auc:.4f}, "
                   f"Val AP: {val_ap:.4f}, "
                   f"Time: {epoch_time:.2f}s")
        
        # Check if this is the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            
            # Save the best model
            checkpoint_path = os.path.join(args.output_dir, "best_phase_1_model.pt")
            model.save(checkpoint_path)
            logger.info(f"Saved best model with val AUC: {val_auc:.4f} to {checkpoint_path}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
            
        # Resample data splits if needed
        if args.resample_interval > 0 and (epoch + 1) % args.resample_interval == 0:
            logger.info(f"Resampling data splits at epoch {epoch+1}")
            train_pos_edge_index, train_neg_edge_index = create_edge_indices(train_graph, 'train')
            val_pos_edge_index, val_neg_edge_index = create_edge_indices(val_graph, 'val')
            
            # Move to device
            train_pos_edge_index = train_pos_edge_index.to(device)
            train_neg_edge_index = train_neg_edge_index.to(device)
            val_pos_edge_index = val_pos_edge_index.to(device)
            val_neg_edge_index = val_neg_edge_index.to(device)
    
    # Load best model
    best_model_path = os.path.join(args.output_dir, "best_phase_1_model.pt")
    model.load(best_model_path)
    logger.info(f"Loaded best Phase 1 model from {best_model_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, "phase_1_history.json")
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], label='AUC')
    plt.plot(history['val_ap'], label='AP')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "phase_1_training_curves.png"))
    
    return model, history

def train_phase_2(args, model, train_graph, val_graph, device):
    """
    Phase 2: Train the model for autoregressive generation.
    
    Args:
        args: Command line arguments
        model: AutoregressiveCitationModel
        train_graph: GraphData object for training
        val_graph: GraphData object for validation
        device: Device to use
        
    Returns:
        model: Trained model
        dict: Training history
    """
    logger.info("Starting Phase 2: Autoregressive Generation Training")
    
    # Define the split for this phase
    split = 'train'  # Since this is the training phase
    
    # Add new command line argument for fixed number of future papers, defaulting to 50
    if not hasattr(args, 'num_future_papers'):
        args.num_future_papers = 50
        logger.info(f"Setting default num_future_papers to {args.num_future_papers}")
        
    # Set the minimum number of future papers required
    min_future_papers = min(5, args.num_future_papers // 10)  # At least 5 or 10% of requested papers
    max_threshold_retries = 10  # Maximum number of times to retry finding a good threshold
    
    # Create optimizer with lower learning rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr_phase_2,
        weight_decay=args.weight_decay
    )
    
    # Get timestamps range
    if train_graph.node_timestamps is not None:
        node_timestamps = train_graph.node_timestamps
    elif hasattr(train_graph, 'paper_times') and train_graph.paper_times is not None:
        node_timestamps = train_graph.paper_times
    else:
        raise ValueError("Training graph does not have node timestamps")
    
    # Ensure node_timestamps is on the correct device
    node_timestamps = node_timestamps.to(device)
    
    # Set time thresholds if not specified
    if args.t1 is None:
        args.t1 = float(node_timestamps.min().item())
    if args.t2 is None:
        args.t2 = float(node_timestamps.max().item() - args.delta_t)
    
    logger.info(f"Time thresholds - T1: {args.t1}, T2: {args.t2}, Initial delta_t: {args.delta_t}, Future papers per epoch: {args.num_future_papers}")
    
    # Move data to device
    train_graph = train_graph.to(device)
    val_graph = val_graph.to(device)
    
    # Create positive and negative edge indices for validation
    val_pos_edge_index, val_neg_edge_index = create_edge_indices(val_graph, 'val')
    val_pos_edge_index = val_pos_edge_index.to(device)
    val_neg_edge_index = val_neg_edge_index.to(device)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_autoregressive_loss': [],
        'val_link_prediction_auc': [],
        'current_threshold': [],
        'delta_t': [],
        'num_future_papers': [],
        'threshold_retries': [],  # Track how many retries per epoch
        
        # Add additional metrics to track
        'val_ar_precision': [],
        'val_ar_recall': [],
        'val_ar_f1': [],
        'val_ar_accuracy': [],
        'val_ar_precision@1': [],
        'val_ar_precision@5': [],
        'val_ar_precision@10': [],
        'val_ar_num_edges_predicted': [],
        'val_ar_num_edges_actual': []
    }
    
    # Track previous threshold for calculating delta
    previous_threshold = None
    
    for epoch in range(args.phase_2_epochs):
        start_time = time.time()
        
        # Initialize variables for the threshold retry loop
        found_valid_threshold = False
        num_threshold_retries = 0
        
        # Keep trying random thresholds until we find one with enough future papers
        while not found_valid_threshold and num_threshold_retries < max_threshold_retries:
            # Update time threshold and calculate dynamic delta_t
            current_threshold, delta_t = update_time_threshold(args, epoch, args.phase_2_epochs, previous_threshold)
            
            # Split the graph into past and future based on current threshold
            current_threshold_tensor = torch.tensor(current_threshold, device=device)
            delta_t_tensor = torch.tensor(delta_t, device=device)
            
            # First filter by time
            past_nodes = node_timestamps < current_threshold_tensor
            future_candidates_by_time = (node_timestamps >= current_threshold_tensor) & (node_timestamps < current_threshold_tensor + delta_t_tensor)
            
            # Now filter by node mask for the current split
            if split == 'train':
                # For training, only use nodes in the training mask
                if train_graph.train_index is not None:
                    future_candidates = future_candidates_by_time & train_graph.train_index
                else:
                    future_candidates = future_candidates_by_time
            elif split == 'val':
                # For validation, only use nodes in the validation mask
                if val_graph.val_index is not None:
                    future_candidates = future_candidates_by_time & val_graph.val_index
                else:
                    future_candidates = future_candidates_by_time
            else:  # test
                # This block would be for testing, but we're in training code
                future_candidates = future_candidates_by_time
            
            # Count how many future candidates we have
            num_future_candidates = future_candidates.sum().item()
            
            # Check if we have enough nodes in both sets
            if past_nodes.sum() >= 2 and num_future_candidates >= min_future_papers:
                found_valid_threshold = True
                logger.info(f"Found valid threshold {current_threshold:.4f} after {num_threshold_retries} retries")
            else:
                num_threshold_retries += 1
                logger.warning(f"Retry {num_threshold_retries}/{max_threshold_retries}: Threshold {current_threshold:.4f} has insufficient nodes "
                              f"(Past: {past_nodes.sum().item()}, Future candidates: {num_future_candidates}, Min required: {min_future_papers})")
        
        # If we couldn't find a valid threshold after retries, skip this epoch
        if not found_valid_threshold:
            logger.error(f"Failed to find valid threshold after {max_threshold_retries} retries - skipping epoch {epoch}")
            continue
            
        # Update previous_threshold for next epoch
        previous_threshold = current_threshold
        
        # Store in history
        history['current_threshold'].append(current_threshold)
        history['delta_t'].append(delta_t)
        history['threshold_retries'].append(num_threshold_retries)
        
        # Debug info
        logger.info(f"Epoch {epoch+1}/{args.phase_2_epochs} - Threshold: {current_threshold:.4f}, Delta_t: {delta_t:.4f}")
        logger.info(f"Past nodes: {past_nodes.sum().item()}/{len(past_nodes)}, Future candidates: {num_future_candidates}/{len(past_nodes)}")
        
        # Randomly sample a fixed number of future papers
        if num_future_candidates <= args.num_future_papers:
            # Use all available future candidates if we have fewer than requested
            future_nodes = future_candidates
            actual_future_papers = num_future_candidates
        else:
            # Randomly sample the requested number of future papers
            future_candidate_indices = torch.nonzero(future_candidates).squeeze(1)
            perm = torch.randperm(num_future_candidates)[:args.num_future_papers]
            selected_indices = future_candidate_indices[perm]
            
            # Create a new mask with only the selected papers
            future_nodes = torch.zeros_like(future_candidates)
            future_nodes[selected_indices] = True
            actual_future_papers = args.num_future_papers
        
        # Log the actual number of future papers used
        logger.info(f"Using {actual_future_papers} future papers for training")
        history['num_future_papers'].append(actual_future_papers)
        
        # Create past and future graphs
        past_graph = train_graph.subgraph(past_nodes)
        future_graph = train_graph.subgraph(future_nodes)
        
        # Only check if past graph has edges - future graph can have 0 edges between future nodes
        # and still be valid for autoregressive training (we're predicting edges from future to past)
        if past_graph.num_edges == 0:
            logger.warning(f"Skipping epoch due to no edges in past graph. "
                          f"Past edges: {past_graph.num_edges}")
            continue
            
        # Log the edge counts for monitoring
        logger.info(f"Edge counts - Past graph: {past_graph.num_edges}, Future graph: {future_graph.num_edges}")
        
        # Create positive and negative edge indices for link prediction
        train_pos_edge_index, train_neg_edge_index = create_edge_indices(past_graph, 'train')
        train_pos_edge_index = train_pos_edge_index.to(device)
        train_neg_edge_index = train_neg_edge_index.to(device)
        
        # Check if we have enough edges for training
        if train_pos_edge_index.shape[1] == 0:
            logger.warning(f"Skipping epoch due to no positive edges in past graph.")
            continue
        
        # Training step with multi-task loss
        model.train()
        try:
            train_loss_dict = model.train_step(
                graph=past_graph,
                pos_edge_index=train_pos_edge_index,
                neg_edge_index=train_neg_edge_index,
                past_graph=past_graph,
                future_graph=future_graph,
                optimizer=optimizer,
                task_weights={'link_prediction': 0.3, 'autoregressive': 0.7}  # Focus more on autoregressive task
            )
        except Exception as e:
            logger.error(f"Error during training step: {str(e)}")
            logger.info(f"Past graph: {past_graph.num_nodes} nodes, {past_graph.num_edges} edges")
            logger.info(f"Future graph: {future_graph.num_nodes} nodes, {future_graph.num_edges} edges")
            logger.info(f"Positive edges: {train_pos_edge_index.shape[1]}, Negative edges: {train_neg_edge_index.shape[1]}")
            # Skip this epoch
            continue
        
        # Validation step
        model.eval()
        with torch.no_grad():
            # Try to find a valid validation set
            found_valid_val_threshold = False
            val_threshold_retries = 0
            
            # Ideally, we want to validate on the same threshold as training, but we might need to adjust
            # if there are insufficient validation samples at that threshold
            val_current_threshold = current_threshold
            val_current_threshold_tensor = current_threshold_tensor
            
            while not found_valid_val_threshold and val_threshold_retries < max_threshold_retries:
                # Split validation graph
                val_node_timestamps = val_graph.node_timestamps
                val_past_nodes = val_node_timestamps < val_current_threshold_tensor
                val_future_candidates_by_time = (val_node_timestamps >= val_current_threshold_tensor) & (val_node_timestamps < val_current_threshold_tensor + delta_t_tensor)
                
                # Filter by validation mask
                if val_graph.val_index is not None:
                    val_future_candidates = val_future_candidates_by_time & val_graph.val_index
                else:
                    val_future_candidates = val_future_candidates_by_time
                
                val_num_future_candidates = val_future_candidates.sum().item()
                
                # Check if we have enough nodes for validation
                if val_past_nodes.sum() >= 2 and val_num_future_candidates >= min_future_papers:
                    found_valid_val_threshold = True
                    if val_threshold_retries > 0:
                        logger.info(f"Found valid validation threshold after {val_threshold_retries} retries")
                else:
                    val_threshold_retries += 1
                    logger.warning(f"Validation retry {val_threshold_retries}/{max_threshold_retries}: "
                                  f"Threshold {val_current_threshold:.4f} has insufficient validation nodes "
                                  f"(Past: {val_past_nodes.sum().item()}, Future: {val_num_future_candidates}, Min required: {min_future_papers})")
                    
                    # Try a slightly different threshold
                    val_current_threshold = args.t1 + (args.t2 - args.t1) * torch.rand(1).item()
                    val_current_threshold_tensor = torch.tensor(val_current_threshold, device=device)
            
            # Skip validation if we couldn't find a valid threshold
            if not found_valid_val_threshold:
                logger.warning(f"Failed to find valid validation threshold after {max_threshold_retries} retries")
                # Use training loss as validation loss for this epoch
                val_metrics = {
                    'total_loss': train_loss_dict['total_loss'],
                    'link_prediction_loss': train_loss_dict.get('link_prediction_loss', 0.0),
                    'autoregressive_loss': train_loss_dict.get('autoregressive_loss', 0.0),
                    'link_prediction_auc': 0.5  # Default value
                }
            else:
                # Randomly sample validation future papers
                if val_num_future_candidates <= args.num_future_papers:
                    val_future_nodes = val_future_candidates
                    val_actual_future_papers = val_num_future_candidates
                else:
                    val_future_candidate_indices = torch.nonzero(val_future_candidates).squeeze(1)
                    val_perm = torch.randperm(val_num_future_candidates)[:args.num_future_papers]
                    val_selected_indices = val_future_candidate_indices[val_perm]
                    
                    val_future_nodes = torch.zeros_like(val_future_candidates)
                    val_future_nodes[val_selected_indices] = True
                    val_actual_future_papers = args.num_future_papers
                
                logger.info(f"Validation - Past nodes: {val_past_nodes.sum().item()}, Future papers: {val_actual_future_papers}")
                
                val_past_graph = val_graph.subgraph(val_past_nodes)
                val_future_graph = val_graph.subgraph(val_future_nodes)
                
                # Check if validation past graph has edges
                if val_past_graph.num_edges == 0:
                    logger.warning(f"No edges in validation past graph - using training loss")
                    # Use training loss as validation loss for this epoch
                    val_metrics = {
                        'total_loss': train_loss_dict['total_loss'],
                        'link_prediction_loss': train_loss_dict.get('link_prediction_loss', 0.0),
                        'autoregressive_loss': train_loss_dict.get('autoregressive_loss', 0.0),
                        'link_prediction_auc': 0.5  # Default value
                    }
                else:
                    try:
                        # Validation
                        val_metrics = model.validation_step(
                            graph=val_graph,
                            val_pos_edge_index=val_pos_edge_index,
                            val_neg_edge_index=val_neg_edge_index,
                            val_past_graph=val_past_graph,
                            val_future_graph=val_future_graph,
                            task_weights={'link_prediction': 0.3, 'autoregressive': 0.7}
                        )
                    except Exception as e:
                        logger.error(f"Error during validation step: {str(e)}")
                        # Use training loss as validation loss for this epoch
                        val_metrics = {
                            'total_loss': train_loss_dict['total_loss'],
                            'link_prediction_loss': train_loss_dict.get('link_prediction_loss', 0.0),
                            'autoregressive_loss': train_loss_dict.get('autoregressive_loss', 0.0),
                            'link_prediction_auc': 0.5  # Default value
                        }
        
        # Extract metrics
        train_loss = train_loss_dict['total_loss']
        val_loss = val_metrics['total_loss']
        val_autoregressive_loss = val_metrics.get('autoregressive_loss', 0.0)
        val_link_auc = val_metrics.get('link_prediction_auc', 0.0)
        
        # Extract additional autoregressive metrics
        val_ar_precision = val_metrics.get('precision', 0.0)
        val_ar_recall = val_metrics.get('recall', 0.0)
        val_ar_f1 = val_metrics.get('f1', 0.0)
        val_ar_accuracy = val_metrics.get('accuracy', 0.0)
        val_ar_precision_1 = val_metrics.get('precision@1', 0.0)
        val_ar_precision_5 = val_metrics.get('precision@5', 0.0)
        val_ar_precision_10 = val_metrics.get('precision@10', 0.0)
        
        # Calculate edge prediction statistics
        val_ar_num_predicted = val_metrics.get('num_predicted_edges', 0)
        val_ar_num_actual = val_metrics.get('num_actual_edges', 0)
        
        # Save metrics to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_autoregressive_loss'].append(val_autoregressive_loss)
        history['val_link_prediction_auc'].append(val_link_auc)
        history['current_threshold'].append(current_threshold)
        history['delta_t'].append(delta_t)
        history['num_future_papers'].append(len(future_nodes))
        history['threshold_retries'].append(num_threshold_retries)
        
        # Save additional metrics to history
        history['val_ar_precision'].append(val_ar_precision)
        history['val_ar_recall'].append(val_ar_recall)
        history['val_ar_f1'].append(val_ar_f1)
        history['val_ar_accuracy'].append(val_ar_accuracy)
        history['val_ar_precision@1'].append(val_ar_precision_1)
        history['val_ar_precision@5'].append(val_ar_precision_5)
        history['val_ar_precision@10'].append(val_ar_precision_10)
        history['val_ar_num_edges_predicted'].append(val_ar_num_predicted)
        history['val_ar_num_edges_actual'].append(val_ar_num_actual)
        
        # Log metrics with more detailed information
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{args.phase_2_epochs} - "
                   f"Threshold: {current_threshold:.2f}, "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val AR Loss: {val_autoregressive_loss:.4f}, "
                   f"Val Link AUC: {val_link_auc:.4f}, "
                   f"Time: {epoch_time:.2f}s")
        
        # Log additional autoregressive metrics
        if val_autoregressive_loss > 0:  # Only log if there's actual autoregressive training
            logger.info(f"Citation Prediction Metrics - "
                       f"Precision: {val_ar_precision:.4f}, "
                       f"Recall: {val_ar_recall:.4f}, "
                       f"F1: {val_ar_f1:.4f}, "
                       f"Accuracy: {val_ar_accuracy:.4f}")
            
            logger.info(f"Citation Ranking Metrics - "
                       f"P@1: {val_ar_precision_1:.4f}, "
                       f"P@5: {val_ar_precision_5:.4f}, "
                       f"P@10: {val_ar_precision_10:.4f}, "
                       f"Edges: {val_ar_num_actual} actual, {val_ar_num_predicted} predicted")
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save the best model
            checkpoint_path = os.path.join(args.output_dir, "best_phase_2_model.pt")
            model.save(checkpoint_path)
            logger.info(f"Saved best model with val loss: {val_loss:.4f} to {checkpoint_path}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model
    best_model_path = os.path.join(args.output_dir, "best_phase_2_model.pt")
    model.load(best_model_path)
    logger.info(f"Loaded best Phase 2 model from {best_model_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, "phase_2_history.json")
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot 2: Autoregressive metrics
    plt.subplot(2, 2, 2)
    plt.plot(history['val_autoregressive_loss'], label='Autoregressive Loss')
    plt.plot(history['val_link_prediction_auc'], label='Link Prediction AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics')
    
    # Plot 3: Edge counts and threshold
    plt.subplot(2, 2, 3)
    plt.plot(history['val_ar_num_edges_actual'], label='Actual Edges')
    plt.plot(history['val_ar_num_edges_predicted'], label='Predicted Edges')
    plt.xlabel('Epoch')
    plt.ylabel('Edge Count')
    plt.legend()
    plt.title('Edge Prediction Analysis')
    
    # Plot 4: Precision, Recall, F1
    plt.subplot(2, 2, 4)
    plt.plot(history['val_ar_precision'], label='Precision')
    plt.plot(history['val_ar_recall'], label='Recall')
    plt.plot(history['val_ar_f1'], label='F1 Score')
    plt.plot(history['val_ar_precision@1'], label='P@1')
    plt.plot(history['val_ar_precision@5'], label='P@5')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Citation Prediction Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "phase_2_training_curves.png"))
    
    # Create a second figure for additional metrics
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Precision@k
    plt.subplot(1, 2, 1)
    plt.plot(history['val_ar_precision@1'], label='P@1')
    plt.plot(history['val_ar_precision@5'], label='P@5')
    plt.plot(history['val_ar_precision@10'], label='P@10')
    plt.xlabel('Epoch')
    plt.ylabel('Precision@k')
    plt.legend()
    plt.title('Ranking Precision at k')
    
    # Plot 2: Temporal Analysis
    plt.subplot(1, 2, 2)
    plt.plot(history['current_threshold'], label='Time Threshold')
    plt.plot(history['threshold_retries'], label='Threshold Retries')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Temporal Threshold Analysis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "phase_2_detailed_metrics.png"))
    
    # Log summary of performance
    logger.info("\nPhase 2 Training Summary:")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # Calculate averages for the last 5 epochs
    last_epochs = min(5, len(history['train_loss']))
    avg_precision = sum(history['val_ar_precision'][-last_epochs:]) / last_epochs
    avg_recall = sum(history['val_ar_recall'][-last_epochs:]) / last_epochs
    avg_f1 = sum(history['val_ar_f1'][-last_epochs:]) / last_epochs
    avg_p1 = sum(history['val_ar_precision@1'][-last_epochs:]) / last_epochs
    avg_p5 = sum(history['val_ar_precision@5'][-last_epochs:]) / last_epochs
    avg_link_auc = sum(history['val_link_prediction_auc'][-last_epochs:]) / last_epochs
    
    logger.info(f"Average metrics over last {last_epochs} epochs:")
    logger.info(f"Link AUC: {avg_link_auc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    logger.info(f"Precision@1: {avg_p1:.4f}, Precision@5: {avg_p5:.4f}")
    
    return model, history

def test_model(args, model, test_graph, device):
    """
    Test the trained model on the test set.
    
    Args:
        args: Command line arguments
        model: Trained AutoregressiveCitationModel
        test_graph: GraphData object for testing
        device: Device to use
        
    Returns:
        dict: Test metrics
    """
    logger.info("Testing model on test set")
    
    # Move data to device
    test_graph = test_graph.to(device)
    
    # Create test edge indices
    test_pos_edge_index, test_neg_edge_index = create_edge_indices(test_graph, 'test')
    
    # Rebalance test edges for fair evaluation
    logger.info("Rebalancing test edges for fair evaluation")
    test_pos_edge_index, test_neg_edge_index = rebalance_edge_indices(
        test_pos_edge_index, test_neg_edge_index, max_edges=args.max_edges_per_batch if hasattr(args, 'max_edges_per_batch') else None
    )
    
    logger.info(f"Test edges after rebalancing: {test_pos_edge_index.shape[1]} positive, {test_neg_edge_index.shape[1]} negative")
    
    test_pos_edge_index = test_pos_edge_index.to(device)
    test_neg_edge_index = test_neg_edge_index.to(device)
    
    # Initialize metrics dictionary
    test_metrics = {}
    
    # Evaluate and report Link Prediction metrics separately (unweighted)
    logger.info("Evaluating Link Prediction performance...")
    model.eval()
    with torch.no_grad():
        # Note: compute_link_prediction_metrics now uses edge masking internally
        # to ensure proper transductive link prediction
        link_pred_metrics = model.compute_link_prediction_metrics(
            graph=test_graph,
            pos_edge_index=test_pos_edge_index,
            neg_edge_index=test_neg_edge_index
        )
        # Store link prediction metrics
        test_metrics['link_prediction'] = link_pred_metrics
    
    # Get timestamps
    if test_graph.node_timestamps is not None:
        node_timestamps = test_graph.node_timestamps
    elif hasattr(test_graph, 'paper_times') and test_graph.paper_times is not None:
        node_timestamps = test_graph.paper_times
    else:
        raise ValueError("Test graph does not have node timestamps")
    
    # Move timestamps to the correct device
    node_timestamps = node_timestamps.to(device)
    
    # If args specify to dynamically adjust thresholds for testing, do so
    if hasattr(args, 'dynamic_test_threshold') and args.dynamic_test_threshold:
        # Get sorted timestamps to find appropriate thresholds
        sorted_timestamps, _ = torch.sort(node_timestamps)
        
        # Required minimum nodes in each category
        min_past_nodes = 5  # At least 5 past nodes
        min_future_nodes = 5  # At least 5 future nodes
        total_nodes = len(sorted_timestamps)
        
        # Ensure we have enough nodes overall
        if total_nodes < min_past_nodes + min_future_nodes:
            logger.warning(f"Not enough nodes overall ({total_nodes}) for autoregressive evaluation")
            autoregressive_metrics = {
                'num_generated_papers': 0,
                'total_citations': 0,
                'avg_citations_per_paper': 0,
                'error': 'Not enough nodes overall for evaluation'
            }
            
            # Store autoregressive metrics and return early
            test_metrics['autoregressive'] = autoregressive_metrics
            
            # Log results as before
            logger.info("-" * 50)
            # ... rest of logging code ...
            return test_metrics
        
        # Compute threshold to ensure min_past_nodes
        past_threshold_idx = min(min_past_nodes, total_nodes - min_future_nodes - 1)
        past_threshold_idx = max(past_threshold_idx, 0)  # Ensure non-negative
        
        # Set threshold between past_threshold_idx and past_threshold_idx+1
        # to ensure min_past_nodes nodes are in the past
        if past_threshold_idx < total_nodes - 1:
            current_threshold = sorted_timestamps[past_threshold_idx].item()
            # Add a small epsilon to ensure the node at the threshold is included
            current_threshold += 1e-6
        else:
            # Not enough nodes to perform meaningful split
            logger.warning("Cannot create a meaningful time threshold split")
            current_threshold = sorted_timestamps[-1].item() - 1  # Place before the last node
        
        # Compute delta_t to ensure min_future_nodes in the future window
        future_end_idx = min(past_threshold_idx + min_future_nodes, total_nodes - 1)
        
        if future_end_idx > past_threshold_idx:
            delta_t = sorted_timestamps[future_end_idx].item() - current_threshold + 1e-6
        else:
            # Default to a small value if not enough future nodes
            delta_t = 1.0
            
        logger.info(f"Dynamically adjusted thresholds - T2: {current_threshold}, delta_t: {delta_t}")
        
        # Create tensors from the computed values
        current_threshold_tensor = torch.tensor(current_threshold, device=device)
        delta_t_tensor = torch.tensor(delta_t, device=device)
    else:
        # For testing, use T2 as the threshold from args
        current_threshold_tensor = torch.tensor(args.t2, device=device)
        delta_t_tensor = torch.tensor(args.delta_t, device=device)
    
    # Identify past and future candidate papers
    past_nodes = node_timestamps < current_threshold_tensor
    future_candidates_by_time = (node_timestamps >= current_threshold_tensor) & (node_timestamps < current_threshold_tensor + delta_t_tensor)
    
    # Filter candidates by test mask
    if test_graph.test_index is not None:
        future_candidates = future_candidates_by_time & test_graph.test_index
    else:
        future_candidates = future_candidates_by_time
    
    num_past_nodes = past_nodes.sum().item()
    num_future_candidates = future_candidates.sum().item()
    logger.info(f"Test split - Past nodes: {num_past_nodes}, Future candidates: {num_future_candidates}")
    
    # Evaluate and report Autoregressive metrics separately (unweighted)
    logger.info("Evaluating Autoregressive performance...")
    
    # Only proceed with autoregressive evaluation if we have enough future nodes
    if num_past_nodes < 2 or num_future_candidates < 5:
        logger.warning(f"Too few nodes for meaningful autoregressive evaluation. Consider using --dynamic_test_threshold flag.")
        autoregressive_metrics = {
            'num_generated_papers': 0,
            'total_citations': 0,
            'avg_citations_per_paper': 0,
            'error': 'Too few nodes for evaluation'
        }
    else:
        # Sample future papers
        if num_future_candidates <= args.num_future_papers:
            future_nodes = future_candidates
            actual_future_papers = num_future_candidates
        else:
            # Randomly sample future papers
            future_candidate_indices = torch.nonzero(future_candidates).squeeze(1)
            perm = torch.randperm(num_future_candidates)[:args.num_future_papers]
            selected_indices = future_candidate_indices[perm]
            
            future_nodes = torch.zeros_like(future_candidates)
            future_nodes[selected_indices] = True
            actual_future_papers = args.num_future_papers
        
        logger.info(f"Using {actual_future_papers} future papers for testing")
        
        # Create test subgraphs
        test_past_graph = test_graph.subgraph(past_nodes)
        test_future_graph = test_graph.subgraph(future_nodes)
        
        # Compute autoregressive loss for evaluation
        with torch.no_grad():
            # Attempt to compute autoregressive loss if possible
            try:
                auto_loss, auto_detailed_metrics = model.compute_autoregressive_loss(
                    past_graph=test_past_graph,
                    future_graph=test_future_graph
                )
                
                # Add the detailed metrics
                autoregressive_metrics = {
                    'loss': auto_loss.item() if isinstance(auto_loss, torch.Tensor) else auto_loss,
                    'num_generated_papers': actual_future_papers
                }
                
                # Add detailed metrics from the loss computation
                for k, v in auto_detailed_metrics.items():
                    if isinstance(v, torch.Tensor):
                        autoregressive_metrics[k] = v.item()
                    else:
                        autoregressive_metrics[k] = v
            except Exception as e:
                logger.warning(f"Error computing autoregressive loss: {str(e)}")
                autoregressive_metrics = {
                    'error': str(e),
                    'num_generated_papers': 0,
                    'total_citations': 0,
                    'avg_citations_per_paper': 0
                }
        
        # Generate citations for future papers
        with torch.no_grad():
            # Get embeddings
            embeddings = model.forward(test_past_graph, task='link_prediction')
            
            # Generate citations
            future_features = test_future_graph.x if hasattr(test_future_graph, 'x') else None
            if future_features is not None and len(future_features) > 0:
                try:
                    predictions, metadata = model.generate_future_papers_autoregressive(
                        graph=test_past_graph,
                        time_threshold=args.t2,
                        future_window=args.delta_t,  # Use the fixed delta_t from args
                        paper_features=future_features,
                        top_k=10,  # Get top 10 citations per paper
                        num_iterations=args.num_iterations,
                        citation_threshold=args.citation_threshold
                    )
                    
                    # Evaluate autoregressive generation
                    logger.info(f"Generated citations for {len(metadata)} future papers")
                    
                    # Count total citations generated
                    total_citations = sum(paper.get('num_citations', 0) for paper in metadata)
                    
                    # Add generation metrics to autoregressive metrics
                    autoregressive_metrics.update({
                        'num_generated_papers': len(metadata),
                        'total_citations': total_citations,
                        'avg_citations_per_paper': total_citations / max(1, len(metadata))
                    })
                except Exception as e:
                    logger.warning(f"Error in autoregressive generation: {str(e)}")
                    autoregressive_metrics.update({
                        'generation_error': str(e)
                    })
            else:
                logger.warning("No future papers available for autoregressive generation")
                autoregressive_metrics.update({
                    'generation_error': 'No future papers available'
                })
    
    # Store autoregressive metrics
    test_metrics['autoregressive'] = autoregressive_metrics
    
    # Log results separately by task
    logger.info("-" * 50)
    logger.info("TEST RESULTS")
    logger.info("-" * 50)
    
    # Link prediction results
    logger.info("LINK PREDICTION METRICS:")
    for metric_name, value in link_pred_metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # Autoregressive results
    logger.info("-" * 30)
    logger.info("AUTOREGRESSIVE METRICS:")
    for metric_name, value in autoregressive_metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            logger.info(f"  {metric_name}: {value:.4f}" if not metric_name.startswith('num_') else f"  {metric_name}: {int(value)}")
        else:
            logger.info(f"  {metric_name}: {value}")
    
    logger.info("-" * 50)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        # Convert tensor values to float for JSON serialization
        serializable_metrics = {}
        for task, metrics in test_metrics.items():
            serializable_metrics[task] = {}
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    serializable_metrics[task][k] = v.item()
                else:
                    serializable_metrics[task][k] = v
        
        json.dump(serializable_metrics, f, indent=2)
    
    return test_metrics

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    graph_data, train_graph, val_graph, test_graph = load_and_prepare_data(args)
    
    # Save arguments
    args_path = os.path.join(args.output_dir, "args.json")
    with open(args_path, 'w') as f:
        # Convert args to dictionary, handling non-serializable types
        args_dict = {k: v if not isinstance(v, (type, torch.device)) else str(v) 
                    for k, v in vars(args).items()}
        json.dump(args_dict, f, indent=2)
    
    # Test only mode
    if args.test_only:
        if args.phase_2_checkpoint:
            # Initialize model
            model = initialize_model(args, graph_data)
            
            # Load checkpoint
            model.load(args.phase_2_checkpoint)
            logger.info(f"Loaded model from {args.phase_2_checkpoint}")
            
            # Move model to device
            model = model.to(device)
            
            # Test model
            test_metrics = test_model(args, model, test_graph, device)
            return
        else:
            raise ValueError("For test_only mode, phase_2_checkpoint must be specified")
    
    # Phase 1: Link Prediction Training
    if not args.skip_phase_1:
        # Initialize model
        model = initialize_model(args, graph_data)
        
        # Train model
        model, phase_1_history = train_phase_1(args, model, train_graph, val_graph, device)
    else:
        # Load model from checkpoint
        if args.phase_1_checkpoint is None:
            args.phase_1_checkpoint = os.path.join(args.output_dir, "best_phase_1_model.pt")
        
        # Initialize model
        model = initialize_model(args, graph_data)
        
        # Load checkpoint
        model.load(args.phase_1_checkpoint)
        logger.info(f"Loaded Phase 1 model from {args.phase_1_checkpoint}")
        
        # Move model to device
        model = model.to(device)
    
    # Phase 2: Autoregressive Generation Training
    if not args.skip_phase_2:
        # Train model
        model, phase_2_history = train_phase_2(args, model, train_graph, val_graph, device)
    else:
        # Load model from checkpoint
        if args.phase_2_checkpoint is None:
            args.phase_2_checkpoint = os.path.join(args.output_dir, "best_phase_2_model.pt")
        
        # Load checkpoint
        model.load(args.phase_2_checkpoint)
        logger.info(f"Loaded Phase 2 model from {args.phase_2_checkpoint}")
    
    # Test model
    logger.info("Testing model on test set")
    test_metrics = test_model(args, model, test_graph, device)
    
    logger.info("Training and testing complete")

if __name__ == "__main__":
    main() 