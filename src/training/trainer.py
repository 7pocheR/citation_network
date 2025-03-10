import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import time
import os
import json
from tqdm import tqdm
import logging
from .metrics import CitationPredictionMetrics, FeatureGenerationMetrics, NetworkEvaluationMetrics
import torch.nn.functional as F
from collections import defaultdict


class DynamicCitationNetworkTrainer:
    """
    Trainer class for the Dynamic Citation Network Model.
    
    Handles training loops, evaluation, and model checkpointing.
    """
    
    def __init__(self, 
                 model, 
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None,
                 encoder_optimizer: Optional[torch.optim.Optimizer] = None,
                 generator_optimizer: Optional[torch.optim.Optimizer] = None,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 checkpoint_dir: str = './checkpoints',
                 log_dir: str = './logs'):
        """
        Initialize the trainer.
        
        Args:
            model: Dynamic Citation Network Model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            test_loader: DataLoader for test data (optional)
            encoder_optimizer: Optimizer for the encoder (optional)
            generator_optimizer: Optimizer for the generator (optional)
            device: Device to run the model on
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory to save training logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.encoder_optimizer = encoder_optimizer
        self.generator_optimizer = generator_optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
        # Ensure model is on the specified device
        self.model.to(self.device)
        
        # Enable cuDNN autotuner if using CUDA
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            self.logger.info(f"CUDA enabled, using {torch.cuda.get_device_name(0)}")
        
        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Training history
        self.training_history = {
            'encoder_train_loss': [],
            'encoder_val_loss': [],
            'generator_train_loss': [],
            'generator_val_loss': [],
            'validation_metrics': []
        }
        
        # Best validation metrics
        self.best_val_loss = float('inf')
        
        # Early stopping parameters
        self.early_stopping_patience = 10
        self.checkpoint_interval = 5
    
    def save_checkpoint(self, epoch: int, filename: str = None):
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch
            filename: Filename for the checkpoint (optional)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict() if self.encoder_optimizer else None,
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict() if self.generator_optimizer else None,
            'training_history': self.training_history
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Epoch number of the checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.encoder_optimizer and 'encoder_optimizer_state_dict' in checkpoint:
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
            
        if self.generator_optimizer and 'generator_optimizer_state_dict' in checkpoint:
            self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
            
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
            
        return checkpoint['epoch']
    
    def train_encoder(self, 
                     num_epochs: int, 
                     link_prediction_callback: Optional[Callable] = None,
                     extra_metrics_callback: Optional[Callable] = None) -> Dict[str, List[float]]:
        """
        Train the encoder model.
        
        Args:
            num_epochs: Number of epochs to train
            link_prediction_callback: Function to generate link prediction data
            extra_metrics_callback: Function to compute additional metrics
            
        Returns:
            Dictionary of training history
        """
        # Log GPU memory usage
        if self.device.type == 'cuda':
            gpu_memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9
            self.logger.info(f"GPU memory allocated at start of encoder training: {gpu_memory_allocated:.2f} GB")
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        history = defaultdict(list)
        
        self.logger.info(f"Starting encoder training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_metrics = defaultdict(list)
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                for key in batch:
                    if key == 'historical_graph':
                        # Move each GraphData object to the device
                        for i in range(len(batch[key])):
                            batch[key][i] = batch[key][i].to(self.device)
                    elif isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Reset gradients
                self.encoder_optimizer.zero_grad()
                
                # Generate link prediction data
                if link_prediction_callback:
                    link_pred_edges, link_pred_labels = link_prediction_callback(batch)
                    link_pred_edges = link_pred_edges.to(self.device)
                    link_pred_labels = link_pred_labels.to(self.device)
                
                    # Train encoder with link prediction
                    loss = self.model.train_encoder(
                        batch['historical_graph'],
                        link_pred_edges,
                        link_pred_labels
                    )
                    
                    # Backward pass
                    loss.backward()
                    self.encoder_optimizer.step()
                    
                    # Record metrics
                    epoch_metrics['encoder_loss'].append(loss.item())
                
            # Compute average metrics for epoch
            for key in epoch_metrics:
                avg_value = np.mean(epoch_metrics[key])
                history[key].append(avg_value)
                progress_bar.set_postfix({key: f"{avg_value:.4f}"})
            
            # Evaluate on validation set
            val_metrics = self.evaluate_encoder(self.val_loader, link_prediction_callback)
            
            # Add validation metrics to history
            for key, value in val_metrics.items():
                history[f"val_{key}"].append(value)
            
            # Display validation metrics
            val_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}, " + val_metrics_str)
            
            # Early stopping
            if 'val_link_pred_loss' in val_metrics:
                val_loss = val_metrics['val_link_pred_loss']
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint(os.path.join(self.checkpoints_dir, "best_encoder.pt"), encoder_only=True)
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(os.path.join(self.checkpoints_dir, f"encoder_epoch_{epoch+1}.pt"), encoder_only=True)
        
        return history
    
    def evaluate_encoder(self, 
                        data_loader: DataLoader,
                        link_prediction_callback: Optional[Callable] = None) -> Dict[str, float]:
        """
        Evaluate the encoder model.
        
        Args:
            data_loader: DataLoader for evaluation data
            link_prediction_callback: Function to generate link prediction data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        metrics_list = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move batch to device
                for key in batch:
                    if key == 'historical_graph':
                        # Move each GraphData object to the device
                        for i in range(len(batch[key])):
                            batch[key][i] = batch[key][i].to(self.device)
                    elif isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Generate link prediction data for evaluation
                if link_prediction_callback:
                    link_pred_edges, link_pred_labels = link_prediction_callback(batch)
                    link_pred_edges = link_pred_edges.to(self.device)
                    link_pred_labels = link_pred_labels.to(self.device)
                else:
                    link_pred_edges = None
                    link_pred_labels = None
                
                # Get node embeddings
                embeddings = self.model.encoder(batch['historical_graph'])
                
                # Compute link prediction metrics
                batch_metrics = {}
                if link_pred_edges is not None and link_pred_labels is not None:
                    # Extract source and target node embeddings
                    source_nodes = link_pred_edges[0]
                    target_nodes = link_pred_edges[1]
                    
                    source_embeddings = embeddings[source_nodes]
                    target_embeddings = embeddings[target_nodes]
                    
                    # Compute scores
                    if self.model.encoder.hyperbolic:
                        # Hyperbolic distance for link prediction
                        scores = -self.model.encoder.hyp_tangent_space.distance(source_embeddings, target_embeddings)
                    else:
                        # Dot product for link prediction
                        scores = torch.sum(source_embeddings * target_embeddings, dim=1)
                    
                    # Compute loss
                    link_pred_loss = F.binary_cross_entropy_with_logits(scores, link_pred_labels)
                    
                    # Compute metrics
                    predictions = torch.sigmoid(scores) > 0.5
                    accuracy = (predictions == link_pred_labels).float().mean().item()
                    
                    batch_metrics['link_pred_loss'] = link_pred_loss.item()
                    batch_metrics['link_pred_accuracy'] = accuracy
                
                metrics_list.append(batch_metrics)
        
        # Compute average metrics
        avg_metrics = {}
        if metrics_list:
            for key in metrics_list[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in metrics_list])
        
        self.model.train()
        return avg_metrics
    
    def train_generator(self, 
                       num_epochs: int,
                       kl_weight: float = 1.0,
                       citation_weight: float = 1.0,
                       feature_weight: float = 1.0,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       early_stopping_patience: int = 10):
        """
        Train the generator component of the model.
        
        Args:
            num_epochs: Number of epochs to train for
            kl_weight: Weight for KL divergence loss
            citation_weight: Weight for citation prediction loss
            feature_weight: Weight for feature reconstruction loss
            scheduler: Learning rate scheduler (optional)
            early_stopping_patience: Number of epochs to wait before early stopping
        """
        if self.generator_optimizer is None:
            raise ValueError("Generator optimizer not provided")
        
        self.model.train()
        
        # Log GPU memory usage at start of training if using CUDA
        if self.device.type == 'cuda':
            self.logger.info(f"GPU memory allocated at start of generator training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            # Empty CUDA cache to free up memory
            torch.cuda.empty_cache()
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"Starting generator training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training metrics
            train_metrics = []
            
            # Training loop
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Train generator
                batch_metrics = self.model.train_generator(
                    snapshots=batch['historical_graph'],
                    paper_features=batch['paper_features'],
                    target_citations=batch['target_citations'],
                    optimizer=self.generator_optimizer,
                    kl_weight=kl_weight,
                    citation_weight=citation_weight,
                    feature_weight=feature_weight
                )
                
                train_metrics.append(batch_metrics)
            
            # Compute average training metrics
            avg_train_metrics = {}
            for key in train_metrics[0].keys():
                avg_train_metrics[key] = np.mean([m[key] for m in train_metrics])
            
            # Update learning rate
            if scheduler:
                scheduler.step()
            
            # Validation
            val_loss = 0.0
            if self.val_loader:
                val_metrics = self.evaluate_generator(
                    self.val_loader, 
                    kl_weight, 
                    citation_weight, 
                    feature_weight
                )
                val_loss = val_metrics.get('total_loss', float('inf'))
            
            # Log metrics
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                            f"Time: {time.time() - epoch_start_time:.2f}s - "
                            f"Train Loss: {avg_train_metrics.get('total_loss', 'N/A'):.4f} - "
                            f"Val Loss: {val_loss:.4f}")
            
            # Update history
            self.training_history['generator_train_loss'].append(avg_train_metrics.get('total_loss', 0.0))
            self.training_history['generator_val_loss'].append(val_loss)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        self.logger.info("Generator training completed")
    
    def evaluate_generator(self, 
                          data_loader: DataLoader,
                          kl_weight: float = 1.0,
                          citation_weight: float = 1.0,
                          feature_weight: float = 1.0) -> Dict[str, float]:
        """
        Evaluate the generator component of the model.
        
        Args:
            data_loader: DataLoader for evaluation data
            kl_weight: Weight for KL divergence loss
            citation_weight: Weight for citation prediction loss
            feature_weight: Weight for feature reconstruction loss
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_losses = []
        all_feature_metrics = []
        all_citation_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(batch['historical_graph'], batch['paper_features'])
                
                # Prepare targets
                targets = {
                    'features': batch['paper_features'],
                    'citations': batch['target_citations']
                }
                
                # Compute losses
                if hasattr(self.model.generator, 'compute_loss'):
                    losses = self.model.generator.compute_loss(
                        outputs, targets,
                        kl_weight=kl_weight,
                        citation_weight=citation_weight,
                        feature_weight=feature_weight
                    )
                    all_losses.append({k: v.item() for k, v in losses.items()})
                
                # Compute feature metrics
                feature_metrics = FeatureGenerationMetrics.compute_feature_metrics(
                    outputs['reconstructed_features'], batch['paper_features']
                )
                all_feature_metrics.append(feature_metrics)
                
                # Compute citation metrics
                citation_metrics = CitationPredictionMetrics.compute_link_prediction_metrics(
                    outputs['citation_probs'], batch['target_citations']
                )
                all_citation_metrics.append(citation_metrics)
        
        # Compute average metrics
        avg_losses = {}
        for key in all_losses[0].keys():
            avg_losses[key] = np.mean([m[key] for m in all_losses])
            
        avg_feature_metrics = {}
        for key in all_feature_metrics[0].keys():
            avg_feature_metrics[key] = np.mean([m[key] for m in all_feature_metrics])
            
        avg_citation_metrics = {}
        for key in all_citation_metrics[0].keys():
            avg_citation_metrics[key] = np.mean([m[key] for m in all_citation_metrics])
        
        # Combine all metrics
        combined_metrics = {}
        combined_metrics.update(avg_losses)
        combined_metrics.update({f"feature_{k}": v for k, v in avg_feature_metrics.items()})
        combined_metrics.update({f"citation_{k}": v for k, v in avg_citation_metrics.items()})
        
        # Save to history
        self.training_history['validation_metrics'].append(combined_metrics)
        
        self.model.train()
        return combined_metrics
    
    def generate_and_evaluate_papers(self, 
                                    num_papers: int = 10, 
                                    data_loader: Optional[DataLoader] = None,
                                    temperature: float = 1.0) -> List[Dict]:
        """
        Generate papers and evaluate their quality.
        
        Args:
            num_papers: Number of papers to generate
            data_loader: DataLoader to get graph snapshots from
            temperature: Sampling temperature
            
        Returns:
            List of generated papers with evaluation metrics
        """
        self.model.eval()
        
        generated_papers = []
        
        # Get a batch from data loader if provided
        if data_loader:
            batch = next(iter(data_loader))
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            snapshots = batch['historical_graph']
        else:
            # Use training data if no data loader provided
            batch = next(iter(self.train_loader))
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            snapshots = batch['historical_graph']
        
        with torch.no_grad():
            for i in range(num_papers):
                # Generate a paper
                paper = self.model.generate_paper(
                    snapshots=snapshots,
                    temperature=temperature
                )
                
                # Ensure all numpy arrays are converted to lists for JSON serialization
                for key, value in paper.items():
                    if isinstance(value, np.ndarray):
                        paper[key] = value.tolist()
                
                generated_papers.append(paper)
        
        self.model.train()
        return generated_papers
    
    def save_training_history(self, filepath: str = None):
        """
        Save training history to a JSON file.
        
        Args:
            filepath: Path to save the file (default: log_dir/training_history.json)
        """
        if filepath is None:
            filepath = os.path.join(self.log_dir, 'training_history.json')
            
        # Convert any NumPy or torch values to standard Python types
        history_to_save = {}
        for key, value in self.training_history.items():
            if isinstance(value, list):
                if isinstance(value[0], dict):
                    # List of dictionaries
                    history_to_save[key] = []
                    for item in value:
                        new_item = {}
                        for k, v in item.items():
                            if isinstance(v, (np.integer, np.floating)):
                                new_item[k] = float(v)
                            elif isinstance(v, (np.ndarray, torch.Tensor)):
                                new_item[k] = v.tolist()
                            else:
                                new_item[k] = v
                        history_to_save[key].append(new_item)
                else:
                    # List of values
                    history_to_save[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in value]
            else:
                history_to_save[key] = value
                
        with open(filepath, 'w') as f:
            json.dump(history_to_save, f, indent=4)
            
        self.logger.info(f"Training history saved to {filepath}")


class TopicAwareCitationNetworkTrainer(DynamicCitationNetworkTrainer):
    """
    Trainer class for the Topic-Aware Citation Network Model.
    
    Extends the base trainer with topic-conditioning capabilities.
    """
    
    def train_generator(self, 
                       num_epochs: int,
                       kl_weight: float = 1.0,
                       citation_weight: float = 1.0,
                       feature_weight: float = 1.0,
                       topic_callback: Optional[Callable] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       early_stopping_patience: int = 10):
        """
        Train the generator component of the topic-aware model.
        
        Args:
            num_epochs: Number of epochs to train for
            kl_weight: Weight for KL divergence loss
            citation_weight: Weight for citation prediction loss
            feature_weight: Weight for feature reconstruction loss
            topic_callback: Function to generate topic IDs for training
            scheduler: Learning rate scheduler (optional)
            early_stopping_patience: Number of epochs to wait before early stopping
        """
        if self.generator_optimizer is None:
            raise ValueError("Generator optimizer not provided")
        
        self.model.train()
        
        # Log GPU memory usage at start of training if using CUDA
        if self.device.type == 'cuda':
            self.logger.info(f"GPU memory allocated at start of generator training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            # Empty CUDA cache to free up memory
            torch.cuda.empty_cache()
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"Starting topic-aware generator training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training metrics
            train_metrics = []
            
            # Training loop
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Generate topic IDs for training
                topic_data = topic_callback(batch) if topic_callback else None
                topic_ids = None
                
                if topic_data is not None:
                    # Check if we have the new format (dictionary) or the old format (tensor)
                    if isinstance(topic_data, dict):
                        topic_ids = topic_data['topic_ids'].to(self.device)
                        # Add the topic count to the batch if available
                        if 'loader' in batch and hasattr(batch['loader'], 'get_feature_info'):
                            feature_info = batch['loader'].get_feature_info()
                            batch['topic_count'] = feature_info.get('topic_count', None)
                    else:
                        # Legacy format - direct tensor
                        topic_ids = topic_data.to(self.device)

                # Train generator
                batch_metrics = self.model.train_generator(
                    snapshots=batch['historical_graph'],
                    paper_features=batch['paper_features'],
                    target_citations=batch['target_citations'],
                    topic_ids=topic_ids,
                    optimizer=self.generator_optimizer,
                    kl_weight=kl_weight,
                    citation_weight=citation_weight,
                    feature_weight=feature_weight
                )
                
                train_metrics.append(batch_metrics)
            
            # Compute average training metrics
            avg_train_metrics = {}
            for key in train_metrics[0].keys():
                avg_train_metrics[key] = np.mean([m[key] for m in train_metrics])
            
            # Update learning rate
            if scheduler:
                scheduler.step()
            
            # Validation
            val_loss = 0.0
            if self.val_loader:
                val_metrics = self.evaluate_generator(
                    self.val_loader, 
                    kl_weight, 
                    citation_weight, 
                    feature_weight,
                    topic_callback
                )
                val_loss = val_metrics.get('total_loss', float('inf'))
            
            # Log metrics
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                            f"Time: {time.time() - epoch_start_time:.2f}s - "
                            f"Train Loss: {avg_train_metrics.get('total_loss', 'N/A'):.4f} - "
                            f"Val Loss: {val_loss:.4f}")
            
            # Update history
            self.training_history['generator_train_loss'].append(avg_train_metrics.get('total_loss', 0.0))
            self.training_history['generator_val_loss'].append(val_loss)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        self.logger.info("Topic-aware generator training completed")
    
    def evaluate_generator(self, 
                          data_loader: DataLoader,
                          kl_weight: float = 1.0,
                          citation_weight: float = 1.0,
                          feature_weight: float = 1.0,
                          topic_callback: Optional[Callable] = None) -> Dict[str, float]:
        """
        Evaluate the generator component of the topic-aware model.
        
        Args:
            data_loader: DataLoader for evaluation data
            kl_weight: Weight for KL divergence loss
            citation_weight: Weight for citation prediction loss
            feature_weight: Weight for feature reconstruction loss
            topic_callback: Function to generate topic IDs for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_losses = []
        all_feature_metrics = []
        all_citation_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Generate topic IDs for evaluation
                topic_data = topic_callback(batch) if topic_callback else None
                topic_ids = None
                
                if topic_data is not None:
                    # Check if we have the new format (dictionary) or the old format (tensor)
                    if isinstance(topic_data, dict):
                        topic_ids = topic_data['topic_ids'].to(self.device)
                        # Add the topic count to the batch if available
                        if 'loader' in batch and hasattr(batch['loader'], 'get_feature_info'):
                            feature_info = batch['loader'].get_feature_info()
                            batch['topic_count'] = feature_info.get('topic_count', None)
                    else:
                        # Legacy format - direct tensor
                        topic_ids = topic_data.to(self.device)

                # Forward pass
                outputs = self.model(batch['historical_graph'], batch['paper_features'], topic_ids)
                
                # Prepare targets
                targets = {
                    'features': batch['paper_features'],
                    'citations': batch['target_citations']
                }
                
                # Compute losses
                if hasattr(self.model.generator, 'compute_loss'):
                    losses = self.model.generator.compute_loss(
                        outputs, targets,
                        kl_weight=kl_weight,
                        citation_weight=citation_weight,
                        feature_weight=feature_weight
                    )
                    all_losses.append({k: v.item() for k, v in losses.items()})
                
                # Compute feature metrics
                feature_metrics = FeatureGenerationMetrics.compute_feature_metrics(
                    outputs['reconstructed_features'], batch['paper_features']
                )
                all_feature_metrics.append(feature_metrics)
                
                # Compute citation metrics
                citation_metrics = CitationPredictionMetrics.compute_link_prediction_metrics(
                    outputs['citation_probs'], batch['target_citations']
                )
                all_citation_metrics.append(citation_metrics)
        
        # Compute average metrics
        avg_losses = {}
        for key in all_losses[0].keys():
            avg_losses[key] = np.mean([m[key] for m in all_losses])
            
        avg_feature_metrics = {}
        for key in all_feature_metrics[0].keys():
            avg_feature_metrics[key] = np.mean([m[key] for m in all_feature_metrics])
            
        avg_citation_metrics = {}
        for key in all_citation_metrics[0].keys():
            avg_citation_metrics[key] = np.mean([m[key] for m in all_citation_metrics])
        
        # Combine all metrics
        combined_metrics = {}
        combined_metrics.update(avg_losses)
        combined_metrics.update({f"feature_{k}": v for k, v in avg_feature_metrics.items()})
        combined_metrics.update({f"citation_{k}": v for k, v in avg_citation_metrics.items()})
        
        # Save to history
        self.training_history['validation_metrics'].append(combined_metrics)
        
        self.model.train()
        return combined_metrics
    
    def generate_and_evaluate_papers(self, 
                                    num_papers: int = 10, 
                                    data_loader: Optional[DataLoader] = None,
                                    topic_ids: Optional[torch.Tensor] = None,
                                    temperature: float = 1.0) -> List[Dict]:
        """
        Generate papers and evaluate their quality.
        
        Args:
            num_papers: Number of papers to generate
            data_loader: DataLoader to get graph snapshots from
            topic_ids: Topic IDs to condition on
            temperature: Sampling temperature
            
        Returns:
            List of generated papers with evaluation metrics
        """
        self.model.eval()
        
        generated_papers = []
        
        # Get a batch from data loader if provided
        if data_loader:
            batch = next(iter(data_loader))
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            snapshots = batch['historical_graph']
        else:
            # Use training data if no data loader provided
            batch = next(iter(self.train_loader))
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            snapshots = batch['historical_graph']
        
        if topic_ids is not None:
            topic_ids = topic_ids.to(self.device)
        
        with torch.no_grad():
            for i in range(num_papers):
                # Generate a paper
                paper = self.model.generate_paper(
                    snapshots=snapshots,
                    topic_ids=topic_ids,
                    temperature=temperature
                )
                
                # Ensure all numpy arrays are converted to lists for JSON serialization
                for key, value in paper.items():
                    if isinstance(value, np.ndarray):
                        paper[key] = value.tolist()
                
                generated_papers.append(paper)
        
        self.model.train()
        return generated_papers 