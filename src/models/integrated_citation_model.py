"""
Integrated Citation Model

This module provides an integrated model for citation network analysis and generation,
combining a graph encoder, link predictor, and paper generator into a single model.
"""

import os
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as PyGData
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from src.data.dataset import GraphData
from src.models.encoder.hyperbolic_encoder import HyperbolicEncoder
from src.models.predictors.attention_predictor import AttentionPredictor, EnhancedAttentionPredictor

logger = logging.getLogger(__name__)

class IntegratedCitationModel(nn.Module):
    """
    Integrated model that combines HyperbolicEncoder, AttentionPredictor,
    and a Generator (CVAE) for paper generation.
    
    This model supports multi-task training for both link prediction 
    (between existing nodes) and paper generation (creating new papers
    with realistic features and citation patterns).
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
        device: str = "cuda",
        # Pre-initialized components
        encoder: Optional[nn.Module] = None,
        predictor: Optional[nn.Module] = None,
        generator: Optional[nn.Module] = None
    ):
        """
        Initialize the integrated citation model.
        
        Args:
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            device: Device to use for model
            encoder: Pre-initialized encoder component
            predictor: Pre-initialized predictor component
            generator: Pre-initialized generator component
        """
        super().__init__()
        
        # Store parameters
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        
        # Set model components
        self.encoder = encoder
        self.predictor = predictor
        self.generator = generator
        
        logger.info("Initialized IntegratedCitationModel")
        logger.info(f"  Encoder: {type(encoder).__name__ if encoder else 'None'}")
        logger.info(f"  Predictor: {type(predictor).__name__ if predictor else 'None'}")
        logger.info(f"  Generator: {type(generator).__name__ if generator else 'None'}")
    
    def forward(self, graph, pos_edge_index=None, neg_edge_index=None):
        """
        Forward pass for link prediction task.
        
        Args:
            graph: Input graph
            pos_edge_index: Positive edge indices
            neg_edge_index: Negative edge indices
            
        Returns:
            Dict containing predictions
        """
        # Get node embeddings
        node_embeddings = self.encoder(graph)
        
        # Predict for positive edges
        if pos_edge_index is not None:
            pos_pred = self.predictor(node_embeddings, pos_edge_index)
            
            # Predict for negative edges
            if neg_edge_index is not None:
                neg_pred = self.predictor(node_embeddings, neg_edge_index)
                return {'pos_pred': pos_pred, 'neg_pred': neg_pred}
            return {'pos_pred': pos_pred}
            
        return {'embeddings': node_embeddings}
    
    def mask_edges(self, graph, edges_to_mask):
        """
        Create a graph with specified edges masked out.
        
        Args:
            graph: Input graph
            edges_to_mask: Edges to mask [2, num_edges]
            
        Returns:
            Masked graph
        """
        # Create a copy of the graph with edges removed
        if not isinstance(graph, GraphData):
            raise TypeError(f"Expected GraphData, got {type(graph)}")
        
        # Create boolean mask (True = keep edge, False = mask edge)
        edge_mask = torch.ones(graph.num_edges, dtype=torch.bool, device=graph.edge_index.device)
        
        # Set mask indices to False
        for i in range(edges_to_mask.shape[1]):
            src, dst = edges_to_mask[0, i], edges_to_mask[1, i]
            mask = (graph.edge_index[0] == src) & (graph.edge_index[1] == dst)
            edge_mask[mask] = False
        
        # Create new graph with masked edges
        masked_graph = GraphData(
            x=graph.x,
            edge_index=graph.edge_index[:, edge_mask],
            edge_attr=graph.edge_attr[edge_mask] if graph.edge_attr is not None else None,
            node_timestamps=graph.node_timestamps
        )
        
        return masked_graph
    
    def compute_link_prediction_loss(self, pos_pred, neg_pred):
        """
        Compute link prediction loss.
        
        Args:
            pos_pred: Predictions for positive edges
            neg_pred: Predictions for negative edges
            
        Returns:
            Link prediction loss
        """
        # Create targets
        pos_targets = torch.ones_like(pos_pred)
        neg_targets = torch.zeros_like(neg_pred)
        
        # Combine predictions and targets
        predictions = torch.cat([pos_pred, neg_pred], dim=0)
        targets = torch.cat([pos_targets, neg_targets], dim=0)
        
        # Compute BCE loss
        loss = F.binary_cross_entropy_with_logits(predictions, targets)
        
        return loss
    
    def compute_generation_loss(self, generator_outputs):
        """
        Compute generation loss.
        
        Args:
            generator_outputs: Outputs from generator
            
        Returns:
            Generation loss
        """
        if not isinstance(generator_outputs, dict):
            raise TypeError(f"Expected dict, got {type(generator_outputs)}")
        
        # Extract loss from generator outputs
        if 'loss' in generator_outputs:
            return generator_outputs['loss']
        
        # Fallback to computing loss manually
        recon_loss = generator_outputs.get('recon_loss', 0)
        kl_loss = generator_outputs.get('kl_loss', 0)
        kl_weight = generator_outputs.get('kl_weight', 0.1)
        
        loss = recon_loss + kl_weight * kl_loss
        return loss
    
    def train_step(
        self,
        graph,
        pos_edge_index,
        neg_edge_index,
        optimizer,
        past_graph=None,
        future_features=None,
        future_timestamps=None,
        task_weights={'link_prediction': 1.0, 'generation': 1.0}
    ):
        """
        Train the model for one step.
        
        Args:
            graph: Input graph
            pos_edge_index: Positive edge indices
            neg_edge_index: Negative edge indices
            optimizer: Optimizer
            past_graph: Past graph for generation task
            future_features: Features of future papers
            future_timestamps: Timestamps of future papers
            task_weights: Weights for each task
            
        Returns:
            Dict of metrics
        """
        # Reset gradients
        optimizer.zero_grad()
        metrics = {}
        
        # Compute link prediction loss if weight > 0
        link_pred_loss = 0
        if task_weights.get('link_prediction', 0) > 0:
            # Mask positive edges to prevent data leakage
            masked_graph = self.mask_edges(graph, pos_edge_index)
            
            # Get node embeddings
            node_embeddings = self.encoder(masked_graph)
            
            # Get predictions
            pos_src, pos_dst = pos_edge_index
            neg_src, neg_dst = neg_edge_index
            
            pos_pred = self.predictor(node_embeddings, pos_edge_index)
            neg_pred = self.predictor(node_embeddings, neg_edge_index)
            
            # Compute loss
            link_pred_loss = self.compute_link_prediction_loss(pos_pred, neg_pred)
            metrics['link_prediction'] = link_pred_loss.item()
        
        # Compute generation loss if weight > 0 and we have generation components
        gen_loss = 0
        if task_weights.get('generation', 0) > 0 and self.generator is not None and future_features is not None:
            if past_graph is None:
                past_graph = graph
            
            # Get node embeddings from past graph
            node_embeddings = self.encoder(past_graph)
            
            # Run generator forward pass
            generator_outputs = self.generator(
                node_embeddings=node_embeddings,
                features=future_features,
                timestamps=future_timestamps if past_graph.node_timestamps is not None else None
            )
            
            # Compute loss
            gen_loss = self.compute_generation_loss(generator_outputs)
            metrics['generation'] = gen_loss.item()
            
            # Additional metrics
            if 'recon_loss' in generator_outputs:
                metrics['gen_recon_loss'] = generator_outputs['recon_loss'].mean().item()
            if 'kl_loss' in generator_outputs:
                metrics['gen_kl_loss'] = generator_outputs['kl_loss'].mean().item()
        
        # Combine losses
        combined_loss = 0
        if task_weights.get('link_prediction', 0) > 0:
            combined_loss += task_weights['link_prediction'] * link_pred_loss
            
        if task_weights.get('generation', 0) > 0 and self.generator is not None and future_features is not None:
            combined_loss += task_weights['generation'] * gen_loss
        
        # Backward pass and optimization
        if torch.is_tensor(combined_loss) and combined_loss.requires_grad:
            try:
                combined_loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
            except Exception as e:
                logger.error(f"Error in backward pass: {e}")
                logger.error(traceback.format_exc())
                # Return metrics without updating weights
                metrics['combined'] = float('inf')
                return metrics
        
        metrics['combined'] = combined_loss.item() if torch.is_tensor(combined_loss) else 0
        return metrics
    
    def validation_step(
        self,
        graph,
        val_pos_edge_index,
        val_neg_edge_index,
        val_past_graph=None,
        val_future_features=None,
        val_future_timestamps=None,
        task_weights={'link_prediction': 1.0, 'generation': 1.0}
    ):
        """
        Validate the model.
        
        Args:
            graph: Input graph
            val_pos_edge_index: Positive edge indices for validation
            val_neg_edge_index: Negative edge indices for validation
            val_past_graph: Past graph for generation validation
            val_future_features: Features of future papers for validation
            val_future_timestamps: Timestamps of future papers for validation
            task_weights: Weights for each task
            
        Returns:
            Dict of metrics
        """
        # No gradient tracking for validation
        with torch.no_grad():
            metrics = {}
            
            # Compute link prediction loss if weight > 0
            if task_weights.get('link_prediction', 0) > 0:
                # Mask positive edges to prevent data leakage
                masked_graph = self.mask_edges(graph, val_pos_edge_index)
                
                # Get node embeddings
                node_embeddings = self.encoder(masked_graph)
                
                # Get predictions
                pos_pred = self.predictor(node_embeddings, val_pos_edge_index)
                neg_pred = self.predictor(node_embeddings, val_neg_edge_index)
                
                # Compute loss
                link_pred_loss = self.compute_link_prediction_loss(pos_pred, neg_pred)
                metrics['link_prediction'] = link_pred_loss.item()
                
                # Compute AUC and AP
                pos_scores = torch.sigmoid(pos_pred).cpu().numpy()
                neg_scores = torch.sigmoid(neg_pred).cpu().numpy()
                y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
                y_score = np.concatenate([pos_scores, neg_scores])
                
                try:
                    metrics['auc'] = roc_auc_score(y_true, y_score)
                    metrics['ap'] = average_precision_score(y_true, y_score)
                except Exception as e:
                    logger.warning(f"Could not compute AUC/AP: {e}")
            
            # Compute generation loss if weight > 0 and we have generation components
            if task_weights.get('generation', 0) > 0 and self.generator is not None and val_future_features is not None:
                if val_past_graph is None:
                    val_past_graph = graph
                
                # Get node embeddings from past graph
                node_embeddings = self.encoder(val_past_graph)
                
                # Run generator forward pass
                generator_outputs = self.generator(
                    node_embeddings=node_embeddings,
                    features=val_future_features,
                    timestamps=val_future_timestamps if val_past_graph.node_timestamps is not None else None
                )
                
                # Compute loss
                gen_loss = self.compute_generation_loss(generator_outputs)
                metrics['generation'] = gen_loss.item()
                
                # Additional metrics
                if 'recon_loss' in generator_outputs:
                    metrics['gen_recon_loss'] = generator_outputs['recon_loss'].mean().item()
                if 'kl_loss' in generator_outputs:
                    metrics['gen_kl_loss'] = generator_outputs['kl_loss'].mean().item()
            
            # Combine losses
            combined_loss = 0
            if task_weights.get('link_prediction', 0) > 0:
                combined_loss += task_weights['link_prediction'] * metrics.get('link_prediction', 0)
                
            if task_weights.get('generation', 0) > 0 and self.generator is not None and val_future_features is not None:
                combined_loss += task_weights['generation'] * metrics.get('generation', 0)
            
            metrics['combined'] = combined_loss
            return metrics
    
    def _predict_citations_for_new_paper(self, node_embeddings, paper_features):
        """
        Predict citation probabilities for new papers.
        
        Args:
            node_embeddings: Node embeddings from the encoder
            paper_features: Features of new papers
            
        Returns:
            Citation probabilities for each (new paper, existing paper) pair
        """
        if self.generator is None:
            raise RuntimeError("Generator component is required for citation prediction")
        
        # Get number of existing and new papers
        num_existing = node_embeddings.size(0)
        num_new = paper_features.size(0)
        
        # Create indices for all possible citations
        # Each new paper could cite any existing paper
        src_indices = torch.arange(num_new, device=node_embeddings.device).repeat_interleave(num_existing)
        dst_indices = torch.arange(num_existing, device=node_embeddings.device).repeat(num_new)
        edge_index = torch.stack([src_indices, dst_indices])
        
        # TODO: Implement actual citation prediction
        # This is a placeholder - in practice, we'd use the predictor component
        # or a specialized citation prediction method from the generator
        citation_probs = torch.rand(num_new, num_existing, device=node_embeddings.device)
        
        return citation_probs
    
    def save(self, path):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'generator_state_dict': self.generator.state_dict() if self.generator else None,
            'model_config': {
                'embed_dim': self.embed_dim,
                'hidden_dim': self.hidden_dim,
                'num_heads': self.num_heads,
                'dropout': self.dropout
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load encoder
        if self.encoder is not None and 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        # Load predictor
        if self.predictor is not None and 'predictor_state_dict' in checkpoint:
            self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        
        # Load generator
        if self.generator is not None and 'generator_state_dict' in checkpoint and checkpoint['generator_state_dict'] is not None:
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
        
        logger.info(f"Model loaded from {path}")
        return self 