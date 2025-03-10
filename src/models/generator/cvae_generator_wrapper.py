"""
CVAE Generator Wrapper

This module provides a wrapper around the enhanced_cvae.py implementation,
adapting it to work within the integrated citation model framework.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from src.models.generator.enhanced_cvae import EnhancedCVAE
from src.models.generator.feature_preprocessing import FeaturePreprocessor
from src.data.dataset import GraphData

logger = logging.getLogger(__name__)

class CVAEGenerator(nn.Module):
    """
    Wrapper around EnhancedCVAE for generating paper features and citation patterns.
    This adapts the CVAE implementation to work with the IntegratedCitationModel.
    """
    
    def __init__(
        self,
        embed_dim: int,
        node_feature_dim: int,
        condition_dim: int,
        latent_dim: int = 128,
        hidden_dims: List[int] = [256, 256],
        n_pca_components: float = 0.95,
        kl_weight: float = 0.1,
        beta_warmup_steps: int = 1000,
        dropout: float = 0.1,
        standardize_features: bool = True,
        projection_pooling: str = 'combined',
        projection_layers: int = 2,
        timestamp_embed_dim: int = 16,
        use_layer_norm: bool = True,
        global_embed_multiplier: int = 12
    ):
        """
        Initialize the CVAE Generator wrapper.
        
        Args:
            embed_dim: Dimension of node embeddings from encoder
            node_feature_dim: Dimension of original node features
            condition_dim: Dimension of conditioning vectors
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions
            n_pca_components: Number of PCA components (or variance ratio)
            kl_weight: Weight for KL divergence loss
            beta_warmup_steps: Steps for annealing KL weight
            dropout: Dropout rate
            standardize_features: Whether to standardize features
            projection_pooling: Method for pooling embeddings ('mean', 'max', 'weighted', 'combined')
            projection_layers: Number of layers in projection networks
            timestamp_embed_dim: Dimension for timestamp embeddings
            use_layer_norm: Whether to use layer normalization
            global_embed_multiplier: Multiplier for global embedding dimension
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.node_feature_dim = node_feature_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.kl_weight = kl_weight
        self.beta_warmup_steps = beta_warmup_steps
        self.step_count = 0
        self.projection_pooling = projection_pooling
        
        # Feature preprocessing
        self.feature_preprocessor = FeaturePreprocessor(
            n_components=n_pca_components,
            standardize=standardize_features
        )
        self.feature_dim = node_feature_dim  # Will be updated after fitting
        
        # Initialize projection networks for conditioning
        self.projection_modules = self._build_projection_networks(
            embed_dim=embed_dim,
            condition_dim=condition_dim,
            num_layers=projection_layers,
            timestamp_dim=timestamp_embed_dim,
            use_layer_norm=use_layer_norm
        )
        
        # Global embedding for conditioning
        global_embed_dim = condition_dim * global_embed_multiplier
        self.global_embedding = nn.Parameter(torch.randn(global_embed_dim))
        self.global_projection = nn.Sequential(
            nn.Linear(global_embed_dim, condition_dim),
            nn.LayerNorm(condition_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, condition_dim)
        )
        
        # Timestamp embeddings
        self.timestamp_embedding = nn.Linear(1, timestamp_embed_dim)
        
        # Will be initialized after feature_preprocessor is fitted
        self.cvae = None
        self.is_fitted = False
        
        # Register forward hook for debugging
        self.register_forward_hook(self._forward_hook)
    
    def _forward_hook(self, module, input, output):
        """Debug hook to trace forward pass."""
        if not hasattr(self, '_forward_count'):
            self._forward_count = 0
        self._forward_count += 1
        
        if self._forward_count % 100 == 0:
            logger.debug(f"Forward pass {self._forward_count} - Input shapes: {[x.shape if isinstance(x, torch.Tensor) else type(x) for x in input]}")
    
    def _build_projection_networks(self, embed_dim, condition_dim, num_layers=2, 
                                  timestamp_dim=16, use_layer_norm=True):
        """Build projection networks for different pooling methods."""
        modules = nn.ModuleDict()
        
        # Mean projection
        mean_layers = []
        in_dim = embed_dim
        for i in range(num_layers - 1):
            mean_layers.append(nn.Linear(in_dim, condition_dim))
            if use_layer_norm:
                mean_layers.append(nn.LayerNorm(condition_dim))
            mean_layers.append(nn.ReLU())
            mean_layers.append(nn.Dropout(0.1))
            in_dim = condition_dim
        mean_layers.append(nn.Linear(in_dim, condition_dim))
        modules['mean'] = nn.Sequential(*mean_layers)
        
        # Max projection (same architecture)
        max_layers = []
        in_dim = embed_dim
        for i in range(num_layers - 1):
            max_layers.append(nn.Linear(in_dim, condition_dim))
            if use_layer_norm:
                max_layers.append(nn.LayerNorm(condition_dim))
            max_layers.append(nn.ReLU())
            max_layers.append(nn.Dropout(0.1))
            in_dim = condition_dim
        max_layers.append(nn.Linear(in_dim, condition_dim))
        modules['max'] = nn.Sequential(*max_layers)
        
        # Weighted projection
        weighted_layers = []
        in_dim = embed_dim + timestamp_dim
        for i in range(num_layers - 1):
            weighted_layers.append(nn.Linear(in_dim, condition_dim))
            if use_layer_norm:
                weighted_layers.append(nn.LayerNorm(condition_dim))
            weighted_layers.append(nn.ReLU())
            weighted_layers.append(nn.Dropout(0.1))
            in_dim = condition_dim
        weighted_layers.append(nn.Linear(in_dim, condition_dim))
        modules['weighted'] = nn.Sequential(*weighted_layers)
        
        # Attention weights
        modules['attention'] = nn.Sequential(
            nn.Linear(embed_dim + timestamp_dim, condition_dim),
            nn.Tanh(),
            nn.Linear(condition_dim, 1)
        )
        
        # Combined projection
        if self.projection_pooling == 'combined':
            combined_layers = []
            in_dim = condition_dim * 3  # Mean, max, and weighted
            for i in range(num_layers - 1):
                combined_layers.append(nn.Linear(in_dim, condition_dim))
                if use_layer_norm:
                    combined_layers.append(nn.LayerNorm(condition_dim))
                combined_layers.append(nn.ReLU())
                combined_layers.append(nn.Dropout(0.1))
                in_dim = condition_dim
            combined_layers.append(nn.Linear(in_dim, condition_dim))
            modules['combined'] = nn.Sequential(*combined_layers)
        
        return modules
    
    def _create_condition_vector(self, node_embeddings, timestamps=None):
        """
        Create fixed-dimensional condition vector from node embeddings.
        
        Args:
            node_embeddings: Tensor of node embeddings [num_nodes, embed_dim]
            timestamps: Optional tensor of timestamps [num_nodes]
            
        Returns:
            condition_vector: Fixed-dimensional conditioning vector
        """
        device = node_embeddings.device
        
        # Mean pooling
        mean_embed = torch.mean(node_embeddings, dim=0)
        mean_projected = self.projection_modules['mean'](mean_embed)
        
        # Max pooling
        max_embed, _ = torch.max(node_embeddings, dim=0)
        max_projected = self.projection_modules['max'](max_embed)
        
        # Weighted pooling with temporal attention
        if timestamps is not None:
            # Convert timestamps to embeddings
            time_embeds = self.timestamp_embedding(timestamps.view(-1, 1))
            
            # Concatenate with node embeddings
            node_time_embeds = torch.cat([node_embeddings, time_embeds], dim=1)
            
            # Compute attention weights
            attn_weights = self.projection_modules['attention'](node_time_embeds)
            attn_weights = F.softmax(attn_weights, dim=0)
            
            # Apply weighted pooling
            weighted_embed = torch.sum(node_embeddings * attn_weights, dim=0)
            
            # Project the weighted embedding
            weighted_projected = self.projection_modules['weighted'](
                torch.cat([weighted_embed, time_embeds.mean(dim=0)], dim=0)
            )
        else:
            # Fallback without timestamps
            weighted_projected = torch.zeros_like(mean_projected)
        
        # Global bias term
        global_proj = self.global_projection(self.global_embedding)
        
        # Combine projections based on strategy
        if self.projection_pooling == 'mean':
            condition = mean_projected + 0.1 * global_proj
        elif self.projection_pooling == 'max':
            condition = max_projected + 0.1 * global_proj
        elif self.projection_pooling == 'weighted':
            condition = weighted_projected + 0.1 * global_proj
        else:  # combined
            combined = torch.cat([mean_projected, max_projected, weighted_projected], dim=0)
            condition = self.projection_modules['combined'](combined) + 0.1 * global_proj
        
        return condition
    
    def fit_preprocessor(self, features):
        """
        Fit the feature preprocessor on the input features.
        
        Args:
            features: Node features [num_nodes, feature_dim]
            
        Returns:
            self
        """
        self.feature_preprocessor.fit(features)
        self.feature_dim = self.feature_preprocessor.output_dim
        
        # Now initialize the CVAE with the correct feature dimension
        self.cvae = EnhancedCVAE(
            input_dim=self.feature_dim,
            condition_dim=self.condition_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=0.1
        )
        
        self.is_fitted = True
        logger.info(f"Initialized CVAE with input dim {self.feature_dim}, condition dim {self.condition_dim}")
        return self
    
    def forward(self, node_embeddings, features, timestamps=None):
        """
        Forward pass for the CVAE generator.
        
        Args:
            node_embeddings: Tensor of node embeddings [num_nodes, embed_dim]
            features: Tensor of node features to reconstruct [batch_size, feature_dim]
            timestamps: Optional tensor of timestamps [num_nodes]
            
        Returns:
            Dict with reconstructed features and loss information
        """
        # Make sure preprocessor is fitted
        if not self.is_fitted:
            with torch.no_grad():
                self.fit_preprocessor(features.detach().cpu())
        
        # Preprocess features
        processed_features = self.feature_preprocessor.transform(features)
        
        # Create condition vector from embeddings
        condition = self._create_condition_vector(node_embeddings, timestamps)
        
        # Run CVAE forward pass
        outputs = self.cvae(processed_features, condition, kl_weight=self.get_kl_weight())
        
        # Inverse transform the reconstructed features
        reconstructed_features = self.feature_preprocessor.inverse_transform(outputs['reconstructed'])
        
        # Update step count for KL annealing
        self.step_count += 1
        
        return {
            'reconstructed': reconstructed_features,
            'processed_reconstructed': outputs['reconstructed'],
            'processed_input': processed_features,
            'latent': outputs['z'],
            'mu': outputs['mu'],
            'logvar': outputs['logvar'],
            'kl_loss': outputs['kl_loss'],
            'recon_loss': outputs['recon_loss'],
            'loss': outputs['loss']
        }
    
    def get_kl_weight(self):
        """Get the current KL weight based on annealing schedule."""
        if self.beta_warmup_steps > 0:
            weight = min(1.0, self.step_count / self.beta_warmup_steps) * self.kl_weight
        else:
            weight = self.kl_weight
        return weight
    
    def generate(self, node_embeddings, num_samples=1, temperature=1.0, timestamps=None):
        """
        Generate new paper features.
        
        Args:
            node_embeddings: Tensor of node embeddings [num_nodes, embed_dim]
            num_samples: Number of samples to generate
            temperature: Temperature for sampling
            timestamps: Optional tensor of timestamps [num_nodes]
            
        Returns:
            generated_features: Tensor of generated features [num_samples, feature_dim]
        """
        if not self.is_fitted:
            raise RuntimeError("Generator not fitted. Call fit_preprocessor first.")
        
        # Create condition vector
        condition = self._create_condition_vector(node_embeddings, timestamps)
        
        # Expand condition for each sample
        condition_expanded = condition.unsqueeze(0).expand(num_samples, -1)
        
        # Sample from the latent space
        z = torch.randn(num_samples, self.latent_dim, device=condition.device) * temperature
        
        # Decode to get features
        processed_features = self.cvae.decode(z, condition_expanded)
        
        # Inverse transform to original feature space
        generated_features = self.feature_preprocessor.inverse_transform(processed_features)
        
        return generated_features
    
    def compute_loss(self, outputs, reduction='mean'):
        """
        Compute the loss for the generator.
        
        Args:
            outputs: Dict of outputs from forward pass
            reduction: Loss reduction method
            
        Returns:
            Dict of loss components
        """
        # Extract components
        recon_loss = outputs['recon_loss']
        kl_loss = outputs['kl_loss']
        
        # Apply reduction
        if reduction == 'mean':
            recon_loss = recon_loss.mean()
            kl_loss = kl_loss.mean()
        elif reduction == 'sum':
            recon_loss = recon_loss.sum()
            kl_loss = kl_loss.sum()
        
        # Compute total loss
        loss = recon_loss + self.get_kl_weight() * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'kl_weight': self.get_kl_weight()
        } 