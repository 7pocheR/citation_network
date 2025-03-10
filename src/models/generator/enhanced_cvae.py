import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    """
    Encoder network for the Enhanced CVAE.
    
    Maps inputs (paper features) and conditions (graph embeddings) to 
    latent distribution parameters (mu, logvar).
    """
    
    def __init__(self, 
                 input_dim: int, 
                 condition_dim: int, 
                 latent_dim: int, 
                 hidden_dims: List[int],
                 dropout: float = 0.2):
        """
        Initialize the encoder network.
        
        Args:
            input_dim: Dimension of input features
            condition_dim: Dimension of conditioning vectors
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        
        # Input layer (features + conditioning)
        total_input_dim = input_dim + condition_dim
        prev_dim = total_input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder_layers = nn.Sequential(*layers)
        
        # Output layers for mu and logvar
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input features [batch_size, input_dim]
            condition: Conditioning vectors [batch_size, condition_dim]
            
        Returns:
            Tuple of (mu, logvar) for the latent distribution
        """
        # Ensure tensors are on the same device
        device = x.device
        condition = condition.to(device)
        
        # Check shapes and adjust if needed
        if x.size(0) != condition.size(0):
            logger.warning(f"Shape mismatch in encoder: x={x.shape}, condition={condition.shape}")
            
            # Get the batch size we need to match
            batch_size = x.size(0)
            
            # Adjust condition tensor to match x's batch size
            if condition.size(0) < batch_size:
                # Repeat condition to match batch size
                repeat_factor = (batch_size + condition.size(0) - 1) // condition.size(0)
                condition = condition.repeat(repeat_factor, 1)[:batch_size]
            else:
                # Truncate condition to match batch size
                condition = condition[:batch_size]
        
        # Concatenate inputs and conditions
        try:
            inputs = torch.cat([x, condition], dim=1)
            
            # Encode
            hidden = self.encoder_layers(inputs)
            
            # Get distribution parameters
            mu = self.fc_mu(hidden)
            logvar = self.fc_logvar(hidden)
            
            return mu, logvar
        except RuntimeError as e:
            logger.error(f"Error in encoder forward: {str(e)}")
            logger.error(f"Shapes: x={x.shape}, condition={condition.shape}")
            raise

class Decoder(nn.Module):
    """
    Decoder network for the Enhanced CVAE.
    
    Maps latent vectors and conditions to reconstructed outputs.
    """
    
    def __init__(self, 
                 latent_dim: int, 
                 condition_dim: int, 
                 output_dim: int, 
                 hidden_dims: List[int],
                 dropout: float = 0.2):
        """
        Initialize the decoder network.
        
        Args:
            latent_dim: Dimension of latent space
            condition_dim: Dimension of conditioning vectors
            output_dim: Dimension of output features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        
        # Build decoder layers
        layers = []
        
        # Input layer (latent + conditioning)
        total_input_dim = latent_dim + condition_dim
        prev_dim = total_input_dim
        
        # Hidden layers (reverse order of encoder for symmetry)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.decoder_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            condition: Conditioning vectors [batch_size, condition_dim]
            
        Returns:
            Reconstructed features [batch_size, output_dim]
        """
        # Ensure tensors are on the same device
        device = z.device
        condition = condition.to(device)
        
        # Check shapes and adjust if needed
        if z.size(0) != condition.size(0):
            logger.warning(f"Shape mismatch in decoder: z={z.shape}, condition={condition.shape}")
            
            # Get the batch size we need to match
            batch_size = z.size(0)
            
            # Adjust condition tensor to match z's batch size
            if condition.size(0) < batch_size:
                # Repeat condition to match batch size
                repeat_factor = (batch_size + condition.size(0) - 1) // condition.size(0)
                condition = condition.repeat(repeat_factor, 1)[:batch_size]
            else:
                # Truncate condition to match batch size
                condition = condition[:batch_size]
        
        # Concatenate latent and condition
        try:
            inputs = torch.cat([z, condition], dim=1)
            
            # Decode
            decoded = self.decoder_layers(inputs)
            
            # Final output layer with appropriate activation
            reconstructed = self.output_layer(decoded)
            
            return reconstructed
        except RuntimeError as e:
            logger.error(f"Error in decoder forward: {str(e)}")
            logger.error(f"Shapes: z={z.shape}, condition={condition.shape}")
            raise

class EnhancedCVAE(nn.Module):
    """
    Enhanced Conditional Variational Autoencoder (CVAE) model with improved architecture.
    
    This CVAE implementation provides:
    1. Clear separation between reconstruction and generation modes
    2. Proper KL annealing for stable training
    3. Enhanced conditioning mechanisms
    4. Support for guided generation
    
    The model takes node features as input and learns to generate similar features
    conditioned on embeddings from the encoder.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 condition_dim: int, 
                 latent_dim: int,
                 hidden_dims: List[int],
                 kl_weight: float = 0.1,
                 beta_warmup_steps: int = 1000,
                 dropout: float = 0.2,
                 citation_weight: float = 1.0,
                 feature_weight: float = 0.01):
        """
        Initialize the enhanced CVAE.
        
        Args:
            input_dim: Dimension of input features
            condition_dim: Dimension of conditioning vectors
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden dimensions for encoder/decoder networks
            kl_weight: Weight for KL divergence loss term
            beta_warmup_steps: Number of steps for KL annealing (warmup)
            dropout: Dropout rate
            citation_weight: Weight for citation loss
            feature_weight: Weight for feature reconstruction loss
        """
        super().__init__()
        
        # Store dimensions and hyperparameters
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.kl_weight = kl_weight
        self.beta_warmup_steps = beta_warmup_steps
        self.dropout = dropout
        self.citation_weight = citation_weight
        self.feature_weight = feature_weight
        
        # Set up KL annealing parameters
        self.current_step = 0
        self.beta = 0.0  # Initial KL weight (will be annealed)
        
        # Create encoder and decoder networks
        self.encoder = Encoder(
            input_dim=input_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            output_dim=input_dim,
            hidden_dims=hidden_dims[::-1],  # Reverse hidden_dims for symmetric architecture
            dropout=dropout
        )
        
        # Optional citation predictor (for future development)
        self.use_citation_predictor = False
        self.citation_predictor = None
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized EnhancedCVAE with input_dim={input_dim}, "
                    f"condition_dim={condition_dim}, latent_dim={latent_dim}")
    
    def _init_weights(self):
        """Initialize weights for the CVAE."""
        # Initialize Gaussian prior parameters
        self.register_buffer('prior_mean', torch.zeros(1, self.latent_dim))
        self.register_buffer('prior_std', torch.ones(1, self.latent_dim))
    
    def _adapt_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Adapt condition tensor to appropriate dimensions if needed.
        
        This handles cases where the condition tensor might have the wrong shape.
        
        Args:
            condition: Input condition tensor
            
        Returns:
            Adapted condition tensor
        """
        # If condition is None, create zero tensor
        if condition is None:
            if hasattr(self, 'prior_mean'):
                device = self.prior_mean.device
            else:
                device = 'cpu'
            batch_size = 1  # Default batch size
            return torch.zeros(batch_size, self.condition_dim, device=device)
        
        # Ensure condition is a tensor
        if not isinstance(condition, torch.Tensor):
            condition = torch.tensor(condition, dtype=torch.float32)
        
        # Check if dimensions need to be adapted
        if condition.dim() == 1:
            # Add batch dimension
            condition = condition.unsqueeze(0)
        
        # Check if condition dimension matches expected
        if condition.size(-1) != self.condition_dim:
            logger.warning(f"Condition dimension mismatch. Expected: {self.condition_dim}, "
                          f"Got: {condition.size(-1)}. Adapting condition.")
            
            device = condition.device
            batch_size = condition.size(0)
            
            # Two adaptation strategies based on size:
            if condition.size(-1) > self.condition_dim:
                # If larger, truncate to expected size
                condition = condition[:, :self.condition_dim]
            else:
                # If smaller, pad with zeros
                padding = torch.zeros(batch_size, self.condition_dim - condition.size(-1), device=device)
                condition = torch.cat([condition, padding], dim=-1)
        
        return condition
    
    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode inputs to latent distribution parameters.
        
        Args:
            x: Input features [batch_size, input_dim]
            condition: Conditioning vectors [batch_size, condition_dim]
            
        Returns:
            Tuple of (mu, logvar) for the latent distribution
        """
        # Adapt condition if needed
        condition = self._adapt_condition(condition)
        
        # Ensure x and condition are on the same device
        device = x.device
        condition = condition.to(device)
        
        # Get distribution parameters from encoder
        mu, logvar = self.encoder(x, condition)
        
        return mu, logvar
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to reconstructions.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
            condition: Conditioning vectors [batch_size, condition_dim]
            
        Returns:
            Reconstructed features [batch_size, input_dim]
        """
        # Adapt condition if needed
        condition = self._adapt_condition(condition)
        
        # Ensure z and condition are on the same device
        device = z.device
        condition = condition.to(device)
        
        # Decode
        reconstructed = self.decoder(z, condition)
        
        return reconstructed
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from the latent distribution.
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
            
        Returns:
            Sampled latent vectors [batch_size, latent_dim]
        """
        # Calculate standard deviation from log variance
        std = torch.exp(0.5 * logvar)
        
        # Sample epsilon from standard normal distribution
        eps = torch.randn_like(std)
        
        # Reparameterize: z = mu + eps * std
        z = mu + eps * std
        
        return z
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training (reconstruction mode).
        
        Args:
            x: Input features [batch_size, input_dim]
            condition: Conditioning vectors [batch_size, condition_dim]
            
        Returns:
            Tuple of (reconstructed, mu, logvar)
        """
        # Encode inputs to latent distribution parameters
        mu, logvar = self.encode(x, condition)
        
        # Sample from distribution using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decode latent vectors to reconstructions
        reconstructed = self.decode(z, condition)
        
        return reconstructed, mu, logvar
    
    def generate(self, 
                condition: torch.Tensor, 
                num_samples: Optional[int] = None,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Generate samples from the prior distribution (generation mode).
        
        Args:
            condition: Conditioning vectors [batch_size, condition_dim]
            num_samples: Number of samples to generate per condition (if None, uses batch_size)
            temperature: Temperature parameter for sampling (higher = more diverse)
            
        Returns:
            Generated features [batch_size * num_samples, input_dim]
        """
        # Adapt condition if needed
        condition = self._adapt_condition(condition)
        
        # Determine batch size and sample count
        batch_size = condition.size(0)
        num_samples = num_samples or 1
        
        # Get device from condition
        device = condition.device
        
        # Repeat conditions if generating multiple samples per condition
        if num_samples > 1:
            # Create [batch_size * num_samples, condition_dim] tensor
            condition = condition.repeat_interleave(num_samples, dim=0)
        
        # Sample from prior distribution N(0, I)
        z = torch.randn(batch_size * num_samples, self.latent_dim, device=device) * temperature
        
        # Decode latent vectors to generated samples
        generated = self.decode(z, condition)
        
        return generated
    
    def generate_with_seed(self,
                          seed_features: torch.Tensor,
                          condition: torch.Tensor,
                          noise_ratio: float = 0.5,
                          temperature: float = 1.0) -> torch.Tensor:
        """
        Generate samples with guidance from seed features.
        
        This allows for controlled generation by interpolating between
        the encoded seed features and random noise.
        
        Args:
            seed_features: Seed features to guide generation [batch_size, input_dim]
            condition: Conditioning vectors [batch_size, condition_dim]
            noise_ratio: Ratio of random noise to encoded features (0 = deterministic, 1 = random)
            temperature: Temperature for the random component
            
        Returns:
            Generated features [batch_size, input_dim]
        """
        # Encode seed features
        mu, logvar = self.encode(seed_features, condition)
        
        # Get device
        device = mu.device
        
        # Sample random noise
        z_random = torch.randn_like(mu) * temperature
        
        # Interpolate between encoded features and random noise
        z = mu * (1 - noise_ratio) + z_random * noise_ratio
        
        # Decode interpolated latent vectors
        generated = self.decode(z, condition)
        
        return generated
    
    def generate_with_guidance(self,
                             guidance_features: torch.Tensor,
                             condition: torch.Tensor,
                             guidance_strength: float = 0.5,
                             temperature: float = 1.0) -> torch.Tensor:
        """
        Generate samples with guidance from partial features.
        
        This is useful for controlled generation with partial specifications.
        
        Args:
            guidance_features: Partial features to guide generation [batch_size, input_dim]
            condition: Conditioning vectors [batch_size, condition_dim]
            guidance_strength: Strength of guidance (0 = random, 1 = deterministic)
            temperature: Temperature for the random component
            
        Returns:
            Generated features [batch_size, input_dim]
        """
        # Encode guidance features to get latent representation
        with torch.no_grad():
            mu, logvar = self.encode(guidance_features, condition)
        
        # Sample random noise
        device = mu.device
        z_random = torch.randn_like(mu) * temperature
        
        # Interpolate between random noise and guidance in latent space
        z = z_random * (1 - guidance_strength) + mu * guidance_strength
        
        # Decode to get generated features
        generated = self.decode(z, condition)
        
        return generated
    
    def predict_citations(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Predict citation probabilities from latent vectors.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
            condition: Conditioning vectors [batch_size, condition_dim]
            
        Returns:
            Citation probability scores [batch_size, num_nodes]
        """
        if not self.use_citation_predictor or self.citation_predictor is None:
            # Default: no citation prediction
            return None
        
        # Use citation predictor if available
        citation_scores = self.citation_predictor(z, condition)
        
        return citation_scores
    
    def compute_loss(self, 
                    x: torch.Tensor, 
                    condition: torch.Tensor,
                    citation_labels: Optional[torch.Tensor] = None,
                    step: Optional[int] = None,
                    feature_weight: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        Compute the CVAE loss for training.
        
        Args:
            x: Input features [batch_size, input_dim]
            condition: Conditioning vectors [batch_size, condition_dim]
            citation_labels: Optional citation ground truth labels [batch_size, num_nodes]
            step: Current training step (for KL annealing)
            feature_weight: Weight for feature reconstruction loss
            
        Returns:
            Dictionary of loss components
        """
        # Ensure all tensors are on the same device
        device = x.device
        condition = condition.to(device)
        if citation_labels is not None:
            citation_labels = citation_labels.to(device)
        
        # Adapt condition if needed
        condition = self._adapt_condition(condition)
        
        # Update KL annealing factor
        if step is not None:
            self.current_step = step
        
        if self.beta_warmup_steps > 0:
            self.beta = min(1.0, self.current_step / self.beta_warmup_steps)
        else:
            self.beta = 1.0
        
        # Forward pass
        reconstructed, mu, logvar = self.forward(x, condition)
        
        # Reconstruction loss (feature reconstruction)
        # Mean squared error for feature values
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # KL divergence loss (regularization)
        # Analytical form for Gaussian prior
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size
        
        # Calculate total loss with weighting
        total_loss = feature_weight * recon_loss + self.beta * self.kl_weight * kl_loss
        
        # Initialize citation loss
        citation_loss = torch.tensor(0.0, device=device)
        
        # Add citation loss if labels are provided and predictor exists
        if citation_labels is not None and self.use_citation_predictor and self.citation_predictor is not None:
            # Predict citations using citation predictor
            citation_probs = self.predict_citations(mu, condition)
            
            # Binary cross entropy loss for citation prediction
            citation_loss = F.binary_cross_entropy(citation_probs, citation_labels, reduction='mean')
            
            # Add to total loss
            total_loss = total_loss + self.citation_weight * citation_loss
        
        # Collect all loss components
        loss_dict = {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'weighted_kl_loss': self.beta * self.kl_weight * kl_loss,
            'citation_loss': citation_loss
        }
        
        # Ensure all loss components require gradients
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor) and not value.requires_grad:
                loss_dict[key] = torch.tensor(value.item(), device=device, requires_grad=True)
        
        return loss_dict
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'condition_dim': self.condition_dim,
            'latent_dim': self.latent_dim,
            'hidden_dims': self.hidden_dims,
            'kl_weight': self.kl_weight,
            'beta_warmup_steps': self.beta_warmup_steps,
            'dropout': self.dropout,
            'citation_weight': self.citation_weight,
            'feature_weight': self.feature_weight
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"EnhancedCVAE(input_dim={self.input_dim}, "
                f"condition_dim={self.condition_dim}, "
                f"latent_dim={self.latent_dim}, "
                f"hidden_dims={self.hidden_dims})") 