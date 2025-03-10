import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union

from src.data.datasets import GraphData


class BaseGenerator(nn.Module, ABC):
    """Base interface for all paper generators in the citation network project.
    
    The generator creates new paper nodes with attributes and connections, conditional
    on both the current state of the citation network and user-provided inputs such as
    desired topics or research areas.
    
    This is the main task of the system, aimed at creating a system that can suggest
    new research directions by generating complete paper nodes that extend the existing
    citation network in a meaningful and realistic way.
    
    All generator implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, 
                 embed_dim: int, 
                 node_feature_dim: int,
                 condition_dim: Optional[int] = None,
                 latent_dim: int = 128,
                 **kwargs):
        """Initialize the base generator.
        
        Args:
            embed_dim (int): Dimensionality of node embeddings from the encoder
            node_feature_dim (int): Dimensionality of node features to generate
            condition_dim (Optional[int]): Dimensionality of conditional inputs
                (e.g., topics, keywords). If None, no conditioning is used.
            latent_dim (int): Dimensionality of the latent space
            **kwargs: Additional parameters specific to the generator implementation
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.node_feature_dim = node_feature_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
    @abstractmethod
    def forward(self, 
                node_embeddings: torch.Tensor, 
                conditions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate new paper node features.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings from the encoder
                [num_nodes, embed_dim]
            conditions (Optional[torch.Tensor]): Conditional inputs such as desired
                topics or research areas [batch_size, condition_dim]
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Generated node features [batch_size, node_feature_dim]
                - Mean of the latent distribution (μ) [batch_size, latent_dim]
                - Log variance of the latent distribution (log σ²) [batch_size, latent_dim]
        """
        pass
    
    @abstractmethod
    def encode(self, 
              node_features: torch.Tensor, 
              node_embeddings: torch.Tensor,
              conditions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode paper features into the latent space.
        
        Args:
            node_features (torch.Tensor): Node features to encode
                [batch_size, node_feature_dim]
            node_embeddings (torch.Tensor): Node embeddings from the encoder
                [batch_size, embed_dim]
            conditions (Optional[torch.Tensor]): Conditional inputs
                [batch_size, condition_dim]
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Mean of the latent distribution (μ) [batch_size, latent_dim]
                - Log variance of the latent distribution (log σ²) [batch_size, latent_dim]
        """
        pass
    
    @abstractmethod
    def decode(self, 
              z: torch.Tensor, 
              node_embeddings: Optional[torch.Tensor] = None,
              conditions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode latent representations into paper features.
        
        Args:
            z (torch.Tensor): Latent representations [batch_size, latent_dim]
            node_embeddings (Optional[torch.Tensor]): Node embeddings from the encoder
                [batch_size, embed_dim]
            conditions (Optional[torch.Tensor]): Conditional inputs
                [batch_size, condition_dim]
                
        Returns:
            torch.Tensor: Generated node features [batch_size, node_feature_dim]
        """
        pass
    
    @abstractmethod
    def sample(self, 
              num_samples: int, 
              node_embeddings: Optional[torch.Tensor] = None,
              conditions: Optional[torch.Tensor] = None,
              temperature: float = 1.0) -> torch.Tensor:
        """Sample new paper features from the generative model.
        
        Args:
            num_samples (int): Number of papers to generate
            node_embeddings (Optional[torch.Tensor]): Node embeddings from the encoder.
                If provided, these are used for conditioning the generation.
                [num_nodes, embed_dim]
            conditions (Optional[torch.Tensor]): Conditional inputs such as desired
                topics or research areas [num_samples, condition_dim]
            temperature (float): Temperature parameter for sampling (higher = more diverse)
                
        Returns:
            torch.Tensor: Generated node features [num_samples, node_feature_dim]
        """
        pass
    
    @abstractmethod
    def generate_papers_for_network(self,
                                   graph: GraphData,
                                   encoder_embeddings: torch.Tensor,
                                   num_papers: int,
                                   conditions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Generate new papers that extend the citation network.
        
        This is the main interface for paper generation, creating both the paper 
        features and metadata such as title, abstract, and potential citations.
        
        Args:
            graph (GraphData): The current citation network
            encoder_embeddings (torch.Tensor): Node embeddings from the encoder
                [num_nodes, embed_dim]
            num_papers (int): Number of papers to generate
            conditions (Optional[torch.Tensor]): Conditional inputs such as desired
                topics or research areas [num_papers, condition_dim]
                
        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]: 
                - Generated node features [num_papers, node_feature_dim]
                - List of paper metadata dictionaries, containing:
                  * 'title': Generated paper title
                  * 'abstract': Generated paper abstract
                  * 'topics': List of relevant topics
                  * 'year': Publication year
                  * 'citations': List of papers this paper cites (indices)
                  * 'cited_by': List of existing papers that would cite this paper (indices)
        """
        pass
    
    @abstractmethod
    def generate_temporal_papers(self,
                                graph: GraphData,
                                encoder_embeddings: torch.Tensor,
                                time_threshold: float,
                                future_window: float,
                                num_papers: int,
                                conditions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Generate new papers for a future time window.
        
        Generate papers that would be published in the time window
        [time_threshold, time_threshold + future_window], based on the state
        of the citation network at time_threshold.
        
        Args:
            graph (GraphData): The current citation network
            encoder_embeddings (torch.Tensor): Node embeddings from the encoder
                [num_nodes, embed_dim]
            time_threshold (float): The timestamp threshold for the snapshot
            future_window (float): The length of the future time window
            num_papers (int): Number of papers to generate
            conditions (Optional[torch.Tensor]): Conditional inputs such as desired
                topics or research areas [num_papers, condition_dim]
                
        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]: 
                - Generated node features [num_papers, node_feature_dim]
                - List of paper metadata dictionaries (as in generate_papers_for_network)
        """
        pass
    
    def reparameterize(self, 
                      mu: torch.Tensor, 
                      logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
                
        Returns:
            torch.Tensor: Sampled latent vectors
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_config(self) -> Dict[str, Any]:
        """Get generator configuration parameters.
        
        Returns:
            Dict[str, Any]: Dictionary containing configuration parameters
        """
        return {
            'embed_dim': self.embed_dim,
            'node_feature_dim': self.node_feature_dim,
            'condition_dim': self.condition_dim,
            'latent_dim': self.latent_dim,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseGenerator':
        """Create a generator instance from configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
                
        Returns:
            BaseGenerator: An instance of the generator
        """
        return cls(**config) 