import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging
from src.data.dataset import GraphData
from src.models.encoder.tgn_encoder import TGNEncoder
from src.models.encoder.tgn import TemporalGraphNetwork
from src.models.predictors.distance_predictor import DistancePredictor

# Setup logger
logger = logging.getLogger(__name__)

# Legacy generator imports - only attempt if needed
try:
    from src.models.generator.generator import CitationPaperGenerator, TopicConditionedGenerator
except ImportError:
    logger.warning("Generator modules not available. Using placeholder implementations.")
    
    # Placeholder classes to prevent errors
    class CitationPaperGenerator(nn.Module):
        """Placeholder for legacy CitationPaperGenerator."""
        def __init__(self, *args, **kwargs):
            super().__init__()
            logger.warning("Using placeholder CitationPaperGenerator - actual implementation not available")
            
    class TopicConditionedGenerator(nn.Module):
        """Placeholder for legacy TopicConditionedGenerator."""
        def __init__(self, *args, **kwargs):
            super().__init__()
            logger.warning("Using placeholder TopicConditionedGenerator - actual implementation not available")

class DynamicCitationNetworkModel(nn.Module):
    """
    Complete dynamic citation network model.
    
    This model integrates:
    1. A temporal graph network encoder with hyperbolic embeddings
    2. A paper generator that creates new nodes with intrinsic features and citation links
    """
    
    def __init__(self, 
                 num_nodes: int,
                 node_feature_dim: int,
                 memory_dim: int = 100,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 latent_dim: int = 128,
                 time_dim: int = 10,
                 message_dim: int = 100,
                 num_gnn_layers: int = 2,
                 edge_dim: Optional[int] = None,
                 use_memory: bool = True,
                 use_hierarchical_generator: bool = True,
                 use_hyperbolic: bool = True,
                 curvature: float = 1.0,
                 dropout: float = 0.2):
        """
        Initialize the dynamic citation network model.
        
        Args:
            num_nodes: Number of nodes in the graph
            node_feature_dim: Dimension of node features
            memory_dim: Dimension of memory vectors
            embedding_dim: Dimension of node embeddings
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
            time_dim: Dimension of time encoding
            message_dim: Dimension of message vectors
            num_gnn_layers: Number of GNN layers
            edge_dim: Dimension of edge features (if available)
            use_memory: Whether to use memory module or not
            use_hierarchical_generator: Whether to use hierarchical or basic generator
            use_hyperbolic: Whether to use hyperbolic embeddings
            curvature: Curvature of hyperbolic space
            dropout: Dropout rate
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.embedding_dim = embedding_dim
        
        # Initialize TGN encoder
        self.encoder = TemporalGraphNetwork(
            num_nodes=num_nodes,
            node_feature_dim=node_feature_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            embedding_dim=embedding_dim,
            message_dim=message_dim,
            edge_dim=edge_dim,
            num_gnn_layers=num_gnn_layers,
            use_memory=use_memory,
            hyperbolic=use_hyperbolic,
            curvature=curvature,
            dropout=dropout
        )
        
        # Initialize paper generator
        self.generator = CitationPaperGenerator(
            encoder_model=self.encoder,
            feature_dim=node_feature_dim,
            embedding_dim=embedding_dim,
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            use_hierarchical=use_hierarchical_generator,
            dropout=dropout
        )
    
    def forward(self, 
               snapshots: List[Dict[str, torch.Tensor]], 
               paper_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            snapshots: List of graph snapshots
            paper_features: Features of papers to generate/reconstruct
            
        Returns:
            Dictionary of outputs from the generator
        """
        return self.generator(snapshots, paper_features)
    
    def train_encoder(self, 
                     snapshots: List[GraphData],
                     link_prediction_edges: torch.Tensor,
                     link_prediction_labels: torch.Tensor) -> torch.Tensor:
        """
        Train the encoder model with link prediction.
        
        Args:
            snapshots: List of graph snapshots
            link_prediction_edges: Edges for link prediction (2, num_edges)
            link_prediction_labels: Labels for link prediction (num_edges)
            
        Returns:
            Loss value
        """
        # Get node embeddings
        embeddings = self.encoder(snapshots)
        
        # Extract source and target node embeddings
        source_nodes = link_prediction_edges[0]
        target_nodes = link_prediction_edges[1]
        
        source_embeddings = embeddings[source_nodes]
        target_embeddings = embeddings[target_nodes]
        
        # Make sure embeddings have gradients
        if not source_embeddings.requires_grad:
            source_embeddings = source_embeddings.detach().clone().requires_grad_(True)
        if not target_embeddings.requires_grad:
            target_embeddings = target_embeddings.detach().clone().requires_grad_(True)
        
        # Compute scores
        if self.encoder.hyperbolic:
            # Hyperbolic distance for link prediction
            scores = -self.encoder.hyp_tangent_space.distance(source_embeddings, target_embeddings)
        else:
            # Dot product for link prediction
            scores = torch.sum(source_embeddings * target_embeddings, dim=1)
        
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(scores, link_prediction_labels)
        
        return loss
    
    def train_generator(self, 
                       snapshots: List[Dict[str, torch.Tensor]], 
                       paper_features: torch.Tensor,
                       target_citations: torch.Tensor,
                       optimizer: torch.optim.Optimizer,
                       kl_weight: float = 1.0,
                       citation_weight: float = 1.0,
                       feature_weight: float = 1.0) -> Dict[str, float]:
        """
        Train the generator model.
        
        Args:
            snapshots: List of graph snapshots
            paper_features: Features of papers to generate/reconstruct
            target_citations: Target citation links
            optimizer: Optimizer for generator parameters
            kl_weight: Weight for KL divergence loss
            citation_weight: Weight for citation prediction loss
            feature_weight: Weight for feature reconstruction loss
            
        Returns:
            Dictionary of training metrics
        """
        self.generator.train()
        optimizer.zero_grad()
        
        # Forward pass and loss computation
        loss_dict = self.generator.train_step(
            snapshots, 
            paper_features, 
            target_citations,
            kl_weight=kl_weight,
            citation_weight=citation_weight,
            feature_weight=feature_weight
        )
        
        # Backward pass
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        # Convert tensors to float for logging
        metrics = {k: v.item() for k, v in loss_dict.items()}
        
        return metrics
    
    def generate_paper(self, 
                      snapshots: List[Dict[str, torch.Tensor]], 
                      initial_features: Optional[torch.Tensor] = None,
                      temperature: float = 1.0,
                      top_k_citations: int = 10) -> Dict:
        """
        Generate a new paper with citation links.
        
        Args:
            snapshots: List of graph snapshots
            initial_features: Optional initial features for the paper
            temperature: Temperature for sampling
            top_k_citations: Number of top citations to return
            
        Returns:
            Dictionary with generated paper information
        """
        return self.generator.generate_paper(
            snapshots=snapshots,
            initial_features=initial_features,
            temperature=temperature,
            top_k_citations=top_k_citations
        )
    
    def evaluate_paper_generation(self, 
                                 snapshots: List[Dict[str, torch.Tensor]], 
                                 test_features: torch.Tensor,
                                 test_citations: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the paper generation quality on test data.
        
        Args:
            snapshots: List of graph snapshots
            test_features: Features of test papers
            test_citations: Citation links of test papers
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.encoder.eval()
        self.generator.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(snapshots, test_features)
            
            # Prepare targets
            targets = {
                'features': test_features,
                'citations': test_citations
            }
            
            # Compute metrics
            feature_loss = F.mse_loss(outputs['reconstructed_features'], test_features).item()
            
            # Citation prediction metrics
            citation_preds = (outputs['citation_probs'] > 0.5).float()
            citation_accuracy = (citation_preds == test_citations).float().mean().item()
            
            # Compute precision, recall, F1 for citation links
            true_positives = (citation_preds * test_citations).sum().item()
            predicted_positives = citation_preds.sum().item()
            actual_positives = test_citations.sum().item()
            
            precision = true_positives / max(predicted_positives, 1)
            recall = true_positives / max(actual_positives, 1)
            f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
            
            return {
                'feature_loss': feature_loss,
                'citation_accuracy': citation_accuracy,
                'citation_precision': precision,
                'citation_recall': recall,
                'citation_f1': f1_score
            }
    
    def save_model(self, path: str):
        """Save the model to the specified path."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'num_nodes': self.num_nodes,
            'node_feature_dim': self.node_feature_dim,
            'embedding_dim': self.embedding_dim
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: torch.device) -> 'DynamicCitationNetworkModel':
        """
        Load a saved model from the specified path.
        
        Args:
            path: Path to the saved model
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with same parameters
        model = cls(
            num_nodes=checkpoint['num_nodes'],
            node_feature_dim=checkpoint['node_feature_dim'],
            embedding_dim=checkpoint['embedding_dim']
        )
        
        # Load state dictionaries
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        
        return model


class TopicAwareCitationNetworkModel(DynamicCitationNetworkModel):
    """
    Extended citation network model with topic-specific generation capabilities.
    
    This model extends the basic model by adding the ability to generate
    papers conditioned on specific research topics.
    """
    
    def __init__(self, 
                 num_nodes: int,
                 node_feature_dim: int,
                 num_topics: int,
                 memory_dim: int = 100,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 latent_dim: int = 128,
                 time_dim: int = 10,
                 message_dim: int = 100,
                 topic_embedding_dim: int = 64,
                 num_gnn_layers: int = 2,
                 edge_dim: Optional[int] = None,
                 use_memory: bool = True,
                 use_hierarchical_generator: bool = True,
                 use_hyperbolic: bool = True,
                 curvature: float = 1.0,
                 dropout: float = 0.2):
        """
        Initialize the topic-aware citation network model.
        
        Args:
            num_nodes: Number of nodes in the graph
            node_feature_dim: Dimension of node features
            num_topics: Number of possible topics
            memory_dim: Dimension of memory vectors
            embedding_dim: Dimension of node embeddings
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
            time_dim: Dimension of time encoding
            message_dim: Dimension of message vectors
            topic_embedding_dim: Dimension of topic embeddings
            num_gnn_layers: Number of GNN layers
            edge_dim: Dimension of edge features (if available)
            use_memory: Whether to use memory module or not
            use_hierarchical_generator: Whether to use hierarchical or basic generator
            use_hyperbolic: Whether to use hyperbolic embeddings
            curvature: Curvature of hyperbolic space
            dropout: Dropout rate
        """
        nn.Module.__init__(self)
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.embedding_dim = embedding_dim
        self.num_topics = num_topics
        
        # Initialize TGN encoder
        self.encoder = TemporalGraphNetwork(
            num_nodes=num_nodes,
            node_feature_dim=node_feature_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            embedding_dim=embedding_dim,
            message_dim=message_dim,
            edge_dim=edge_dim,
            num_gnn_layers=num_gnn_layers,
            use_memory=use_memory,
            hyperbolic=use_hyperbolic,
            curvature=curvature,
            dropout=dropout
        )
        
        # Initialize topic-conditioned generator
        self.generator = TopicConditionedGenerator(
            encoder_model=self.encoder,
            feature_dim=node_feature_dim,
            embedding_dim=embedding_dim,
            num_topics=num_topics,
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            topic_embedding_dim=topic_embedding_dim,
            use_hierarchical=use_hierarchical_generator,
            dropout=dropout
        )
    
    def forward(self, 
               snapshots: List[Dict[str, torch.Tensor]], 
               paper_features: torch.Tensor,
               topic_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the topic-aware model.
        
        Args:
            snapshots: List of graph snapshots
            paper_features: Features of papers to generate/reconstruct
            topic_ids: Optional topic IDs to condition on
            
        Returns:
            Dictionary of outputs from the generator
        """
        return self.generator(snapshots, paper_features, topic_ids)
    
    def train_generator(self, 
                       snapshots: List[Dict[str, torch.Tensor]], 
                       paper_features: torch.Tensor,
                       target_citations: torch.Tensor,
                       topic_ids: Optional[torch.Tensor] = None,
                       optimizer: torch.optim.Optimizer = None,
                       kl_weight: float = 1.0,
                       citation_weight: float = 1.0,
                       feature_weight: float = 1.0) -> Dict[str, float]:
        """
        Train the topic-aware generator model.
        
        Args:
            snapshots: List of graph snapshots
            paper_features: Features of papers to generate/reconstruct
            target_citations: Target citation links
            topic_ids: Optional topic IDs to condition on
            optimizer: Optimizer for generator parameters
            kl_weight: Weight for KL divergence loss
            citation_weight: Weight for citation prediction loss
            feature_weight: Weight for feature reconstruction loss
            
        Returns:
            Dictionary of training metrics
        """
        self.generator.train()
        if optimizer is not None:
            optimizer.zero_grad()
        
        # Forward pass and loss computation
        loss_dict = self.generator.train_step(
            snapshots, 
            paper_features, 
            target_citations,
            topic_ids=topic_ids,
            kl_weight=kl_weight,
            citation_weight=citation_weight,
            feature_weight=feature_weight
        )
        
        # Backward pass
        if optimizer is not None:
            loss_dict['total_loss'].backward()
            optimizer.step()
        
        # Convert tensors to float for logging
        metrics = {k: v.item() for k, v in loss_dict.items()}
        
        return metrics
    
    def generate_paper(self, 
                      snapshots: List[Dict[str, torch.Tensor]], 
                      topic_ids: Optional[torch.Tensor] = None,
                      initial_features: Optional[torch.Tensor] = None,
                      temperature: float = 1.0,
                      top_k_citations: int = 10) -> Dict:
        """
        Generate a new paper with citation links, optionally conditioned on topics.
        
        Args:
            snapshots: List of graph snapshots
            topic_ids: Optional topic IDs to condition on
            initial_features: Optional initial features to condition on
            temperature: Sampling temperature
            top_k_citations: Number of top citation links to return
            
        Returns:
            Dictionary with generated paper information
        """
        self.encoder.eval()
        self.generator.eval()
        
        with torch.no_grad():
            return self.generator.generate_paper(
                snapshots=snapshots,
                topic_ids=topic_ids,
                initial_features=initial_features,
                temperature=temperature,
                top_k_citations=top_k_citations
            )
    
    def save_model(self, path: str):
        """Save the model to the specified path."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'num_nodes': self.num_nodes,
            'node_feature_dim': self.node_feature_dim,
            'embedding_dim': self.embedding_dim,
            'num_topics': self.num_topics
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: torch.device) -> 'TopicAwareCitationNetworkModel':
        """
        Load a saved model from the specified path.
        
        Args:
            path: Path to the saved model
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with same parameters
        model = cls(
            num_nodes=checkpoint['num_nodes'],
            node_feature_dim=checkpoint['node_feature_dim'],
            num_topics=checkpoint['num_topics'],
            embedding_dim=checkpoint['embedding_dim']
        )
        
        # Load state dictionaries
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        
        return model 