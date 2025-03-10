import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import datetime
import uuid
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Define placeholder classes to replace legacy imports
class ConditionalVAE(nn.Module):
    """Placeholder for legacy ConditionalVAE."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.warning("Using placeholder ConditionalVAE - actual implementation not available")
        self.latent_dim = kwargs.get('latent_dim', 128)
        
    def encode(self, *args, **kwargs):
        return torch.randn(args[0].size(0), self.latent_dim).to(args[0].device)
        
    def decode(self, *args, **kwargs):
        return torch.zeros(*args[0].shape).to(args[0].device)

class HierarchicalPaperGenerator(nn.Module):
    """Placeholder for legacy HierarchicalPaperGenerator."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.warning("Using placeholder HierarchicalPaperGenerator - actual implementation not available")
        self.latent_dim = kwargs.get('latent_dim', 128)
        
    def encode(self, *args, **kwargs):
        return torch.randn(args[0].size(0), self.latent_dim).to(args[0].device)
        
    def decode(self, *args, **kwargs):
        return torch.zeros(*args[0].shape).to(args[0].device)


class CitationPaperGenerator(nn.Module):
    """
    Generator for new papers in a citation network.
    
    This model combines the dynamic GNN encoder and generative module
    to create new papers with intrinsic features and citation links.
    """
    
    def __init__(self, 
                 encoder_model,  # TGN encoder model
                 feature_dim: int,
                 embedding_dim: int,
                 num_nodes: int, 
                 hidden_dim: int = 256,
                 latent_dim: int = 128,
                 use_hierarchical: bool = True,
                 dropout: float = 0.2):
        """
        Initialize the citation paper generator.
        
        Args:
            encoder_model: Pre-trained TGN encoder model
            feature_dim: Dimension of paper features (topics/keywords)
            embedding_dim: Dimension of node embeddings
            num_nodes: Number of nodes in the graph
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
            use_hierarchical: Whether to use hierarchical or basic generator
            dropout: Dropout rate
        """
        super().__init__()
        self.encoder = encoder_model
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.use_hierarchical = use_hierarchical
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Initialize generator model
        if use_hierarchical:
            self.generator = HierarchicalPaperGenerator(
                input_feature_dim=feature_dim,
                condition_dim=embedding_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_nodes=num_nodes,
                dropout=dropout
            )
        else:
            self.generator = ConditionalVAE(
                input_feature_dim=feature_dim,
                condition_dim=embedding_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_nodes=num_nodes,
                dropout=dropout
            )
        
        # Relevance scorer for conditioning the generator
        self.relevance_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
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
        # Debug prints
        print(f"DEBUG - paper_features shape: {paper_features.shape}, dim: {paper_features.dim()}")
        
        # Get node embeddings from encoder
        with torch.no_grad():
            node_embeddings = self.encoder(snapshots)
        
        print(f"DEBUG - node_embeddings shape: {node_embeddings.shape}")
        
        # Compute relevance of each existing node as conditioning
        relevance_scores = self.relevance_scorer(node_embeddings).squeeze(-1)
        relevance_weights = F.softmax(relevance_scores, dim=0)
        
        print(f"DEBUG - relevance_weights shape: {relevance_weights.shape}")
        
        # Create conditioning vector as weighted average of node embeddings
        condition = torch.matmul(relevance_weights.unsqueeze(0), node_embeddings)
        
        print(f"DEBUG - condition shape: {condition.shape}")
        
        # Handle 3D paper_features
        if paper_features.dim() == 3:
            print("DEBUG - Using 3D paper_features path")
            # For now, just use the first item in the batch for simplicity
            # In a real implementation, you would process each item separately
            paper_features_2d = paper_features[0]  # Shape: [num_nodes, feature_dim]
            print(f"DEBUG - paper_features_2d shape: {paper_features_2d.shape}")
            
            # Forward pass through generator
            outputs = self.generator(paper_features_2d, condition)
            
            # Add batch dimension back to outputs
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    outputs[key] = value.unsqueeze(0)
        else:
            print("DEBUG - Using 2D paper_features path")
            # If paper_features is [batch_size, feature_dim]
            # Ensure condition has the same batch dimension
            if paper_features.size(0) > 1 and condition.size(0) == 1:
                condition = condition.expand(paper_features.size(0), -1)
                print(f"DEBUG - expanded condition shape: {condition.shape}")
            
            # Forward pass through generator
            outputs = self.generator(paper_features, condition)
        
        # Add relevance weights to outputs
        outputs['relevance_weights'] = relevance_weights
        
        return outputs
    
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
        self.eval()
        
        # Get node embeddings from encoder
        with torch.no_grad():
            node_embeddings = self.encoder(snapshots)
        
        # Compute relevance of each existing node as conditioning
        relevance_scores = self.relevance_scorer(node_embeddings).squeeze(-1)
        relevance_weights = F.softmax(relevance_scores, dim=0)
        
        # Create conditioning vector as weighted average of node embeddings
        condition = torch.matmul(relevance_weights.unsqueeze(0), node_embeddings)
        
        # Generate paper features
        if initial_features is None:
            # Sample from prior
            prior_mu, prior_log_var = self.generator.compute_prior(condition)
            z = self.generator.cvae.reparameterize(prior_mu, prior_log_var)
            
            # Apply temperature
            if temperature != 1.0:
                z = z * temperature
                
            # Decode - this returns a tuple of (reconstructed_features, citation_probs)
            features, citations = self.generator.cvae.decode(z, condition)
        else:
            # Use provided features - this returns a dictionary
            outputs = self.forward(snapshots, initial_features)
            features = outputs['reconstructed_features']
            citations = outputs['citation_probs']
        
        # Get top-k citations
        top_k_values, top_k_indices = torch.topk(citations, min(top_k_citations, citations.size(1)))
        
        # Convert feature vector back to topics and keywords
        # Assume first 10 elements are topic scores, next elements are keywords
        topic_scores = features.squeeze(0)[:10].cpu().numpy()
        
        # Create topics list similar to the original dataset format
        topics = []
        for i, score in enumerate(topic_scores):
            if score > 0.1:  # Only include topics with significant scores
                topics.append({
                    "id": f"topic_{i+1}",
                    "display_name": f"Generated Topic {i+1}",
                    "score": float(score),
                    "field": "Computer Science"
                })
        
        # Create keywords from the remaining feature elements (if any)
        keywords = []
        if features.shape[1] > 10:
            keyword_scores = features.squeeze(0)[10:].cpu().numpy()
            for i, score in enumerate(keyword_scores):
                if score > 0.1:  # Only include keywords with significant scores
                    keywords.append({
                        "id": f"keyword_{i+1}",
                        "display_name": f"Generated Keyword {i+1}",
                        "score": float(score)
                    })
        
        # Generate UUID for paper ID
        paper_id = str(uuid.uuid4())
        
        # Use current date as publication date
        pub_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Debug print
        print(f"DEBUG - Generated features shape: {features.shape}, type: {type(features)}")
        print(f"DEBUG - Topics length: {len(topics)}, type: {type(topics)}")
        print(f"DEBUG - Keywords length: {len(keywords)}, type: {type(keywords)}")
        
        # Return formatted output
        return {
            "id": paper_id,
            "title": f"Generated Paper {paper_id[:8]}",
            "publication_date": pub_date,
            "topics": topics,
            "keywords": keywords,
            "features": features.cpu().numpy(),
            "citations": citations.cpu().numpy(),
            "top_citations": top_k_indices.cpu().numpy(),
            "citation_scores": top_k_values.cpu().numpy(),
            "relevance_weights": relevance_weights.cpu().numpy(),
            "z": z.cpu().numpy()
        }
    
    def train_step(self, 
                  snapshots: List[Dict[str, torch.Tensor]], 
                  paper_features: torch.Tensor,
                  target_citations: torch.Tensor,
                  kl_weight: float = 1.0,
                  citation_weight: float = 1.0,
                  feature_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Perform a training step.
        
        Args:
            snapshots: List of graph snapshots
            paper_features: Features of papers to generate/reconstruct
            target_citations: Target citation links
            kl_weight: Weight for KL divergence loss
            citation_weight: Weight for citation prediction loss
            feature_weight: Weight for feature reconstruction loss
            
        Returns:
            Dictionary of loss components
        """
        # Forward pass
        outputs = self.forward(snapshots, paper_features)
        
        # Prepare targets
        targets = {
            'features': paper_features,
            'citations': target_citations
        }
        
        # Compute loss
        if self.use_hierarchical:
            loss_dict = self.generator.compute_loss(
                outputs, targets, 
                kl_weight=kl_weight,
                citation_weight=citation_weight,
                feature_weight=feature_weight
            )
        else:
            loss_dict = self.generator.compute_loss(
                outputs, targets, 
                kl_weight=kl_weight,
                citation_weight=citation_weight,
                feature_weight=feature_weight
            )
        
        return loss_dict


class TopicConditionedGenerator(nn.Module):
    """
    Extended paper generator that can be conditioned on specific topics.
    
    This model allows generating papers in specific research areas by
    conditioning the generation process on topic vectors.
    """
    
    def __init__(self, 
                 encoder_model, 
                 feature_dim: int,
                 embedding_dim: int,
                 num_topics: int,
                 num_nodes: int,
                 hidden_dim: int = 256,
                 latent_dim: int = 128,
                 topic_embedding_dim: int = 64,
                 use_hierarchical: bool = True,
                 dropout: float = 0.2):
        """
        Initialize the topic-conditioned generator.
        
        Args:
            encoder_model: Pre-trained TGN encoder model
            feature_dim: Dimension of paper features (topics/keywords)
            embedding_dim: Dimension of node embeddings
            num_topics: Number of possible topics
            num_nodes: Number of nodes in the graph
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
            topic_embedding_dim: Dimension of topic embeddings
            use_hierarchical: Whether to use hierarchical or basic generator
            dropout: Dropout rate
        """
        super().__init__()
        self.encoder = encoder_model
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.num_topics = num_topics
        self.num_nodes = num_nodes
        self.use_hierarchical = use_hierarchical
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Topic embedding layer
        self.topic_embedding = nn.Embedding(num_topics, topic_embedding_dim)
        
        # Topic-aware conditioning network
        self.topic_conditioner = nn.Sequential(
            nn.Linear(topic_embedding_dim + embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
            
        # Initialize generator model
        if use_hierarchical:
            self.generator = HierarchicalPaperGenerator(
                input_feature_dim=feature_dim,
                condition_dim=embedding_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_nodes=num_nodes,
                dropout=dropout
            )
        else:
            self.generator = ConditionalVAE(
                input_feature_dim=feature_dim,
                condition_dim=embedding_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_nodes=num_nodes,
                dropout=dropout
            )
        
        # Relevance scorer for conditioning the generator
        self.relevance_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, 
               snapshots: List[Dict[str, torch.Tensor]], 
               paper_features: torch.Tensor,
               topic_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            snapshots: List of graph snapshots
            paper_features: Features of papers to generate/reconstruct
            topic_ids: Optional topic IDs to condition on
            
        Returns:
            Dictionary of outputs from the generator
        """
        # Get node embeddings from encoder
        with torch.no_grad():
            node_embeddings = self.encoder(snapshots)
        
        # Compute relevance of each existing node as conditioning
        relevance_scores = self.relevance_scorer(node_embeddings).squeeze(-1)
        relevance_weights = F.softmax(relevance_scores, dim=0)
        
        # Create base conditioning vector as weighted average of node embeddings
        base_condition = torch.matmul(relevance_weights.unsqueeze(0), node_embeddings)
        
        # If topic IDs are provided, condition on them
        if topic_ids is not None:
            # Get topic embeddings
            topic_emb = self.topic_embedding(topic_ids)
            
            # Combine with base condition
            combined = torch.cat([base_condition, topic_emb], dim=1)
            
            # Generate topic-aware conditioning
            condition = self.topic_conditioner(combined)
        else:
            condition = base_condition
        
        # Handle different dimensions of paper_features
        if paper_features.dim() == 3:
            # If paper_features is [batch_size, num_nodes, feature_dim]
            # We need to reshape it to [batch_size * num_nodes, feature_dim]
            batch_size, num_nodes, feature_dim = paper_features.shape
            paper_features_reshaped = paper_features.reshape(-1, feature_dim)
            
            # Expand condition to match the batch dimension
            condition_expanded = condition.expand(batch_size * num_nodes, -1)
            
            # Forward pass through generator
            outputs_flat = self.generator(paper_features_reshaped, condition_expanded)
            
            # Reshape outputs back to original dimensions
            outputs = {}
            for key, value in outputs_flat.items():
                if isinstance(value, torch.Tensor) and value.dim() > 1:
                    if key == 'citation_probs':
                        # Citation probs might have a different shape
                        if value.size(0) == batch_size * num_nodes:
                            outputs[key] = value.reshape(batch_size, num_nodes, -1)
                    else:
                        # For other tensors, reshape back to batch dimensions
                        if value.size(0) == batch_size * num_nodes:
                            outputs[key] = value.reshape(batch_size, num_nodes, -1)
                else:
                    outputs[key] = value
        else:
            # If paper_features is [batch_size, feature_dim]
            # Ensure condition has the same batch dimension
            if paper_features.size(0) > 1 and condition.size(0) == 1:
                condition = condition.expand(paper_features.size(0), -1)
            
            # Forward pass through generator
            outputs = self.generator(paper_features, condition)
        
        # Add relevance weights to outputs
        outputs['relevance_weights'] = relevance_weights
        
        return outputs
    
    def generate_paper(self,
                      snapshots: List[Dict[str, torch.Tensor]],
                      topic_ids: Optional[List[int]] = None,
                      initial_features: Optional[torch.Tensor] = None,
                      temperature: float = 1.0,
                      top_k_citations: int = 10) -> Dict:
        """
        Generate a new paper with citation links conditioned on topics.
        
        Args:
            snapshots: List of graph snapshots
            topic_ids: Optional list of topic IDs to condition on
            initial_features: Optional initial features for the paper
            temperature: Temperature for sampling
            top_k_citations: Number of top citations to return
            
        Returns:
            Dictionary with generated paper information
        """
        import uuid
        import datetime
        import numpy as np
        
        self.eval()
        
        # Get node embeddings from encoder
        with torch.no_grad():
            node_embeddings = self.encoder(snapshots)
        
        # Compute relevance of each existing node as conditioning
        relevance_scores = self.relevance_scorer(node_embeddings).squeeze(-1)
        relevance_weights = F.softmax(relevance_scores, dim=0)
        
        # Create base conditioning vector as weighted average of node embeddings
        base_condition = torch.matmul(relevance_weights.unsqueeze(0), node_embeddings)
        
        # Add topic conditioning if specified
        if topic_ids is not None:
            # Get topic embeddings
            topic_embs = [self.topic_embedding(torch.tensor(tid, device=base_condition.device)) 
                         for tid in topic_ids]
            
            # Average topic embeddings if multiple are provided
            if topic_embs:
                topic_emb = torch.stack(topic_embs).mean(dim=0).unsqueeze(0)
                
                # Concatenate base condition with topic embedding
                condition = torch.cat([base_condition, topic_emb], dim=1)
                condition = self.topic_conditioner(condition)
            else:
                condition = base_condition
        else:
            condition = base_condition
        
        # Generate paper features
        if initial_features is None:
            # For hierarchical generator
            if self.use_hierarchical:
                # Sample from prior
                prior_mu, prior_log_var = self.generator.compute_prior(condition)
                z = self.generator.cvae.reparameterize(prior_mu, prior_log_var)
                
                # Apply temperature
                if temperature != 1.0:
                    z = z * temperature
                    
                # Decode - returns a tuple of (reconstructed_features, citation_probs)
                features, citations = self.generator.cvae.decode(z, condition)
            else:
                # For basic generator - sample directly from prior
                z = torch.randn(1, self.latent_dim, device=condition.device)
                
                # Apply temperature
                if temperature != 1.0:
                    z = z * temperature
                    
                # Decode - returns a tuple of (reconstructed_features, citation_probs)
                features, citations = self.generator.cvae.decode(z, condition)
        else:
            # Use provided features - returns a dictionary
            outputs = self.forward(snapshots, initial_features)
            features = outputs['reconstructed_features']
            citations = outputs['citation_probs']
            z = outputs['z']
        
        # Get top-k citations
        top_k_values, top_k_indices = torch.topk(citations, min(top_k_citations, citations.size(1)))
        
        # Convert feature vector back to topics and keywords
        # Assume first 10 elements are topic scores, next elements are keywords
        topic_scores = features.squeeze(0)[:10].cpu().numpy()
        
        # Create topics list similar to the original dataset format
        topics = []
        for i, score in enumerate(topic_scores):
            if score > 0.1:  # Only include topics with significant scores
                topics.append({
                    "id": f"topic_{i+1}",
                    "display_name": f"Generated Topic {i+1}",
                    "score": float(score),
                    "field": "Computer Science"
                })
        
        # Create keywords from the remaining feature elements (if any)
        keywords = []
        if features.shape[1] > 10:
            keyword_scores = features.squeeze(0)[10:].cpu().numpy()
            for i, score in enumerate(keyword_scores):
                if score > 0.1:  # Only include keywords with significant scores
                    keywords.append({
                        "id": f"keyword_{i+1}",
                        "display_name": f"Generated Keyword {i+1}",
                        "score": float(score)
                    })
        
        # Generate UUID for paper ID
        paper_id = str(uuid.uuid4())
        
        # Use current date as publication date
        pub_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Debug print
        print(f"DEBUG - Generated features shape: {features.shape}, type: {type(features)}")
        print(f"DEBUG - Topics length: {len(topics)}, type: {type(topics)}")
        print(f"DEBUG - Keywords length: {len(keywords)}, type: {type(keywords)}")
        
        # Return formatted output
        return {
            "id": paper_id,
            "title": f"Generated Paper {paper_id[:8]}",
            "publication_date": pub_date,
            "topics": topics,
            "keywords": keywords,
            "features": features.cpu().numpy(),
            "citations": citations.cpu().numpy(),
            "top_citations": top_k_indices.cpu().numpy(),
            "citation_scores": top_k_values.cpu().numpy(),
            "relevance_weights": relevance_weights.cpu().numpy(),
            "z": z.cpu().numpy()
        }
    
    def train_step(self, 
                  snapshots: List[Dict[str, torch.Tensor]], 
                  paper_features: torch.Tensor,
                  target_citations: torch.Tensor,
                  topic_ids: Optional[torch.Tensor] = None,
                  kl_weight: float = 1.0,
                  citation_weight: float = 1.0,
                  feature_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Perform a training step.
        
        Args:
            snapshots: List of graph snapshots
            paper_features: Features of papers to generate/reconstruct
            target_citations: Target citation links
            topic_ids: Optional topic IDs to condition on
            kl_weight: Weight for KL divergence loss
            citation_weight: Weight for citation prediction loss
            feature_weight: Weight for feature reconstruction loss
            
        Returns:
            Dictionary of loss components
        """
        # Forward pass
        outputs = self.forward(snapshots, paper_features, topic_ids)
        
        # Prepare targets
        targets = {
            'features': paper_features,
            'citations': target_citations
        }
        
        # Compute loss
        if self.use_hierarchical:
            loss_dict = self.generator.compute_loss(
                outputs, targets, 
                kl_weight=kl_weight,
                citation_weight=citation_weight,
                feature_weight=feature_weight
            )
        else:
            loss_dict = self.generator.compute_loss(
                outputs, targets, 
                kl_weight=kl_weight,
                citation_weight=citation_weight,
                feature_weight=feature_weight
            )
        
        return loss_dict 