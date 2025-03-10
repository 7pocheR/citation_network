import os
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as GraphData
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from src.models.encoder.hyperbolic_encoder import HyperbolicEncoder
from src.models.predictors.attention_predictor import AttentionPredictor
from src.models.predictors.autoregressive_predictor import AutoregressiveLinkPredictor

logger = logging.getLogger(__name__)

class AutoregressiveCitationModel(nn.Module):
    """
    Integrated model that combines HyperbolicEncoder, AttentionPredictor,
    and AutoregressiveLinkPredictor for paper generation.
    
    This model supports multi-task training for both link prediction 
    (between unmasked nodes) and paper generation (predicting future papers 
    after a time threshold) using an autoregressive approach.
    
    The key difference from IntegratedCitationModel is that paper generation
    is treated as a series of link predictions rather than using a 
    dedicated generative model like CVAE.
    """
    
    def __init__(
        self,
        num_nodes: Optional[int] = None,
        node_feature_dim: Optional[int] = None,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        edge_dim: Optional[int] = None,
        use_hierarchical: bool = False,
        topic_vocab_size: Optional[int] = None,
        curvature: float = 1.0,
        dropout: float = 0.2,
        num_encoder_layers: int = 1,
        num_predictor_layers: int = 2,
        ordering_strategy: str = 'citation',
        temperature: float = 1.0,
        reveal_ratio: float = 0.3,
        device_manager: Optional[Any] = None,
        # Pre-initialized components
        encoder: Optional[nn.Module] = None,
        predictor: Optional[nn.Module] = None,
        autoregressive_predictor: Optional[nn.Module] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the autoregressive citation model.
        
        Args:
            num_nodes: Number of nodes in the graph
            node_feature_dim: Dimension of node features
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            edge_dim: Edge feature dimension
            use_hierarchical: Whether to use hierarchical encoding
            topic_vocab_size: Size of topic vocabulary
            curvature: Hyperbolic curvature parameter
            dropout: Dropout rate
            num_encoder_layers: Number of layers in the encoder
            num_predictor_layers: Number of layers in the predictor
            ordering_strategy: Strategy for ordering autoregressive predictions
            temperature: Temperature for controlling prediction randomness
            reveal_ratio: Proportion of links to reveal during training
            device_manager: Device manager
            encoder: Pre-initialized encoder component
            predictor: Pre-initialized predictor component
            autoregressive_predictor: Pre-initialized autoregressive predictor
            device: Device to use for model
        """
        super().__init__()
        
        # Store parameters
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.use_hierarchical = use_hierarchical
        self.curvature = curvature
        self.dropout = dropout
        self.ordering_strategy = ordering_strategy
        self.temperature = temperature
        self.reveal_ratio = reveal_ratio
        
        # Set up device handling
        self.device_manager = device_manager
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize components
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = HyperbolicEncoder(
                node_dim=node_feature_dim if node_feature_dim is not None else embed_dim,
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                curvature=curvature,
                dropout=dropout
            )
            
        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = AttentionPredictor(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_predictor_layers,
                num_heads=num_heads,
                dropout=dropout
            )
            
        if autoregressive_predictor is not None:
            self.autoregressive_predictor = autoregressive_predictor
        else:
            self.autoregressive_predictor = AutoregressiveLinkPredictor(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                ordering_strategy=ordering_strategy,
                temperature=temperature,
                reveal_ratio=reveal_ratio,
                node_feature_dim=node_feature_dim
            )
            
        # Feature projection (to project paper features to embedding space)
        self.feature_projector = nn.Sequential(
            nn.Linear(node_feature_dim if node_feature_dim is not None else embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Move to device
        self.to(self.device)
        
    def to_device(self, x):
        """Move an object to the device where the model is located.
        
        Args:
            x: Object to move to device
            
        Returns:
            Object on the same device as the model
        """
        device = next(self.parameters()).device
        
        if x is None:
            return None
        elif isinstance(x, (list, tuple)):
            return [self.to_device(item) for item in x]
        elif isinstance(x, dict):
            return {k: self.to_device(v) for k, v in x.items()}
        elif hasattr(x, 'to') and callable(getattr(x, 'to')):
            return x.to(device)
        else:
            return x
    
    def forward(self, graph, task='both'):
        """
        Forward pass of the model.
        
        Args:
            graph: Graph data object
            task: Task to perform ('link_prediction', 'generation', or 'both')
            
        Returns:
            Dictionary of outputs depending on task
        """
        # Move graph to device
        graph = self.to_device(graph)
        
        # Run encoder to get node embeddings
        node_embeddings = self.encoder(graph)
        
        outputs = {'node_embeddings': node_embeddings}
        
        # Add link prediction if requested
        if task in ['link_prediction', 'both']:
            # For link prediction, we don't need to do additional computation here
            # It will be handled by specific methods like predict_links
            outputs['link_prediction'] = True
            
        # Add generation if requested
        if task in ['generation', 'both']:
            # For generation, we'll just note that it's enabled
            # Actual generation will be done by specific methods
            outputs['generation'] = True
            
        return outputs
    
    def predict_links(
        self,
        graph: GraphData,
        src_nodes: Optional[torch.Tensor] = None,
        dst_nodes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict links between source and destination nodes.
        
        Args:
            graph: Graph data object
            src_nodes: Source node indices
            dst_nodes: Destination node indices
            
        Returns:
            Predicted link probabilities
        """
        # Move data to device
        graph = self.to_device(graph)
        if src_nodes is not None:
            src_nodes = self.to_device(src_nodes)
        if dst_nodes is not None:
            dst_nodes = self.to_device(dst_nodes)
            
        # Get node embeddings
        node_embeddings = self.encoder(graph)
        
        # If no specific nodes are provided, use all nodes
        if src_nodes is None or dst_nodes is None:
            adj_matrix = self.predictor.predict_adjacency(node_embeddings)
            return adj_matrix
            
        # Get source and destination embeddings
        src_embeddings = node_embeddings[src_nodes]
        dst_embeddings = node_embeddings[dst_nodes]
        
        # Predict links
        scores = self.predictor(src_embeddings, dst_embeddings)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(scores)
        
        return probs
    
    def predict_adjacency_matrix(self, graph: GraphData) -> torch.Tensor:
        """
        Predict adjacency matrix for the graph.
        
        Args:
            graph: Graph data object
            
        Returns:
            Adjacency matrix with link probabilities
        """
        # Move graph to device
        graph = self.to_device(graph)
        
        # Get node embeddings
        node_embeddings = self.encoder(graph)
        
        # Predict adjacency matrix
        adj_matrix = self.predictor.predict_adjacency(node_embeddings)
        
        return adj_matrix
    
    def _generate_paper_features(self, graph, num_papers):
        """Generate or interpolate paper features based on existing graph.
        
        Args:
            graph: Source graph for feature interpolation
            num_papers: Number of paper features to generate
            
        Returns:
            Tensor of generated paper features
        """
        device = next(self.parameters()).device
        
        # If the graph has features, interpolate between existing features
        if hasattr(graph, 'x') and graph.x is not None:
            # Get random pairs of existing nodes
            num_nodes = graph.x.size(0)
            idx1 = torch.randint(0, num_nodes, (num_papers,), device=device)
            idx2 = torch.randint(0, num_nodes, (num_papers,), device=device)
            
            # Interpolate between their features
            alpha = torch.rand(num_papers, 1, device=device)
            paper_features = alpha * graph.x[idx1] + (1 - alpha) * graph.x[idx2]
            
            # Add some noise for diversity
            noise = torch.randn_like(paper_features) * 0.1
            paper_features = torch.clamp(paper_features + noise, 0, 1)
        else:
            # If no features available, create random features
            feature_dim = graph.num_node_features if hasattr(graph, 'num_node_features') else self.node_feature_dim
            paper_features = torch.rand(num_papers, feature_dim, device=device)
            
        return paper_features

    def generate_future_papers_autoregressive(
        self,
        graph: GraphData,
        time_threshold: Optional[float] = None,
        future_window: Optional[float] = None,
        num_papers: int = 10,
        paper_features: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        threshold: float = 0.5,
        temperature: Optional[float] = None,
        num_iterations: int = 5,
        citation_threshold: float = 0.8
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Generate future papers and predict their citation links autoregressively.
        
        Args:
            graph: Citation graph
            time_threshold: Time threshold for future papers (if None, use max timestamp)
            future_window: Time window for future papers (if None, use 1.0)
            num_papers: Number of papers to generate
            paper_features: Optional precomputed features for new papers
            top_k: If provided, keep only top k citations per paper
            threshold: Probability threshold for binary predictions
            temperature: Temperature for softmax (higher = more diversity)
            num_iterations: Number of iterations for autoregressive refinement (default=5)
            citation_threshold: Probability threshold for adding citations in refinement (default=0.8)
            
        Returns:
            Tuple containing:
            - Generated paper features
            - List of paper information dictionaries
        """
        # Ensure graph is on the correct device
        graph = self.to_device(graph)
        
        # Get time threshold from graph if not provided
        if time_threshold is None:
            if hasattr(graph, 'node_timestamps') and graph.node_timestamps is not None:
                time_threshold = float(graph.node_timestamps.max().item())
            elif hasattr(graph, 'paper_times') and graph.paper_times is not None:
                time_threshold = float(graph.paper_times.max().item())
            else:
                time_threshold = 0.0
                
        # Set future window if not provided
        if future_window is None:
            future_window = 1.0
            
        # Create past graph (papers before time_threshold)
        if hasattr(graph, 'node_timestamps') and graph.node_timestamps is not None:
            past_mask = graph.node_timestamps < time_threshold
            past_graph = graph.subgraph(past_mask)
        elif hasattr(graph, 'paper_times') and graph.paper_times is not None:
            past_mask = graph.paper_times < time_threshold
            past_graph = graph.subgraph(past_mask)
        else:
            past_graph = graph
        
        # Get node embeddings for past graph
        node_embeddings = self.encoder(past_graph)
        
        # Generate or use provided paper features
        if paper_features is None:
            paper_features = self._generate_paper_features(past_graph, num_papers)
        else:
            paper_features = self.to_device(paper_features)
        
        # Use autoregressive predictor to generate papers
        temp = temperature if temperature is not None else self.temperature
        generated_features, paper_info = self.autoregressive_predictor.generate_papers_autoregressively(
            graph=past_graph,
            node_embeddings=node_embeddings,
            time_threshold=time_threshold,
            future_window=future_window,
            num_papers=num_papers,
            paper_features=paper_features,
            top_k=top_k,
            threshold=threshold,
            num_iterations=num_iterations,
            citation_threshold=citation_threshold
        )
        
        return generated_features, paper_info
    
    def compute_link_prediction_loss(
        self, 
        graph: GraphData,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for link prediction task.
        
        Args:
            graph: Graph data object
            pos_edge_index: Positive edge indices
            neg_edge_index: Negative edge indices
            
        Returns:
            Link prediction loss
        """
        # Check if we have enough edges for loss computation
        if pos_edge_index.shape[1] == 0 or neg_edge_index.shape[1] == 0:
            logger.warning(f"Insufficient edges for link prediction loss. Positive: {pos_edge_index.shape[1]}, Negative: {neg_edge_index.shape[1]}")
            return torch.tensor(0.0, device=self.device_manager.device if self.device_manager else 
                              (next(self.parameters()).device if next(self.parameters(), None) else 'cpu'))
        
        # Move data to device
        graph = self.to_device(graph)
        pos_edge_index = self.to_device(pos_edge_index)
        neg_edge_index = self.to_device(neg_edge_index)
        
        # Create a masked graph with positive edges removed from message passing
        # This ensures proper transductive link prediction without data leakage
        masked_graph = graph.mask_edges(pos_edge_index)
        
        # Get node embeddings from the masked graph
        # The model cannot use the positive edges during message passing
        node_embeddings = self.encoder(masked_graph)
        
        # Get source and destination node embeddings for positive edges
        src_pos, dst_pos = pos_edge_index
        src_pos_embed = node_embeddings[src_pos]
        dst_pos_embed = node_embeddings[dst_pos]
        
        # Get source and destination node embeddings for negative edges
        src_neg, dst_neg = neg_edge_index
        src_neg_embed = node_embeddings[src_neg]
        dst_neg_embed = node_embeddings[dst_neg]
        
        # Predict links
        pos_scores = self.predictor(src_pos_embed, dst_pos_embed)
        neg_scores = self.predictor(src_neg_embed, dst_neg_embed)
        
        # Compute binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, 
            torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, 
            torch.zeros_like(neg_scores)
        )
        
        # Combine positive and negative losses
        link_pred_loss = (pos_loss + neg_loss) / 2
        
        return link_pred_loss
    
    def compute_autoregressive_loss(
        self,
        past_graph: GraphData,
        future_graph: GraphData,
        reveal_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss for autoregressive link prediction task.
        
        Args:
            past_graph: Graph containing nodes before time threshold
            future_graph: Graph containing nodes after time threshold
            reveal_ratio: Proportion of links to reveal (overrides model default)
            
        Returns:
            Tuple containing:
            - Autoregressive prediction loss
            - Dictionary of detailed loss components
        """
        metrics = {}
        device = next(self.parameters()).device
        
        # If we don't have past and future graph, return zero loss
        if past_graph is None or future_graph is None or future_graph.num_nodes == 0:
            return torch.tensor(0.0, device=device), metrics
        
        # Get past embeddings
        past_embeddings = self.encoder(past_graph)
        
        # Set batch size based on available memory
        batch_size = min(32, future_graph.num_nodes)
        num_batches = (future_graph.num_nodes + batch_size - 1) // batch_size
        
        # Initialize metrics
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Track total edge counts for reporting
        total_actual_edges = 0
        total_predicted_edges = 0  # Will count edges predicted with confidence > 0.5
        
        # Process future nodes in batches
        for batch_idx in range(num_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, future_graph.num_nodes)
            batch_indices = list(range(start_idx, end_idx))
            
            # Get batch features
            batch_features = future_graph.x[batch_indices]
            
            # Process each node in batch
            for i, node_idx in enumerate(batch_indices):
                # Find true citations for this node
                true_links = []
                src_indices, dst_indices = future_graph.edge_index
                
                # Collect all outgoing edges from this node
                for j in range(len(src_indices)):
                    if src_indices[j] == node_idx and dst_indices[j] < past_graph.num_nodes:
                        true_links.append((node_idx, dst_indices[j].item()))
                
                # Update total actual edges count
                total_actual_edges += len(true_links)
                
                # Create true link tensor
                revealed_links = None
                if true_links:
                    # Convert to tensor
                    true_links_tensor = torch.tensor(true_links, device=device)
                    
                    # Determine links to reveal
                    effective_reveal_ratio = reveal_ratio if reveal_ratio is not None else self.reveal_ratio
                    num_to_reveal = max(1, int(len(true_links) * effective_reveal_ratio))
                    
                    # Select links to reveal
                    if hasattr(self.autoregressive_predictor, '_select_revealed_links'):
                        revealed_links = self.autoregressive_predictor._select_revealed_links(
                            true_links_tensor, num_to_reveal
                        )
                
                # Predict links
                link_probs, _ = self.autoregressive_predictor.predict_links_autoregressively(
                    past_graph,
                    past_embeddings,
                    masked_node_index=node_idx,
                    node_features=batch_features[i].unsqueeze(0),
                    revealed_links=revealed_links
                )
                
                # Create true labels tensor
                true_labels = torch.zeros(past_graph.num_nodes, device=device)
                
                # Set positive links
                for _, dst in true_links:
                    if dst < true_labels.size(0):
                        true_labels[dst] = 1.0
                
                # Count predicted edges (confidence > 0.5)
                predicted_edges = (link_probs >= 0.5).sum().item()
                total_predicted_edges += predicted_edges
                
                # Reshape if needed
                if link_probs.dim() > 1 and true_labels.dim() == 1:
                    true_labels = true_labels.view(-1, 1)
                
                # Compute loss
                loss = F.binary_cross_entropy(link_probs, true_labels)
                total_loss += loss.item()
                
                # Store predictions and labels for metrics
                all_preds.append(link_probs.detach())
                all_labels.append(true_labels.detach())
        
        # Compute average loss
        avg_loss = total_loss / future_graph.num_nodes
        
        # Compute metrics if predictions are available
        if all_preds:
            # Concatenate all predictions and labels
            all_preds = torch.cat([p.view(-1) for p in all_preds])
            all_labels = torch.cat([l.view(-1) for l in all_labels])
            
            # Compute metrics
            metrics = self._compute_metrics(all_preds, all_labels)
            
            # Add loss to metrics
            metrics['autoregressive'] = avg_loss
            metrics['val_autoregressive'] = avg_loss
            
            # Add edge count metrics
            metrics['num_actual_edges'] = total_actual_edges
            metrics['num_predicted_edges'] = total_predicted_edges
        
        # Create loss tensor for backpropagation
        loss_tensor = torch.tensor(avg_loss, device=device, requires_grad=True)
        
        return loss_tensor, metrics
    
    def compute_multi_task_loss(
        self,
        graph: GraphData,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        past_graph: Optional[GraphData] = None,
        future_graph: Optional[GraphData] = None,
        task_weights: Dict[str, float] = {'link_prediction': 1.0, 'autoregressive': 1.0}
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for multi-task learning.
        
        Args:
            graph: Graph data object
            pos_edge_index: Positive edge indices
            neg_edge_index: Negative edge indices
            past_graph: Graph containing nodes before time threshold
            future_graph: Graph containing nodes after time threshold
            task_weights: Dictionary of task weights
            
        Returns:
            Dictionary of losses
        """
        device = next(self.parameters()).device
        losses = {}
        
        # Get task weights with defaults
        link_pred_weight = task_weights.get('link_prediction', 1.0)
        autoregressive_weight = task_weights.get('autoregressive', 1.0)
        
        # Compute link prediction loss
        if link_pred_weight > 0 and pos_edge_index is not None and neg_edge_index is not None:
            link_pred_loss = self.compute_link_prediction_loss(
                graph=graph,
                pos_edge_index=pos_edge_index,
                neg_edge_index=neg_edge_index
            )
            losses['link_prediction'] = link_pred_weight * link_pred_loss
        else:
            losses['link_prediction'] = torch.tensor(0.0, device=device)
            
        # Compute autoregressive loss
        if autoregressive_weight > 0 and past_graph is not None and future_graph is not None:
            autoregressive_loss, auto_metrics = self.compute_autoregressive_loss(
                past_graph=past_graph,
                future_graph=future_graph
            )
            losses['autoregressive'] = autoregressive_weight * autoregressive_loss
            
            # Store metrics separately
            losses['auto_metrics'] = auto_metrics
        else:
            losses['autoregressive'] = torch.tensor(0.0, device=device)
            losses['auto_metrics'] = {}
            
        # Compute combined loss (excluding metrics dictionary)
        losses['total_loss'] = losses['link_prediction'] + losses['autoregressive']
        
        return losses
    
    def train_step(
        self,
        graph: GraphData,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        past_graph: Optional[GraphData] = None,
        future_graph: Optional[GraphData] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        task_weights: Dict[str, float] = {'link_prediction': 1.0, 'autoregressive': 1.0}
    ) -> Dict[str, float]:
        """
        Training step for the model.
        
        Args:
            graph: Graph data object
            pos_edge_index: Positive edge indices
            neg_edge_index: Negative edge indices
            past_graph: Graph containing nodes before time threshold
            future_graph: Graph containing nodes after time threshold
            optimizer: Optimizer to use
            task_weights: Dictionary of task weights
            
        Returns:
            Dictionary of metrics
        """
        self.train()
        
        # Move data to device
        graph = self.to_device(graph)
        pos_edge_index = self.to_device(pos_edge_index)
        neg_edge_index = self.to_device(neg_edge_index)
        if past_graph is not None:
            past_graph = self.to_device(past_graph)
        if future_graph is not None:
            future_graph = self.to_device(future_graph)
            
        # Compute losses
        losses = self.compute_multi_task_loss(
            graph=graph,
            pos_edge_index=pos_edge_index,
            neg_edge_index=neg_edge_index,
            past_graph=past_graph,
            future_graph=future_graph,
            task_weights=task_weights
        )
        
        # Optimization step if optimizer provided
        if optimizer is not None:
            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()
            
        # Convert losses to float for logging
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v
                  for k, v in losses.items()}
        
        # Add autoregressive metrics
        if 'auto_metrics' in losses:
            for k, v in losses['auto_metrics'].items():
                metrics[f'auto_{k}'] = v
            metrics.pop('auto_metrics', None)
            
        return metrics
    
    def validation_step(
        self,
        graph: GraphData,
        val_pos_edge_index: torch.Tensor,
        val_neg_edge_index: torch.Tensor,
        val_past_graph: Optional[GraphData] = None,
        val_future_graph: Optional[GraphData] = None,
        task_weights: Dict[str, float] = {'link_prediction': 1.0, 'autoregressive': 1.0}
    ) -> Dict[str, float]:
        """
        Validation step for the model.
        
        Args:
            graph: Graph data object
            val_pos_edge_index: Positive edge indices for validation
            val_neg_edge_index: Negative edge indices for validation
            val_past_graph: Past graph for validation
            val_future_graph: Future graph for validation
            task_weights: Dictionary of task weights
            
        Returns:
            Dictionary of validation metrics
        """
        self.eval()
        
        # Move data to device
        graph = self.to_device(graph)
        val_pos_edge_index = self.to_device(val_pos_edge_index)
        val_neg_edge_index = self.to_device(val_neg_edge_index)
        if val_past_graph is not None:
            val_past_graph = self.to_device(val_past_graph)
        if val_future_graph is not None:
            val_future_graph = self.to_device(val_future_graph)
            
        # Compute validation losses
        with torch.no_grad():
            losses = self.compute_multi_task_loss(
                graph=graph,
                pos_edge_index=val_pos_edge_index,
                neg_edge_index=val_neg_edge_index,
                past_graph=val_past_graph,
                future_graph=val_future_graph,
                task_weights=task_weights
            )
            
        # Convert losses to float for logging
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v
                  for k, v in losses.items()}
        
        # Add autoregressive metrics
        if 'auto_metrics' in metrics:
            auto_metrics = metrics.pop('auto_metrics', {})
            for k, v in auto_metrics.items():
                metrics[f'auto_{k}'] = v
        
        # Compute link prediction metrics if available
        if val_pos_edge_index is not None and val_neg_edge_index is not None:
            with torch.no_grad():
                link_metrics = self.compute_link_prediction_metrics(
                    graph=graph,
                    pos_edge_index=val_pos_edge_index,
                    neg_edge_index=val_neg_edge_index
                )
                metrics['link_prediction_auc'] = link_metrics['auc']
                metrics['link_prediction_ap'] = link_metrics['ap']
        
        return metrics
    
    def _compute_metrics(self, scores, labels):
        """
        Compute standard classification metrics.
        
        Args:
            scores: Predicted probabilities
            labels: True labels
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy for sklearn metrics
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            
        # Ensure arrays are flattened
        scores = scores.flatten()
        labels = labels.flatten()
        
        # Binary predictions using threshold 0.5
        binary_preds = (scores >= 0.5).astype(np.float32)
        
        # Compute metrics
        metrics = {}
        
        # ROC AUC and Average Precision
        try:
            # Check if there's more than one class
            if len(np.unique(labels)) > 1:
                metrics['auc'] = roc_auc_score(labels, scores)
                metrics['ap'] = average_precision_score(labels, scores)
            else:
                # If only one class is present, set default values
                metrics['auc'] = 0.5  # Default value for random classifier
                metrics['ap'] = np.mean(labels)  # Average precision is mean of labels (all 0s or all 1s)
                logger.warning(f"Only one class present in labels (all {labels[0]}). Setting default metrics.")
        except Exception as e:
            logger.warning(f"Error computing AUC metrics: {str(e)}. Setting default values.")
            metrics['auc'] = 0.5
            metrics['ap'] = np.mean(labels)
        
        # Accuracy
        metrics['accuracy'] = np.mean(binary_preds == labels)
        
        # True positives, false positives, false negatives
        tp = np.sum((binary_preds == 1) & (labels == 1))
        fp = np.sum((binary_preds == 1) & (labels == 0))
        fn = np.sum((binary_preds == 0) & (labels == 1))
        
        # Precision, recall, F1
        metrics['precision'] = tp / (tp + fp + 1e-10)
        metrics['recall'] = tp / (tp + fn + 1e-10)
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-10)
        
        # Add precision@k for k in [1, 5, 10]
        for k in [1, 5, 10]:
            metrics[f'precision@{k}'] = self.compute_precision_at_k(scores, labels, k)
            
        return metrics

    def compute_link_prediction_metrics(
        self,
        graph: GraphData,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute metrics for link prediction task.
        
        Args:
            graph: Graph data object
            pos_edge_index: Positive edge indices
            neg_edge_index: Negative edge indices
            
        Returns:
            Dictionary of metrics
        """
        self.eval()
        
        # Check if we have enough edges for evaluation
        if pos_edge_index.shape[1] == 0 or neg_edge_index.shape[1] == 0:
            logger.warning(f"Insufficient edges for link prediction metrics. Positive: {pos_edge_index.shape[1]}, Negative: {neg_edge_index.shape[1]}")
            return {
                'auc': 0.5,
                'ap': 0.5,
                'accuracy': 0.5,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        # Move data to device
        graph = self.to_device(graph)
        pos_edge_index = self.to_device(pos_edge_index)
        neg_edge_index = self.to_device(neg_edge_index)
        
        # Get node embeddings using masked graph (transductive setting)
        with torch.no_grad():
            # Create a masked graph with positive edges removed from message passing
            # This ensures proper transductive link prediction without data leakage
            masked_graph = graph.mask_edges(pos_edge_index)
            
            # Get node embeddings from the masked graph
            # The model cannot use the positive edges during message passing
            node_embeddings = self.encoder(masked_graph)
            
            # Get source and destination node embeddings for positive edges
            src_pos, dst_pos = pos_edge_index
            src_pos_embed = node_embeddings[src_pos]
            dst_pos_embed = node_embeddings[dst_pos]
            
            # Get source and destination node embeddings for negative edges
            src_neg, dst_neg = neg_edge_index
            src_neg_embed = node_embeddings[src_neg]
            dst_neg_embed = node_embeddings[dst_neg]
            
            # Predict links
            pos_scores = self.predictor(src_pos_embed, dst_pos_embed)
            neg_scores = self.predictor(src_neg_embed, dst_neg_embed)
            
            # Apply sigmoid to get probabilities
            pos_probs = torch.sigmoid(pos_scores)
            neg_probs = torch.sigmoid(neg_scores)
            
            # Concatenate scores and create labels
            scores = torch.cat([pos_probs, neg_probs]).cpu().numpy()
            labels = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
            
            # Compute metrics
            return self._compute_metrics(scores, labels)
    
    def compute_precision_at_k(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> float:
        """
        Compute precision@k metric.
        
        Args:
            scores: Predicted scores
            labels: True labels
            k: Number of top predictions to consider
            
        Returns:
            Precision@k value
        """
        # Sort by score
        idx = np.argsort(scores)[::-1]
        top_k_idx = idx[:k]
        
        # Compute precision
        precision = np.mean(labels[top_k_idx])
        
        return precision
    
    def save(self, path: str) -> None:
        """
        Save model to file.
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare state dict with model configuration
        state_dict = {
            'model_state': self.state_dict(),
            'model_config': {
                'num_nodes': self.num_nodes,
                'node_feature_dim': self.node_feature_dim,
                'embed_dim': self.embed_dim,
                'hidden_dim': self.hidden_dim,
                'num_heads': self.num_heads,
                'edge_dim': self.edge_dim,
                'use_hierarchical': self.use_hierarchical,
                'curvature': self.curvature,
                'dropout': self.dropout,
                'ordering_strategy': self.ordering_strategy,
                'temperature': self.temperature,
                'reveal_ratio': self.reveal_ratio
            }
        }
        
        torch.save(state_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from file.
        
        Args:
            path: Path to load model from
        """
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)
        
        # Update model config if available
        if 'model_config' in state_dict:
            for key, value in state_dict['model_config'].items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        # Load model state
        if 'model_state' in state_dict:
            self.load_state_dict(state_dict['model_state'])
        else:
            self.load_state_dict(state_dict)
            
        logger.info(f"Model loaded from {path}")
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'AutoregressiveCitationModel':
        """
        Load model from pretrained file.
        
        Args:
            path: Path to load model from
            
        Returns:
            Loaded model
        """
        # Load state dict on CPU first
        state_dict = torch.load(path, map_location='cpu')
        
        # Get model config
        config = state_dict.get('model_config', {})
        
        # Create model
        model = cls(**config)
        
        # Load state dict
        if 'model_state' in state_dict:
            model.load_state_dict(state_dict['model_state'])
        else:
            model.load_state_dict(state_dict)
        
        logger.info(f"Model loaded from {path}")    
        return model 