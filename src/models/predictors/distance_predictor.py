import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .base import BasePredictor
from src.data.datasets import GraphData


class DistancePredictor(BasePredictor):
    """A simple baseline predictor that uses embedding distances to predict citations.
    
    DistancePredictor assumes that papers with similar embeddings are more likely to be
    connected by citations. It uses a simple distance/similarity metric in the embedding
    space to predict citation likelihood.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 distance_metric: str = 'cosine',
                 add_trainable_params: bool = True,
                 **kwargs):
        """Initialize the distance-based predictor.
        
        Args:
            embed_dim (int): Dimensionality of node embeddings
            distance_metric (str): Distance metric to use ('cosine', 'euclidean', or 'dot')
            add_trainable_params (bool): Whether to add trainable parameters
            **kwargs: Additional parameters for the base class
        """
        super().__init__(embed_dim=embed_dim, **kwargs)
        self.distance_metric = distance_metric
        self.add_trainable_params = add_trainable_params
        
        # Add trainable parameters to enable gradient flow
        if add_trainable_params:
            self.src_transform = nn.Linear(embed_dim, embed_dim)
            self.dst_transform = nn.Linear(embed_dim, embed_dim)
            self.score_scale = nn.Parameter(torch.ones(1))
            self.score_shift = nn.Parameter(torch.zeros(1))
        
    def forward(self, 
                src_embeddings: torch.Tensor, 
                dst_embeddings: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict citation likelihood using embedding distances.
        
        Args:
            src_embeddings (torch.Tensor): Embeddings of source nodes (citing papers)
                [batch_size, embed_dim]
            dst_embeddings (torch.Tensor): Embeddings of destination nodes (cited papers)
                [batch_size, embed_dim]
            edge_attr (Optional[torch.Tensor]): Edge attributes, not used in this predictor
                
        Returns:
            torch.Tensor: Predicted citation likelihood scores [batch_size]
        """
        # Ensure tensors are on the same device
        device = src_embeddings.device
        
        # Apply transformations if using trainable parameters
        if self.add_trainable_params:
            src_embeddings = self.src_transform(src_embeddings)
            dst_embeddings = self.dst_transform(dst_embeddings)
        
        if self.distance_metric == 'cosine':
            # Normalize embeddings for cosine similarity
            src_norm = torch.nn.functional.normalize(src_embeddings, p=2, dim=1)
            dst_norm = torch.nn.functional.normalize(dst_embeddings, p=2, dim=1)
            # Cosine similarity (higher values indicate more similarity)
            similarity = torch.sum(src_norm * dst_norm, dim=1)
            
            # Apply trainable scaling and shifting if enabled
            if self.add_trainable_params:
                similarity = self.score_scale * similarity + self.score_shift
                
            return similarity
        
        elif self.distance_metric == 'euclidean':
            # Euclidean distance (lower values indicate more similarity)
            # We negate it so larger values mean higher likelihood
            distance = torch.sqrt(torch.sum((src_embeddings - dst_embeddings) ** 2, dim=1) + 1e-8)
            similarity = 1.0 / (1.0 + distance)  # Convert to similarity score between 0 and 1
            
            # Apply trainable scaling and shifting if enabled
            if self.add_trainable_params:
                similarity = self.score_scale * similarity + self.score_shift
                
            return similarity
        
        elif self.distance_metric == 'dot':
            # Dot product (higher values indicate more similarity)
            similarity = torch.sum(src_embeddings * dst_embeddings, dim=1)
            
            # Apply trainable scaling and shifting if enabled
            if self.add_trainable_params:
                similarity = self.score_scale * similarity + self.score_shift
                
            return similarity
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def predict_batch(self, 
                     node_embeddings: torch.Tensor, 
                     edge_indices: torch.Tensor,
                     edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict citations for a batch of edges.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings for all papers
                [num_nodes, embed_dim]
            edge_indices (torch.Tensor): Edge indices to predict
                [2, num_edges] where edge_indices[0] are source nodes and
                edge_indices[1] are destination nodes
            edge_attr (Optional[torch.Tensor]): Edge attributes, not used in this predictor
                
        Returns:
            torch.Tensor: Predicted citation likelihood scores [num_edges]
        """
        # Get src and dst embeddings for the specified edges
        src_indices = edge_indices[0]
        dst_indices = edge_indices[1]
        
        # Ensure all tensors are on the same device
        device = node_embeddings.device
        src_indices = src_indices.to(device)
        dst_indices = dst_indices.to(device)
        
        src_embeddings = node_embeddings[src_indices]
        dst_embeddings = node_embeddings[dst_indices]
        
        # Use forward method to predict
        return self.forward(src_embeddings, dst_embeddings, edge_attr)
    
    def predict_citations(self,
                         node_embeddings: torch.Tensor,
                         existing_graph: GraphData,
                         k: int = 10,
                         **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict new citations for papers in an existing graph.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings for all papers
                [num_nodes, embed_dim]
            existing_graph (GraphData): The existing citation network
            k (int): Number of top predictions to return
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Edge indices of top predicted citations [2, k]
                - Scores for the top predicted citations [k]
        """
        # Ensure we're using the correct device
        device = node_embeddings.device
        
        # Get existing edges to avoid predicting them again
        existing_edge_set = set([(existing_graph.edge_index[0, i].item(), 
                               existing_graph.edge_index[1, i].item()) 
                              for i in range(existing_graph.edge_index.shape[1])])
        
        # For temporal prediction, we can use publication times if available
        paper_times = None
        if hasattr(existing_graph, 'node_timestamps') and existing_graph.node_timestamps is not None:
            paper_times = existing_graph.node_timestamps
        elif hasattr(existing_graph, 'paper_times') and existing_graph.paper_times is not None:
            # Legacy field name fallback
            paper_times = existing_graph.paper_times
        
        # Create all possible pairs of papers
        num_nodes = node_embeddings.shape[0]
        candidate_edges = []
        
        # Filter based on temporal constraints if time information available
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue  # Skip self-citations
                    
                # Skip existing citations
                if (i, j) in existing_edge_set:
                    continue
                    
                # Apply temporal constraint: can't cite future papers
                if paper_times is not None:
                    if paper_times[i] > paper_times[j]:
                        continue  # Can't cite papers from the future
                
                candidate_edges.append((i, j))
        
        if not candidate_edges:
            # No valid candidates
            return torch.zeros((2, 0), dtype=torch.long).to(device), torch.zeros(0).to(device)
        
        # Convert to tensor and move to the correct device
        candidate_edges = torch.tensor(candidate_edges, dtype=torch.long).t().to(device)
        
        # Compute scores for candidate edges
        scores = []
        for i in range(candidate_edges.shape[1]):
            src, dst = candidate_edges[0, i], candidate_edges[1, i]
            src_emb = node_embeddings[src].unsqueeze(0)
            dst_emb = node_embeddings[dst].unsqueeze(0)
            
            # Use forward method for prediction
            score = self.forward(src_emb, dst_emb).item()
            scores.append(score)
        
        scores = torch.tensor(scores).to(device)
        
        # Get top k predictions
        if len(scores) > k:
            top_k_indices = torch.topk(scores, k=k).indices
            top_edges = candidate_edges[:, top_k_indices]
            top_scores = scores[top_k_indices]
        else:
            # In case there are fewer candidates than k
            top_edges = candidate_edges
            top_scores = scores
        
        return top_edges, top_scores
    
    def predict_temporal_citations(self,
                                  node_embeddings: torch.Tensor,
                                  existing_graph: GraphData,
                                  time_threshold: float,
                                  future_window: Optional[float] = None,
                                  k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict future citations based on a temporal snapshot.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings for all papers
                [num_nodes, embed_dim]
            existing_graph (GraphData): The complete citation network
            time_threshold (float): The timestamp threshold for the snapshot
            future_window (Optional[float]): If provided, only predict citations
                within the window [time_threshold, time_threshold + future_window]
            k (int): Number of top predictions to return
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Edge indices of top predicted future citations [2, k]
                - Scores for the top predicted future citations [k]
        """
        # Ensure we're using the correct device
        device = node_embeddings.device
        
        # Extract papers that exist at the time threshold
        node_timestamps = existing_graph.node_timestamps  # Using standardized field name
        valid_papers = torch.where(node_timestamps <= time_threshold)[0]
        
        # Get edge timestamps using standardized field name
        edge_timestamps = existing_graph.edge_timestamps
        if edge_timestamps is None:
            # If no time information available, use all existing edges
            edge_timestamps = torch.zeros(existing_graph.edge_index.shape[1])
        
        # Move tensors to the correct device
        edge_timestamps = edge_timestamps.to(device)
        
        # Extract edges prior to time threshold
        edge_index = existing_graph.edge_index.to(device)
        edges_at_threshold = edge_index[:, edge_timestamps <= time_threshold]
        
        # Create a snapshot graph at time_threshold
        snapshot_graph = GraphData(
            x=existing_graph.x,
            edge_index=edges_at_threshold,
            node_timestamps=node_timestamps,  # Using standardized field name
            snapshot_time=time_threshold
        )
        
        # Get candidate edges that might form in the future
        candidate_edges = self.get_candidate_edges(snapshot_graph)
        
        # Filter candidates to only include papers that exist at time_threshold
        valid_papers_set = set(valid_papers.tolist())
        filtered_candidates = []
        
        for i in range(candidate_edges.shape[1]):
            src, dst = candidate_edges[0, i].item(), candidate_edges[1, i].item()
            if src in valid_papers_set and dst in valid_papers_set:
                filtered_candidates.append((src, dst))
        
        if not filtered_candidates:
            # Return empty tensors if no valid candidates
            return torch.empty((2, 0), dtype=torch.long).to(device), torch.empty(0).to(device)
        
        filtered_candidate_edges = torch.tensor(filtered_candidates, dtype=torch.long).t().to(device)
        
        # Predict scores for candidate edges
        scores = self.predict_batch(node_embeddings, filtered_candidate_edges)
        
        # Get top k predictions
        if k < len(scores):
            top_k_indices = torch.topk(scores, k=k).indices
            top_edges = filtered_candidate_edges[:, top_k_indices]
            top_scores = scores[top_k_indices]
        else:
            # In case there are fewer candidates than k
            top_edges = filtered_candidate_edges
            top_scores = scores
        
        return top_edges, top_scores
    
    def get_config(self) -> Dict[str, Any]:
        """Get predictor configuration parameters.
        
        Returns:
            Dict[str, Any]: Dictionary containing configuration parameters
        """
        config = super().get_config()
        config.update({
            'distance_metric': self.distance_metric,
            'add_trainable_params': self.add_trainable_params,
        })
        return config 