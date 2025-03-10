import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union

from src.data.datasets import GraphData


class BasePredictor(nn.Module, ABC):
    """Base interface for all citation link predictors in the citation network project.
    
    The predictor uses node embeddings to predict citation relationships between papers.
    It serves as an auxiliary task to effectively train the encoder, providing a way to
    learn the patterns of citations in the network.
    
    All predictor implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, 
                 embed_dim: int, 
                 **kwargs):
        """Initialize the base predictor.
        
        Args:
            embed_dim (int): Dimensionality of node embeddings
            **kwargs: Additional parameters specific to the predictor implementation
        """
        super().__init__()
        self.embed_dim = embed_dim
        
    @abstractmethod
    def forward(self, 
                src_embeddings: torch.Tensor, 
                dst_embeddings: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict the likelihood of citation links between papers.
        
        Args:
            src_embeddings (torch.Tensor): Embeddings of source nodes (citing papers)
                [batch_size, embed_dim]
            dst_embeddings (torch.Tensor): Embeddings of destination nodes (cited papers)
                [batch_size, embed_dim]
            edge_attr (Optional[torch.Tensor]): Edge attributes (if available)
                [batch_size, edge_attr_dim]
                
        Returns:
            torch.Tensor: Predicted citation likelihood scores [batch_size]
        """
        pass
    
    @abstractmethod
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
            edge_attr (Optional[torch.Tensor]): Edge attributes (if available)
                [num_edges, edge_attr_dim]
                
        Returns:
            torch.Tensor: Predicted citation likelihood scores [num_edges]
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def predict_temporal_citations(self,
                                  node_embeddings: torch.Tensor,
                                  existing_graph: GraphData,
                                  time_threshold: float,
                                  future_window: Optional[float] = None,
                                  k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict future citations based on a temporal snapshot.
        
        Predict citations that will appear after the time threshold, based on
        the state of the citation network at the specified time threshold.
        
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
        pass
    
    def get_candidate_edges(self, 
                           existing_graph: GraphData,
                           max_candidates: Optional[int] = None) -> torch.Tensor:
        """Get candidate edges that don't exist in the current graph.
        
        Helper method to generate candidate edges for prediction.
        
        Args:
            existing_graph (GraphData): The current citation network
            max_candidates (Optional[int]): Maximum number of candidate edges to return
                
        Returns:
            torch.Tensor: Candidate edge indices [2, num_candidates]
        """
        # Default implementation samples random non-existing edges
        # Calculate num_nodes from the node features or timestamp fields
        if hasattr(existing_graph, 'x') and existing_graph.x is not None:
            num_nodes = existing_graph.x.shape[0]
        elif hasattr(existing_graph, 'node_timestamps') and existing_graph.node_timestamps is not None:
            num_nodes = len(existing_graph.node_timestamps)
        elif hasattr(existing_graph, 'paper_times') and existing_graph.paper_times is not None:
            num_nodes = len(existing_graph.paper_times)
        else:
            # Fallback: get max node index from edge_index + 1
            num_nodes = existing_graph.edge_index.max().item() + 1
        
        # Create a mask for existing edges
        edge_index = existing_graph.edge_index
        edge_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        edge_mask[edge_index[0], edge_index[1]] = True
        
        # Sample candidate edges (non-existing)
        candidates = []
        
        # If max_candidates is None or unreasonably large, set a reasonable limit
        if max_candidates is None or max_candidates > num_nodes * 100:
            max_candidates = min(num_nodes * 100, 10000)
            
        while len(candidates) < max_candidates:
            # Sample random node pairs
            src = torch.randint(0, num_nodes, (1,))
            dst = torch.randint(0, num_nodes, (1,))
            
            # Check if edge doesn't exist and nodes are different
            if src != dst and not edge_mask[src, dst]:
                candidates.append((src.item(), dst.item()))
                edge_mask[src, dst] = True  # Mark as considered
        
        # Convert to tensor
        candidate_edges = torch.tensor(candidates, dtype=torch.long).t()
        return candidate_edges
    
    def get_config(self) -> Dict[str, Any]:
        """Get predictor configuration parameters.
        
        Returns:
            Dict[str, Any]: Dictionary containing configuration parameters
        """
        return {
            'embed_dim': self.embed_dim,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BasePredictor':
        """Create a predictor instance from configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
                
        Returns:
            BasePredictor: An instance of the predictor
        """
        return cls(**config) 