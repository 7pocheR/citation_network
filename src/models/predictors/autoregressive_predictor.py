import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import traceback
import copy

from src.models.predictors.attention_predictor import AttentionPredictor

logger = logging.getLogger(__name__)

class AutoregressiveLinkPredictor(nn.Module):
    """
    Autoregressive Link Predictor for citation network modeling.
    
    This module extends the AttentionPredictor to support autoregressive 
    prediction of links for new nodes, treating paper generation as a 
    series of link predictions rather than using a dedicated generative model.
    
    The predictor can use different ordering strategies for autoregressive
    prediction, such as citation-based or temporal ordering.
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4, 
        dropout: float = 0.2,
        ordering_strategy: str = 'citation',  # 'citation' or 'temporal'
        temperature: float = 1.0,
        reveal_ratio: float = 0.3,  # Proportion of links to reveal during training
        node_feature_dim: Optional[int] = None  # Add node_feature_dim parameter
    ):
        """
        Initialize the autoregressive link predictor.
        
        Args:
            embed_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension for internal layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            ordering_strategy: Strategy for ordering autoregressive predictions
                               ('citation' or 'temporal')
            temperature: Temperature for controlling randomness in predictions
            reveal_ratio: Proportion of links to reveal during training
            node_feature_dim: Dimension of node features (if different from embed_dim)
        """
        super().__init__()
        
        # Store parameters
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.ordering_strategy = ordering_strategy
        self.temperature = temperature
        self.reveal_ratio = reveal_ratio
        self.node_feature_dim = node_feature_dim if node_feature_dim is not None else embed_dim
        
        # Initialize the base attention predictor
        self.attention_predictor = AttentionPredictor(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feature conditioning layers
        self.feature_projector = nn.Sequential(
            nn.Linear(self.node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Context aggregation layer (for combining predicted links)
        self.context_aggregator = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(
        self, 
        src_embeddings: torch.Tensor, 
        dst_embeddings: torch.Tensor, 
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the attention predictor.
        
        This simply passes through to the base attention predictor.
        
        Args:
            src_embeddings: Source node embeddings [num_edges, embed_dim]
            dst_embeddings: Destination node embeddings [num_edges, embed_dim]
            edge_attr: Optional edge attributes
            
        Returns:
            Probability scores for edges [num_edges]
        """
        return self.attention_predictor(src_embeddings, dst_embeddings, edge_attr)
    
    def predict_adjacency(
        self, 
        node_embeddings: torch.Tensor,
        batch_size: int = 100
    ) -> torch.Tensor:
        """
        Predict adjacency matrix for all node pairs.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, embed_dim]
            batch_size: Batch size for processing large graphs
            
        Returns:
            Adjacency matrix with probability scores [num_nodes, num_nodes]
        """
        return self.attention_predictor.predict_adjacency(node_embeddings, batch_size)
    
    def _get_node_ordering(
        self, 
        graph: Any, 
        node_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Get ordering of nodes for autoregressive prediction based on strategy.
        
        Args:
            graph: Graph data object
            node_indices: Indices of nodes to order
            
        Returns:
            Ordered indices of nodes
        """
        device = node_indices.device
        
        if self.ordering_strategy == 'citation':
            # Order by citation count (highest to lowest)
            if hasattr(graph, 'edge_index'):
                # Count incoming citations
                _, dst = graph.edge_index
                citation_counts = torch.zeros(graph.num_nodes, device=device)
                for node in dst:
                    citation_counts[node] += 1
                    
                # Get counts for the nodes we care about
                node_citations = citation_counts[node_indices]
                
                # Sort by citation count (descending)
                _, sorted_indices = torch.sort(node_citations, descending=True)
                return node_indices[sorted_indices]
            else:
                # logger.warning("No edge_index in graph, falling back to original order")
                return node_indices
                
        elif self.ordering_strategy == 'temporal':
            # Order by timestamp (oldest to newest)
            if hasattr(graph, 'timestamps') and graph.timestamps is not None:
                node_times = graph.timestamps[node_indices]
                _, sorted_indices = torch.sort(node_times)
                return node_indices[sorted_indices]
            else:
                # logger.warning("No timestamps in graph, falling back to original order")
                return node_indices
                
        else:
            # logger.warning(f"Unknown ordering strategy: {self.ordering_strategy}, using original order")
            return node_indices
    
    def _select_revealed_links(
        self, 
        true_links: torch.Tensor, 
        num_to_reveal: int
    ) -> torch.Tensor:
        """
        Randomly select a subset of true links to reveal during training.
        
        Args:
            true_links: Tensor of true link indices [num_links, 2]
            num_to_reveal: Number of links to reveal
            
        Returns:
            Tensor of revealed link indices [num_revealed, 2]
        """
        num_links = true_links.size(0)
        if num_links <= num_to_reveal:
            return true_links
            
        # Randomly select indices
        idx = torch.randperm(num_links)[:num_to_reveal]
        return true_links[idx]
    
    def predict_links_autoregressively(
        self,
        graph: Any,
        node_embeddings: torch.Tensor,
        masked_node_index: int,
        node_features: torch.Tensor,
        revealed_links: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict links autoregressively for a masked node.
        
        Args:
            graph: Graph data object
            node_embeddings: Node embeddings [num_nodes, embed_dim]
            masked_node_index: Index of node to predict links for
            node_features: Features for the masked node [feature_dim]
            revealed_links: Optional tensor of revealed links for the masked node [num_revealed, 2]
            top_k: If provided, keep only top k links
            threshold: Probability threshold for binary predictions
            
        Returns:
            Tensor of link probabilities
            Tensor of selected links (based on top_k or threshold)
        """
        # logger.info("Starting predict_links_autoregressively")
        
        num_nodes = node_embeddings.size(0)
        device = node_embeddings.device
        
        # Project node features to embedding space
        # logger.info(f"Projecting node features with shape {node_features.shape}")
        try:
            # Log more details about the input tensor
            # logger.info(f"Node features dimensions: {node_features.dim()}, device: {node_features.device}")
            
            # Ensure node_features has the right shape [batch_size, feature_dim]
            if node_features.dim() == 1:
                # If it's a 1D tensor, add a batch dimension
                # logger.info("Adding batch dimension to 1D node features")
                node_features = node_features.unsqueeze(0)
            
            node_feature_embedding = self.feature_projector(node_features)
            # logger.info(f"Feature embedding shape: {node_feature_embedding.shape}")
        except Exception as e:
            logger.error(f"Error in feature projection: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Start with feature-only embedding for masked node
        masked_node_embedding = node_feature_embedding
        
        # Log the shape after initial assignment
        # logger.info(f"Initial masked_node_embedding shape: {masked_node_embedding.shape}")
        
        # Ensure masked_node_embedding is 2D with shape [1, embed_dim]
        if masked_node_embedding.dim() > 2:
            # logger.info(f"Reshaping masked_node_embedding from {masked_node_embedding.shape}")
            masked_node_embedding = masked_node_embedding.view(1, -1)
            # logger.info(f"Reshaped to {masked_node_embedding.shape}")
        
        # If we have revealed links, update the embedding to include that information
        if revealed_links is not None and revealed_links.size(0) > 0:
            # logger.info(f"Processing {revealed_links.size(0)} revealed links")
            
            # Get embeddings of nodes with revealed links
            try:
                # Extract destination node indices from revealed links
                revealed_dst_nodes = revealed_links[:, 1]
                revealed_dst_embeds = node_embeddings[revealed_dst_nodes]
                
                # Aggregate these embeddings using the context aggregator
                context_embedding = self.context_aggregator(
                    torch.mean(revealed_dst_embeds, dim=0, keepdim=True)
                )
                
                # Combine with feature embedding
                masked_node_embedding = masked_node_embedding + context_embedding.squeeze(0)
                # logger.info("Updated node embedding with context from revealed links")
            except Exception as e:
                logger.error(f"Error processing revealed links: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue with just feature embedding if there's an error
        else:
            # logger.info("No revealed links provided - using only node features for prediction")
            pass
        
        # Predict links from masked node to all nodes in the graph
        # logger.info("Predicting links to all nodes in the graph")
        try:
            # Create virtual src and dst for batch prediction
            # Handle the dimension properly: check the shape and flatten if necessary
            if masked_node_embedding.dim() == 3:
                # If it's already 3D with shape [1, 1, embed_dim], flatten the first two dimensions
                # logger.info(f"Reshaping tensor from {masked_node_embedding.shape} to [1, {masked_node_embedding.size(-1)}]")
                masked_node_embedding = masked_node_embedding.view(1, -1)
            elif masked_node_embedding.dim() == 2 and masked_node_embedding.size(0) == 1:
                # If it's 2D with shape [1, embed_dim], keep as is
                pass
            elif masked_node_embedding.dim() == 1:
                # If it's 1D with shape [embed_dim], add a batch dimension
                masked_node_embedding = masked_node_embedding.unsqueeze(0)
                
            # Now expand to match the number of nodes
            src_embeddings = masked_node_embedding.expand(num_nodes, -1)
            dst_embeddings = node_embeddings
            
            # Log tensor shapes for debugging
            # logger.info(f"Source embeddings shape: {src_embeddings.shape}, Destination embeddings shape: {dst_embeddings.shape}")
            
            # Use attention predictor to get link probabilities
            # logger.info("Calling attention predictor")
            link_probs_raw = self.attention_predictor(src_embeddings, dst_embeddings)
            
            # Apply sigmoid to ensure values are between 0 and 1
            link_probs = torch.sigmoid(link_probs_raw)
            
            # Ensure consistent output shape
            # The attention predictor might return either [num_nodes] or [num_nodes, 1]
            # We'll standardize to [num_nodes, 1] for consistency
            if len(link_probs.shape) == 1 or link_probs.shape[1] != 1:
                link_probs = link_probs.view(-1, 1)
                
            # Add clipping for extra safety (ensure values are strictly between 0 and 1)
            link_probs = torch.clamp(link_probs, min=1e-6, max=1-1e-6)
                
            # logger.info(f"Got link probabilities with shape {link_probs.shape}")
        except Exception as e:
            logger.error(f"Error in link prediction: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Select links based on threshold or top-k
        if top_k is not None:
            # logger.info(f"Selecting top {top_k} links")
            # Get top-k links
            values, indices = torch.topk(link_probs, min(top_k, num_nodes))
            selected_links = torch.stack([
                torch.full_like(indices, masked_node_index).to(device),
                indices
            ], dim=0)
        else:
            # logger.info(f"Selecting links with threshold {threshold}")
            # Get links above threshold
            indices = torch.nonzero(link_probs >= threshold).squeeze(1)
            if indices.numel() > 0:
                selected_links = torch.stack([
                    torch.full_like(indices, masked_node_index).to(device),
                    indices
                ], dim=0)
            else:
                # No links selected - return empty tensor with correct shape
                selected_links = torch.zeros((2, 0), dtype=torch.long, device=device)
            
        # logger.info(f"Returning {selected_links.size(1)} selected links")
        return link_probs, selected_links
    
    def generate_papers_autoregressively(
        self,
        graph: Any,
        node_embeddings: torch.Tensor,
        time_threshold: float,
        future_window: float,
        num_papers: int,
        paper_features: torch.Tensor,
        top_k: Optional[int] = None,
        threshold: float = 0.5,
        num_iterations: int = 5,
        citation_threshold: float = 0.8
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Generate new papers with citation links using autoregressive prediction.
        
        Args:
            graph: Graph data object
            node_embeddings: Node embeddings for the unmasked graph [num_unmasked, embed_dim]
            time_threshold: Time threshold for masked nodes
            future_window: Time window for generation
            num_papers: Number of papers to generate
            paper_features: Features for the new papers [num_papers, feature_dim]
            top_k: If provided, keep only top k links per paper
            threshold: Probability threshold for binary predictions
            num_iterations: Number of iterations for autoregressive refinement (default=5)
            citation_threshold: Probability threshold for adding citations in refinement (default=0.8)
            
        Returns:
            Tuple containing:
            - Generated paper features [num_papers, feature_dim]
            - List of paper information dictionaries including citation probabilities
        """
        device = node_embeddings.device
        num_unmasked_nodes = node_embeddings.size(0)
        embed_dim = node_embeddings.size(1)
        
        # Generate future timestamps for the new papers
        future_times = torch.linspace(
            time_threshold, 
            time_threshold + future_window,
            num_papers,
            device=device
        )
        
        # Prepare storage for predictions
        all_link_probs = []
        all_binary_preds = []
        
        # FULLY PARALLEL APPROACH:
        # 1. Project all paper features to embeddings in one batch operation
        paper_feature_embeddings = self.feature_projector(paper_features)  # [num_papers, embed_dim]
        
        # ========== STEP 1: INITIAL PREDICTIONS (FULLY BATCHED) ==========
        # Create a large batch with each paper's embedding repeated for all nodes in graph
        # We'll create a tensor of shape [num_papers * num_unmasked_nodes, embed_dim]
        # This represents, for each paper, its embedding paired with every existing node
        
        # For each paper, create src embeddings for all node pairs
        # Reshape to [num_papers, 1, embed_dim] to prepare for expansion
        paper_embed_expanded = paper_feature_embeddings.unsqueeze(1)  # [num_papers, 1, embed_dim]
        
        # Expand to [num_papers, num_unmasked_nodes, embed_dim]
        # This repeats each paper's embedding for all nodes in the graph
        paper_embed_expanded = paper_embed_expanded.expand(-1, num_unmasked_nodes, -1)
        
        # Reshape to [num_papers * num_unmasked_nodes, embed_dim] for batch processing
        src_embeddings_batch = paper_embed_expanded.reshape(-1, embed_dim)
        
        # Repeat node embeddings for each paper
        # First, reshape to [1, num_unmasked_nodes, embed_dim]
        node_embed_expanded = node_embeddings.unsqueeze(0)  # [1, num_unmasked_nodes, embed_dim]
        
        # Expand to [num_papers, num_unmasked_nodes, embed_dim]
        # This repeats the entire node embeddings matrix for each paper
        node_embed_expanded = node_embed_expanded.expand(num_papers, -1, -1)
        
        # Reshape to [num_papers * num_unmasked_nodes, embed_dim]
        dst_embeddings_batch = node_embed_expanded.reshape(-1, embed_dim)
        
        # Now predict all paper-node links in a single batch operation
        link_probs_raw_batch = self.attention_predictor(src_embeddings_batch, dst_embeddings_batch)
        link_probs_batch = torch.sigmoid(link_probs_raw_batch)
        
        # Ensure consistent output shape
        if len(link_probs_batch.shape) == 1 or link_probs_batch.shape[1] != 1:
            link_probs_batch = link_probs_batch.view(-1, 1)
            
        # Reshape back to [num_papers, num_unmasked_nodes, 1]
        link_probs_reshaped = link_probs_batch.view(num_papers, num_unmasked_nodes, 1)
        
        # Add clipping for extra safety
        link_probs_reshaped = torch.clamp(link_probs_reshaped, min=1e-6, max=1-1e-6)
        
        # Process each paper's predictions to select edges
        # This step is harder to fully parallelize due to variable number of edges per paper
        for i in range(num_papers):
            # Get link probabilities for this paper
            paper_link_probs = link_probs_reshaped[i, :, 0]  # [num_unmasked_nodes]
            masked_node_index = num_unmasked_nodes + i
            
            # Select links based on threshold or top-k
            if top_k is not None:
                values, indices = torch.topk(paper_link_probs, min(top_k, num_unmasked_nodes))
                binary_preds = torch.stack([
                    torch.full_like(indices, masked_node_index).to(device),
                    indices
                ], dim=0)
            else:
                indices = torch.nonzero(paper_link_probs >= threshold).squeeze(1)
                if indices.numel() > 0:
                    binary_preds = torch.stack([
                        torch.full_like(indices, masked_node_index).to(device),
                        indices
                    ], dim=0)
                else:
                    binary_preds = torch.zeros((2, 0), dtype=torch.long, device=device)
            
            # Store predictions
            all_link_probs.append(paper_link_probs.view(-1, 1))  # Store as [num_nodes, 1]
            all_binary_preds.append(binary_preds)
        
        # ========== STEP 2: REFINEMENT ITERATIONS (FULLY BATCHED) ==========
        for iteration in range(num_iterations):
            # Create temporary graph with all predicted citations so far
            temp_graph = graph.clone() if hasattr(graph, 'clone') else copy.deepcopy(graph)
            
            # Collect and add all predicted edges
            all_new_edges = []
            for binary_preds in all_binary_preds:
                if binary_preds.size(1) > 0:
                    all_new_edges.append(binary_preds)
                    
            if all_new_edges:
                new_edges_combined = torch.cat(all_new_edges, dim=1)
                
                if hasattr(temp_graph, 'edge_index') and temp_graph.edge_index is not None:
                    temp_graph.edge_index = torch.cat([temp_graph.edge_index, new_edges_combined], dim=1)
                else:
                    temp_graph.edge_index = new_edges_combined
            
            # Recompute node embeddings for the temporary graph
            if hasattr(self, 'encoder') and self.encoder is not None:
                temp_node_embeddings = self.encoder(temp_graph)
            else:
                temp_node_embeddings = node_embeddings
                
            # Batch process all papers together for refinement
            # Setup is similar to initial prediction, but using updated node embeddings
            
            # Repeat each paper's embedding for all nodes
            paper_embed_expanded = paper_feature_embeddings.unsqueeze(1).expand(-1, num_unmasked_nodes, -1)
            src_embeddings_batch = paper_embed_expanded.reshape(-1, embed_dim)
            
            # Repeat updated node embeddings for each paper
            node_embed_expanded = temp_node_embeddings.unsqueeze(0).expand(num_papers, -1, -1)
            dst_embeddings_batch = node_embed_expanded.reshape(-1, embed_dim)
            
            # Predict all refined link probabilities in a single batch
            refined_probs_raw_batch = self.attention_predictor(src_embeddings_batch, dst_embeddings_batch)
            refined_probs_batch = torch.sigmoid(refined_probs_raw_batch)
            
            # Ensure consistent output shape
            if len(refined_probs_batch.shape) == 1 or refined_probs_batch.shape[1] != 1:
                refined_probs_batch = refined_probs_batch.view(-1, 1)
                
            # Reshape to [num_papers, num_unmasked_nodes, 1]
            refined_probs_reshaped = refined_probs_batch.view(num_papers, num_unmasked_nodes, 1)
            
            # Add clipping for extra safety
            refined_probs_reshaped = torch.clamp(refined_probs_reshaped, min=1e-6, max=1-1e-6)
            
            # Apply thresholding for each paper 
            # (must be in a loop due to variable number of edges per paper)
            for i in range(num_papers):
                paper_refined_probs = refined_probs_reshaped[i, :, 0]  # [num_unmasked_nodes]
                masked_node_index = num_unmasked_nodes + i
                
                # Apply refinement threshold (usually higher than initial threshold)
                indices = torch.nonzero(paper_refined_probs >= citation_threshold).squeeze(1)
                if indices.numel() > 0:
                    binary_preds = torch.stack([
                        torch.full_like(indices, masked_node_index).to(device),
                        indices
                    ], dim=0)
                else:
                    binary_preds = torch.zeros((2, 0), dtype=torch.long, device=device)
                
                # Update predictions
                all_link_probs[i] = paper_refined_probs.view(-1, 1)  # Store as [num_nodes, 1]
                all_binary_preds[i] = binary_preds
        
        # Create final paper info dictionaries
        paper_info_list = []
        for i in range(num_papers):
            paper_info = {
                'index': num_unmasked_nodes + i,
                'feature': paper_features[i].cpu().numpy(),
                'timestamp': future_times[i].item(),
                'citation_probs': all_link_probs[i].cpu().numpy(),
                'citations': all_binary_preds[i].cpu().numpy(),
                'num_citations': all_binary_preds[i].size(1)
            }
            
            paper_info_list.append(paper_info)
        
        return paper_features, paper_info_list 