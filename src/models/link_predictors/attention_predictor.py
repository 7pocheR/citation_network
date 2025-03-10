import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPredictor(nn.Module):
    def __init__(self, embed_dim, use_edge_attr=True, dot_product=True, dropout=0.5):
        super(AttentionPredictor, self).__init__()
        self.embed_dim = embed_dim
        self.use_edge_attr = use_edge_attr
        self.dot_product = dot_product
        self.dropout = dropout
        
        # MLP layers
        self.src_mlp = nn.Linear(embed_dim, embed_dim)
        self.dst_mlp = nn.Linear(embed_dim, embed_dim)
        self.final_mlp = nn.Linear(2 * embed_dim, 1)

    def forward(self, src_embeddings, dst_embeddings, edge_attr=None):
        """
        Forward pass of the attention predictor.
        
        Args:
            src_embeddings: Source node embeddings [num_edges, embed_dim]
            dst_embeddings: Destination node embeddings [num_edges, embed_dim]
            edge_attr: Optional edge attributes
            
        Returns:
            Probability scores for edges [num_edges]
        """
        if src_embeddings.shape[0] == 0 or dst_embeddings.shape[0] == 0:
            # Return empty tensor if no embeddings provided
            return torch.tensor([], device=src_embeddings.device)
            
        # Handle edge attributes if provided
        if edge_attr is not None and self.use_edge_attr:
            if isinstance(edge_attr, dict) and 'attr' in edge_attr:
                edge_attr = edge_attr['attr']
                
            if torch.is_tensor(edge_attr) and edge_attr.dim() > 0:
                # Project edge attributes to embed_dim, if needed
                if not hasattr(self, 'edge_proj') or self.edge_proj is None:
                    edge_dim = edge_attr.size(-1)
                    self.edge_proj = nn.Linear(edge_dim, self.embed_dim).to(src_embeddings.device)
                
                edge_proj = self.edge_proj(edge_attr)
                
                # Add edge projections to node embeddings
                src_embeddings = src_embeddings + edge_proj
                dst_embeddings = dst_embeddings + edge_proj
                
        # Apply MLP to src and dst embeddings
        src_h = F.relu(self.src_mlp(src_embeddings))
        dst_h = F.relu(self.dst_mlp(dst_embeddings))
        
        # Compute attention scores
        if self.dot_product:
            # Apply dropout
            if self.dropout > 0:
                src_h = F.dropout(src_h, p=self.dropout, training=self.training)
                dst_h = F.dropout(dst_h, p=self.dropout, training=self.training)
                
            # Compute dot product
            scores = torch.sum(src_h * dst_h, dim=-1)
        else:
            # Concatenate embeddings
            concat = torch.cat([src_h, dst_h], dim=-1)
            
            # Apply dropout
            if self.dropout > 0:
                concat = F.dropout(concat, p=self.dropout, training=self.training)
                
            # Apply final MLP
            scores = self.final_mlp(concat).squeeze(-1)
            
        return scores
        
    def predict_adjacency(self, node_embeddings):
        """
        Predict adjacency matrix for all node pairs.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, embed_dim]
            
        Returns:
            Adjacency matrix with probability scores [num_nodes, num_nodes]
        """
        num_nodes = node_embeddings.size(0)
        device = node_embeddings.device
        
        # For small graphs, compute all pairs
        if num_nodes <= 1000:
            # Create all possible pairs
            src_indices = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
            dst_indices = torch.arange(num_nodes, device=device).repeat(num_nodes)
            
            # Get embeddings for all pairs
            src_embeddings = node_embeddings[src_indices]
            dst_embeddings = node_embeddings[dst_indices]
            
            # Predict scores
            scores = self.forward(src_embeddings, dst_embeddings)
            
            # Reshape to adjacency matrix
            adjacency = scores.reshape(num_nodes, num_nodes)
            
            # Apply sigmoid to get probabilities
            adjacency = torch.sigmoid(adjacency)
            
            return adjacency
        
        # For large graphs, compute in batches to avoid OOM
        else:
            adjacency = torch.zeros((num_nodes, num_nodes), device=device)
            batch_size = 100  # Adjust based on GPU memory
            
            for i in range(0, num_nodes, batch_size):
                end_i = min(i + batch_size, num_nodes)
                src_embeddings = node_embeddings[i:end_i]
                
                for j in range(0, num_nodes, batch_size):
                    end_j = min(j + batch_size, num_nodes)
                    dst_embeddings = node_embeddings[j:end_j]
                    
                    # Create all pairs between batches
                    src_indices = torch.arange(end_i - i, device=device).repeat_interleave(end_j - j)
                    dst_indices = torch.arange(end_j - j, device=device).repeat(end_i - i)
                    
                    # Get embeddings
                    batch_src_embeddings = src_embeddings[src_indices]
                    batch_dst_embeddings = dst_embeddings[dst_indices]
                    
                    # Predict scores
                    batch_scores = self.forward(batch_src_embeddings, batch_dst_embeddings)
                    batch_scores = torch.sigmoid(batch_scores)
                    
                    # Add to adjacency matrix
                    adjacency[i:end_i, j:end_j] = batch_scores.reshape(end_i - i, end_j - j)
            
            return adjacency 