import json
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx
from torch_geometric.data import Data, HeteroData


class CitationNetworkLoader:
    """
    Data loader for citation network data.
    Processes raw citation data into temporal snapshots for dynamic GNN training.
    """
    
    def __init__(self, data_path: str, cut_year: int, min_citations: int = 0):
        """
        Initialize the citation network loader.
        
        Args:
            data_path: Path to the JSON data file
            cut_year: The year T to cut off the data (train on data before T)
            min_citations: Minimum number of citations for a paper to be included
        """
        self.data_path = data_path
        self.cut_year = cut_year
        self.min_citations = min_citations
        self.paper_ids_to_idx = {}  # Maps paper IDs to numeric indices
        self.idx_to_paper_ids = {}  # Maps numeric indices to paper IDs
        self.paper_features = {}    # Maps paper IDs to feature vectors
        self.paper_topics = {}      # Maps paper IDs to topic vectors
        self.citations = {}         # Maps paper IDs to lists of cited paper IDs
        self.publication_dates = {} # Maps paper IDs to publication dates
        self.temporal_edges = []    # List of (source, target, time) for dynamic graph
        
    def load_data(self):
        """Load the citation network data from the JSON file."""
        print(f"Loading data from {self.data_path}...")
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        # Process papers and citations
        papers = data.get('papers', {})
        
        # First pass: collect all paper IDs and their features
        for paper_id, paper_data in tqdm(papers.items(), desc="Processing papers"):
            # Extract publication date
            pub_date_str = paper_data.get('publication_date', '')
            if not pub_date_str:
                continue
                
            pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d')
            pub_year = pub_date.year
            
            # Skip papers published after cut-off year
            if pub_year > self.cut_year:
                continue
                
            # Skip papers with too few citations
            cited_by_count = paper_data.get('cited_by_count', 0)
            if cited_by_count < self.min_citations:
                continue
            
            # Store publication date
            self.publication_dates[paper_id] = pub_date
            
            # Process paper topics
            topics = paper_data.get('topics', [])
            topic_vector = [0.0] * 10  # Assuming max 10 topics, adjust as needed
            
            for i, topic in enumerate(topics[:10]):  # Limit to 10 topics
                topic_name = topic.get('display_name', '')
                topic_score = topic.get('score', 0.0)
                topic_vector[i] = topic_score
            
            self.paper_topics[paper_id] = topic_vector
            
            # Process paper keywords
            keywords = paper_data.get('keywords', [])
            keyword_vector = [0.0] * 20  # Assuming max 20 keywords, adjust as needed
            
            for i, keyword in enumerate(keywords[:20]):  # Limit to 20 keywords
                keyword_score = keyword.get('score', 0.0)
                keyword_vector[i] = keyword_score
            
            # Combine topics and keywords as paper features
            self.paper_features[paper_id] = topic_vector + keyword_vector
            
            # Store citations
            references = paper_data.get('referenced_works', [])
            self.citations[paper_id] = references
        
        # Create mappings between paper IDs and indices
        for idx, paper_id in enumerate(self.paper_features.keys()):
            self.paper_ids_to_idx[paper_id] = idx
            self.idx_to_paper_ids[idx] = paper_id
        
        # Process temporal citation edges
        for source_id, target_ids in tqdm(self.citations.items(), desc="Processing citations"):
            source_idx = self.paper_ids_to_idx.get(source_id)
            if source_idx is None:
                continue
                
            source_date = self.publication_dates[source_id]
            
            for target_id in target_ids:
                target_idx = self.paper_ids_to_idx.get(target_id)
                if target_idx is None:
                    continue
                
                # Ensure the cited paper was published before the citing paper
                if target_id in self.publication_dates:
                    target_date = self.publication_dates[target_id]
                    if target_date >= source_date:
                        continue
                        
                # Add directed edge: source cites target
                time = source_date.timestamp()
                self.temporal_edges.append((source_idx, target_idx, time))
        
        print(f"Loaded {len(self.paper_ids_to_idx)} papers and {len(self.temporal_edges)} citation edges")
        return self
    
    def create_node_features(self) -> torch.Tensor:
        """Create a tensor of node features for all papers."""
        num_nodes = len(self.paper_ids_to_idx)
        feature_dim = len(next(iter(self.paper_features.values())))
        features = torch.zeros((num_nodes, feature_dim))
        
        for paper_id, idx in self.paper_ids_to_idx.items():
            features[idx] = torch.tensor(self.paper_features[paper_id])
        
        return features
    
    def create_edge_index_and_time(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create edge index and edge time tensors for temporal graph."""
        if not self.temporal_edges:
            raise ValueError("No temporal edges found. Call load_data() first.")
            
        # Sort edges by time
        self.temporal_edges.sort(key=lambda x: x[2])
        
        edge_index = torch.tensor([[s, t] for s, t, _ in self.temporal_edges], 
                                  dtype=torch.long).t().contiguous()
        edge_time = torch.tensor([t for _, _, t in self.temporal_edges], 
                               dtype=torch.float)
        
        return edge_index, edge_time
    
    def create_snapshots(self, num_snapshots: int) -> List[Data]:
        """
        Create a series of graph snapshots for temporal analysis.
        
        Args:
            num_snapshots: Number of temporal snapshots to create
            
        Returns:
            List of PyG Data objects representing graph snapshots
        """
        features = self.create_node_features()
        edge_index, edge_time = self.create_edge_index_and_time()
        
        # Find time range
        min_time = edge_time.min().item()
        max_time = edge_time.max().item()
        time_range = max_time - min_time
        
        snapshots = []
        for i in range(num_snapshots):
            # Calculate time threshold for this snapshot
            time_threshold = min_time + (i + 1) * time_range / num_snapshots
            
            # Select edges before this threshold
            mask = edge_time <= time_threshold
            snapshot_edge_index = edge_index[:, mask]
            
            # Create PyG Data object
            snapshot = Data(
                x=features,
                edge_index=snapshot_edge_index,
                timestamp=time_threshold
            )
            snapshots.append(snapshot)
            
        return snapshots
    
    def create_train_test_split(self, test_year: Optional[int] = None) -> Tuple[List[int], List[int]]:
        """
        Split papers into training and testing sets based on publication year.
        
        Args:
            test_year: Year to use for test set. If None, uses cut_year.
            
        Returns:
            Tuple of (train_indices, test_indices)
        """
        if test_year is None:
            test_year = self.cut_year
            
        train_indices = []
        test_indices = []
        
        for paper_id, idx in self.paper_ids_to_idx.items():
            year = self.publication_dates[paper_id].year
            
            if year < test_year:
                train_indices.append(idx)
            elif year == test_year:
                test_indices.append(idx)
        
        return train_indices, test_indices
    
    def get_papers_by_year(self, year: int) -> List[int]:
        """Get indices of papers published in a specific year."""
        indices = []
        for paper_id, idx in self.paper_ids_to_idx.items():
            if self.publication_dates[paper_id].year == year:
                indices.append(idx)
        return indices
    
    def create_pyg_temporal_graph(self) -> Data:
        """Create a PyTorch Geometric temporal graph representation."""
        features = self.create_node_features()
        edge_index, edge_time = self.create_edge_index_and_time()
        
        data = Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_time.unsqueeze(1)  # Edge timestamp as attribute
        )
        
        return data
    
    def get_feature_dim(self) -> int:
        """Get the dimension of node feature vectors."""
        return len(next(iter(self.paper_features.values())))
    
    def get_num_nodes(self) -> int:
        """Get the total number of nodes in the graph."""
        return len(self.paper_ids_to_idx) 