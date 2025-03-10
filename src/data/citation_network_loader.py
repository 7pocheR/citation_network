import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import random
from collections import defaultdict

logger = logging.getLogger(__name__)

class CitationNetworkLoader:
    """
    Data loader for citation network datasets.
    
    Handles loading and preprocessing of citation network data.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the loader.
        
        Args:
            data_path: Path to the citation network data file
        """
        self.data_path = data_path
        self.data = self._load_data(data_path)
        
        # Metadata
        self.paper_ids = list(self.data['papers'].keys())
        self.num_nodes = len(self.paper_ids)
        self.id_to_idx = {paper_id: idx for idx, paper_id in enumerate(self.paper_ids)}
        
        # Extract publication years
        self.paper_years = self._extract_publication_years()
        
        # Placeholder for features and edges
        self.node_features = None
        self.edge_indices = None
        self.edge_timestamps = None
        
        logger.info(f"Loaded {self.num_nodes} papers from {data_path}")
    
    def _load_data(self, data_path: str) -> Dict:
        """
        Load data from a JSON file.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded data as a dictionary
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _extract_publication_years(self) -> Dict[str, int]:
        """
        Extract publication years for all papers.
        
        Returns:
            Dictionary mapping paper IDs to publication years
        """
        years = {}
        for paper_id, paper_data in self.data['papers'].items():
            pub_date = paper_data.get('publication_date', '')
            try:
                # Assuming format like "2020-01-01"
                year = int(pub_date.split('-')[0]) if pub_date else 0
            except (ValueError, IndexError):
                year = 0
            years[paper_id] = year
        return years
    
    def process_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], Dict[str, int]]:
        """
        Process the loaded data into tensors for the model.
        
        Returns:
            node_features: Tensor of node features
            edge_indices: Tensor of edge indices
            edge_timestamps: Tensor of edge timestamps
            paper_ids: List of paper IDs
            paper_years: Dictionary mapping paper IDs to years
        """
        # Create node features
        self.node_features = self.create_node_features()
        
        # Create edges
        self.edge_indices, self.edge_timestamps = self.create_edge_index_and_time()
        
        return self.node_features, self.edge_indices, self.edge_timestamps, self.paper_ids, self.paper_years
    
    def create_node_features(self) -> torch.Tensor:
        """
        Create node feature tensors from the raw data.
        
        For simplicity, we create random features in this implementation.
        In a real implementation, you would extract meaningful features.
        
        Returns:
            Tensor of node features with shape [num_nodes, feature_dim]
        """
        # For testing, use a small feature dimension
        feature_dim = 16
        
        # Create random features
        # In a real implementation, these would be extracted from paper attributes
        features = np.random.randn(self.num_nodes, feature_dim).astype(np.float32)
        return torch.FloatTensor(features)
    
    def create_edge_index_and_time(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edge indices and timestamps from the citation data.
        
        Returns:
            edge_indices: Tensor of edge indices with shape [2, num_edges]
            edge_timestamps: Tensor of edge timestamps with shape [num_edges]
        """
        source_nodes = []
        target_nodes = []
        timestamps = []
        
        for source_id, paper_data in self.data['papers'].items():
            source_idx = self.id_to_idx[source_id]
            source_year = self.paper_years[source_id]
            
            # Process references
            for ref_id in paper_data.get('referenced_works', []):
                if ref_id in self.id_to_idx:  # Check if the referenced paper is in our dataset
                    target_idx = self.id_to_idx[ref_id]
                    target_year = self.paper_years.get(ref_id, 0)
                    
                    # Add the edge
                    source_nodes.append(source_idx)
                    target_nodes.append(target_idx)
                    
                    # Use the source paper's year as timestamp
                    # In a real implementation, you might have more precise timestamps
                    timestamps.append(source_year)
        
        # Convert to tensors
        edge_indices = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_timestamps = torch.tensor(timestamps, dtype=torch.long)
        
        logger.info(f"Created {len(source_nodes)} edges")
        
        return edge_indices, edge_timestamps
    
    def split_data(self, test_year: int, val_ratio: float = 0.2) -> Tuple[Dict, Dict, Dict]:
        """
        Split the data into training, validation, and test sets.
        
        Args:
            test_year: Papers published on or after this year are used for testing
            val_ratio: Ratio of training papers to use for validation
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        # Ensure we have processed data
        if self.node_features is None:
            self.process_data()
        
        # Get indices of papers by year
        train_indices = []
        test_indices = []
        
        for paper_id, year in self.paper_years.items():
            idx = self.id_to_idx[paper_id]
            if year < test_year:
                train_indices.append(idx)
            else:
                test_indices.append(idx)
        
        # Shuffle training indices
        random.shuffle(train_indices)
        
        # Split training into train and validation
        val_size = int(len(train_indices) * val_ratio)
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]
        
        logger.info(f"Data split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test")
        
        # Create data dictionaries
        train_data = {
            'node_features': self.node_features,
            'edge_indices': self.edge_indices,
            'timestamps': self.edge_timestamps,
            'paper_ids': [self.paper_ids[i] for i in train_indices]
        }
        
        val_data = {
            'node_features': self.node_features,
            'edge_indices': self.edge_indices,
            'timestamps': self.edge_timestamps,
            'paper_ids': [self.paper_ids[i] for i in val_indices]
        }
        
        test_data = {
            'node_features': self.node_features,
            'edge_indices': self.edge_indices,
            'timestamps': self.edge_timestamps,
            'paper_ids': [self.paper_ids[i] for i in test_indices]
        }
        
        return train_data, val_data, test_data
    
    def get_papers_by_year(self, year: int) -> List[int]:
        """Get indices of papers published in a specific year."""
        indices = []
        for paper_id, pub_year in self.paper_years.items():
            if pub_year == year:
                indices.append(self.id_to_idx[paper_id])
        return indices
    
    @property
    def node_feature_dim(self) -> int:
        """Get the dimension of node features."""
        if self.node_features is None:
            self.process_data()
        return self.node_features.shape[1] 