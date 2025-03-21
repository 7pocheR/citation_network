import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import random
from collections import defaultdict
import pickle
import os

logger = logging.getLogger(__name__)

class OneHotCitationNetworkLoader:
    """
    Enhanced data loader for citation network datasets that uses one-hot encoding
    of topics and keywords for node features.
    
    Handles loading and preprocessing of citation network data with semantically
    meaningful features based on OpenAlex topics and keywords.
    """
    
    def __init__(self, data_path: str, embedding_dict_path: str = "embedding_dictionaries.pkl"):
        """
        Initialize the loader with paths to the dataset and embedding dictionaries.
        
        Args:
            data_path: Path to the citation network dataset JSON file
            embedding_dict_path: Path to the pickle file containing topic and keyword mappings
        """
        # Load data
        logger.info(f"Loading data from {data_path}")
        self.data_path = data_path
        self.embedding_dict_path = embedding_dict_path
        
        # Check if embedding dictionaries exist
        if not os.path.exists(embedding_dict_path):
            raise FileNotFoundError(f"Embedding dictionaries not found at {embedding_dict_path}. "
                                   f"Run create_embedding_dictionaries.py first.")
        
        # Load embedding dictionaries
        with open(embedding_dict_path, 'rb') as f:
            self.embedding_data = pickle.load(f)
            
        logger.info(f"Loaded embedding dictionaries with {self.embedding_data['topic_count']} topics "
                   f"and {self.embedding_data['keyword_count']} keywords")
        
        # Load dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Extract paper data
        self.papers = self.data.get('papers', {})
        self.num_nodes = len(self.papers)
        logger.info(f"Loaded {self.num_nodes} papers from dataset")
        
        # Map paper IDs to indices
        self.paper_id_to_idx = {}
        self.paper_ids = []
        for idx, paper_id in enumerate(self.papers.keys()):
            self.paper_id_to_idx[paper_id] = idx
            self.paper_ids.append(paper_id)
        
        # Extract publication years if available
        self.paper_years = []
        for paper_id, paper_data in self.papers.items():
            year = None
            
            # Try to extract the year from publication_date
            pub_date = paper_data.get('publication_date')
            if pub_date:
                try:
                    # Check if the format is YYYY-MM-DD
                    if '-' in pub_date:
                        year = int(pub_date.split('-')[0])
                    else:
                        # Try parsing as a numeric year
                        year = int(pub_date)
                except (ValueError, IndexError):
                    year = None
            
            # If year extraction failed, use a default value
            if year is None:
                year = 2020  # Default year
                
            self.paper_years.append(year)
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[int]]:
        """
        Load and preprocess the citation network data.
        
        Returns:
            Tuple containing:
            - Node features: Tensor of shape [num_nodes, feature_dim]
            - Edge indices: Tensor of shape [2, num_edges]
            - Edge timestamps: Tensor of shape [num_edges]
            - Paper IDs: List of paper IDs
            - Paper years: List of publication years
        """
        # Create node features
        logger.info("Creating node features using one-hot encoding of topics and keywords")
        self.node_features = self.create_node_features()
        
        # Create edges
        logger.info("Creating edge indices and timestamps")
        self.edge_indices, self.edge_timestamps = self.create_edge_index_and_time()
        
        return self.node_features, self.edge_indices, self.edge_timestamps, self.paper_ids, self.paper_years
    
    def create_node_features(self) -> torch.Tensor:
        """
        Create node feature tensors using one-hot encoding of topics and keywords.
        
        Returns:
            Tensor of node features with shape [num_nodes, feature_dim]
            where feature_dim = topic_count + keyword_count
        """
        # Get dimensions from embedding data
        topic_count = self.embedding_data['topic_count']
        keyword_count = self.embedding_data['keyword_count']
        feature_dim = topic_count + keyword_count
        
        logger.info(f"Creating feature vectors with dimension {feature_dim} "
                   f"({topic_count} topics + {keyword_count} keywords)")
        
        # Create mapping dictionaries
        topic_id_to_index = self.embedding_data['topic_id_to_index']
        keyword_id_to_index = self.embedding_data['keyword_id_to_index']
        
        # Initialize features with zeros
        features = np.zeros((self.num_nodes, feature_dim), dtype=np.float32)
        
        # Fill in features for each paper
        for i, paper_id in enumerate(self.paper_ids):
            paper_data = self.papers[paper_id]
            
            # Set topic features
            if 'topics' in paper_data:
                for topic in paper_data['topics']:
                    if 'id' in topic and topic['id'] in topic_id_to_index:
                        idx = topic_id_to_index[topic['id']]
                        score = topic.get('score', 1.0)  # Use score if available, else 1.0
                        features[i, idx] = score
                    elif 'id' in topic:  # Topic exists but not in our dictionary
                        logger.debug(f"Topic ID {topic['id']} not found in embedding dictionary")
            
            # Set keyword features
            if 'keywords' in paper_data:
                for keyword in paper_data['keywords']:
                    if 'id' in keyword and keyword['id'] in keyword_id_to_index:
                        idx = topic_count + keyword_id_to_index[keyword['id']]  # Offset by topic_count
                        score = keyword.get('score', 1.0)  # Use score if available, else 1.0
                        features[i, idx] = score
                    elif 'id' in keyword:  # Keyword exists but not in our dictionary
                        logger.debug(f"Keyword ID {keyword['id']} not found in embedding dictionary")
        
        # For papers with no topics or keywords, use average of citation neighborhood
        empty_papers = np.where(np.sum(features, axis=1) == 0)[0]
        if len(empty_papers) > 0:
            logger.warning(f"Found {len(empty_papers)} papers with no matching topics or keywords")
            
            # Build citation network for finding neighborhoods
            citations_dict = {}  # Paper index -> list of cited paper indices
            cited_by_dict = {}   # Paper index -> list of papers citing it
            
            # Build the citation dictionary
            for i, paper_id in enumerate(self.paper_ids):
                paper_data = self.papers[paper_id]
                citations_dict[i] = []
                cited_by_dict[i] = []
                
                # Get outgoing citations
                if 'outgoing_citations' in paper_data and paper_data['outgoing_citations']:
                    for cited_id in paper_data['outgoing_citations']:
                        if cited_id in self.paper_id_to_idx:
                            cited_idx = self.paper_id_to_idx[cited_id]
                            citations_dict[i].append(cited_idx)
            
            # Build the cited-by dictionary (reverse citations)
            for i, outgoing_list in citations_dict.items():
                for cited_idx in outgoing_list:
                    if i not in cited_by_dict[cited_idx]:
                        cited_by_dict[cited_idx].append(i)
            
            # For each empty paper, use the average of its neighborhood
            for empty_idx in empty_papers:
                # Get indices of papers it cites and papers citing it
                neighborhood = set(citations_dict[empty_idx] + cited_by_dict[empty_idx])
                
                # Remove self if present
                if empty_idx in neighborhood:
                    neighborhood.remove(empty_idx)
                
                if neighborhood:
                    # Calculate average of neighborhood features
                    neighborhood_features = features[list(neighborhood)]
                    avg_features = np.mean(neighborhood_features, axis=0)
                    features[empty_idx] = avg_features
                else:
                    # If no neighborhood, use small random noise as fallback
                    logger.warning(f"Paper at index {empty_idx} has no citation neighborhood, using small random noise")
                    features[empty_idx] = np.random.randn(feature_dim).astype(np.float32) * 0.01

        # Convert to tensor
        return torch.FloatTensor(features)
    
    def create_edge_index_and_time(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create edge index and timestamp tensors from the citation network.
        
        Returns:
            Tuple containing:
            - Edge indices: Tensor of shape [2, num_edges]
            - Edge timestamps: Tensor of shape [num_edges]
        """
        # Lists to store edge information
        edges = []
        timestamps = []
        
        # Process each paper and its references
        for paper_id, paper_data in self.papers.items():
            # Get the source node index
            src_idx = self.paper_id_to_idx[paper_id]
            
            # Process references (citations)
            refs = paper_data.get('referenced_works', [])
            
            for ref in refs:
                # Check if the referenced paper is in our dataset
                ref_id = ref.get('paper_id')
                if ref_id in self.paper_id_to_idx:
                    # Get the target node index
                    tgt_idx = self.paper_id_to_idx[ref_id]
                    
                    # Add edge (src_idx cites tgt_idx)
                    edges.append([src_idx, tgt_idx])
                    
                    # Add timestamp (publication year of source paper)
                    timestamps.append(self.paper_years[src_idx])
        
        # Convert to tensors
        if edges:
            edge_indices = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_timestamps = torch.tensor(timestamps, dtype=torch.long)
        else:
            # Create empty tensors if no edges were found
            edge_indices = torch.zeros((2, 0), dtype=torch.long)
            edge_timestamps = torch.zeros((0,), dtype=torch.long)
        
        logger.info(f"Created {edge_indices.size(1)} edges with timestamps")
        
        return edge_indices, edge_timestamps
    
    def get_feature_dimension(self) -> int:
        """
        Get the dimension of node features.
        
        Returns:
            int: Feature dimension (topic_count + keyword_count)
        """
        return self.embedding_data['topic_count'] + self.embedding_data['keyword_count']
    
    def get_feature_info(self) -> Dict:
        """
        Get information about the feature structure.
        
        Returns:
            Dict containing:
            - topic_count: Number of topics
            - keyword_count: Number of keywords
            - topic_index_to_name: Mapping from topic indices to display names
            - keyword_index_to_name: Mapping from keyword indices to display names
        """
        return {
            'topic_count': self.embedding_data['topic_count'],
            'keyword_count': self.embedding_data['keyword_count'],
            'topic_index_to_name': self.embedding_data['index_to_topic_display'],
            'keyword_index_to_name': self.embedding_data['index_to_keyword_display']
        }
    
    def feature_vector_to_topics_keywords(self, feature_vector: np.ndarray) -> Dict:
        """
        Convert a feature vector back to human-readable topics and keywords.
        
        Args:
            feature_vector: Feature vector of shape [feature_dim]
            
        Returns:
            Dict containing:
            - topics: List of (topic_name, score) tuples
            - keywords: List of (keyword_name, score) tuples
        """
        topic_count = self.embedding_data['topic_count']
        
        # Extract topic and keyword vectors
        topic_vector = feature_vector[:topic_count]
        keyword_vector = feature_vector[topic_count:]
        
        # Get non-zero topics
        topics = []
        for idx, score in enumerate(topic_vector):
            if score > 0:
                topic_id = self.embedding_data['index_to_topic_id'][idx]
                topic_name = self.embedding_data['index_to_topic_display'][idx]
                topics.append((topic_name, float(score)))
        
        # Get non-zero keywords
        keywords = []
        for idx, score in enumerate(keyword_vector):
            if score > 0:
                keyword_id = self.embedding_data['index_to_keyword_id'][idx]
                keyword_name = self.embedding_data['index_to_keyword_display'][idx]
                keywords.append((keyword_name, float(score)))
        
        # Sort by score descending
        topics.sort(key=lambda x: x[1], reverse=True)
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'topics': topics,
            'keywords': keywords
        }
    
    def split_data(self, test_year: int, val_ratio: float = 0.2) -> Tuple[Dict, Dict, Dict]:
        """
        Split the data into training, validation, and test sets.
        
        Args:
            test_year: Papers published on or after this year are used for testing
            val_ratio: Ratio of training papers to use for validation
            
        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        # Ensure we have loaded data
        if not hasattr(self, 'node_features') or self.node_features is None:
            self.load_data()
        
        # Get indices of papers by year
        train_indices = []
        test_indices = []
        
        for i, year in enumerate(self.paper_years):
            if year < test_year:
                train_indices.append(i)
            else:
                test_indices.append(i)
        
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