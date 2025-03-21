import os
import json
import time
import torch
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from src.data.dataset import GraphData

logger = logging.getLogger(__name__)

class CitationDataLoader:
    """
    Comprehensive data loader for citation network data.
    
    This loader handles:
    1. Loading data from JSON files
    2. Processing the data with one-hot encoding for topics/keywords
    3. Creating proper train/validation/test splits
    4. Supporting batch processing
    5. Being usable for both testing and actual model training
    """
    
    def __init__(
        self, 
        data_path: str = None,
        dataset_path: str = None,  # Alternative parameter name for backward compatibility
        embedding_dict_path: str = None,
        temporal_split: bool = True,
        batch_size: int = 1024,
        random_seed: int = 42
    ):
        """
        Initialize the citation data loader.
        
        Args:
            data_path: Path to the citation network data JSON file (deprecated, use dataset_path)
            dataset_path: Path to the citation network data JSON file
            embedding_dict_path: Path to the embedding dictionaries pickle file
                                 (if None, will look in the same directory as data_path)
            temporal_split: Whether to split data temporally (by year) or randomly
            batch_size: Size of batches for training
            random_seed: Random seed for reproducibility
        """
        # Handle parameter name change for backward compatibility
        if dataset_path is not None:
            self.data_path = dataset_path
        else:
            self.data_path = data_path
            
        self.temporal_split = temporal_split
        self.batch_size = batch_size
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        if self.data_path is not None:
            # Validate data path
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Set default embedding_dict_path if not provided
            if embedding_dict_path is None:
                embedding_dict_path = os.path.join(os.path.dirname(self.data_path), "embedding_dictionaries.pkl")
            
            # Validate embedding dictionary path
            if not os.path.exists(embedding_dict_path):
                raise FileNotFoundError(f"Embedding dictionaries not found: {embedding_dict_path}")
                
            self.embedding_dict_path = embedding_dict_path
            
            # Load embedding dictionaries and data
            self._load_data()
        else:
            # Initialize as empty for later loading
            self.embedding_dict_path = embedding_dict_path
            self.papers = {}
            self.num_nodes = 0
            self.paper_id_to_idx = {}
            self.paper_ids = []
        
        # Initialize placeholders for processed data
        self.node_features = None
        self.edge_indices = None
        self.edge_timestamps = None
        self.graph_data = None
        
        # Prepare train/val/test indices (will be populated later)
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        # Cached subgraphs
        self.train_graph = None
        self.val_graph = None
        self.test_graph = None
    
    def _load_data(self):
        """Load embedding dictionaries and citation network data."""
        # Load embedding dictionaries
        with open(self.embedding_dict_path, 'rb') as f:
            self.embedding_data = pickle.load(f)
            
        logger.info(f"Loaded embedding dictionaries with {self.embedding_data['topic_count']} topics "
                   f"and {self.embedding_data['keyword_count']} keywords")
        
        # Load citation network data
        start_time = time.time()
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Extract paper data
        self.papers = self.data.get('papers', {})
        self.num_nodes = len(self.papers)
        
        # Map paper IDs to indices
        self.paper_id_to_idx = {}
        self.paper_ids = []
        for idx, paper_id in enumerate(self.papers.keys()):
            self.paper_id_to_idx[paper_id] = idx
            self.paper_ids.append(paper_id)
        
        # Extract metadata
        self.extract_metadata()
        
        logger.info(f"Loaded {self.num_nodes} papers from {self.data_path} in {time.time() - start_time:.2f} seconds")
    
    def extract_metadata(self):
        """Extract metadata from papers, including publication years."""
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
            
            # If year extraction failed, use a default value (most recent year in the dataset)
            if year is None:
                year = 2022  # Default year
                
            self.paper_years.append(year)
        
        # Convert to tensor for easy manipulation
        self.paper_years_tensor = torch.tensor(self.paper_years, dtype=torch.float)
        
        # Find min and max years
        self.min_year = min(year for year in self.paper_years if year > 0)
        self.max_year = max(self.paper_years)
        
        logger.info(f"Publication years range from {self.min_year} to {self.max_year}")
    
    def process_data(self) -> GraphData:
        """
        Process the loaded data into a GraphData object.
        
        Returns:
            GraphData: A GraphData object containing the citation network
        """
        # Create node features
        if self.node_features is None:
            logger.info("Creating node features using one-hot encoding of topics and keywords")
            self.node_features = self.create_node_features()
        
        # Create edges
        if self.edge_indices is None or self.edge_timestamps is None:
            logger.info("Creating edge indices and timestamps")
            self.edge_indices, self.edge_timestamps = self.create_edge_index_and_time()
        
        # Build GraphData object
        if self.graph_data is None:
            # Convert paper_years to tensor
            node_timestamps = torch.tensor(self.paper_years, dtype=torch.float)
            
            # Create GraphData object
            self.graph_data = GraphData(
                x=self.node_features,
                edge_index=self.edge_indices,
                edge_timestamps=self.edge_timestamps,
                node_timestamps=node_timestamps
            )
        
        return self.graph_data
    
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
                    elif 'id' in topic:
                        logger.debug(f"Topic ID {topic['id']} not found in embedding dictionary")
            
            # Set keyword features
            if 'keywords' in paper_data:
                for keyword in paper_data['keywords']:
                    if 'id' in keyword and keyword['id'] in keyword_id_to_index:
                        idx = topic_count + keyword_id_to_index[keyword['id']]
                        score = keyword.get('score', 1.0)  # Use score if available, else 1.0
                        features[i, idx] = score
                    elif 'id' in keyword:
                        logger.debug(f"Keyword ID {keyword['id']} not found in embedding dictionary")
        
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
            # First check referenced_works field (OpenAlex format)
            if 'referenced_works' in paper_data:
                refs = paper_data.get('referenced_works', [])
                for ref_id in refs:
                    if ref_id in self.paper_id_to_idx:
                        # Get the target node index
                        tgt_idx = self.paper_id_to_idx[ref_id]
                        
                        # Add edge (src_idx cites tgt_idx)
                        edges.append([src_idx, tgt_idx])
                        
                        # Add timestamp (publication year of source paper)
                        timestamps.append(self.paper_years[src_idx])
            # Then check references field (legacy format)
            elif 'referenced_works' in paper_data:
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
            edge_timestamps = torch.tensor(timestamps, dtype=torch.float)
        else:
            # Create empty tensors if no edges were found
            edge_indices = torch.zeros((2, 0), dtype=torch.long)
            edge_timestamps = torch.zeros((0,), dtype=torch.float)
        
        logger.info(f"Created {edge_indices.size(1)} edges with timestamps")
        
        return edge_indices, edge_timestamps
    
    def split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[GraphData, GraphData, GraphData]:
        """
        Split the data into train, validation, and test sets.
        
        Args:
            train_ratio: Ratio of training data (if temporal_split=False)
            val_ratio: Ratio of validation data (if temporal_split=False)
            test_ratio: Ratio of test data (if temporal_split=False)
            
        Returns:
            Tuple containing train, validation, and test GraphData objects
        """
        # Make sure data is processed
        if self.graph_data is None:
            self.process_data()
            
        # Return cached subgraphs if already split
        if self.train_graph is not None and self.val_graph is not None and self.test_graph is not None:
            return self.train_graph, self.val_graph, self.test_graph
        
        # Get the number of edges
        num_edges = self.edge_indices.shape[1]
        
        # Handle the case of empty edge sets
        if num_edges == 0:
            logger.warning("No edges found in the dataset. Creating empty splits.")
            self.train_indices = torch.tensor([], dtype=torch.long)
            self.val_indices = torch.tensor([], dtype=torch.long)
            self.test_indices = torch.tensor([], dtype=torch.long)
        else:
            if self.temporal_split:
                # For temporal split, we need to sort edges by timestamp
                sorted_indices = torch.argsort(self.edge_timestamps)
                
                # Determine cut points based on timestamps
                year_range = self.max_year - self.min_year
                train_year = self.min_year + int(year_range * train_ratio)
                val_year = train_year + int(year_range * val_ratio)
                
                # Split indices by year
                self.train_indices = torch.tensor([i for i, idx in enumerate(sorted_indices) 
                                                   if self.edge_timestamps[idx] <= train_year], dtype=torch.long)
                self.val_indices = torch.tensor([i for i, idx in enumerate(sorted_indices) 
                                                if train_year < self.edge_timestamps[idx] <= val_year], dtype=torch.long)
                self.test_indices = torch.tensor([i for i, idx in enumerate(sorted_indices) 
                                                 if self.edge_timestamps[idx] > val_year], dtype=torch.long)
            else:
                # For random split, shuffle indices
                indices = torch.randperm(num_edges)
                train_size = int(num_edges * train_ratio)
                val_size = int(num_edges * val_ratio)
                
                # Split indices
                self.train_indices = indices[:train_size]
                self.val_indices = indices[train_size:train_size+val_size]
                self.test_indices = indices[train_size+val_size:]
        
        logger.info(f"Split data into {len(self.train_indices)} training, {len(self.val_indices)} validation, and {len(self.test_indices)} test edges")
        
        # Create subgraphs
        self.train_graph = self._create_subgraph(self.train_indices)
        self.val_graph = self._create_subgraph(self.val_indices)
        self.test_graph = self._create_subgraph(self.test_indices)
        
        return self.train_graph, self.val_graph, self.test_graph
    
    def _create_subgraph(self, indices: torch.Tensor) -> GraphData:
        """
        Create a subgraph from the given edge indices.
        
        Args:
            indices: Edge indices to include in the subgraph
            
        Returns:
            GraphData: A GraphData object for the subgraph
        """
        # Handle empty indices
        if len(indices) == 0:
            # Create a graph with the same nodes but no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_timestamps = torch.zeros((0,), dtype=torch.float)
            
            subgraph = GraphData(
                x=self.node_features,
                edge_index=edge_index,
                edge_timestamps=edge_timestamps,
                node_timestamps=self.graph_data.node_timestamps
            )
            return subgraph
        
        # Create a new GraphData instance with the selected edges
        subgraph = GraphData(
            x=self.node_features,
            edge_index=self.edge_indices[:, indices],
            edge_timestamps=self.edge_timestamps[indices],
            node_timestamps=self.graph_data.node_timestamps
        )
        
        # Handle edge_attr separately
        if hasattr(self.graph_data, 'edge_attr') and self.graph_data.edge_attr is not None:
            if isinstance(self.graph_data.edge_attr, dict):
                # If edge_attr is a dictionary, create a new dictionary with indexed values
                edge_attr = {}
                for key, value in self.graph_data.edge_attr.items():
                    edge_attr[key] = value[indices]
                subgraph.edge_attr = edge_attr
            else:
                # If edge_attr is a tensor, index it directly
                subgraph.edge_attr = self.graph_data.edge_attr[indices]
        
        return subgraph
    
    def get_batches(self, split: str = 'train', batch_size: Optional[int] = None) -> List[GraphData]:
        """
        Create batches of data for training.
        
        Args:
            split: Which data split to use ('train', 'val', or 'test')
            batch_size: Size of batches (if None, use self.batch_size)
            
        Returns:
            List of GraphData objects, each representing a batch
        """
        # Make sure data is split
        if self.train_graph is None:
            self.split_data()
        
        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.batch_size
        
        # Get the appropriate graph based on split
        if split == 'train':
            graph = self.train_graph
            indices = self.train_indices
        elif split == 'val':
            graph = self.val_graph
            indices = self.val_indices
        elif split == 'test':
            graph = self.test_graph
            indices = self.test_indices
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
        
        # Calculate number of batches
        num_edges = len(indices)
        num_batches = (num_edges + batch_size - 1) // batch_size  # Ceiling division
        
        # Create batches
        batches = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_edges)
            
            # Get indices for this batch
            batch_indices = indices[start_idx:end_idx]
            
            # Create a batch graph
            batch_graph = self._create_subgraph(batch_indices)
            batches.append(batch_graph)
        
        logger.info(f"Created {len(batches)} batches for {split} data with batch size {batch_size}")
        return batches
    
    def get_feature_info(self) -> Dict:
        """
        Get information about the feature structure.
        
        Returns:
            Dictionary with feature information including topic and keyword counts
        """
        return {
            'topic_count': self.embedding_data['topic_count'],
            'keyword_count': self.embedding_data['keyword_count'],
            'feature_dim': self.embedding_data['topic_count'] + self.embedding_data['keyword_count']
        }
    
    def feature_vector_to_topics_keywords(self, feature_vector: np.ndarray) -> Dict:
        """
        Convert a feature vector back to topics and keywords.
        
        Args:
            feature_vector: Feature vector to convert
            
        Returns:
            Dictionary with topics and keywords
        """
        topic_count = self.embedding_data['topic_count']
        
        # Extract topics and keywords from feature vector
        topic_vector = feature_vector[:topic_count]
        keyword_vector = feature_vector[topic_count:]
        
        # Convert indices back to topic/keyword IDs
        index_to_topic_id = self.embedding_data['index_to_topic_id']
        index_to_keyword_id = self.embedding_data['index_to_keyword_id']
        
        # Build topics list
        topics = []
        for idx in np.where(topic_vector > 0)[0]:
            if idx < len(index_to_topic_id):
                topic_id = index_to_topic_id[int(idx)]
                topics.append({
                    'id': topic_id,
                    'score': float(topic_vector[idx])
                })
        
        # Build keywords list
        keywords = []
        for idx in np.where(keyword_vector > 0)[0]:
            if idx < len(index_to_keyword_id):
                keyword_id = index_to_keyword_id[int(idx)]
                keywords.append({
                    'id': keyword_id,
                    'score': float(keyword_vector[idx])
                })
                
        return {
            'topics': topics,
            'keywords': keywords
        }
    
    def load(self) -> GraphData:
        """
        Load and process data into a GraphData object. 
        This is a simplified method that combines loading and processing.
        
        Returns:
            GraphData: A GraphData object containing the citation network
        """
        # If data_path wasn't provided during init, ensure they're provided as parameters
        if not hasattr(self, 'data') or self.data is None:
            if not hasattr(self, 'data_path') or self.data_path is None:
                raise ValueError("Data path must be provided either during initialization or as a parameter")
            self._load_data()
            
        # Process the data
        return self.process_data()


class CitationBatchDataset(Dataset):
    """
    Dataset for batched citation network data.
    
    This dataset is designed to work with PyTorch DataLoader for efficient batch processing.
    """
    
    def __init__(self, 
                 graph_data: GraphData,
                 edge_indices: torch.Tensor,
                 batch_size: int = 1024,
                 shuffle: bool = True,
                 negative_samples: int = 1):
        """
        Initialize the dataset.
        
        Args:
            graph_data: The full GraphData object
            edge_indices: Indices of edges to include in this dataset
            batch_size: Batch size
            shuffle: Whether to shuffle indices
            negative_samples: Number of negative samples per positive edge
        """
        self.graph_data = graph_data
        self.edge_indices = edge_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.negative_samples = negative_samples
        
        # Calculate number of batches
        self.num_edges = len(edge_indices)
        self.num_batches = (self.num_edges + batch_size - 1) // batch_size
        
        # Shuffle indices if requested
        if shuffle:
            self.batch_order = torch.randperm(self.num_batches)
        else:
            self.batch_order = torch.arange(self.num_batches)
    
    def __len__(self) -> int:
        """Get the number of batches."""
        return self.num_batches
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a batch by index.
        
        Args:
            idx: Batch index
            
        Returns:
            Dictionary with batch data
        """
        # Get the actual batch index from the shuffled order
        batch_idx = self.batch_order[idx]
        
        # Calculate start and end indices for this batch
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, self.num_edges)
        
        # Get indices for this batch
        batch_indices = self.edge_indices[start_idx:end_idx]
        
        # Create a batch graph
        batch_graph = GraphData(
            x=self.graph_data.x,
            edge_index=self.graph_data.edge_index[:, batch_indices],
            edge_timestamps=self.graph_data.edge_timestamps[batch_indices],
            node_timestamps=self.graph_data.node_timestamps
        )
        
        # Handle edge_attr if it exists
        if hasattr(self.graph_data, 'edge_attr') and self.graph_data.edge_attr is not None:
            if isinstance(self.graph_data.edge_attr, dict):
                # If edge_attr is a dictionary, create a new dictionary with indexed values
                edge_attr = {}
                for key, value in self.graph_data.edge_attr.items():
                    edge_attr[key] = value[batch_indices]
                batch_graph.edge_attr = edge_attr
            else:
                # If edge_attr is a tensor, index it directly
                batch_graph.edge_attr = self.graph_data.edge_attr[batch_indices]
        
        return {
            'graph': batch_graph,
            'batch_indices': batch_indices
        }
    
    def sample_negative_edges(self, num_samples: int, exclude_edges: torch.Tensor) -> torch.Tensor:
        """
        Sample negative (non-existent) edges.
        
        Args:
            num_samples: Number of negative edges to sample
            exclude_edges: Existing edges to exclude
            
        Returns:
            Tensor of sampled negative edges with shape [2, num_samples]
        """
        num_nodes = self.graph_data.x.shape[0]
        
        # Create a set of existing edges for fast lookup
        existing_edges = set()
        for i in range(exclude_edges.shape[1]):
            src, dst = exclude_edges[0, i].item(), exclude_edges[1, i].item()
            existing_edges.add((src, dst))
        
        # Sample negative edges
        negative_edges = []
        while len(negative_edges) < num_samples:
            # Sample random source and target nodes
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            
            # Check if this is a valid negative edge (not in existing edges)
            if src != dst and (src, dst) not in existing_edges:
                negative_edges.append([src, dst])
                # Add to existing edges to avoid duplicates
                existing_edges.add((src, dst))
        
        return torch.tensor(negative_edges, dtype=torch.long).t() 