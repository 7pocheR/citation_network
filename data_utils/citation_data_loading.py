"""
Citation Data Loading Utilities

This module provides functions for loading citation network data from different formats,
including standard JSON and OpenAlex format, and converting them to GraphData objects.
It properly handles embedding dictionaries for consistent topic and keyword encoding.
"""

import os
import json
import logging
import torch
import numpy as np
import pickle
from typing import Dict, Optional, List, Union, Tuple, Any

# Add src directory to path if needed
import sys
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from src.data.dataset import GraphData

logger = logging.getLogger(__name__)

def load_embedding_dictionaries(embedding_dict_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load embedding dictionaries for topics and keywords.
    
    Args:
        embedding_dict_path: Path to the embedding dictionaries file
            If None, looks in standard locations
            
    Returns:
        Dictionary containing the embedding dictionaries
    """
    # Try a few standard locations if not specified
    if embedding_dict_path is None:
        potential_paths = [
            "data/embedding_dictionaries.pkl",
            "../data/embedding_dictionaries.pkl",
            "embedding_dictionaries.pkl"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                embedding_dict_path = path
                break
        
        if embedding_dict_path is None:
            raise FileNotFoundError(
                "Embedding dictionaries file not found in standard locations. "
                "Please specify the path explicitly."
            )
    
    # Load the dictionaries
    logger.info(f"Loading embedding dictionaries from {embedding_dict_path}")
    with open(embedding_dict_path, 'rb') as f:
        embedding_data = pickle.load(f)
    
    topic_count = embedding_data.get('topic_count', 0)
    keyword_count = embedding_data.get('keyword_count', 0)
    logger.info(f"Loaded dictionaries with {topic_count} topics and {keyword_count} keywords")
    
    return embedding_data

def create_features_from_paper_data(
    paper_data: Dict[str, Any],
    embedding_data: Dict[str, Any],
    normalize_scores: bool = True
) -> torch.Tensor:
    """
    Create feature tensor from paper data using embedding dictionaries.
    
    Args:
        paper_data: Dictionary containing paper data with topics and keywords
        embedding_data: Dictionary containing embedding dictionaries
        normalize_scores: Whether to normalize topic and keyword scores within each paper
        
    Returns:
        Feature tensor for the paper
    """
    # Get dimensions and mapping dictionaries
    topic_count = embedding_data['topic_count']
    keyword_count = embedding_data['keyword_count']
    feature_dim = topic_count + keyword_count
    
    topic_id_to_index = embedding_data['topic_id_to_index']
    keyword_id_to_index = embedding_data['keyword_id_to_index']
    
    # Initialize feature vector
    features = np.zeros(feature_dim, dtype=np.float32)
    
    # Fill in topic features
    if 'topics' in paper_data:
        for topic in paper_data['topics']:
            if 'id' in topic and topic['id'] in topic_id_to_index:
                idx = topic_id_to_index[topic['id']]
                score = topic.get('score', 1.0)  # Use score if available, else 1.0
                features[idx] = score
    
    # Fill in keyword features
    if 'keywords' in paper_data:
        for keyword in paper_data['keywords']:
            if 'id' in keyword and keyword['id'] in keyword_id_to_index:
                idx = topic_count + keyword_id_to_index[keyword['id']]
                score = keyword.get('score', 1.0)  # Use score if available, else 1.0
                features[idx] = score
    
    # Normalize scores if requested
    if normalize_scores and features.sum() > 1e-6:
        features /= features.sum()
    
    return torch.FloatTensor(features)

def load_openalex_format(
    file_path: str, 
    embedding_dict_path: Optional[str] = None,
    fallback_feature_dim: int = 16,
    normalize_scores: bool = True
) -> GraphData:
    """
    Load citation network data from a JSON file in the OpenAlex format.
    
    Args:
        file_path: Path to the OpenAlex format JSON file
        embedding_dict_path: Path to the embedding dictionaries file
        fallback_feature_dim: Dimension of feature vectors if embedding dictionaries not available
        normalize_scores: Whether to normalize topic and keyword scores within each paper
        
    Returns:
        GraphData object with the loaded data
    """
    
    logger.info(f"Loading OpenAlex format data from {file_path}")
    
    # Load JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract papers data
    papers = data.get('papers', {})
    if not papers:
        raise ValueError("No papers found in the data")
    
    # Get list of paper IDs
    paper_ids = list(papers.keys())
    logger.info(f"Found {len(paper_ids)} papers")
    
    # Try to load embedding dictionaries
    use_embedding_dict = False
    embedding_data = None
    embedding_data = load_embedding_dictionaries(embedding_dict_path)
    
    if embedding_data is not None:
        # Calculate feature_dim from topic_count and keyword_count
        feature_dim = embedding_data['topic_count'] + embedding_data['keyword_count']
        # Store it for future use
        embedding_data['feature_dim'] = feature_dim
        use_embedding_dict = True
        logger.info(f"Using embedding dictionaries with dimension {feature_dim}")
    else:
        logger.warning("Failed to load embedding dictionaries, using fallback")
    
    if not use_embedding_dict:
        logger.info(f"Using fallback feature dimension {fallback_feature_dim}")
        feature_dim = fallback_feature_dim
    
    # Create node features tensor
    num_papers = len(paper_ids)
    x = torch.zeros((num_papers, feature_dim), dtype=torch.float32)
    
    # Process papers
    for paper_idx, paper_id in enumerate(paper_ids):
        paper = papers[paper_id]
        
        # Set features based on embedding dictionaries or fallback
        if use_embedding_dict:
            x[paper_idx] = create_features_from_paper_data(
                paper, 
                embedding_data, 
                normalize_scores=normalize_scores
            )
        else:
            # Fallback to simple feature creation based on topics
            if 'topics' in paper and paper['topics']:
                # Create a simple embedding from the topic scores
                topic_scores = []
                for topic in paper['topics']:
                    score = topic.get('score', 0.5)
                    topic_scores.append(score)
                
                # Normalize scores if requested
                if normalize_scores and topic_scores:
                    max_score = max(topic_scores)
                    if max_score > 1e-6:
                        topic_scores = [score / max_score for score in topic_scores]
                
                # Create a simple feature vector using topic scores
                simple_features = torch.zeros(fallback_feature_dim, dtype=torch.float32)
                for i, score in enumerate(topic_scores[:fallback_feature_dim]):
                    simple_features[i] = score
                
                # Assign to node features
                x[paper_idx] = simple_features
    
    # Create edge index
    edges = []
    for src_idx, paper_id in enumerate(paper_ids):
        paper = papers[paper_id]
        
        # Add citation edges
        if 'referenced_works' in paper:
            for ref_id in paper['referenced_works']:
                if ref_id in papers:  # Only include references to papers in the dataset
                    try:
                        dst_idx = paper_ids.index(ref_id)
                        edges.append((src_idx, dst_idx))
                    except ValueError:
                        # Reference paper not in the list of paper IDs
                        continue
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    logger.info(f"Created edge index with {len(edges)} edges")
    
    # Create metadata
    metadata = {
        'paper_ids': paper_ids,
        'papers': papers
    }
    
    # Create graph data
    graph_data = GraphData(
        x=x,
        edge_index=edge_index,
        metadata=metadata
    )
    
    # Add timestamp data if available
    # Check both timestamps and publication_date fields
    timestamps = []
    has_timestamps = False
    
    # First check if we have direct timestamps field
    if all('timestamp' in papers[paper_id] for paper_id in paper_ids):
        for paper_id in paper_ids:
            paper = papers[paper_id]
            timestamps.append(float(paper['timestamp']))
        has_timestamps = True
        logger.info("Using timestamp field for paper dates")
    
    # Then check if we have publication_date field
    elif all('publication_date' in papers[paper_id] for paper_id in paper_ids):
        logger.info("Found publication_date field in all papers")
        
        # Log a few examples to debug
        sample_paper_ids = paper_ids[:3]  # First 3 papers
        logger.debug("Sample publication_date examples:")
        for paper_id in sample_paper_ids:
            logger.debug(f"  Paper {paper_id}: publication_date = {papers[paper_id].get('publication_date', 'None')}")
        
        for paper_id in paper_ids:
            paper = papers[paper_id]
            date_str = paper['publication_date']
            try:
                # Try to parse the date string (format: YYYY-MM-DD)
                year = int(date_str[:4])  # Extract year
                # If we have month info, add fractional component
                if len(date_str) >= 7:
                    month = int(date_str[5:7])
                    year_fraction = (month - 1) / 12.0
                    timestamps.append(float(year) + year_fraction)
                else:
                    timestamps.append(float(year))
            except (ValueError, TypeError, IndexError):
                logger.warning(f"Could not parse publication_date: {date_str}")
                timestamps.append(0.0)  # Default timestamp if parsing fails
        has_timestamps = True
        logger.info("Using publication_date field for paper dates")
    
    # If we have timestamps, add them to the graph data
    if has_timestamps:
        graph_data.node_timestamps = torch.tensor(timestamps, dtype=torch.float)
        # Also set the timestamps attribute for backward compatibility
        graph_data.timestamps = graph_data.node_timestamps 
        logger.info(f"Added timestamp data with range: {min(timestamps):.2f} to {max(timestamps):.2f}")
    else:
        logger.warning("No timestamp information found")
    
    logger.info(f"Created graph data: {graph_data}")
    return graph_data

def load_standard_format(
    file_path: str, 
    embedding_dict_path: Optional[str] = None,
    fallback_feature_dim: int = 16,
    normalize_scores: bool = True
) -> GraphData:
    """
    Load citation network data from a JSON file in standard format.
    
    Args:
        file_path: Path to the standard format JSON file
        embedding_dict_path: Path to the embedding dictionaries file
        fallback_feature_dim: Dimension of feature vectors if embedding dictionaries not available
        normalize_scores: Whether to normalize topic and keyword scores within each paper
        
    Returns:
        GraphData object with the loaded data
    """
    logger.info(f"Loading standard format data from {file_path}")
    
    # Load JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract nodes and edges
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])
    
    if not nodes:
        raise ValueError("No nodes found in the data")
    
    logger.info(f"Found {len(nodes)} nodes and {len(edges)} edges")
    
    # Try to load embedding dictionaries
    use_embedding_dict = False
    embedding_data = None
    if embedding_dict_path is not None:
        embedding_data = load_embedding_dictionaries(embedding_dict_path)
        
        if embedding_data is not None:
            # Calculate feature_dim from topic_count and keyword_count
            feature_dim = embedding_data['topic_count'] + embedding_data['keyword_count']
            # Store it for future use
            embedding_data['feature_dim'] = feature_dim
            use_embedding_dict = True
            logger.info(f"Using embedding dictionaries with dimension {feature_dim}")
    
    if not use_embedding_dict:
        logger.info(f"Using fallback feature dimension {fallback_feature_dim}")
        feature_dim = fallback_feature_dim
    
    # Create node features tensor
    num_nodes = len(nodes)
    x = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)
    
    # Process nodes
    for i, node in enumerate(nodes):
        # Set features based on embedding dictionaries or fallback
        if use_embedding_dict:
            x[i] = create_features_from_paper_data(
                node, 
                embedding_data, 
                normalize_scores=normalize_scores
            )
        else:
            # Fallback to simple feature creation based on topics
            if 'topics' in node and node['topics']:
                # Create a simple embedding from the topic scores
                topic_scores = []
                for topic in node['topics']:
                    score = topic.get('score', 0.5)
                    topic_scores.append(score)
                
                # Normalize scores if requested
                if normalize_scores and topic_scores:
                    max_score = max(topic_scores)
                    if max_score > 1e-6:
                        topic_scores = [score / max_score for score in topic_scores]
                
                # Create a simple feature vector using topic scores
                simple_features = torch.zeros(fallback_feature_dim, dtype=torch.float32)
                for j, score in enumerate(topic_scores[:fallback_feature_dim]):
                    simple_features[j] = score
                
                # Assign to node features
                x[i] = simple_features
    
    # Create edge index
    edge_index = torch.zeros((2, len(edges)), dtype=torch.long)
    for i, edge in enumerate(edges):
        edge_index[0, i] = edge['source']
        edge_index[1, i] = edge['target']
    
    # Create metadata
    metadata = {
        'nodes': nodes,
        'edges': edges
    }
    
    # Create graph data
    graph_data = GraphData(
        x=x,
        edge_index=edge_index,
        metadata=metadata
    )
    
    # Add timestamp data if available
    # Check for different timestamp fields in the standard format
    timestamps = []
    has_timestamps = False
    
    # Check if we have 'timestamp' field
    if all('timestamp' in node for node in nodes):
        for node in nodes:
            try:
                timestamps.append(float(node['timestamp']))
            except (ValueError, TypeError):
                timestamps.append(0.0)  # Default timestamp if parsing fails
        has_timestamps = True
        logger.info("Using timestamp field for node dates")
    
    # Check if we have 'time' field
    elif all('time' in node for node in nodes):
        for node in nodes:
            try:
                timestamps.append(float(node['time']))
            except (ValueError, TypeError):
                timestamps.append(0.0)  # Default timestamp if parsing fails
        has_timestamps = True
        logger.info("Using time field for node dates")
    
    # Check if we have 'publication_date' field (for compatibility with OpenAlex format)
    elif all('publication_date' in node for node in nodes):
        for node in nodes:
            date_str = node['publication_date']
            try:
                # Try to parse the date string (format: YYYY-MM-DD)
                year = int(date_str[:4])  # Extract year
                # If we have month info, add fractional component
                if len(date_str) >= 7:
                    month = int(date_str[5:7])
                    year_fraction = (month - 1) / 12.0
                    timestamps.append(float(year) + year_fraction)
                else:
                    timestamps.append(float(year))
            except (ValueError, TypeError, IndexError):
                logger.warning(f"Could not parse publication_date: {date_str}")
                timestamps.append(0.0)  # Default timestamp if parsing fails
        has_timestamps = True
        logger.info("Using publication_date field for node dates")
    
    # If we have timestamps, add them to the graph data
    if has_timestamps:
        graph_data.node_timestamps = torch.tensor(timestamps, dtype=torch.float)
        # Also set the timestamps attribute for backward compatibility
        graph_data.timestamps = graph_data.node_timestamps
        logger.info(f"Added timestamp data with range: {min(timestamps):.2f} to {max(timestamps):.2f}")
    else:
        logger.warning("No timestamp information found in standard format")
    
    logger.info(f"Created graph data: {graph_data}")
    return graph_data

def load_graph_data(
    file_path: str, 
    embedding_dict_path: Optional[str] = "data/embedding_dictionaries.pkl",
    fallback_feature_dim: int = 16,
    normalize_scores: bool = True
) -> GraphData:
    """
    Unified function to load graph data from different formats.
    
    This function automatically detects the format of the data and calls
    the appropriate loader. It also handles embedding dictionaries for
    feature creation.
    
    Args:
        file_path: Path to the data file
        embedding_dict_path: Path to the embedding dictionaries file
        fallback_feature_dim: Dimension of feature vectors if embedding dictionaries not available
        normalize_scores: Whether to normalize topic and keyword scores within each paper
        
    Returns:
        GraphData object with the loaded data
    """
    logger.info(f"Loading graph data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            # Try to parse the first few bytes to detect format
            data_start = f.read(1000)
            f.seek(0)  # Reset file pointer
            
            # Try to determine the format from the data structure
            if '"papers"' in data_start and '"metadata"' in data_start:
                logger.info("Detected OpenAlex format")
                return load_openalex_format(
                    file_path, 
                    embedding_dict_path, 
                    fallback_feature_dim,
                    normalize_scores
                )
            elif '"nodes"' in data_start and '"edges"' in data_start:
                logger.info("Detected standard format")
                return load_standard_format(
                    file_path, 
                    embedding_dict_path, 
                    fallback_feature_dim,
                    normalize_scores
                )
            else:
                logger.warning("Could not automatically detect format, using standard format")
                return load_standard_format(
                    file_path, 
                    embedding_dict_path, 
                    fallback_feature_dim,
                    normalize_scores
                )
                
        except json.JSONDecodeError:
            # If JSON parsing fails, try each format with proper error handling
            logger.warning("Error parsing JSON, trying each format explicitly")
            
            try:
                return load_openalex_format(
                    file_path, 
                    embedding_dict_path, 
                    fallback_feature_dim,
                    normalize_scores
                )
            except Exception as e:
                logger.warning(f"OpenAlex format failed: {str(e)}")
                
                try:
                    return load_standard_format(
                        file_path, 
                        embedding_dict_path, 
                        fallback_feature_dim,
                        normalize_scores
                    )
                except Exception as e:
                    logger.error(f"Standard format failed: {str(e)}")
                    raise ValueError(f"Could not load data from {file_path} in any format: {str(e)}")

# Add extra utility functions for data processing as needed
def create_train_val_test_split(
    graph_data: GraphData, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1, 
    temporal_split: bool = True
) -> Tuple[GraphData, GraphData, GraphData]:
    """
    Split a graph into train, validation, and test sets.
    
    Args:
        graph_data: Input graph data
        val_ratio: Ratio of edges for validation
        test_ratio: Ratio of edges for testing
        temporal_split: Whether to use temporal splitting
        
    Returns:
        Tuple of (train_graph, val_graph, test_graph)
    """
    # This is just a placeholder - actual implementation would depend on
    # how GraphData handles edge splitting
    logger.warning("create_train_val_test_split is not fully implemented yet")
    
    # Get the total number of nodes and device
    total_nodes = graph_data.x.shape[0]
    device = graph_data.x.device
    
    # Calculate split sizes
    test_size = int(total_nodes * test_ratio)
    val_size = int(total_nodes * val_ratio)
    train_size = total_nodes - test_size - val_size
    
    # Create initial ordered boolean masks (not shuffled yet)
    # Initialize all to False
    train_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    
    # Set the appropriate segments to True
    train_mask[:train_size] = True  # First 80% for training
    val_mask[train_size:train_size+val_size] = True  # Next 10% for validation
    test_mask[train_size+val_size:] = True  # Last 10% for testing
    
    # Create random permutation to shuffle the masks
    perm = torch.randperm(total_nodes)
    
    # Apply the permutation to shuffle the masks
    train_mask = train_mask[perm]
    val_mask = val_mask[perm]
    test_mask = test_mask[perm]
    
    # Assign masks to graph_data
    graph_data.train_index = train_mask
    graph_data.val_index = val_mask
    graph_data.test_index = test_mask
    
    return graph_data, graph_data, graph_data

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the functions
    test_file = "../data/test_dataset.json"
    if os.path.exists(test_file):
        # First try to find embedding dictionaries
        embedding_dict_path = None
        for potential_path in ["../data/embedding_dictionaries.pkl", "embedding_dictionaries.pkl"]:
            if os.path.exists(potential_path):
                embedding_dict_path = potential_path
                break
        
        graph_data = load_graph_data(test_file, embedding_dict_path=embedding_dict_path)
        print(f"Loaded graph with {graph_data.x.shape[0]} nodes and {graph_data.edge_index.shape[1]} edges")
        print(f"Feature dimension: {graph_data.x.shape[1]}")
    else:
        print(f"Test file {test_file} not found") 