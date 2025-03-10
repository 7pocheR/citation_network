import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import networkx as nx


class CitationPredictionMetrics:
    """
    Metrics for evaluating citation prediction performance.
    """
    
    @staticmethod
    def compute_link_prediction_metrics(predictions: torch.Tensor, 
                                       targets: torch.Tensor, 
                                       threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute metrics for link prediction.
        
        Args:
            predictions: Prediction scores/probabilities
            targets: Ground truth binary labels
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy for sklearn metrics
        if isinstance(predictions, torch.Tensor):
            # Handle dimension mismatch - reshape if needed
            if targets.dim() > predictions.dim():
                print(f"DEBUG METRICS - Reshaping targets from {targets.shape} to match predictions {predictions.shape}")
                if targets.dim() == 3 and predictions.dim() == 2:
                    # If targets has shape [batch, n, n] and predictions has shape [n, n]
                    # Reshape targets to [n, n]
                    targets = targets.squeeze(0)
            
            predictions_np = predictions.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
        else:
            predictions_np = predictions
            targets_np = targets
            
        # Convert scores to binary predictions
        binary_preds = (predictions_np > threshold).astype(np.float32)
        
        # Compute precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_np.flatten(), binary_preds.flatten(), average='binary', zero_division=0
        )
        
        # Compute AUC if possible (not possible if all targets are one class)
        try:
            roc_auc = roc_auc_score(targets_np.flatten(), predictions_np.flatten())
        except ValueError:
            roc_auc = float('nan')
            
        # Compute AP (Average Precision)
        try:
            ap = average_precision_score(targets_np.flatten(), predictions_np.flatten())
        except ValueError:
            ap = float('nan')
            
        # Compute basic accuracy
        accuracy = (binary_preds == targets_np).mean()
        
        # Count positive predictions
        num_positive_preds = binary_preds.sum()
        num_positive_targets = targets_np.sum()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'average_precision': ap,
            'num_positive_preds': num_positive_preds,
            'num_positive_targets': num_positive_targets
        }
    
    @staticmethod
    def compute_precision_at_k(scores: torch.Tensor, 
                              ground_truth: torch.Tensor, 
                              k: int) -> float:
        """
        Compute Precision@k for link prediction.
        
        Args:
            scores: Prediction scores for each potential link
            ground_truth: Binary ground truth values for each potential link
            k: Number of top predictions to consider
            
        Returns:
            Precision@k value
        """
        if isinstance(scores, torch.Tensor):
            # Handle dimension mismatch - reshape if needed
            if ground_truth.dim() > scores.dim():
                if ground_truth.dim() == 3 and scores.dim() == 2:
                    # If ground_truth has shape [batch, n, n] and scores has shape [n, n]
                    # Reshape ground_truth to [n, n]
                    ground_truth = ground_truth.squeeze(0)
                    
            scores = scores.detach().cpu().numpy()
            ground_truth = ground_truth.detach().cpu().numpy()
            
        # Get indices of top-k predictions
        top_k_indices = np.argsort(scores.flatten())[-k:]
        
        # Get ground truth values for top-k predictions
        top_k_ground_truth = ground_truth.flatten()[top_k_indices]
        
        # Compute precision@k
        precision_at_k = top_k_ground_truth.mean()
        
        return precision_at_k


class FeatureGenerationMetrics:
    """
    Metrics for evaluating feature generation performance.
    """
    
    @staticmethod
    def compute_feature_metrics(generated_features: torch.Tensor, 
                               target_features: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics for feature generation.
        
        Args:
            generated_features: Generated feature vectors
            target_features: Ground truth feature vectors
            
        Returns:
            Dictionary of metrics
        """
        if isinstance(generated_features, torch.Tensor):
            generated_features = generated_features.detach().cpu().numpy()
            target_features = target_features.detach().cpu().numpy()
            
        # Compute mean squared error
        mse = np.mean((generated_features - target_features) ** 2)
        
        # Compute cosine similarity
        norm_gen = np.linalg.norm(generated_features, axis=1, keepdims=True)
        norm_target = np.linalg.norm(target_features, axis=1, keepdims=True)
        dot_product = np.sum(generated_features * target_features, axis=1, keepdims=True)
        cosine_sim = (dot_product / (norm_gen * norm_target)).mean()
        
        # Compute L1 distance
        l1_distance = np.abs(generated_features - target_features).sum(axis=1).mean()
        
        return {
            'mse': mse,
            'cosine_similarity': cosine_sim,
            'l1_distance': l1_distance
        }
    
    @staticmethod
    def top_k_feature_overlap(generated_features: torch.Tensor, 
                             target_features: torch.Tensor, 
                             k: int) -> float:
        """
        Compute overlap between top-k features in generated and target.
        
        Args:
            generated_features: Generated feature vectors
            target_features: Ground truth feature vectors
            k: Number of top features to consider
            
        Returns:
            Average overlap ratio (0-1)
        """
        if isinstance(generated_features, torch.Tensor):
            generated_features = generated_features.detach().cpu().numpy()
            target_features = target_features.detach().cpu().numpy()
            
        # Calculate overlap for each sample
        num_samples = generated_features.shape[0]
        overlaps = []
        
        for i in range(num_samples):
            # Get indices of top-k features
            gen_top_k = np.argsort(generated_features[i])[-k:]
            target_top_k = np.argsort(target_features[i])[-k:]
            
            # Count overlap
            overlap = len(set(gen_top_k).intersection(set(target_top_k)))
            overlap_ratio = overlap / k
            overlaps.append(overlap_ratio)
        
        return np.mean(overlaps)


class NetworkEvaluationMetrics:
    """
    Metrics for evaluating the structure of the generated citation network.
    """
    
    @staticmethod
    def create_networkx_graph(edge_index: torch.Tensor) -> nx.DiGraph:
        """
        Create a NetworkX graph from an edge index tensor.
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            
        Returns:
            NetworkX directed graph
        """
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.detach().cpu().numpy()
            
        # Create directed graph
        G = nx.DiGraph()
        
        # Add edges
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(src, dst)
            
        return G
    
    @staticmethod
    def compute_graph_statistics(G: nx.DiGraph) -> Dict[str, float]:
        """
        Compute statistics of a graph.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary of graph statistics
        """
        # Basic stats
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)
        
        # Try to compute average clustering coefficient (may not work for directed graphs)
        try:
            clustering_coeff = nx.average_clustering(G)
        except:
            clustering_coeff = float('nan')
            
        # Compute in-degree and out-degree statistics
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        
        avg_in_degree = np.mean(in_degrees) if in_degrees else 0
        avg_out_degree = np.mean(out_degrees) if out_degrees else 0
        max_in_degree = max(in_degrees) if in_degrees else 0
        max_out_degree = max(out_degrees) if out_degrees else 0
        
        # Check if graph is connected (for undirected version)
        undirected_G = G.to_undirected()
        is_connected = nx.is_connected(undirected_G)
        
        # Number of connected components
        num_components = nx.number_connected_components(undirected_G)
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'clustering_coefficient': clustering_coeff,
            'avg_in_degree': avg_in_degree,
            'avg_out_degree': avg_out_degree,
            'max_in_degree': max_in_degree,
            'max_out_degree': max_out_degree,
            'is_connected': int(is_connected),
            'num_components': num_components
        }
    
    @staticmethod
    def compare_graph_statistics(real_stats: Dict[str, float], 
                                generated_stats: Dict[str, float]) -> Dict[str, float]:
        """
        Compare statistics between real and generated graphs.
        
        Args:
            real_stats: Statistics of the real graph
            generated_stats: Statistics of the generated graph
            
        Returns:
            Dictionary of comparison metrics (ratios or differences)
        """
        comparison = {}
        
        # Compute relative differences/ratios for each metric
        for key in real_stats:
            if key in ['is_connected', 'num_components']:
                # For boolean/count metrics, compute absolute difference
                comparison[f"{key}_diff"] = abs(real_stats[key] - generated_stats[key])
            elif real_stats[key] > 0:
                # For positive metrics, compute ratio (generated/real)
                comparison[f"{key}_ratio"] = generated_stats[key] / real_stats[key]
            else:
                # Skip division by zero cases
                comparison[f"{key}_ratio"] = float('nan')
        
        return comparison


class NodePredictionMetrics:
    """
    Metrics for evaluating the quality of generated papers.
    """
    
    @staticmethod
    def evaluate_generated_paper(generated_features: np.ndarray,
                                generated_citations: np.ndarray,
                                all_papers_features: np.ndarray,
                                all_papers_citations: np.ndarray,
                                papers_from_year_T: List[int],
                                top_k: int = 5) -> Dict[str, float]:
        """
        Evaluate a generated paper against actual papers from year T.
        
        Args:
            generated_features: Features of the generated paper
            generated_citations: Citation vector of the generated paper
            all_papers_features: Features of all papers
            all_papers_citations: Citation vectors of all papers
            papers_from_year_T: Indices of papers from year T
            top_k: Number of top papers to consider
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get features and citations for papers from year T
        year_T_features = all_papers_features[papers_from_year_T]
        year_T_citations = all_papers_citations[papers_from_year_T]
        
        # Compute feature similarities to all papers from year T
        feature_similarities = []
        for features in year_T_features:
            # Normalize the features
            norm_gen = np.linalg.norm(generated_features)
            norm_real = np.linalg.norm(features)
            
            # Compute cosine similarity
            if norm_gen > 0 and norm_real > 0:
                sim = np.dot(generated_features, features) / (norm_gen * norm_real)
            else:
                sim = 0.0
                
            feature_similarities.append(sim)
        
        # Compute citation similarities to all papers from year T
        citation_similarities = []
        for citations in year_T_citations:
            # Compute Jaccard similarity of citation sets
            intersection = np.sum(generated_citations * citations)
            union = np.sum(np.clip(generated_citations + citations, 0, 1))
            
            if union > 0:
                sim = intersection / union
            else:
                sim = 0.0
                
            citation_similarities.append(sim)
        
        # Compute combined similarity (weighted average)
        feature_weight = 0.5
        citation_weight = 0.5
        combined_similarities = [
            feature_weight * fs + citation_weight * cs
            for fs, cs in zip(feature_similarities, citation_similarities)
        ]
        
        # Find top-k similar real papers
        top_k_indices = np.argsort(combined_similarities)[-top_k:]
        top_k_similarities = [combined_similarities[i] for i in top_k_indices]
        
        # Get best matching real paper
        best_match_idx = top_k_indices[-1]
        best_match_paper_idx = papers_from_year_T[best_match_idx]
        best_match_similarity = top_k_similarities[-1]
        
        # Calculate feature and citation similarity with best matching paper
        best_match_feature_sim = feature_similarities[best_match_idx]
        best_match_citation_sim = citation_similarities[best_match_idx]
        
        return {
            'best_match_similarity': best_match_similarity,
            'best_match_feature_sim': best_match_feature_sim,
            'best_match_citation_sim': best_match_citation_sim,
            'best_match_paper_idx': best_match_paper_idx,
            'avg_top_k_similarity': np.mean(top_k_similarities),
            'avg_feature_similarity': np.mean(feature_similarities),
            'avg_citation_similarity': np.mean(citation_similarities)
        }
        
    @staticmethod
    def evaluate_citation_impact(actual_citation_counts: np.ndarray,
                                predicted_citation_scores: np.ndarray) -> Dict[str, float]:
        """
        Evaluate how well the model predicts future citation impact.
        
        Args:
            actual_citation_counts: Actual citation counts for papers
            predicted_citation_scores: Predicted citation scores
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Normalize actual citation counts to [0, 1] range
        max_citations = max(actual_citation_counts)
        if max_citations > 0:
            normalized_citations = actual_citation_counts / max_citations
        else:
            normalized_citations = actual_citation_counts
            
        # Compute correlation between predicted scores and actual citations
        try:
            correlation = np.corrcoef(predicted_citation_scores, normalized_citations)[0, 1]
        except:
            correlation = float('nan')
            
        # Compute rank correlation (Spearman)
        try:
            from scipy.stats import spearmanr
            rank_correlation, _ = spearmanr(predicted_citation_scores, actual_citation_counts)
        except:
            rank_correlation = float('nan')
            
        # Identify top-k papers by actual citation count and by prediction
        k = min(len(actual_citation_counts) // 10, 100)  # Use 10% or max 100 papers
        top_k_by_actual = np.argsort(actual_citation_counts)[-k:]
        top_k_by_predicted = np.argsort(predicted_citation_scores)[-k:]
        
        # Compute overlap between top-k sets
        overlap = len(set(top_k_by_actual).intersection(set(top_k_by_predicted)))
        overlap_ratio = overlap / k if k > 0 else 0
        
        return {
            'correlation': correlation,
            'rank_correlation': rank_correlation,
            'top_k_overlap_ratio': overlap_ratio
        } 