import torch
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from src.data.datasets import GraphData

logger = logging.getLogger(__name__)

class GenerationEvaluator:
    """Evaluator for generated papers with metrics for quality, novelty, and diversity.
    
    This class provides comprehensive evaluation metrics for generated papers:
    - Quality metrics: How realistic are the generated papers
    - Novelty metrics: How different are generated papers from training
    - Diversity metrics: How varied are the generated papers
    - Citation analysis: How well generated papers follow citation patterns
    """
    
    def __init__(self, original_graph: GraphData, generated_features: torch.Tensor, 
                 generated_metadata: List[Dict[str, Any]]):
        """Initialize the generation evaluator.
        
        Args:
            original_graph (GraphData): Original citation graph
            generated_features (torch.Tensor): Generated paper features
            generated_metadata (List[Dict[str, Any]]): Generated paper metadata
        """
        self.original_graph = original_graph
        self.generated_features = generated_features
        self.generated_metadata = generated_metadata
        
        # Cache for computed metrics
        self.metrics_cache = {}
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics.
        
        Returns:
            Dict[str, float]: Dictionary of all computed metrics
        """
        all_metrics = {}
        
        # Quality metrics
        quality_metrics = self.compute_quality_metrics()
        all_metrics.update({f"quality_{k}": v for k, v in quality_metrics.items()})
        
        # Novelty metrics
        novelty_metrics = self.compute_novelty_metrics()
        all_metrics.update({f"novelty_{k}": v for k, v in novelty_metrics.items()})
        
        # Diversity metrics
        diversity_metrics = self.compute_diversity_metrics()
        all_metrics.update({f"diversity_{k}": v for k, v in diversity_metrics.items()})
        
        # Citation pattern metrics
        citation_metrics = self.compute_citation_pattern_metrics()
        all_metrics.update({f"citation_{k}": v for k, v in citation_metrics.items()})
        
        return all_metrics
    
    def compute_quality_metrics(self) -> Dict[str, float]:
        """Compute quality metrics for generated papers.
        
        Returns:
            Dict[str, float]: Dictionary of quality metrics
        """
        if "quality" in self.metrics_cache:
            return self.metrics_cache["quality"]
        
        # Convert tensors to numpy arrays
        orig_features = self.original_graph.x.cpu().numpy()
        gen_features = self.generated_features.cpu().numpy()
        
        # Feature distribution metrics
        orig_mean = np.mean(orig_features, axis=0)
        gen_mean = np.mean(gen_features, axis=0)
        
        orig_std = np.std(orig_features, axis=0)
        gen_std = np.std(gen_features, axis=0)
        
        # Mean absolute error of feature distributions
        mean_error = np.mean(np.abs(orig_mean - gen_mean))
        std_error = np.mean(np.abs(orig_std - gen_std))
        
        # Feature correlation
        feature_correlation = np.corrcoef(orig_mean, gen_mean)[0, 1]
        
        # Earth Mover's Distance (if scipy available)
        try:
            # Compute EMD for each feature dimension
            emd_values = []
            for i in range(min(10, orig_features.shape[1])):  # Limit to first 10 features
                emd = scipy.stats.wasserstein_distance(
                    orig_features[:, i], gen_features[:, i])
                emd_values.append(emd)
            
            avg_emd = np.mean(emd_values)
        except:
            avg_emd = np.nan
        
        metrics = {
            "mean_error": float(mean_error),
            "std_error": float(std_error),
            "feature_correlation": float(feature_correlation),
            "avg_emd": float(avg_emd)
        }
        
        # Cache results
        self.metrics_cache["quality"] = metrics
        
        return metrics
    
    def compute_novelty_metrics(self) -> Dict[str, float]:
        """Compute novelty metrics comparing generated papers to training data.
        
        Returns:
            Dict[str, float]: Dictionary of novelty metrics
        """
        if "novelty" in self.metrics_cache:
            return self.metrics_cache["novelty"]
        
        # Convert tensors to numpy arrays
        orig_features = self.original_graph.x.cpu().numpy()
        gen_features = self.generated_features.cpu().numpy()
        
        # Minimum distance to any training example
        min_distances = []
        
        # Limit computation for large datasets
        if len(gen_features) > 100 or len(orig_features) > 10000:
            # Sample a subset
            sample_size = min(100, len(gen_features))
            sample_indices = np.random.choice(
                len(gen_features), sample_size, replace=False)
            gen_sample = gen_features[sample_indices]
            
            orig_sample_size = min(10000, len(orig_features))
            orig_sample_indices = np.random.choice(
                len(orig_features), orig_sample_size, replace=False)
            orig_sample = orig_features[orig_sample_indices]
            
            # Compute pairwise distances
            distances = pairwise_distances(gen_sample, orig_sample, metric='euclidean')
            min_distances = np.min(distances, axis=1)
        else:
            # Compute for all examples
            distances = pairwise_distances(gen_features, orig_features, metric='euclidean')
            min_distances = np.min(distances, axis=1)
        
        # Average minimum distance
        avg_min_distance = float(np.mean(min_distances))
        
        # Percentage of examples with distance > threshold
        novelty_threshold = np.percentile(min_distances, 75)  # 75th percentile
        novelty_percentage = float(np.mean(min_distances > novelty_threshold))
        
        metrics = {
            "avg_min_distance": avg_min_distance,
            "novelty_percentage": novelty_percentage
        }
        
        # Cache results
        self.metrics_cache["novelty"] = metrics
        
        return metrics
    
    def compute_diversity_metrics(self) -> Dict[str, float]:
        """Compute diversity metrics for generated papers.
        
        Returns:
            Dict[str, float]: Dictionary of diversity metrics
        """
        if "diversity" in self.metrics_cache:
            return self.metrics_cache["diversity"]
        
        # Convert tensors to numpy arrays
        gen_features = self.generated_features.cpu().numpy()
        
        # Pairwise distances
        if len(gen_features) > 100:
            # Sample a subset for large datasets
            sample_size = min(100, len(gen_features))
            sample_indices = np.random.choice(
                len(gen_features), sample_size, replace=False)
            gen_sample = gen_features[sample_indices]
            distances = pairwise_distances(gen_sample, metric='euclidean')
        else:
            distances = pairwise_distances(gen_features, metric='euclidean')
        
        # Remove self-distances (diagonal)
        np.fill_diagonal(distances, np.nan)
        mean_distance = float(np.nanmean(distances))
        
        # Compute pairwise cosine similarities
        if len(gen_features) > 100:
            # Reuse the sample
            norms = np.linalg.norm(gen_sample, axis=1, keepdims=True)
            normalized = gen_sample / (norms + 1e-8)
            cosine_similarities = np.dot(normalized, normalized.T)
        else:
            norms = np.linalg.norm(gen_features, axis=1, keepdims=True)
            normalized = gen_features / (norms + 1e-8)
            cosine_similarities = np.dot(normalized, normalized.T)
        
        # Remove self-similarities (diagonal)
        np.fill_diagonal(cosine_similarities, np.nan)
        mean_similarity = float(np.nanmean(cosine_similarities))
        
        # Perform PCA to measure variance along principal components
        try:
            pca = PCA(n_components=min(10, gen_features.shape[1], gen_features.shape[0]))
            pca.fit(gen_features)
            explained_variance_ratio = pca.explained_variance_ratio_
            
            # How much variance is explained by the first component
            first_component_ratio = float(explained_variance_ratio[0])
            
            # How many components needed to explain 90% variance
            cumulative_variance = np.cumsum(explained_variance_ratio)
            components_for_90 = np.sum(cumulative_variance < 0.9) + 1
        except:
            first_component_ratio = np.nan
            components_for_90 = np.nan
        
        metrics = {
            "mean_pairwise_distance": mean_distance,
            "mean_cosine_similarity": mean_similarity,
            "first_component_ratio": first_component_ratio,
            "components_for_90pct": float(components_for_90)
        }
        
        # Cache results
        self.metrics_cache["diversity"] = metrics
        
        return metrics
    
    def compute_citation_pattern_metrics(self) -> Dict[str, float]:
        """Compute metrics related to citation patterns.
        
        Returns:
            Dict[str, float]: Dictionary of citation pattern metrics
        """
        if "citation" in self.metrics_cache:
            return self.metrics_cache["citation"]
        
        # Extract citation data from metadata
        citation_counts = []
        for paper in self.generated_metadata:
            if 'citations' in paper:
                citation_counts.append(len(paper['citations']))
            else:
                citation_counts.append(0)
        
        # No citations found
        if len(citation_counts) == 0 or all(count == 0 for count in citation_counts):
            return {
                "avg_citations": 0.0,
                "citation_distribution_kl": 1.0,  # Maximum divergence
                "citation_variety": 0.0
            }
        
        # Average citations per paper
        avg_citations = float(np.mean(citation_counts))
        
        # Citation variety - ratio of unique cited papers to total citations
        all_citations = []
        for paper in self.generated_metadata:
            if 'citations' in paper:
                all_citations.extend(paper['citations'])
        
        if len(all_citations) > 0:
            citation_variety = float(len(set(all_citations)) / len(all_citations))
        else:
            citation_variety = 0.0
        
        # KL divergence between generated and original citation distribution
        # First get original citation distribution
        edge_index = self.original_graph.edge_index.cpu().numpy()
        original_cited_counts = Counter(edge_index[1, :])
        
        # Convert to probability distributions
        orig_dist = np.array(list(original_cited_counts.values()))
        orig_dist = orig_dist / orig_dist.sum()
        
        gen_cited_counts = Counter(all_citations)
        gen_dist = np.array(list(gen_cited_counts.values()))
        gen_dist = gen_dist / gen_dist.sum()
        
        # Make distributions the same length
        if len(gen_dist) < len(orig_dist):
            gen_dist = np.pad(gen_dist, (0, len(orig_dist) - len(gen_dist)))
        elif len(orig_dist) < len(gen_dist):
            orig_dist = np.pad(orig_dist, (0, len(gen_dist) - len(orig_dist)))
        
        # Compute KL divergence
        try:
            # Add small epsilon to avoid division by zero
            kl_divergence = scipy.stats.entropy(gen_dist + 1e-10, orig_dist + 1e-10)
        except:
            kl_divergence = 1.0  # Default to maximum divergence
        
        metrics = {
            "avg_citations": avg_citations,
            "citation_distribution_kl": float(kl_divergence),
            "citation_variety": citation_variety
        }
        
        # Cache results
        self.metrics_cache["citation"] = metrics
        
        return metrics
    
    def save_evaluation_results(self, output_path: str, prefix: str = "") -> None:
        """Save evaluation results to a file.
        
        Args:
            output_path (str): Directory to save results
            prefix (str): Optional prefix for filenames
        """
        # Compute all metrics
        all_metrics = self.compute_all_metrics()
        
        # Create output file path
        if prefix:
            filename = f"{prefix}_evaluation_metrics.txt"
        else:
            filename = "evaluation_metrics.txt"
        
        output_file = os.path.join(output_path, filename)
        
        # Save metrics to file
        with open(output_file, "w") as f:
            f.write("=== Generator Evaluation Metrics ===\n\n")
            
            # Quality metrics
            f.write("-- Quality Metrics --\n")
            for key, value in all_metrics.items():
                if key.startswith("quality_"):
                    metric_name = key.replace("quality_", "")
                    f.write(f"{metric_name}: {value:.4f}\n")
            f.write("\n")
            
            # Novelty metrics
            f.write("-- Novelty Metrics --\n")
            for key, value in all_metrics.items():
                if key.startswith("novelty_"):
                    metric_name = key.replace("novelty_", "")
                    f.write(f"{metric_name}: {value:.4f}\n")
            f.write("\n")
            
            # Diversity metrics
            f.write("-- Diversity Metrics --\n")
            for key, value in all_metrics.items():
                if key.startswith("diversity_"):
                    metric_name = key.replace("diversity_", "")
                    f.write(f"{metric_name}: {value:.4f}\n")
            f.write("\n")
            
            # Citation pattern metrics
            f.write("-- Citation Pattern Metrics --\n")
            for key, value in all_metrics.items():
                if key.startswith("citation_"):
                    metric_name = key.replace("citation_", "")
                    f.write(f"{metric_name}: {value:.4f}\n")
        
        logger.info(f"Evaluation results saved to {output_file}")
        
        # Also save as JSON for programmatic access
        try:
            import json
            json_file = os.path.join(output_path, filename.replace(".txt", ".json"))
            with open(json_file, "w") as f:
                json.dump(all_metrics, f, indent=2)
            logger.info(f"Evaluation results also saved as JSON to {json_file}")
        except:
            logger.warning("Could not save results as JSON")


class TemporalGenerationEvaluator(GenerationEvaluator):
    """Evaluator specifically for temporally generated papers.
    
    This extends the base evaluator with metrics specific to temporal generation:
    - Time consistency: Do papers only cite previous works
    - Temporal pattern matching: How well generated papers match temporal patterns
    """
    
    def __init__(self, 
                 original_graph: GraphData, 
                 generated_features: torch.Tensor,
                 generated_metadata: List[Dict[str, Any]],
                 time_threshold: float,
                 future_window: float):
        """Initialize the temporal generation evaluator.
        
        Args:
            original_graph (GraphData): Original citation graph
            generated_features (torch.Tensor): Generated paper features
            generated_metadata (List[Dict[str, Any]]): Generated paper metadata
            time_threshold (float): Starting time for generation
            future_window (float): Time window for generation
        """
        super().__init__(original_graph, generated_features, generated_metadata)
        self.time_threshold = time_threshold
        self.future_window = future_window
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics including temporal ones.
        
        Returns:
            Dict[str, float]: Dictionary of all computed metrics
        """
        # Get base metrics
        all_metrics = super().compute_all_metrics()
        
        # Add temporal metrics
        temporal_metrics = self.compute_temporal_metrics()
        all_metrics.update({f"temporal_{k}": v for k, v in temporal_metrics.items()})
        
        return all_metrics
    
    def compute_temporal_metrics(self) -> Dict[str, float]:
        """Compute metrics related to temporal aspects of generation.
        
        Returns:
            Dict[str, float]: Dictionary of temporal metrics
        """
        if "temporal" in self.metrics_cache:
            return self.metrics_cache["temporal"]
        
        # Check for required temporal information
        has_timestamps = hasattr(self.original_graph, 'node_timestamps') or hasattr(self.original_graph, 'timestamps')
        if not has_timestamps:
            logger.warning("Original graph does not have timestamp information")
            return {
                "time_consistency": 0.0,
                "temporal_pattern_match": 0.0
            }
        
        # Get timestamps from graph
        if hasattr(self.original_graph, 'node_timestamps'):
            orig_timestamps = self.original_graph.node_timestamps.cpu().numpy()
        else:
            orig_timestamps = self.original_graph.timestamps.cpu().numpy()
        
        # Get timestamps from generated papers
        gen_timestamps = []
        for paper in self.generated_metadata:
            if 'timestamp' in paper:
                gen_timestamps.append(paper['timestamp'])
            else:
                # Fallback - generate a random timestamp in the future window
                random_offset = np.random.random() * self.future_window
                gen_timestamps.append(self.time_threshold + random_offset)
        
        # Time consistency - papers should only cite previous works
        citation_consistency = []
        for i, paper in enumerate(self.generated_metadata):
            if 'citations' not in paper or 'timestamp' not in paper:
                continue
                
            paper_time = paper['timestamp']
            consistent_citations = 0
            total_citations = len(paper['citations'])
            
            if total_citations == 0:
                continue
                
            for citation in paper['citations']:
                if citation < len(orig_timestamps):
                    # Check if cited paper was published before citing paper
                    if orig_timestamps[citation] < paper_time:
                        consistent_citations += 1
            
            if total_citations > 0:
                citation_consistency.append(consistent_citations / total_citations)
        
        # Average citation consistency
        if citation_consistency:
            time_consistency = float(np.mean(citation_consistency))
        else:
            time_consistency = 0.0
        
        # Temporal pattern matching - distribution of timestamps
        # Compare distribution of time differences between original and generated
        orig_time_diffs = []
        for i in range(1, len(orig_timestamps)):
            orig_time_diffs.append(orig_timestamps[i] - orig_timestamps[i-1])
        
        gen_time_diffs = []
        for i in range(1, len(gen_timestamps)):
            gen_time_diffs.append(gen_timestamps[i] - gen_timestamps[i-1])
        
        if not orig_time_diffs or not gen_time_diffs:
            temporal_pattern_match = 0.0
        else:
            # Normalize time differences
            orig_time_diffs = np.array(orig_time_diffs)
            gen_time_diffs = np.array(gen_time_diffs)
            
            # Compute KL divergence between distributions
            try:
                # Create histograms with the same bins
                combined = np.concatenate([orig_time_diffs, gen_time_diffs])
                min_val, max_val = np.min(combined), np.max(combined)
                bins = 20
                
                orig_hist, _ = np.histogram(orig_time_diffs, bins=bins, range=(min_val, max_val), density=True)
                gen_hist, _ = np.histogram(gen_time_diffs, bins=bins, range=(min_val, max_val), density=True)
                
                # Add small epsilon to avoid division by zero
                orig_hist = orig_hist + 1e-10
                gen_hist = gen_hist + 1e-10
                
                # Normalize to create probability distributions
                orig_hist = orig_hist / np.sum(orig_hist)
                gen_hist = gen_hist / np.sum(gen_hist)
                
                # Compute KL divergence
                kl_divergence = scipy.stats.entropy(gen_hist, orig_hist)
                
                # Convert to similarity score (1 - normalized KL)
                max_kl = np.log(bins)  # Maximum possible KL
                temporal_pattern_match = 1.0 - min(kl_divergence / max_kl, 1.0)
            except:
                temporal_pattern_match = 0.0
        
        metrics = {
            "time_consistency": time_consistency,
            "temporal_pattern_match": float(temporal_pattern_match)
        }
        
        # Cache results
        self.metrics_cache["temporal"] = metrics
        
        return metrics
    
    def save_evaluation_results(self, output_path: str, prefix: str = "") -> None:
        """Save evaluation results to a file, including temporal metrics.
        
        Args:
            output_path (str): Directory to save results
            prefix (str): Optional prefix for filenames
        """
        # Compute all metrics
        all_metrics = self.compute_all_metrics()
        
        # Create output file path
        if prefix:
            filename = f"{prefix}_temporal_evaluation_metrics.txt"
        else:
            filename = "temporal_evaluation_metrics.txt"
        
        output_file = os.path.join(output_path, filename)
        
        # Save metrics to file
        with open(output_file, "w") as f:
            f.write("=== Temporal Generator Evaluation Metrics ===\n\n")
            
            # Quality metrics
            f.write("-- Quality Metrics --\n")
            for key, value in all_metrics.items():
                if key.startswith("quality_"):
                    metric_name = key.replace("quality_", "")
                    f.write(f"{metric_name}: {value:.4f}\n")
            f.write("\n")
            
            # Novelty metrics
            f.write("-- Novelty Metrics --\n")
            for key, value in all_metrics.items():
                if key.startswith("novelty_"):
                    metric_name = key.replace("novelty_", "")
                    f.write(f"{metric_name}: {value:.4f}\n")
            f.write("\n")
            
            # Diversity metrics
            f.write("-- Diversity Metrics --\n")
            for key, value in all_metrics.items():
                if key.startswith("diversity_"):
                    metric_name = key.replace("diversity_", "")
                    f.write(f"{metric_name}: {value:.4f}\n")
            f.write("\n")
            
            # Citation pattern metrics
            f.write("-- Citation Pattern Metrics --\n")
            for key, value in all_metrics.items():
                if key.startswith("citation_"):
                    metric_name = key.replace("citation_", "")
                    f.write(f"{metric_name}: {value:.4f}\n")
            f.write("\n")
            
            # Temporal metrics
            f.write("-- Temporal Metrics --\n")
            for key, value in all_metrics.items():
                if key.startswith("temporal_"):
                    metric_name = key.replace("temporal_", "")
                    f.write(f"{metric_name}: {value:.4f}\n")
        
        logger.info(f"Temporal evaluation results saved to {output_file}")
        
        # Also save as JSON for programmatic access
        try:
            import json
            json_file = os.path.join(output_path, filename.replace(".txt", ".json"))
            with open(json_file, "w") as f:
                json.dump(all_metrics, f, indent=2)
            logger.info(f"Evaluation results also saved as JSON to {json_file}")
        except:
            logger.warning("Could not save results as JSON")


# Helper function to evaluate generation
def evaluate_generation(
    original_graph: GraphData,
    generated_features: torch.Tensor,
    generated_metadata: List[Dict[str, Any]],
    output_path: str,
    prefix: str = "",
    time_threshold: Optional[float] = None,
    future_window: Optional[float] = None
) -> Dict[str, float]:
    """Evaluate generated papers and save results.
    
    Args:
        original_graph (GraphData): Original citation graph
        generated_features (torch.Tensor): Generated paper features
        generated_metadata (List[Dict[str, Any]]): Generated paper metadata
        output_path (str): Directory to save results
        prefix (str): Optional prefix for filenames
        time_threshold (Optional[float]): Starting time for temporal generation
        future_window (Optional[float]): Time window for temporal generation
        
    Returns:
        Dict[str, float]: Dictionary of all computed metrics
    """
    # Create appropriate evaluator
    if time_threshold is not None and future_window is not None:
        evaluator = TemporalGenerationEvaluator(
            original_graph=original_graph,
            generated_features=generated_features,
            generated_metadata=generated_metadata,
            time_threshold=time_threshold,
            future_window=future_window
        )
    else:
        evaluator = GenerationEvaluator(
            original_graph=original_graph,
            generated_features=generated_features,
            generated_metadata=generated_metadata
        )
    
    # Compute metrics
    all_metrics = evaluator.compute_all_metrics()
    
    # Save results
    evaluator.save_evaluation_results(output_path, prefix)
    
    return all_metrics 