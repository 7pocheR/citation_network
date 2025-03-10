import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance, entropy
from torch.distributions import kl_divergence

from src.data.dataset import GraphData
from src.models.integrated_citation_model import IntegratedCitationModel

logger = logging.getLogger(__name__)

class TemporalEvaluator:
    """
    Evaluator for temporal hold-out evaluation of citation network models.
    
    This class implements evaluation strategies that compare generated papers
    with actual papers published in a future time window.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the temporal evaluator.
        
        Args:
            device: Device to use for computations ('cpu' or 'cuda')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
    def temporal_split(self, 
                      graph: GraphData, 
                      time_threshold: float) -> Tuple[GraphData, GraphData]:
        """
        Split a graph into past and future based on node timestamps.
        
        Args:
            graph: The complete graph
            time_threshold: Timestamp to split past and future
            
        Returns:
            Tuple of (past_graph, future_graph)
        """
        if not hasattr(graph, 'node_timestamps') or graph.node_timestamps is None:
            raise ValueError("Graph must have node_timestamps for temporal split")
            
        # Create past/future masks
        past_mask = graph.node_timestamps <= time_threshold
        future_mask = graph.node_timestamps > time_threshold
        
        # Get indices
        past_indices = torch.where(past_mask)[0]
        future_indices = torch.where(future_mask)[0]
        
        # Create subgraphs
        past_graph = graph.subgraph(past_indices)
        future_graph = graph.subgraph(future_indices)
        
        logger.info(f"Split graph at time {time_threshold}: "
                   f"{past_indices.size(0)} past nodes, {future_indices.size(0)} future nodes")
                   
        return past_graph, future_graph
        
    def evaluate_feature_distribution(self,
                                    generated_features: torch.Tensor,
                                    real_features: torch.Tensor) -> Dict[str, float]:
        """
        Compare distributions of generated vs. real paper features.
        
        Args:
            generated_features: Features of generated papers [num_papers, feature_dim]
            real_features: Features of real papers [num_papers, feature_dim]
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy for some metrics
        gen_np = generated_features.detach().cpu().numpy()
        real_np = real_features.detach().cpu().numpy()
        
        # Basic statistical metrics
        metrics = {}
        
        # Feature-wise mean and std difference
        metrics['mean_diff'] = np.mean(np.abs(np.mean(gen_np, axis=0) - np.mean(real_np, axis=0)))
        metrics['std_diff'] = np.mean(np.abs(np.std(gen_np, axis=0) - np.std(real_np, axis=0)))
        
        # MSE and MAE
        metrics['mse'] = mean_squared_error(real_np, gen_np)
        metrics['mae'] = mean_absolute_error(real_np, gen_np)
        
        # Cosine similarity (average pairwise similarity)
        cos_sim = cosine_similarity(gen_np, real_np)
        metrics['cosine_sim'] = np.mean(np.diag(cos_sim))
        
        # Distribution difference metrics
        # 1. Wasserstein distance for each feature dimension
        w_distances = []
        for i in range(gen_np.shape[1]):
            w_distances.append(wasserstein_distance(gen_np[:, i], real_np[:, i]))
        metrics['wasserstein_dist'] = np.mean(w_distances)
        
        # Maximum Mean Discrepancy (MMD) - simplified version
        def mmd_linear(x, y):
            """Linear MMD"""
            xx = np.mean(np.dot(x, x.T))
            yy = np.mean(np.dot(y, y.T))
            xy = np.mean(np.dot(x, y.T))
            return xx + yy - 2 * xy
            
        metrics['mmd'] = mmd_linear(gen_np, real_np)
        
        return metrics
        
    def evaluate_citation_patterns(self,
                                 generated_citations: torch.Tensor,
                                 real_citations: torch.Tensor) -> Dict[str, float]:
        """
        Compare citation patterns between generated and real papers.
        
        Args:
            generated_citations: Citation probability matrix for generated papers
            real_citations: Citation adjacency matrix for real papers
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to numpy
        gen_np = generated_citations.detach().cpu().numpy()
        real_np = real_citations.detach().cpu().numpy()
        
        # Ensure binary values for real citations
        if np.max(real_np) > 1:
            real_np = (real_np > 0).astype(float)
            
        # Thresholded accuracy
        gen_binary = (gen_np > 0.5).astype(float)
        metrics['citation_accuracy'] = np.mean((gen_binary == real_np).astype(float))
        
        # Degree distribution comparison
        gen_out_degree = np.sum(gen_binary, axis=1)
        real_out_degree = np.sum(real_np, axis=1)
        
        metrics['degree_mse'] = mean_squared_error(real_out_degree, gen_out_degree)
        metrics['degree_mae'] = mean_absolute_error(real_out_degree, gen_out_degree)
        
        # Average citation count difference
        metrics['avg_citation_diff'] = np.abs(np.mean(gen_out_degree) - np.mean(real_out_degree))
        
        # Citation distribution divergence (KL)
        # Add a small epsilon to avoid division by zero
        eps = 1e-8
        
        # Create histograms/distributions
        gen_hist, _ = np.histogram(gen_out_degree, bins=10, density=True)
        real_hist, _ = np.histogram(real_out_degree, bins=10, density=True)
        
        # Ensure non-zero probabilities
        gen_hist = gen_hist + eps
        gen_hist = gen_hist / np.sum(gen_hist)
        
        real_hist = real_hist + eps
        real_hist = real_hist / np.sum(real_hist)
        
        # Calculate KL divergence
        metrics['citation_kl_div'] = entropy(real_hist, gen_hist)
        
        return metrics
        
    def evaluate_topic_evolution(self,
                               generated_features: torch.Tensor,
                               real_features: torch.Tensor,
                               feature_to_topic_map: Optional[Dict[int, str]] = None) -> Dict[str, float]:
        """
        Evaluate how well generated papers capture emerging topics.
        
        Args:
            generated_features: Features of generated papers [num_papers, feature_dim]
            real_features: Features of real papers [num_papers, feature_dim]
            feature_to_topic_map: Optional mapping from feature index to topic name
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to numpy
        gen_np = generated_features.detach().cpu().numpy()
        real_np = real_features.detach().cpu().numpy()
        
        # Identify top features in real papers
        real_mean = np.mean(real_np, axis=0)
        top_real_features = np.argsort(real_mean)[-10:]  # Top 10 features
        
        # Check if these top features are also prominent in generated papers
        gen_mean = np.mean(gen_np, axis=0)
        gen_rank = np.argsort(gen_mean)
        
        # Calculate how many of the top real features are in the top generated features
        top_overlap_10 = len(set(top_real_features) & set(np.argsort(gen_mean)[-10:]))
        top_overlap_20 = len(set(np.argsort(real_mean)[-20:]) & set(np.argsort(gen_mean)[-20:]))
        
        metrics['top_topic_overlap_10'] = top_overlap_10 / 10.0
        metrics['top_topic_overlap_20'] = top_overlap_20 / 20.0
        
        # Calculate rank correlation of features
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(real_mean, gen_mean)
        metrics['topic_rank_correlation'] = rank_corr
        
        # Calculate average feature importance error
        metrics['topic_importance_mse'] = mean_squared_error(real_mean, gen_mean)
        
        # If we have a mapping from features to topics, include topic-specific metrics
        if feature_to_topic_map:
            topic_metrics = {}
            for feat_idx, topic_name in feature_to_topic_map.items():
                real_importance = real_mean[feat_idx]
                gen_importance = gen_mean[feat_idx]
                topic_metrics[f"topic_{topic_name}"] = {
                    'real_importance': float(real_importance),
                    'gen_importance': float(gen_importance),
                    'diff': float(abs(real_importance - gen_importance))
                }
            metrics['topic_specific'] = topic_metrics
            
        return metrics
        
    def evaluate_temporal_coherence(self,
                                  generated_papers: List[Dict[str, Any]],
                                  time_order: List[int]) -> Dict[str, float]:
        """
        Evaluate temporal coherence of generated papers.
        
        Args:
            generated_papers: List of generated paper info dictionaries
            time_order: Order of papers by time (indices)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Check if papers have timestamps
        if 'timestamp' not in generated_papers[0]:
            return {'error': 'No timestamps in generated papers'}
            
        # Extract timestamps
        timestamps = [paper['timestamp'] for paper in generated_papers]
        
        # Check if timestamps increase monotonically for the time_order
        ordered_timestamps = [timestamps[i] for i in time_order]
        is_monotonic = all(ordered_timestamps[i] <= ordered_timestamps[i+1] 
                          for i in range(len(ordered_timestamps)-1))
        
        metrics['time_monotonic'] = float(is_monotonic)
        
        # Measure time consistency (correlation between index and time)
        from scipy.stats import pearsonr
        time_corr, _ = pearsonr(list(range(len(ordered_timestamps))), ordered_timestamps)
        metrics['time_correlation'] = time_corr
        
        # Check if citation patterns respect time (papers only cite earlier papers)
        time_consistent_citations = 0
        total_citations = 0
        
        for i, paper in enumerate(generated_papers):
            if 'cited_papers' in paper:
                for cited in paper['cited_papers']:
                    total_citations += 1
                    cited_idx = cited.get('id')
                    if cited_idx < len(timestamps):  # Ensure the cited paper exists
                        # Check if cited paper is earlier in time
                        if timestamps[cited_idx] < timestamps[i]:
                            time_consistent_citations += 1
        
        if total_citations > 0:
            metrics['citation_time_consistency'] = time_consistent_citations / total_citations
        else:
            metrics['citation_time_consistency'] = 1.0  # No citations = perfect consistency
            
        return metrics
        
    def evaluate_model(self,
                     model: IntegratedCitationModel,
                     graph: GraphData,
                     time_threshold: float,
                     future_window: Optional[float] = None,
                     num_papers: int = 10,
                     temperature: float = 1.0) -> Dict[str, Any]:
        """
        Evaluate a model using temporal hold-out strategy.
        
        Args:
            model: The model to evaluate
            graph: The complete graph
            time_threshold: Time to split past/future
            future_window: Time window for future papers (from threshold)
            num_papers: Number of papers to generate
            temperature: Temperature for generation diversity
            
        Returns:
            Dictionary of evaluation results
        """
        # Split graph into past and future
        past_graph, future_graph = self.temporal_split(graph, time_threshold)
        
        # Set default future window if not provided
        if future_window is None:
            if hasattr(graph, 'node_timestamps') and graph.node_timestamps is not None:
                max_time = torch.max(graph.node_timestamps).item()
                future_window = max_time - time_threshold
                logger.info(f"Setting future window to {future_window}")
            else:
                future_window = 1.0
                logger.warning(f"No timestamps available, using default future window {future_window}")
        
        # Extract future features for comparison
        future_features = future_graph.x
        
        # Generate papers with the model
        with torch.no_grad():
            model.eval()
            generated_features, paper_info = model.generate_future_papers(
                past_graph, 
                time_threshold=time_threshold,
                future_window=future_window,
                num_papers=num_papers,
                temperature=temperature
            )
        
        # Match the number of papers for evaluation
        min_papers = min(len(paper_info), future_features.size(0))
        generated_features = generated_features[:min_papers]
        future_features = future_features[:min_papers]
        paper_info = paper_info[:min_papers]
        
        # Calculate adjacency matrix for real future papers
        # Get node embeddings for past papers
        with torch.no_grad():
            node_embeddings = model.encoder(past_graph)
            
            # Predict citation probabilities between future and past papers
            if hasattr(model, '_predict_citations_for_new_paper'):
                real_citation_probs = model._predict_citations_for_new_paper(
                    node_embeddings, future_features
                )
            else:
                # Fallback
                logger.warning("Model doesn't have _predict_citations_for_new_paper method")
                real_citation_probs = torch.zeros(
                    (min_papers, past_graph.num_nodes), 
                    device=self.device
                )
        
        # Extract citation probabilities from generated papers
        gen_citation_probs = torch.zeros(
            (min_papers, past_graph.num_nodes), 
            device=self.device
        )
        
        for i, paper in enumerate(paper_info[:min_papers]):
            if 'cited_papers' in paper:
                for cited in paper['cited_papers']:
                    idx = cited.get('id')
                    prob = cited.get('probability', 1.0)
                    if idx < past_graph.num_nodes:
                        gen_citation_probs[i, idx] = prob
        
        # Create a time-ordered list of paper indices
        if 'timestamp' in paper_info[0]:
            timestamps = [paper.get('timestamp', 0) for paper in paper_info[:min_papers]]
            time_order = np.argsort(timestamps).tolist()
        else:
            time_order = list(range(min_papers))
        
        # Run evaluations
        results = {}
        
        # Feature distribution evaluation
        results['feature_distribution'] = self.evaluate_feature_distribution(
            generated_features, future_features
        )
        
        # Citation pattern evaluation
        results['citation_patterns'] = self.evaluate_citation_patterns(
            gen_citation_probs, real_citation_probs
        )
        
        # Topic evolution evaluation
        results['topic_evolution'] = self.evaluate_topic_evolution(
            generated_features, future_features
        )
        
        # Temporal coherence evaluation
        results['temporal_coherence'] = self.evaluate_temporal_coherence(
            paper_info[:min_papers], time_order
        )
        
        # Overall scores (weighted combination)
        overall_score = (
            0.3 * (1.0 - results['feature_distribution']['mse'] / 
                  max(1.0, results['feature_distribution']['mse'])) +
            0.3 * results['citation_patterns']['citation_accuracy'] +
            0.2 * results['topic_evolution']['topic_rank_correlation'] +
            0.2 * results['temporal_coherence']['time_correlation']
        )
        
        results['overall_score'] = max(0.0, min(1.0, overall_score))  # Clamp to [0, 1]
        
        return results
        
    def compare_generation_approaches(self,
                                    model: IntegratedCitationModel,
                                    graph: GraphData,
                                    time_threshold: float,
                                    future_window: Optional[float] = None,
                                    num_papers: int = 10,
                                    temperature: float = 1.0) -> Dict[str, Any]:
        """
        Compare different paper generation approaches.
        
        Args:
            model: The model to evaluate
            graph: The complete graph
            time_threshold: Time to split past/future
            future_window: Time window for future papers
            num_papers: Number of papers to generate
            temperature: Temperature for generation diversity
            
        Returns:
            Dictionary of comparative evaluation results
        """
        # Split graph into past and future
        past_graph, future_graph = self.temporal_split(graph, time_threshold)
        
        # Set future window if not provided
        if future_window is None:
            if hasattr(graph, 'node_timestamps') and graph.node_timestamps is not None:
                max_time = torch.max(graph.node_timestamps).item()
                future_window = max_time - time_threshold
                logger.info(f"Setting future window to {future_window}")
            else:
                future_window = 1.0
                logger.warning(f"No timestamps available, using default future window {future_window}")
        
        # Full CVAE approach evaluation
        logger.info("Evaluating CVAE-based generation...")
        cvae_results = self.evaluate_model(
            model, graph, time_threshold, future_window, num_papers, temperature
        )
        
        # Feature distribution metrics are most relevant for comparing approaches
        comparative_results = {
            'cvae_approach': {
                'feature_mse': cvae_results['feature_distribution']['mse'],
                'citation_accuracy': cvae_results['citation_patterns']['citation_accuracy'],
                'topic_correlation': cvae_results['topic_evolution']['topic_rank_correlation'],
                'temporal_coherence': cvae_results['temporal_coherence']['time_correlation'],
                'overall_score': cvae_results['overall_score']
            }
        }
        
        # Additional approaches could be implemented here
        # For example, predictor-only approach, hybrid approach, etc.
        
        # TODO: Implement predictor-only approach
        # TODO: Implement hybrid approach
        
        return comparative_results
        
    def visualize_comparison(self, 
                           comparison_results: Dict[str, Any],
                           save_path: Optional[str] = None) -> None:
        """
        Visualize comparison of different generation approaches.
        
        Args:
            comparison_results: Results from compare_generation_approaches
            save_path: Optional path to save the visualization
        """
        # Extract metrics for each approach
        approaches = list(comparison_results.keys())
        metrics = ['feature_mse', 'citation_accuracy', 'topic_correlation', 
                  'temporal_coherence', 'overall_score']
        
        # Create subplots
        fig, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [comparison_results[approach][metric] for approach in approaches]
            axs[i].bar(approaches, values)
            axs[i].set_title(metric)
            axs[i].set_ylim(0, 1 if metric != 'feature_mse' else max(values) * 1.2)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show() 