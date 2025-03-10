import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Type
import logging

from src.data.datasets import GraphData
from src.models.encoder.base import BaseEncoder
from src.models.encoder.tgn_encoder import TGNEncoder
from src.models.encoder.hyperbolic_gnn import HyperbolicGNN
from src.models.encoder.simple_hyperbolic import SimpleHyperbolicEncoder
from src.models.predictors.attention_predictor import AttentionPredictor

logger = logging.getLogger(__name__)

# Placeholder class for legacy generator
class EnhancedCVAEGenerator(nn.Module):
    """Placeholder for legacy EnhancedCVAEGenerator."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.warning("Using placeholder EnhancedCVAEGenerator - actual implementation not available")
        self.latent_dim = kwargs.get('latent_dim', 128)
        
    def encode(self, *args, **kwargs):
        return torch.randn(1, self.latent_dim).to(next(self.parameters()).device)
        
    def decode(self, *args, **kwargs):
        return torch.zeros(1, kwargs.get('feature_dim', 256)).to(next(self.parameters()).device)
        
    def forward(self, *args, **kwargs):
        return {'reconstructed_features': torch.zeros(1, 1)}

class DeviceManager:
    """Utility class to manage consistent device placement across model components.
    
    This ensures that all components (encoder, predictor, generator) operate on the
    same device, and handles proper device transfer for inputs and outputs.
    """
    
    def __init__(self, default_device: Optional[torch.device] = None):
        """Initialize the device manager.
        
        Args:
            default_device (Optional[torch.device]): Default device to use if none specified
        """
        if default_device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = default_device
        
        self.module_devices = {}
    
    def register_module(self, name: str, module: nn.Module) -> None:
        """Register a module with the device manager.
        
        Args:
            name (str): Name of the module
            module (nn.Module): The module to register
        """
        # Store the current device of the module
        try:
            device = next(module.parameters()).device
            self.module_devices[name] = device
        except StopIteration:
            # Module has no parameters
            self.module_devices[name] = self.device
    
    def get_module_device(self, name: str) -> torch.device:
        """Get the device of a registered module.
        
        Args:
            name (str): Name of the module
            
        Returns:
            torch.device: Device of the module
        """
        if name not in self.module_devices:
            return self.device
        return self.module_devices[name]
    
    def ensure_same_device(self, tensor: torch.Tensor, target_name: str) -> torch.Tensor:
        """Ensure a tensor is on the same device as a registered module.
        
        Args:
            tensor (torch.Tensor): Tensor to check
            target_name (str): Name of the module for device matching
            
        Returns:
            torch.Tensor: Tensor on the correct device
        """
        target_device = self.get_module_device(target_name)
        
        if tensor.device != target_device:
            return tensor.to(target_device)
        return tensor
    
    def sync_modules(self, target_device: Optional[torch.device] = None) -> None:
        """Synchronize all modules to use the same device.
        
        Args:
            target_device (Optional[torch.device]): Target device to move all modules to.
                If None, uses the device manager's default device.
        """
        if target_device is None:
            target_device = self.device
            
        # Update the target device
        self.device = target_device
        
        # Notify about device synchronization
        logger.info(f"Synchronizing all modules to device: {target_device}")
        
        # Reset the module device tracking
        self.module_devices = {}


class EncoderAdapterLayer(nn.Module):
    """Adapter layer to handle different encoder output spaces.
    
    This layer provides proper transformation from various encoder spaces 
    (Euclidean, hyperbolic) to a consistent format that can be consumed by
    both the predictor and generator.
    """
    
    def __init__(self, 
                 embed_dim: int, 
                 encoder_type: str,
                 use_hyperbolic: bool = False,
                 curvature: float = 1.0):
        """Initialize the encoder adapter layer.
        
        Args:
            embed_dim (int): Dimension of embeddings
            encoder_type (str): Type of encoder ('tgn', 'hyperbolic_gnn', etc.)
            use_hyperbolic (bool): Whether the encoder uses hyperbolic space
            curvature (float): Curvature of hyperbolic space if applicable
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder_type = encoder_type
        self.use_hyperbolic = use_hyperbolic
        self.curvature = curvature
        
        # Create adaptation layers based on encoder type
        if use_hyperbolic:
            # For hyperbolic encoders, add tangent space projection
            self.adapter = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
        else:
            # For standard Euclidean encoders, minimal adaptation
            self.adapter = nn.Identity()
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Adapt encoder embeddings to standard format.
        
        Args:
            embeddings (torch.Tensor): Encoder embeddings [num_nodes, embed_dim]
            
        Returns:
            torch.Tensor: Adapted embeddings [num_nodes, embed_dim]
        """
        if self.use_hyperbolic:
            # Convert from hyperbolic to Euclidean via logarithmic map
            norm = torch.norm(embeddings, dim=-1, keepdim=True)
            # Avoid numerical issues with very small norms
            mask = (norm > 1e-10).float()
            norm = norm * mask + (1 - mask) * 1e-10
            
            # Apply logarithmic map (tanh^-1(r) * x/r)
            factor = torch.arctanh(norm.clamp(max=0.99)) / norm
            euclidean = embeddings * factor * mask
            
            # Process through adapter
            return self.adapter(euclidean)
        else:
            # No special adaptation needed for Euclidean embeddings
            return self.adapter(embeddings)


class UnifiedCitationNetworkModel(nn.Module):
    """Unified model integrating encoder, predictor, and generator components.
    
    This model implements the complete pipeline:
    1. Encoder processes the graph to create node embeddings
    2. These embeddings are used by the AttentionPredictor for link prediction
    3. The same embeddings are used by the generator for creating new nodes
    
    All components share a common device manager for consistent device usage.
    """
    
    def __init__(self,
                 encoder: BaseEncoder,
                 predictor: AttentionPredictor,
                 generator: EnhancedCVAEGenerator,
                 device_manager: Optional[DeviceManager] = None):
        """Initialize the unified model.
        
        Args:
            encoder (BaseEncoder): Graph encoder module
            predictor (AttentionPredictor): Edge prediction module
            generator (EnhancedCVAEGenerator): Node generation module
            device_manager (Optional[DeviceManager]): Device manager for consistent
                device placement. If None, a new one is created.
        """
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.generator = generator
        
        # Setup device management
        if device_manager is None:
            self.device_manager = DeviceManager()
        else:
            self.device_manager = device_manager
            
        # Register all modules with the device manager
        self.device_manager.register_module('encoder', encoder)
        self.device_manager.register_module('predictor', predictor)
        self.device_manager.register_module('generator', generator)
        
        # Determine encoder type for appropriate adaptation
        encoder_type = self._determine_encoder_type(encoder)
        use_hyperbolic = 'hyperbolic' in encoder_type.lower()
        
        # Create encoder adapter for proper embedding transformation
        self.encoder_adapter = EncoderAdapterLayer(
            embed_dim=encoder.embed_dim,
            encoder_type=encoder_type,
            use_hyperbolic=use_hyperbolic,
            curvature=getattr(encoder, 'curvature', 1.0)
        )
        
        # Register adapter with device manager
        self.device_manager.register_module('encoder_adapter', self.encoder_adapter)
    
    def _determine_encoder_type(self, encoder: BaseEncoder) -> str:
        """Determine the type of encoder.
        
        Args:
            encoder (BaseEncoder): The encoder to classify
            
        Returns:
            str: Encoder type identifier
        """
        if isinstance(encoder, TGNEncoder):
            return 'tgn'
        elif isinstance(encoder, HyperbolicGNN):
            return 'hyperbolic_gnn'
        elif isinstance(encoder, SimpleHyperbolicEncoder):
            return 'simple_hyperbolic'
        else:
            # Default case for unknown encoders
            return 'generic'
    
    def forward(self, 
                graph: GraphData,
                mode: str = 'full') -> Dict[str, Any]:
        """Forward pass through the unified model.
        
        Args:
            graph (GraphData): Input graph data
            mode (str): Operation mode:
                'full' - Run encoder, predictor, and generator
                'encode' - Run only the encoder
                'predict' - Run encoder and predictor
                'generate' - Run encoder and generator
                
        Returns:
            Dict[str, Any]: Dictionary containing outputs from the enabled components
        """
        results = {}
        
        # Always run encoder
        node_embeddings = self.encoder(graph)
        
        # Adapt embeddings for unified usage
        adapted_embeddings = self.encoder_adapter(node_embeddings)
        
        # Store in results
        results['embeddings'] = adapted_embeddings
        
        # Run predictor if requested
        if mode in ['full', 'predict']:
            # Predict citations using the attention predictor
            edge_scores = self.predictor(adapted_embeddings, graph)
            results['edge_scores'] = edge_scores
        
        # Run generator if requested
        if mode in ['full', 'generate']:
            # Create conditions based on graph data
            conditions = self._prepare_generator_conditions(graph)
            
            # Generate new papers
            generated_features, mu, logvar = self.generator(
                node_embeddings=adapted_embeddings,
                conditions=conditions
            )
            
            results['generated_features'] = generated_features
            results['mu'] = mu
            results['logvar'] = logvar
            results['conditions'] = conditions
        
        return results
    
    def _prepare_generator_conditions(self, graph: GraphData) -> Dict[str, torch.Tensor]:
        """Prepare conditioning inputs for the generator from graph data.
        
        Args:
            graph (GraphData): Input graph data
            
        Returns:
            Dict[str, torch.Tensor]: Conditioning inputs for the generator
        """
        conditions = {}
        
        # Extract timestamps if available
        if hasattr(graph, 'node_timestamps') and graph.node_timestamps is not None:
            conditions['timestamps'] = graph.node_timestamps
        elif hasattr(graph, 'timestamps') and graph.timestamps is not None:
            conditions['timestamps'] = graph.timestamps
        
        # Extract topic information if available
        if hasattr(graph, 'node_topics') and graph.node_topics is not None:
            conditions['topic_ids'] = graph.node_topics
        
        return conditions
    
    def predict_links(self, 
                     graph: GraphData,
                     src_nodes: Optional[torch.Tensor] = None,
                     dst_nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict links between nodes.
        
        Args:
            graph (GraphData): Input graph data
            src_nodes (Optional[torch.Tensor]): Source node indices
            dst_nodes (Optional[torch.Tensor]): Target node indices
            
        Returns:
            torch.Tensor: Edge prediction scores
        """
        # Encode the graph
        node_embeddings = self.encoder(graph)
        
        # Adapt embeddings
        adapted_embeddings = self.encoder_adapter(node_embeddings)
        
        # Predict links
        if src_nodes is not None and dst_nodes is not None:
            # Predict specific links
            edge_scores = self.predictor.predict_edge_scores(
                node_embeddings=adapted_embeddings,
                src_nodes=src_nodes,
                dst_nodes=dst_nodes
            )
        else:
            # Predict all links
            edge_scores = self.predictor(adapted_embeddings, graph)
        
        return edge_scores
    
    def generate_papers(self,
                        graph: GraphData,
                        num_papers: int,
                        conditions: Optional[Dict[str, torch.Tensor]] = None,
                        temperature: float = 1.0) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Generate new papers.
        
        Args:
            graph (GraphData): Input graph data
            num_papers (int): Number of papers to generate
            conditions (Optional[Dict[str, torch.Tensor]]): Additional conditions
            temperature (float): Sampling temperature
            
        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]:
                - Generated features
                - Paper metadata
        """
        # Encode the graph
        node_embeddings = self.encoder(graph)
        
        # Adapt embeddings
        adapted_embeddings = self.encoder_adapter(node_embeddings)
        
        # Merge with graph conditions
        graph_conditions = self._prepare_generator_conditions(graph)
        if conditions is not None:
            graph_conditions.update(conditions)
        
        # Generate papers
        return self.generator.generate_papers_for_network(
            graph=graph,
            encoder_embeddings=adapted_embeddings,
            num_papers=num_papers,
            conditions=graph_conditions
        )
    
    def generate_temporal_papers(self,
                                graph: GraphData,
                                time_threshold: float,
                                future_window: float,
                                num_papers: int,
                                conditions: Optional[Dict[str, torch.Tensor]] = None,
                                temperature: float = 1.0) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Generate papers for a future time window.
        
        Args:
            graph (GraphData): Input graph data
            time_threshold (float): Start time for generation
            future_window (float): Window of time to generate for
            num_papers (int): Number of papers to generate
            conditions (Optional[Dict[str, torch.Tensor]]): Additional conditions
            temperature (float): Sampling temperature
            
        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]:
                - Generated features
                - Paper metadata
        """
        # Encode the graph
        node_embeddings = self.encoder(graph)
        
        # Adapt embeddings
        adapted_embeddings = self.encoder_adapter(node_embeddings)
        
        # Merge with graph conditions
        graph_conditions = self._prepare_generator_conditions(graph)
        if conditions is not None:
            graph_conditions.update(conditions)
        
        # Generate temporal papers
        return self.generator.generate_temporal_papers(
            graph=graph,
            encoder_embeddings=adapted_embeddings,
            time_threshold=time_threshold,
            future_window=future_window,
            num_papers=num_papers,
            conditions=graph_conditions
        )
    
    def train_generator(self,
                       graph: GraphData,
                       node_features: torch.Tensor,
                       optimizer: torch.optim.Optimizer,
                       batch_size: int = 32,
                       kl_weight: float = 1.0) -> Dict[str, float]:
        """Train the generator component.
        
        Args:
            graph (GraphData): Input graph data
            node_features (torch.Tensor): Ground truth node features
            optimizer (torch.optim.Optimizer): Optimizer for training
            batch_size (int): Training batch size
            kl_weight (float): Weight for KL divergence loss
            
        Returns:
            Dict[str, float]: Training losses
        """
        # Set to training mode
        self.generator.train()
        
        # Encode the graph
        node_embeddings = self.encoder(graph)
        
        # Adapt embeddings
        adapted_embeddings = self.encoder_adapter(node_embeddings)
        
        # Prepare conditions
        conditions = self._prepare_generator_conditions(graph)
        
        # Train in batches if necessary
        num_nodes = node_features.size(0)
        
        losses = {
            'loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0
        }
        
        if num_nodes <= batch_size:
            # Single batch training
            optimizer.zero_grad()
            
            # Compute loss
            batch_losses = self.generator.compute_loss(
                node_features=node_features,
                node_embeddings=adapted_embeddings,
                conditions=conditions
            )
            
            # Backward pass
            batch_losses['loss'].backward()
            optimizer.step()
            
            # Update losses
            for k, v in batch_losses.items():
                losses[k] = v.item()
        else:
            # Multi-batch training
            num_batches = (num_nodes + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_nodes)
                
                # Get batch
                batch_features = node_features[start_idx:end_idx]
                batch_embeddings = adapted_embeddings[start_idx:end_idx]
                
                # Get batch conditions
                batch_conditions = {}
                for k, v in conditions.items():
                    if v is not None:
                        batch_conditions[k] = v[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                # Compute loss
                batch_losses = self.generator.compute_loss(
                    node_features=batch_features,
                    node_embeddings=batch_embeddings,
                    conditions=batch_conditions
                )
                
                # Backward pass
                batch_losses['loss'].backward()
                optimizer.step()
                
                # Update losses
                batch_weight = (end_idx - start_idx) / num_nodes
                for k, v in batch_losses.items():
                    losses[k] += v.item() * batch_weight
        
        return losses
    
    def to(self, device: torch.device) -> 'UnifiedCitationNetworkModel':
        """Move the model to the specified device.
        
        Args:
            device (torch.device): Target device
            
        Returns:
            UnifiedCitationNetworkModel: Self with updated device
        """
        super().to(device)
        
        # Update device manager
        self.device_manager.sync_modules(device)
        
        return self


def create_unified_model(
    encoder_type: str,
    encoder_config: Dict[str, Any],
    predictor_config: Dict[str, Any],
    generator_config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> UnifiedCitationNetworkModel:
    """Factory function to create a unified model.
    
    Args:
        encoder_type (str): Type of encoder to use
        encoder_config (Dict[str, Any]): Encoder configuration
        predictor_config (Dict[str, Any]): Predictor configuration
        generator_config (Dict[str, Any]): Generator configuration
        device (Optional[torch.device]): Target device
        
    Returns:
        UnifiedCitationNetworkModel: Instantiated unified model
    """
    # Create device manager
    device_manager = DeviceManager(device)
    
    # Create encoder based on type
    if encoder_type == 'tgn':
        encoder = TGNEncoder(**encoder_config)
    elif encoder_type == 'hyperbolic_gnn':
        encoder = HyperbolicGNN(**encoder_config)
    elif encoder_type == 'simple_hyperbolic':
        encoder = SimpleHyperbolicEncoder(**encoder_config)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Create attention predictor
    predictor = AttentionPredictor(**predictor_config)
    
    # Create enhanced generator
    # Make sure node_feature_dim matches the encoder's node dimension
    if 'node_feature_dim' not in generator_config and 'node_dim' in encoder_config:
        generator_config['node_feature_dim'] = encoder_config['node_dim']
    
    # Ensure embed_dim matches between components
    if 'embed_dim' not in generator_config and 'embed_dim' in encoder_config:
        generator_config['embed_dim'] = encoder_config['embed_dim']
    
    generator = EnhancedCVAEGenerator(**generator_config)
    
    # Create and return unified model
    model = UnifiedCitationNetworkModel(
        encoder=encoder,
        predictor=predictor,
        generator=generator,
        device_manager=device_manager
    )
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    return model 