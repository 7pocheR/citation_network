# Enhanced MLPPredictor for Citation Network Modeling

This directory contains predictor implementations for the Citation Network Modeling project. The enhanced MLPPredictor is a sophisticated neural network-based predictor that uses embedding pairs to predict citation probabilities.

## Overview

The enhanced MLPPredictor improves upon the original implementation with several new features:

1. **Dimension Adaptation**: Automatically handles embeddings of different dimensions
2. **Feature Interaction Layer**: Creates sophisticated interactions between source and destination embeddings
3. **Residual Connections**: Improves gradient flow with residual blocks
4. **Metadata Integration**: Incorporates paper metadata (topics, keywords) into prediction
5. **Advanced Activation Functions**: Uses SiLU (Swish) or Mish for better performance
6. **Normalization Options**: Options for both batch and layer normalization
7. **Edge Case Handling**: Robust handling of empty graphs, single nodes, etc.

## Components

### AdaptiveInputLayer

This layer handles embeddings of different dimensions, allowing the MLPPredictor to work with different encoder outputs without requiring manual dimension matching.

```python
# Example: Creating an adaptive layer that transforms to 64-dim embeddings
adapter = AdaptiveInputLayer(target_dim=64)

# Using with various input dimensions
output_32 = adapter(torch.randn(10, 32))  # [10, 64]
output_128 = adapter(torch.randn(10, 128))  # [10, 64]
```

### FeatureInteractionLayer

Creates sophisticated interactions between source and destination embeddings to capture complex citation patterns.

```python
# Example: Creating an interaction layer
interaction = FeatureInteractionLayer(embed_dim=32, interaction_dim=16)

# Using the layer
src_emb = torch.randn(10, 32)
dst_emb = torch.randn(10, 32)
features = interaction(src_emb, dst_emb)  # [10, 65]
```

### NodeMetadataProcessor

Processes node metadata (topics, keywords) to incorporate into prediction.

```python
# Example: Creating a metadata processor
metadata_dims = {'topics': 10, 'keywords': 15}
processor = NodeMetadataProcessor(metadata_dims, output_dim=32)

# Using the processor
metadata_features = processor(graph, src_indices, dst_indices)
```

### ResidualMLPBlock

Enhances gradient flow with residual connections and advanced activations.

```python
# Example: Creating a residual block with Mish activation
block = ResidualMLPBlock(
    input_dim=64,
    hidden_dim=32,
    use_layer_norm=True,
    use_batch_norm=True,
    activation='mish'
)
```

## Usage

### Basic Usage

```python
# Create a basic enhanced predictor
predictor = MLPPredictor(
    embed_dim=64,
    hidden_dims=[128, 64, 32],
    dropout=0.2,
    use_batch_norm=True,
    use_layer_norm=True,
    activation='silu'
)

# Predict a single pair
score = predictor(src_embedding, dst_embedding)

# Predict batch of pairs
scores = predictor.predict_batch(node_embeddings, edge_indices)

# Find top k predictions
top_edges, top_scores = predictor.predict_citations(
    node_embeddings, 
    existing_graph, 
    k=10
)
```

### With Metadata

```python
# Define metadata dimensions
metadata_fields = {
    'topics': 10,  # 10-dimensional topic vectors
    'keywords': 15  # 15-dimensional keyword vectors
}

# Create predictor with metadata support
predictor = MLPPredictor(
    embed_dim=64,
    metadata_fields=metadata_fields
)

# Predict with metadata
scores = predictor.predict_batch(
    node_embeddings, 
    edge_indices, 
    graph_data=graph_with_metadata
)
```

### Temporal Prediction

```python
# Predict future citations based on temporal snapshot
future_edges, future_scores = predictor.predict_temporal_citations(
    node_embeddings,
    existing_graph,
    time_threshold=2020.0,
    future_window=2.0,  # Look 2 years into the future
    k=10
)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_dim` | int | required | Base dimensionality for node embeddings |
| `hidden_dims` | List[int] | [256, 128, 64] | Dimensions of hidden layers |
| `dropout` | float | 0.2 | Dropout rate for regularization |
| `use_batch_norm` | bool | True | Whether to use batch normalization |
| `use_layer_norm` | bool | True | Whether to use layer normalization |
| `activation` | str | 'silu' | Activation function ('relu', 'silu', 'mish') |
| `interaction_dim` | int | 64 | Dimension for feature interactions |
| `metadata_fields` | Dict[str, int] | None | Dictionary mapping metadata field names to dimensions |

## Performance

The enhanced MLPPredictor generally shows improved performance over the basic implementation, especially for complex citation patterns. Key improvements include:

1. Better handling of non-transitive relations due to the sophisticated feature interaction layer
2. Improved gradient flow due to residual connections
3. More efficient information extraction with SiLU/Mish activations
4. Better utilization of metadata when available
5. Robust handling of dimension mismatches across different encoder outputs 