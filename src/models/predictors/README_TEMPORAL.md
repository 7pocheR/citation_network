# Enhanced TemporalPredictor

The Enhanced TemporalPredictor is a sophisticated citation prediction model that explicitly incorporates temporal dynamics into the prediction process. It is designed to capture complex temporal patterns in citation networks, including recency bias, citation velocity, and domain-specific temporal patterns.

## Key Features

### 1. Advanced Temporal Encoding

The TemporalPredictor uses a multi-strategy approach to encode time differences between papers:

- **Linear Encoding**: Simple linear transformation of time differences
- **Log-scaled Encoding**: Better captures varying time scales by using logarithmic transformation
- **Periodic Encoding**: Captures seasonal/cyclical patterns in publication and citation behavior
- **Multi-mode Combination**: Combines different encoding strategies for richer temporal representations

### 2. Recency Bias Modeling

Research shows that papers tend to cite recent work more frequently. The TemporalPredictor models this recency bias with:

- **Parameterized Decay Functions**: Learnable parameters control how citation likelihood decays with time
- **Domain-specific Recency**: Different research fields have different citation patterns and decay rates
- **Adaptive Recency**: The model can learn optimal recency parameters during training

### 3. Citation Velocity Awareness

Some papers gain citations more rapidly than others. The TemporalPredictor tracks this "citation velocity":

- **Velocity Tracking**: Measures how quickly papers accumulate citations over time
- **Rising Star Detection**: Identifies papers gaining citations rapidly
- **Momentum-based Prediction**: Uses citation momentum to predict future citation patterns

### 4. Temporal Prediction Horizons

The model supports different prediction time horizons:

- **Configurable Future Windows**: Predict citations within specific time windows
- **Confidence Estimation**: Optional confidence scores for predictions
- **Temporal Constraints**: Ensures predictions respect temporal ordering (papers can only cite earlier papers)

### 5. Domain-Time Interaction

Different research domains have different temporal citation patterns:

- **Field-specific Decay Rates**: Models how citation patterns vary across research fields
- **Domain Weighting**: Configurable weights for different domains
- **Hybrid Modeling**: Combines domain knowledge with temporal patterns

### 6. Temporal Robustness

The model is designed to handle challenges in temporal data:

- **Sparsity Handling**: Graceful handling of sparse temporal data
- **Adaptive Sampling**: Stratified sampling by time period
- **Fallback Mechanisms**: Robust prediction even with limited history

## Usage Examples

### Basic Usage

```python
from src.models.predictors.temporal_predictor import TemporalPredictor

# Initialize the predictor
predictor = TemporalPredictor(
    embed_dim=128,
    time_encoding_dim=32,
    hidden_dims=[256, 128],
    dropout=0.2
)

# Predict citations
predicted_edges, scores = predictor.predict_citations(
    node_embeddings=embeddings,
    existing_graph=graph_data,
    k=10
)
```

### Temporal Prediction with Future Window

```python
# Predict citations that will form in the next 2 time units
predicted_edges, scores = predictor.predict_temporal_citations(
    node_embeddings=embeddings,
    existing_graph=graph_data,
    time_threshold=current_time,
    future_window=2.0,
    k=10
)
```

### Domain-specific Recency

```python
# Initialize with domain-specific recency
predictor = TemporalPredictor(
    embed_dim=128,
    time_encoding_dim=32,
    domain_specific_recency=True,
    num_domains=5,
    learn_recency_params=True
)

# Predict with domain weights
predicted_edges, scores = predictor.predict_citations(
    node_embeddings=embeddings,
    existing_graph=graph_data,
    k=10,
    domain_weights=[1.2, 0.8, 1.0, 1.5, 0.9]  # Weight different domains
)
```

### Citation Velocity Tracking

```python
# Initialize with citation velocity tracking
predictor = TemporalPredictor(
    embed_dim=128,
    use_citation_velocity=True,
    velocity_encoding_dim=16
)

# Predict with custom velocity parameters
predicted_edges, scores = predictor.predict_citations(
    node_embeddings=embeddings,
    existing_graph=graph_data,
    k=10,
    time_window=0.5,  # Look at citations in the last 0.5 time units
    min_citations=3   # Minimum citations for velocity calculation
)
```

### Confidence Estimation

```python
# Initialize with confidence estimation
predictor = TemporalPredictor(
    embed_dim=128,
    confidence_estimation=True
)

# Predict with confidence threshold
predicted_edges, scores = predictor.predict_temporal_citations(
    node_embeddings=embeddings,
    existing_graph=graph_data,
    time_threshold=current_time,
    k=10,
    confidence_threshold=0.7  # Only return high-confidence predictions
)

# scores will be a tensor of shape [k, 2] with prediction and confidence scores
predictions = scores[:, 0]
confidence = scores[:, 1]
```

## Configuration Parameters

### Basic Parameters

- `embed_dim`: Dimensionality of node embeddings
- `time_encoding_dim`: Dimensionality of time encoding
- `hidden_dims`: Dimensions of hidden layers
- `dropout`: Dropout rate
- `use_batch_norm`: Whether to use batch normalization

### Temporal Encoding Parameters

- `num_encoding_modes`: Number of encoding modes/strategies to combine
- `use_log_scale`: Whether to use log-scaled time encoding
- `use_periodic`: Whether to use periodic time encoding

### Recency Bias Parameters

- `recency_bias`: Whether to use recency bias modeling
- `domain_specific_recency`: Whether to learn domain-specific recency factors
- `num_domains`: Number of research domains for domain-specific parameters
- `learn_recency_params`: Whether to learn recency parameters or use fixed values
- `default_decay_factor`: Default decay factor when not learning

### Citation Velocity Parameters

- `use_citation_velocity`: Whether to use citation velocity features
- `velocity_encoding_dim`: Dimensionality of velocity encoding

### Temporal Prediction Parameters

- `confidence_estimation`: Whether to estimate prediction confidence

### Device and Threading Parameters

- `cache_size`: Size of cache for dimension handling

## Implementation Details

### Temporal Encoding Layer

The `TemporalEncodingLayer` transforms time differences into rich temporal feature representations using multiple encoding strategies:

```python
# Example of how time is encoded
time_encoding = temporal_encoder(time_diff)
```

### Recency Bias Application

Recency bias is applied to prediction scores:

```python
# Example of how recency bias is applied
adjusted_scores = apply_recency_bias(scores, time_diff, domain_ids)
```

### Citation Velocity Calculation

Citation velocity is calculated based on recent citation patterns:

```python
# Example of how citation velocity is calculated
velocities = _calculate_citation_velocities(
    existing_graph, 
    candidate_edges,
    time_window=1.0,
    min_citations=5
)
```

## Performance Considerations

- The model uses batched processing to handle large citation networks efficiently
- Adaptive input layers handle embeddings of varying dimensions
- Thread-safe caching improves performance for repeated operations
- Domain identification uses caching to avoid redundant computation

## Limitations and Future Work

- The current domain identification is a placeholder and should be replaced with actual paper category information
- Citation velocity calculation could be optimized for very large graphs
- More sophisticated temporal patterns (e.g., seasonal effects) could be incorporated
- Integration with hyperbolic embeddings could better capture hierarchical citation structures 