import torch
import numpy as np
import os
import pickle
import logging
import time
from typing import Optional, Tuple, Dict, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class OptimizedFeaturePreprocessor:
    """
    Optimized feature preprocessing for paper features with standardization and dimensionality reduction.
    
    This class handles preprocessing of paper features with several optimizations:
    1. Uses more efficient PCA algorithms (Incremental PCA, Truncated SVD)
    2. Implements optional initial dimensionality reduction via random projection
    3. Provides caching to avoid recomputing PCA
    4. Includes performance logging
    
    The preprocessor can be saved and loaded to ensure consistent transformations
    across training and inference.
    """
    
    def __init__(self, 
                 n_components: Optional[Union[int, float]] = None, 
                 standardize: bool = True,
                 reduction_method: str = 'pca',  # 'pca', 'ipca', 'svd'
                 random_projection_dim: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 batch_size: int = 1000,
                 random_state: int = 42):
        """
        Initialize the optimized feature preprocessor.
        
        Args:
            n_components: Number of components to keep. If float between 0 and 1,
                          it represents the proportion of variance to be retained.
                          If None, no dimensionality reduction is performed.
            standardize: Whether to standardize features before dimensionality reduction
            reduction_method: Method for dimensionality reduction:
                             'pca': Standard PCA (good for small datasets)
                             'ipca': Incremental PCA (good for large datasets)
                             'svd': Truncated SVD (faster, works with sparse data)
            random_projection_dim: If set, apply random projection to this dimension first
                                  before the main dimensionality reduction
            cache_dir: Directory to cache fitted preprocessors
            batch_size: Batch size for incremental PCA
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.standardize = standardize
        self.reduction_method = reduction_method
        self.random_projection_dim = random_projection_dim
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Initialize cache directory if provided
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize preprocessors
        self.scaler = StandardScaler() if standardize else None
        self.random_projector = None
        
        # Initialize the appropriate dimension reduction method
        if n_components is not None:
            if reduction_method == 'ipca':
                # For incremental PCA, convert float n_components (variance ratio) to an integer
                # Since IncrementalPCA doesn't accept float values for n_components
                if isinstance(n_components, float) and 0 < n_components < 1:
                    # Use a reasonable default - 50% of features or 100, whichever is smaller
                    n_components_int = min(100, int(min(batch_size, 1000) * 0.5))
                    logger.info(f"Converting float n_components={n_components} to int={n_components_int} for IncrementalPCA")
                    self.n_components_target = n_components  # Save the target explained variance
                    self.reducer = IncrementalPCA(
                        n_components=n_components_int,
                        batch_size=batch_size
                    )
                else:
                    self.reducer = IncrementalPCA(
                        n_components=n_components,
                        batch_size=batch_size
                    )
            elif reduction_method == 'svd':
                self.reducer = TruncatedSVD(
                    n_components=n_components if isinstance(n_components, int) else 100,
                    random_state=random_state
                )
            else:  # Default to standard PCA
                self.reducer = PCA(
                    n_components=n_components,
                    random_state=random_state
                )
        else:
            self.reducer = None
        
        # Tracking state
        self.fitted = False
        self.input_dim = None
        self.transformed_dim = None
        self.explained_variance_ratio = None
        self.fit_time = None
    
    def _get_cache_path(self, features_shape):
        """Get path for cached preprocessor based on data characteristics."""
        if not self.cache_dir:
            return None
            
        # Create a cache key based on dataset shape and preprocessing parameters
        cache_key = f"preproc_{features_shape[0]}x{features_shape[1]}_{self.n_components}_{self.reduction_method}"
        if self.random_projection_dim:
            cache_key += f"_rp{self.random_projection_dim}"
        if self.standardize:
            cache_key += "_std"
            
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _check_cache(self, features_shape):
        """Check if a fitted preprocessor exists in cache."""
        cache_path = self._get_cache_path(features_shape)
        if cache_path and os.path.exists(cache_path):
            try:
                logger.info(f"Loading preprocessor from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    cached_preprocessor = pickle.load(f)
                    
                # Copy attributes from cached preprocessor
                self.scaler = cached_preprocessor.scaler
                self.random_projector = cached_preprocessor.random_projector
                self.reducer = cached_preprocessor.reducer
                self.fitted = cached_preprocessor.fitted
                self.input_dim = cached_preprocessor.input_dim
                self.transformed_dim = cached_preprocessor.transformed_dim
                self.explained_variance_ratio = cached_preprocessor.explained_variance_ratio
                self.fit_time = cached_preprocessor.fit_time
                
                return True
            except Exception as e:
                logger.warning(f"Error loading cached preprocessor: {str(e)}. Will fit a new one.")
        
        return False
    
    def _save_to_cache(self, features_shape):
        """Save fitted preprocessor to cache."""
        if not self.cache_dir:
            return
            
        cache_path = self._get_cache_path(features_shape)
        try:
            logger.info(f"Saving preprocessor to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            logger.warning(f"Error saving preprocessor to cache: {str(e)}")
    
    def fit(self, features: torch.Tensor) -> 'OptimizedFeaturePreprocessor':
        """
        Fit the preprocessor to the features.
        
        Args:
            features: Features to fit the preprocessor to [num_samples, feature_dim]
            
        Returns:
            Self for chaining
        """
        # Start timing
        start_time = time.time()
        
        # Convert to numpy for preprocessing
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        # Store original dimension
        self.input_dim = features.shape[1]
        
        # Check cache first
        if self._check_cache(features.shape):
            logger.info(f"Using cached preprocessor (fit time was {self.fit_time:.2f}s)")
            return self
            
        logger.info(f"Fitting preprocessor to features with shape {features.shape}")
        
        # Apply standardization if requested
        if self.standardize:
            logger.info("Applying standardization")
            features = self.scaler.fit_transform(features)
        
        # Apply random projection if requested (for initial dimensionality reduction)
        if self.random_projection_dim and self.random_projection_dim < features.shape[1]:
            logger.info(f"Applying random projection to {self.random_projection_dim} dimensions")
            self.random_projector = SparseRandomProjection(
                n_components=self.random_projection_dim,
                random_state=self.random_state
            )
            features = self.random_projector.fit_transform(features)
            logger.info(f"Random projection reduced dimension from {self.input_dim} to {features.shape[1]}")
        
        # Apply main dimensionality reduction if requested
        if self.reducer is not None:
            reduction_start = time.time()
            
            if self.reduction_method == 'ipca':
                # For large datasets, use incremental PCA with batches
                batch_size = min(self.batch_size, features.shape[0])
                
                # Initialize with first batch
                logger.info(f"Fitting Incremental PCA with batch size {batch_size}")
                
                for i in range(0, features.shape[0], batch_size):
                    end_idx = min(i + batch_size, features.shape[0])
                    if i == 0:
                        self.reducer.partial_fit(features[i:end_idx])
                    else:
                        self.reducer.partial_fit(features[i:end_idx])
                        
                features = self.reducer.transform(features)
            else:
                # Standard PCA or SVD
                logger.info(f"Fitting {self.reduction_method.upper()}")
                features = self.reducer.fit_transform(features)
            
            reduction_time = time.time() - reduction_start
            logger.info(f"Dimension reduction took {reduction_time:.2f}s")
            
            # Store dimensionality reduction results
            self.transformed_dim = features.shape[1]
            
            if hasattr(self.reducer, 'explained_variance_ratio_'):
                self.explained_variance_ratio = self.reducer.explained_variance_ratio_
                total_variance = sum(self.explained_variance_ratio)
                logger.info(f"Reduced dimension from {self.input_dim} to {self.transformed_dim} "
                           f"(explained variance: {total_variance:.2%})")
            else:
                logger.info(f"Reduced dimension from {self.input_dim} to {self.transformed_dim}")
        else:
            self.transformed_dim = features.shape[1]
            logger.info(f"No dimension reduction applied, keeping {self.transformed_dim} features")
        
        # Record fit time
        self.fit_time = time.time() - start_time
        logger.info(f"Preprocessing completed in {self.fit_time:.2f}s")
        
        # Set fitted flag and save to cache
        self.fitted = True
        self._save_to_cache(features.shape)
        
        return self
    
    def transform(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Transform features using fitted preprocessor.
        
        Args:
            features: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted yet. Call fit() first.")
        
        # Convert to numpy if tensor
        is_tensor = torch.is_tensor(features)
        original_device = features.device if is_tensor else None
        original_dtype = features.dtype if is_tensor else None
        
        if is_tensor:
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features
            
        # Standardize if needed
        if self.standardize and self.scaler is not None:
            features_np = self.scaler.transform(features_np)
            
        # Apply dimension reduction
        if self.reducer is not None:
            transformed_features = self.reducer.transform(features_np)
        else:
            transformed_features = features_np
            
        # Apply random projection if specified
        if self.random_projector is not None:
            transformed_features = self.random_projector.transform(transformed_features)
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            transformed_features = torch.tensor(
                transformed_features, 
                dtype=torch.float32,  # Always use float32 for consistency
                device=original_device
            )
        
        return transformed_features
    
    def fit_transform(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Fit preprocessor to features and transform them.
        
        Args:
            features: Features to fit and transform [num_samples, feature_dim]
            
        Returns:
            Transformed features
        """
        return self.fit(features).transform(features)
    
    def inverse_transform(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Reverse the transformation to recover original features.
        
        Args:
            features: Transformed features [batch_size, transformed_dim]
            
        Returns:
            Approximately reconstructed original features
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
            
        # Check input type
        is_torch = isinstance(features, torch.Tensor)
        if is_torch:
            device = features.device
            features = features.detach().cpu().numpy()
            
        # Apply inverse transforms in reverse order
        if self.reducer is not None:
            features = self.reducer.inverse_transform(features)
            
        if self.random_projector is not None:
            # Random projection is lossy, this is approximate
            logger.warning("Inverting random projection is approximate and lossy")
            if hasattr(self.random_projector, 'inverse_transform'):
                features = self.random_projector.inverse_transform(features)
            
        if self.standardize and self.scaler is not None:
            features = self.scaler.inverse_transform(features)
            
        # Convert back to torch if needed
        if is_torch:
            features = torch.tensor(features, device=device)
            
        return features
    
    def save(self, filepath: str) -> None:
        """
        Save preprocessor to disk.
        
        Args:
            filepath: Path to save the preprocessor
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
        logger.info(f"OptimizedFeaturePreprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'OptimizedFeaturePreprocessor':
        """
        Load preprocessor from disk.
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor
        """
        # Load using pickle
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
            
        logger.info(f"OptimizedFeaturePreprocessor loaded from {filepath}")
        return preprocessor 