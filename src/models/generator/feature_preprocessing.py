import torch
import numpy as np
import os
import pickle
import logging
from typing import Optional, Tuple, Dict, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class FeaturePreprocessor:
    """
    Feature preprocessing for paper features with standardization and PCA.
    
    This class handles preprocessing of paper features before feeding them to the CVAE:
    1. Standardization (centering and scaling to unit variance)
    2. PCA transformation to create orthogonal feature dimensions
    
    The preprocessor can be saved and loaded to ensure consistent transformations
    across training and inference.
    """
    
    def __init__(self, 
                 n_components: Optional[Union[int, float]] = None, 
                 standardize: bool = True,
                 random_state: int = 42):
        """
        Initialize the feature preprocessor.
        
        Args:
            n_components: Number of PCA components to keep. If float between 0 and 1,
                          it represents the proportion of variance to be retained.
                          If None, no dimensionality reduction is performed.
            standardize: Whether to standardize features before PCA
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.standardize = standardize
        self.random_state = random_state
        
        # Initialize preprocessors
        self.pca = PCA(n_components=n_components, random_state=random_state) if n_components is not None else None
        self.scaler = StandardScaler() if standardize else None
        
        # Tracking state
        self.fitted = False
        self.input_dim = None
        self.transformed_dim = None
        self.explained_variance_ratio = None
    
    def fit(self, features: torch.Tensor) -> 'FeaturePreprocessor':
        """
        Fit the preprocessor to the features.
        
        Args:
            features: Features to fit the preprocessor to [num_samples, feature_dim]
            
        Returns:
            Self for chaining
        """
        # Convert to numpy for preprocessing
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        # Store original dimension
        self.input_dim = features.shape[1]
        
        # Apply standardization if requested
        if self.standardize:
            logger.info(f"Fitting standardization to features with shape {features.shape}")
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        
        # Apply PCA if requested
        if self.n_components is not None:
            logger.info(f"Fitting PCA to features with n_components={self.n_components}")
            self.pca = PCA(n_components=self.n_components)
            features = self.pca.fit_transform(features)
            self.transformed_dim = features.shape[1]
            logger.info(f"PCA reduced dimension from {self.input_dim} to {self.transformed_dim}")
        else:
            self.transformed_dim = self.input_dim
            logger.info(f"No dimension reduction applied, keeping {self.transformed_dim} features")
        
        # Set fitted flag
        self.fitted = True
        
        return self
    
    def transform(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Transform features using fitted preprocessor.
        
        Args:
            features: Features to transform [batch_size, feature_dim]
            
        Returns:
            Transformed features [batch_size, transformed_dim]
            
        Raises:
            ValueError: If the preprocessor is not fitted
        """
        if not self.fitted:
            raise ValueError("Feature preprocessor must be fitted before transformation")
        
        # Check if features is a tensor and remember its device
        is_tensor = isinstance(features, torch.Tensor)
        if is_tensor:
            original_device = features.device
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features
        
        # Apply standardization if used during fitting
        if self.standardize and self.scaler is not None:
            features_np = self.scaler.transform(features_np)
        
        # Apply PCA if used during fitting
        if self.pca is not None:
            transformed = self.pca.transform(features_np)
        else:
            transformed = features_np
        
        # Convert back to tensor if input was a tensor
        if is_tensor:
            transformed = torch.tensor(transformed, dtype=torch.float32, device=original_device)
        
        return transformed
    
    def inverse_transform(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Transform features back to original space.
        
        Args:
            features: Transformed features [batch_size, transformed_dim]
            
        Returns:
            Features in original space [batch_size, original_dim]
            
        Raises:
            ValueError: If the preprocessor is not fitted
        """
        if not self.fitted:
            raise ValueError("Feature preprocessor must be fitted before inverse transformation")
        
        # Check if features is a tensor and remember its device
        is_tensor = isinstance(features, torch.Tensor)
        if is_tensor:
            original_device = features.device
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features
        
        # Apply inverse PCA if used during fitting
        if self.pca is not None:
            features_np = self.pca.inverse_transform(features_np)
        
        # Apply inverse standardization if used during fitting
        if self.standardize and self.scaler is not None:
            features_np = self.scaler.inverse_transform(features_np)
        
        # Convert back to tensor if input was a tensor
        if is_tensor:
            features_np = torch.tensor(features_np, dtype=torch.float32, device=original_device)
        
        return features_np
    
    def fit_transform(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Fit the preprocessor and transform features in one step.
        
        Args:
            features: Paper features tensor or array
            
        Returns:
            Transformed features in the same format as input
        """
        self.fit(features)
        return self.transform(features)
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to a file.
        
        Args:
            filepath: Path to save the preprocessor
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {
            'n_components': self.n_components,
            'standardize': self.standardize,
            'random_state': self.random_state,
            'fitted': self.fitted,
            'input_dim': self.input_dim,
            'transformed_dim': self.transformed_dim,
            'explained_variance_ratio': self.explained_variance_ratio,
            'pca': self.pca,
            'scaler': self.scaler
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"FeaturePreprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeaturePreprocessor':
        """
        Load a fitted preprocessor from a file.
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            Loaded FeaturePreprocessor
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create a new instance
        preprocessor = cls(
            n_components=state['n_components'],
            standardize=state['standardize'],
            random_state=state['random_state']
        )
        
        # Restore state
        preprocessor.fitted = state['fitted']
        preprocessor.input_dim = state['input_dim']
        preprocessor.transformed_dim = state['transformed_dim']
        preprocessor.explained_variance_ratio = state['explained_variance_ratio']
        preprocessor.pca = state['pca']
        preprocessor.scaler = state['scaler']
        
        logger.info(f"FeaturePreprocessor loaded from {filepath}")
        return preprocessor
    
    def visualize_explained_variance(self, filepath: Optional[str] = None) -> plt.Figure:
        """
        Visualize the explained variance ratio from PCA.
        
        Args:
            filepath: If provided, save the plot to this path
            
        Returns:
            Matplotlib figure object
        """
        if not self.fitted or self.pca is None:
            raise ValueError("PCA must be fitted before visualization")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Cumulative explained variance
        cumulative = np.cumsum(self.pca.explained_variance_ratio_)
        
        # Component indices
        components = range(1, len(self.pca.explained_variance_ratio_) + 1)
        
        # Plot individual and cumulative explained variance
        ax.bar(components, self.pca.explained_variance_ratio_, alpha=0.5, label='Individual')
        ax.step(components, cumulative, where='mid', label='Cumulative')
        
        # Add threshold line
        if isinstance(self.n_components, float):
            ax.axhline(y=self.n_components, color='r', linestyle='--', 
                      label=f'Threshold ({self.n_components:.2f})')
        
        # Formatting
        ax.set_xlabel('Principal Components')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Explained Variance')
        ax.legend(loc='best')
        ax.grid(True)
        
        # Save if filepath provided
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"PCA explained variance plot saved to {filepath}")
        
        return fig
    
    def visualize_feature_correlation(self, 
                                     original_features: Union[torch.Tensor, np.ndarray],
                                     transformed_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
                                     feature_names: Optional[list] = None,
                                     n_features: int = 10,
                                     filepath: Optional[str] = None) -> plt.Figure:
        """
        Visualize feature correlations before and after transformation.
        
        Args:
            original_features: Original features before transformation
            transformed_features: Transformed features (if None, will be computed)
            feature_names: Names of the features (if available)
            n_features: Number of features to display (for readability)
            filepath: If provided, save the plot to this path
            
        Returns:
            Matplotlib figure object
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before visualization")
        
        # Convert to numpy if tensor
        if isinstance(original_features, torch.Tensor):
            original_np = original_features.detach().cpu().numpy()
        else:
            original_np = original_features
        
        # Transform if not provided
        if transformed_features is None:
            transformed_np = self.transform(original_np)
        elif isinstance(transformed_features, torch.Tensor):
            transformed_np = transformed_features.detach().cpu().numpy()
        else:
            transformed_np = transformed_features
        
        # Limit to n_features for readability
        original_subset = original_np[:, :n_features]
        transformed_subset = transformed_np[:, :min(n_features, transformed_np.shape[1])]
        
        # Create feature names if not provided
        if feature_names is None:
            orig_names = [f"Feature {i+1}" for i in range(original_subset.shape[1])]
            trans_names = [f"PC {i+1}" for i in range(transformed_subset.shape[1])]
        else:
            orig_names = feature_names[:n_features]
            trans_names = [f"PC {i+1}" for i in range(transformed_subset.shape[1])]
        
        # Calculate correlation matrices
        orig_corr = np.corrcoef(original_subset, rowvar=False)
        trans_corr = np.corrcoef(transformed_subset, rowvar=False)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot original correlation
        sns.heatmap(orig_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   xticklabels=orig_names, yticklabels=orig_names, ax=ax1)
        ax1.set_title('Original Feature Correlation')
        
        # Plot transformed correlation
        sns.heatmap(trans_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                   xticklabels=trans_names, yticklabels=trans_names, ax=ax2)
        ax2.set_title('Transformed Feature Correlation')
        
        plt.tight_layout()
        
        # Save if filepath provided
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Feature correlation plot saved to {filepath}")
        
        return fig
    
    def visualize_reconstruction(self,
                               original_features: Union[torch.Tensor, np.ndarray],
                               n_samples: int = 5,
                               n_features: int = 10,
                               feature_names: Optional[list] = None,
                               filepath: Optional[str] = None) -> plt.Figure:
        """
        Visualize original vs reconstructed features to assess information loss.
        
        Args:
            original_features: Original features before transformation
            n_samples: Number of samples to visualize
            n_features: Number of features to display per sample
            feature_names: Names of the features (if available)
            filepath: If provided, save the plot to this path
            
        Returns:
            Matplotlib figure object
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before visualization")
        
        # Convert to numpy if tensor
        if isinstance(original_features, torch.Tensor):
            original_np = original_features.detach().cpu().numpy()
        else:
            original_np = original_features
        
        # Select random samples
        indices = np.random.choice(original_np.shape[0], size=min(n_samples, original_np.shape[0]), replace=False)
        samples = original_np[indices]
        
        # Transform and inverse transform
        transformed = self.transform(samples)
        reconstructed = self.inverse_transform(transformed)
        
        # Limit features for display
        samples = samples[:, :n_features]
        reconstructed = reconstructed[:, :n_features]
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(n_features)]
        else:
            feature_names = feature_names[:n_features]
        
        # Create plot
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for i, (ax, orig, recon) in enumerate(zip(axes, samples, reconstructed)):
            # Calculate reconstruction error
            error = np.mean(np.abs(orig - recon))
            
            # Set up bar positions
            x = np.arange(len(feature_names))
            width = 0.35
            
            # Plot original and reconstructed
            ax.bar(x - width/2, orig, width, label='Original')
            ax.bar(x + width/2, recon, width, label='Reconstructed')
            
            # Formatting
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_title(f'Sample {i+1} - Mean Abs Error: {error:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if filepath provided
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Reconstruction visualization saved to {filepath}")
        
        return fig
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        return {
            'n_components': self.n_components,
            'standardize': self.standardize,
            'random_state': self.random_state,
            'input_dim': self.input_dim,
            'transformed_dim': self.transformed_dim
        }
    
    def __repr__(self) -> str:
        """String representation of the preprocessor."""
        status = "fitted" if self.fitted else "not fitted"
        if self.fitted:
            return (f"FeaturePreprocessor(n_components={self.n_components}, "
                   f"standardize={self.standardize}, {status}, "
                   f"input_dim={self.input_dim}, transformed_dim={self.transformed_dim})")
        else:
            return (f"FeaturePreprocessor(n_components={self.n_components}, "
                   f"standardize={self.standardize}, {status})")
    
    def get_output_dim(self) -> int:
        """
        Get the dimension of the transformed features.
        
        Returns:
            The dimension of features after preprocessing
        
        Raises:
            ValueError: If the preprocessor is not fitted yet
        """
        if not self.fitted:
            raise ValueError("Feature preprocessor is not fitted yet")
        
        if self.n_components is None:
            # If n_components is None, the output dim is the same as input
            return self.input_dim
        
        # For integer n_components, return that value
        if isinstance(self.n_components, int):
            return self.n_components
        
        # For float values (explained variance ratio), use the actual number 
        # of components from the fitted PCA
        if hasattr(self, 'pca') and hasattr(self.pca, 'n_components_'):
            return self.pca.n_components_
        
        # Fallback if PCA doesn't have n_components_ attribute
        if hasattr(self, 'input_dim'):
            return self.input_dim
        
        # If we can't determine output dim, raise error
        raise ValueError("Could not determine output dimension. Preprocessor may not be correctly fitted.") 