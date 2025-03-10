import json
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        json.JSONDecodeError: If the config file is not valid JSON
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate the config
    validate_config(config)
    
    return config


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration.
    
    Returns:
        Dictionary containing the default configuration
    """
    config = {
        "data": {
            "path": "dataset.json",
            "test_year": 2018,
            "val_ratio": 0.2,
            "sequence_length": 5,
            "time_window": 365,  # Days (1 year)
            "stride": 180  # Days (6 months)
        },
        "model": {
            "embedding_dim": 128,
            "memory_dim": 128,
            "time_dim": 32,
            "hidden_dim": 256,
            "latent_dim": 64,
            "num_heads": 4,
            "dropout": 0.1,
            "use_memory": True,
            "hyperbolic": True,
            "num_topics": 10,
            "topic_embedding_dim": 64
        },
        "training": {
            "batch_size": 32,
            "num_workers": 4,
            "encoder_epochs": 50,
            "generator_epochs": 50,
            "encoder_lr": 0.001,
            "generator_lr": 0.001,
            "weight_decay": 1e-5,
            "kl_weight": 0.1,
            "citation_weight": 1.0,
            "feature_weight": 1.0,
            "negative_sampling_ratio": 1.0,
            "early_stopping_patience": 10
        },
        "generation": {
            "num_papers": 100,
            "num_papers_per_topic": 10,
            "temperature": 1.0
        }
    }
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If the configuration is invalid
    """
    default_config = create_default_config()
    
    # Check if all required sections exist
    for section in default_config:
        if section not in config:
            raise ValueError(f"Missing section in config: {section}")
    
    # Check if all required parameters exist in each section
    for section, params in default_config.items():
        if section not in config:
            continue
            
        for param in params:
            if param not in config[section]:
                # Add the default value
                config[section][param] = default_config[section][param]
    
    # Additional validation
    # Data validation
    if config["data"]["sequence_length"] < 1:
        raise ValueError("sequence_length must be >= 1")
    
    if config["data"]["time_window"] < 1:
        raise ValueError("time_window must be >= 1")
    
    if config["data"]["stride"] < 1:
        raise ValueError("stride must be >= 1")
    
    if not (0 < config["data"]["val_ratio"] < 1):
        raise ValueError("val_ratio must be between 0 and 1")
    
    # Model validation
    if config["model"]["embedding_dim"] < 1:
        raise ValueError("embedding_dim must be >= 1")
    
    if config["model"]["latent_dim"] < 1:
        raise ValueError("latent_dim must be >= 1")
    
    if config["model"]["num_heads"] < 1:
        raise ValueError("num_heads must be >= 1")
    
    if not (0 <= config["model"]["dropout"] < 1):
        raise ValueError("dropout must be between 0 and 1")
    
    # Training validation
    if config["training"]["batch_size"] < 1:
        raise ValueError("batch_size must be >= 1")
    
    if config["training"]["encoder_epochs"] < 1:
        raise ValueError("encoder_epochs must be >= 1")
    
    if config["training"]["generator_epochs"] < 1:
        raise ValueError("generator_epochs must be >= 1")
    
    if config["training"]["encoder_lr"] <= 0:
        raise ValueError("encoder_lr must be > 0")
    
    if config["training"]["generator_lr"] <= 0:
        raise ValueError("generator_lr must be > 0")
    
    if config["training"]["weight_decay"] < 0:
        raise ValueError("weight_decay must be >= 0")
    
    if config["training"]["kl_weight"] < 0:
        raise ValueError("kl_weight must be >= 0")
    
    if config["training"]["citation_weight"] < 0:
        raise ValueError("citation_weight must be >= 0")
    
    if config["training"]["feature_weight"] < 0:
        raise ValueError("feature_weight must be >= 0")
    
    # Generation validation
    if config["generation"]["num_papers"] < 1:
        raise ValueError("num_papers must be >= 1")
    
    if config["generation"]["num_papers_per_topic"] < 1:
        raise ValueError("num_papers_per_topic must be >= 1")
    
    if config["generation"]["temperature"] <= 0:
        raise ValueError("temperature must be > 0")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def create_default_config_file(config_path: str = "config/default.json") -> None:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path to save the configuration file
    """
    config = create_default_config()
    save_config(config, config_path) 