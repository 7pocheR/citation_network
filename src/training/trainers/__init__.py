"""
Training modules for citation network modeling project.

This package contains trainers for different components of the citation network system.
"""

from src.training.trainers.citation_prediction_trainer import CitationPredictionTrainer, TrainingConfig

__all__ = ['CitationPredictionTrainer', 'TrainingConfig'] 