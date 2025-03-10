"""
Loss functions for citation network modeling project.

This package contains loss functions for different tasks in the citation network system.
"""

import torch
import torch.nn.functional as F


def binary_cross_entropy_with_logits(pos_scores, neg_scores):
    """
    Compute binary cross entropy loss for link prediction.
    
    Args:
        pos_scores: Scores for positive examples (edges that exist)
        neg_scores: Scores for negative examples (edges that don't exist)
        
    Returns:
        Loss value
    """
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    scores = torch.cat([pos_scores, neg_scores])
    return F.binary_cross_entropy_with_logits(scores, labels)


def margin_ranking_loss(pos_scores, neg_scores, margin=0.1):
    """
    Compute margin ranking loss for link prediction.
    
    Args:
        pos_scores: Scores for positive examples (edges that exist)
        neg_scores: Scores for negative examples (edges that don't exist)
        margin: Margin value
        
    Returns:
        Loss value
    """
    ones = torch.ones_like(pos_scores)
    return F.margin_ranking_loss(pos_scores, neg_scores, ones, margin=margin) 