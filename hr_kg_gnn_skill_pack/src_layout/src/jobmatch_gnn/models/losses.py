"""Loss functions for recommendation and KG reconstruction."""
from __future__ import annotations


def bpr_loss(pos_scores: object, neg_scores: object) -> object:
    """Bayesian Personalized Ranking loss."""
    raise NotImplementedError


def contrastive_loss(anchor: object, positive: object, negatives: object, temperature: float = 0.07) -> object:
    """InfoNCE contrastive loss."""
    raise NotImplementedError
