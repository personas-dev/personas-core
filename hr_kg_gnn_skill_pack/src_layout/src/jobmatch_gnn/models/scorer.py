"""Matching score heads."""
from __future__ import annotations


def cosine_score(candidate_embedding: object, job_embedding: object) -> object:
    """Compute recall-stage cosine scores."""
    raise NotImplementedError


def rank_score(pair_features: object) -> object:
    """Compute reranking scores from embedding/path/rule/semantic features."""
    raise NotImplementedError
