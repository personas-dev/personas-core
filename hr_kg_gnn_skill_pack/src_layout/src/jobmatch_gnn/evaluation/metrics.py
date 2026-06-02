"""Ranking metrics for person-job recommendation."""
from __future__ import annotations

import math


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Compute Recall@K."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Compute Precision@K."""
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Compute binary NDCG@K."""
    dcg = 0.0
    for i, item_id in enumerate(ranked[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def mrr(relevant: set[str], ranked: list[str]) -> float:
    """Compute reciprocal rank for a single query."""
    for i, item_id in enumerate(ranked, start=1):
        if item_id in relevant:
            return 1.0 / i
    return 0.0
