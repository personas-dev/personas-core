"""Ranking metrics for person-job recommendation."""
from __future__ import annotations

import math
from statistics import mean


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Compute Recall@K for one query."""

    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Compute Precision@K for one query."""

    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Compute binary NDCG@K for one query."""

    dcg = 0.0
    for index, item_id in enumerate(ranked[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def reciprocal_rank(relevant: set[str], ranked: list[str]) -> float:
    """Compute reciprocal rank for one query."""

    for index, item_id in enumerate(ranked, start=1):
        if item_id in relevant:
            return 1.0 / index
    return 0.0


def hitrate_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Compute binary HitRate@K for one query."""

    return float(bool(set(ranked[:k]) & relevant))


def evaluate_rankings(relevant_by_user: dict[str, set[str]], rankings: dict[str, list[str]], k: int) -> dict[str, float]:
    """Aggregate ranking metrics across users."""

    users = [user_id for user_id, relevant in relevant_by_user.items() if relevant]
    if not users:
        return {f"recall@{k}": 0.0, f"precision@{k}": 0.0, f"ndcg@{k}": 0.0, "mrr": 0.0, f"hitrate@{k}": 0.0, "user_count": 0.0}
    recalls = [recall_at_k(relevant_by_user[user_id], rankings.get(user_id, []), k) for user_id in users]
    precisions = [precision_at_k(relevant_by_user[user_id], rankings.get(user_id, []), k) for user_id in users]
    ndcgs = [ndcg_at_k(relevant_by_user[user_id], rankings.get(user_id, []), k) for user_id in users]
    mrrs = [reciprocal_rank(relevant_by_user[user_id], rankings.get(user_id, [])) for user_id in users]
    hits = [hitrate_at_k(relevant_by_user[user_id], rankings.get(user_id, []), k) for user_id in users]
    return {
        f"recall@{k}": float(mean(recalls)),
        f"precision@{k}": float(mean(precisions)),
        f"ndcg@{k}": float(mean(ndcgs)),
        "mrr": float(mean(mrrs)),
        f"hitrate@{k}": float(mean(hits)),
        "user_count": float(len(users)),
    }
