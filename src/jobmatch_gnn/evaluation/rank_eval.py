"""Full-corpus ranking evaluation (docs/13).

Models produce, per evaluated user, a truncated top-N ranking (N >= 1000 by
default). Metrics@K with K <= 50 are exact; a positive outside the truncation
contributes rank = +inf (MRR term 0), which is stated in docs/13.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KS = (10, 50)


@dataclass(frozen=True)
class EvalResult:
    metrics: dict[str, float]
    per_user_ndcg: dict[int, float]


def evaluate_rankings(
    rankings: dict[int, np.ndarray],
    relevant: dict[int, set[int]],
    ks: tuple[int, ...] = KS,
) -> EvalResult:
    """Compute macro-averaged Recall/Precision/NDCG/HitRate@K and MRR."""

    sums: dict[str, float] = {}
    per_user_ndcg: dict[int, float] = {}
    count = 0
    main_k = ks[0]
    for user, rel in relevant.items():
        ranked = rankings.get(user)
        if ranked is None or not rel:
            continue
        count += 1
        rel_arr = np.fromiter(rel, dtype=np.int64)
        hits_mask = np.isin(ranked, rel_arr)
        hit_positions = np.flatnonzero(hits_mask)
        first = hit_positions[0] + 1 if hit_positions.size else None
        sums["mrr"] = sums.get("mrr", 0.0) + (1.0 / first if first else 0.0)
        for k in ks:
            top_hits = int(hits_mask[:k].sum())
            ideal = min(len(rel), k)
            idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal + 2))))
            dcg = float(np.sum(1.0 / np.log2(hit_positions[hit_positions < k] + 2)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            sums[f"recall@{k}"] = sums.get(f"recall@{k}", 0.0) + top_hits / len(rel)
            sums[f"precision@{k}"] = sums.get(f"precision@{k}", 0.0) + top_hits / k
            sums[f"ndcg@{k}"] = sums.get(f"ndcg@{k}", 0.0) + ndcg
            sums[f"hitrate@{k}"] = sums.get(f"hitrate@{k}", 0.0) + (1.0 if top_hits else 0.0)
            if k == main_k:
                per_user_ndcg[user] = ndcg
    metrics = {key: value / max(1, count) for key, value in sums.items()}
    metrics["eval_users"] = float(count)
    return EvalResult(metrics=metrics, per_user_ndcg=per_user_ndcg)


def topn_from_scores(
    scores: np.ndarray,
    exclude: set[int],
    topn: int,
) -> np.ndarray:
    """Return top-N job indices from a full score vector, excluding known positives."""

    if exclude:
        scores = scores.copy()
        scores[np.fromiter(exclude, dtype=np.int64)] = -np.inf
    n = min(topn, scores.shape[0])
    part = np.argpartition(-scores, n - 1)[:n]
    return part[np.argsort(-scores[part])]
