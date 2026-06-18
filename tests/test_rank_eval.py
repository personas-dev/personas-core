"""Anchor tests for full-corpus ranking metrics (docs/13 §3)."""
import numpy as np

from jobmatch_gnn.evaluation.rank_eval import evaluate_rankings, topn_from_scores


def test_perfect_ranking():
    rankings = {0: np.array([5, 1, 2])}
    result = evaluate_rankings(rankings, {0: {5}}, ks=(10,))
    assert result.metrics["ndcg@10"] == 1.0
    assert result.metrics["mrr"] == 1.0
    assert result.metrics["recall@10"] == 1.0


def test_rank_three():
    rankings = {0: np.array([9, 8, 5, 1])}
    result = evaluate_rankings(rankings, {0: {5}}, ks=(10,))
    assert abs(result.metrics["ndcg@10"] - 1.0 / np.log2(4)) < 1e-9
    assert abs(result.metrics["mrr"] - 1.0 / 3.0) < 1e-9


def test_miss_outside_topn():
    rankings = {0: np.array([1, 2, 3])}
    result = evaluate_rankings(rankings, {0: {99}}, ks=(10,))
    assert result.metrics["ndcg@10"] == 0.0
    assert result.metrics["mrr"] == 0.0


def test_topn_excludes_known_positives():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    top = topn_from_scores(scores, exclude={0}, topn=3)
    assert list(top) == [1, 2, 3]
