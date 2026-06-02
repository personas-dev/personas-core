from jobmatch_gnn.evaluation.metrics import evaluate_rankings, ndcg_at_k, recall_at_k


def test_basic_ranking_metrics() -> None:
    relevant = {"a", "c"}
    ranked = ["a", "b", "c"]
    assert recall_at_k(relevant, ranked, 1) == 0.5
    assert ndcg_at_k(relevant, ranked, 3) > 0.0


def test_evaluate_rankings() -> None:
    metrics = evaluate_rankings({"u1": {"j1"}}, {"u1": ["j2", "j1"]}, 2)
    assert metrics["recall@2"] == 1.0
    assert metrics["hitrate@2"] == 1.0
