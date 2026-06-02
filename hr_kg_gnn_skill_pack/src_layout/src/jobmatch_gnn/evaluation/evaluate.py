"""Evaluation entrypoint."""
from __future__ import annotations


def evaluate_model(model: object, dataset: object, top_k: list[int]) -> dict[str, float]:
    """Evaluate a model on ranking metrics."""
    raise NotImplementedError
