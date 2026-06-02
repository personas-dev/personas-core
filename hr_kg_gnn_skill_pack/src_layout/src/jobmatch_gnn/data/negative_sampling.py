"""Negative sampling strategies."""
from __future__ import annotations


def sample_negatives(candidate_id: str, positive_job_ids: set[str], all_job_ids: list[str], num_negatives: int, seed: int) -> list[str]:
    """Sample random negatives. Extend with BM25/SBERT hard negatives."""
    import random

    rng = random.Random(seed)
    pool = [job_id for job_id in all_job_ids if job_id not in positive_job_ids]
    rng.shuffle(pool)
    return pool[:num_negatives]
