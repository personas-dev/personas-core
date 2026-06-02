"""Inference ranker."""
from __future__ import annotations

from jobmatch_gnn.data.schema import CandidateProfile, InferenceMode, JobProfile, RecommendationResponse


def rank_jobs_for_candidate(
    candidate: CandidateProfile,
    jobs: list[JobProfile],
    top_k: int = 10,
    mode: InferenceMode = "full",
) -> RecommendationResponse:
    """Return Top-K job recommendations with structured explanations."""
    raise NotImplementedError
