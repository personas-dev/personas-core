"""Skill gap analysis."""
from __future__ import annotations

from jobmatch_gnn.data.schema import CandidateProfile, JobProfile


def find_missing_required_skills(candidate: CandidateProfile, job: JobProfile) -> list[str]:
    """Return required skills missing from candidate profile."""
    return sorted(set(job.required_skills) - set(candidate.skills))
