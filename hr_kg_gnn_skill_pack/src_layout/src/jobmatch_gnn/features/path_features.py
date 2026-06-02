"""Skill-path features for Candidate -> Skill <- Job."""
from __future__ import annotations

from jobmatch_gnn.data.schema import CandidateProfile, JobProfile


def compute_skill_path_features(candidate: CandidateProfile, job: JobProfile) -> dict[str, float]:
    """Compute explicit skill path features for a candidate-job pair."""
    c_skills = set(candidate.skills)
    required = set(job.required_skills)
    preferred = set(job.preferred_skills)
    return {
        "shared_required_skill_count": float(len(c_skills & required)),
        "shared_preferred_skill_count": float(len(c_skills & preferred)),
        "missing_required_skill_count": float(len(required - c_skills)),
        "skill_path_count": float(len(c_skills & (required | preferred))),
    }
