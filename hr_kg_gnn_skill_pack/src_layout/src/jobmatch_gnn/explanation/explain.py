"""Recommendation explanation generation."""
from __future__ import annotations

from jobmatch_gnn.data.schema import CandidateProfile, JobProfile


def explain_match(candidate: CandidateProfile, job: JobProfile) -> dict[str, list[str]]:
    """Return matched skills, missing skills, graph paths, and reasons."""
    c_skills = set(candidate.skills)
    required = set(job.required_skills)
    preferred = set(job.preferred_skills)
    matched = sorted(c_skills & (required | preferred))
    missing = sorted(required - c_skills)
    paths = [f"Candidate:{candidate.candidate_id} -> Skill:{s} <- Job:{job.job_id}" for s in matched]
    reasons: list[str] = []
    if matched:
        reasons.append("技能覆盖度较高")
    if candidate.expected_city and job.city and candidate.expected_city == job.city:
        reasons.append("城市匹配")
    return {"matched_skills": matched, "missing_skills": missing, "graph_paths": paths, "reasons": reasons}
