"""Rule-based features for person-job matching."""
from __future__ import annotations

from jobmatch_gnn.data.schema import CandidateProfile, JobProfile


def compute_rule_features(candidate: CandidateProfile, job: JobProfile) -> dict[str, float]:
    """Compute skill/education/experience/city matching features."""
    candidate_skills = set(candidate.skills)
    required = set(job.required_skills)
    preferred = set(job.preferred_skills)
    required_coverage = len(candidate_skills & required) / max(len(required), 1)
    preferred_coverage = len(candidate_skills & preferred) / max(len(preferred), 1)
    city_match = float(bool(candidate.expected_city and job.city and candidate.expected_city == job.city))
    education_match = float(bool(candidate.education and job.education_requirement and candidate.education == job.education_requirement))
    experience_match = float(
        candidate.experience_years is not None
        and job.experience_requirement is not None
        and candidate.experience_years >= job.experience_requirement
    )
    return {
        "required_skill_coverage": required_coverage,
        "preferred_skill_coverage": preferred_coverage,
        "city_match": city_match,
        "education_match": education_match,
        "experience_match": experience_match,
    }
