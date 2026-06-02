"""Rule-based matcher baseline."""
from __future__ import annotations

from jobmatch_gnn.data.schema import CandidateProfile, JobProfile
from jobmatch_gnn.features.rule_features import compute_rule_features


def rule_score(candidate: CandidateProfile, job: JobProfile) -> float:
    """Compute weighted rule matching score."""
    f = compute_rule_features(candidate, job)
    skill = 0.8 * f["required_skill_coverage"] + 0.2 * f["preferred_skill_coverage"]
    return 0.5 * skill + 0.2 * f["education_match"] + 0.2 * f["experience_match"] + 0.1 * f["city_match"]
