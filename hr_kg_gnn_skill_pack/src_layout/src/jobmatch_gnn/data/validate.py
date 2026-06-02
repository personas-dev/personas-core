"""Data validation utilities."""
from __future__ import annotations

from .schema import CandidateProfile, JobProfile


def validate_candidate(candidate: CandidateProfile) -> None:
    """Validate a candidate profile. Raise ValueError if invalid."""
    if not candidate.candidate_id:
        raise ValueError("candidate_id is required")


def validate_job(job: JobProfile) -> None:
    """Validate a job profile. Raise ValueError if invalid."""
    if not job.job_id:
        raise ValueError("job_id is required")
    if not job.job_title:
        raise ValueError("job_title is required")
