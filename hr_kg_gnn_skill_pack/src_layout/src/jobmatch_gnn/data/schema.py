"""Pydantic schemas for person-job matching."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    """Structured candidate profile."""

    candidate_id: str
    education: str | None = None
    major: str | None = None
    experience_years: float | None = None
    skills: list[str] = Field(default_factory=list)
    project_domains: list[str] = Field(default_factory=list)
    expected_city: str | None = None
    expected_position: str | None = None
    resume_text: str | None = None


class JobProfile(BaseModel):
    """Structured job profile."""

    job_id: str
    job_title: str
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    education_requirement: str | None = None
    experience_requirement: float | None = None
    city: str | None = None
    industry: str | None = None
    jd_text: str | None = None


class RecommendationItem(BaseModel):
    """Single recommendation result with structured explanation."""

    job_id: str
    score: float
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    graph_paths: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    """Top-K recommendation response."""

    candidate_id: str
    model_version: str
    schema_version: str = "1.0.0"
    graph_snapshot_id: str | None = None
    items: list[RecommendationItem]


InferenceMode = Literal["recall", "rerank", "full"]
