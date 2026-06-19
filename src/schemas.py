from __future__ import annotations

from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
	skills: list[str] | None = None
	experience_text: str | None = None
	desired_cities: list[str] | None = None
	current_city: str | None = None
	industries: list[str] | None = None
	desired_roles: list[str] | None = None
	age: int | None = None
	degree: str | None = None
	years_experience: int | None = None
	salary_expectation: str | None = None


class JobProfile(BaseModel):
	title: str | None = None
	description: str | None = None
	city: str | None = None
	industry: str | None = None
	job_type: str | None = None
	required_skills: list[str] | None = None
	degree_requirement: str | None = None
	years_requirement: str | None = None
	salary_range: str | None = None


class RecommendOptions(BaseModel):
	page: int = Field(default=1, ge=1)
	page_size: int = Field(default=20, ge=1, le=100)
	sort_by: str = Field(default='match')


class JobRecommendRequest(BaseModel):
	profile: CandidateProfile
	options: RecommendOptions = Field(default_factory=RecommendOptions)
	raw_query: str | None = None


class CandidateRecommendRequest(BaseModel):
	profile: JobProfile
	options: RecommendOptions = Field(default_factory=RecommendOptions)
	raw_query: str | None = None


class RecommendedJob(BaseModel):
	id: int
	title: str
	company: str
	category: str
	salary: str
	city: str
	district: str | None = None
	education: str
	experience: str
	match_score: int = Field(ge=0, le=100)
	match_reasons: list[str]
	level: str
	highlight: str
	reason: str
	duty: str
	filter_tags: list[str]
	keywords: list[str]
	detail_bullets: list[str]


class RecommendedCandidate(BaseModel):
	id: int
	name: str
	gender: str
	age: int
	degree: str
	years: int
	current: str
	salary: str
	match_score: int = Field(ge=0, le=100)
	match_reasons: list[str]
	tags: list[str]
	reason: str
	avatar: str
	filter_tags: list[str]
	keywords: list[str]
	target_roles: list[str]
	city: str


class Pagination(BaseModel):
	page: int
	page_size: int
	total: int
	total_pages: int


class JobRecommendResponse(BaseModel):
	items: list[RecommendedJob]
	pagination: Pagination


class CandidateRecommendResponse(BaseModel):
	items: list[RecommendedCandidate]
	pagination: Pagination
