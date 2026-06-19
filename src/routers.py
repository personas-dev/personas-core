from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.matcher import match_candidates_to_job, match_jobs_to_candidate, paginate
from src.schemas import (
	CandidateRecommendRequest,
	CandidateRecommendResponse,
	JobRecommendRequest,
	JobRecommendResponse,
	Pagination,
	RecommendedCandidate,
	RecommendedJob,
)

api_router = APIRouter(prefix='/api/v1')


def _job_to_recommended(job: dict, score: int, reasons: list[str]) -> RecommendedJob:
	return RecommendedJob(
		id=job['id'],
		title=job['title'],
		company=job['company'],
		category=job['category'],
		salary=job['salary'],
		city=job['city'],
		district=job.get('district'),
		education=job['education'],
		experience=job['experience'],
		match_score=score,
		match_reasons=reasons,
		level=job['level'],
		highlight=job['highlight'],
		reason=job['reason'],
		duty=job['duty'],
		filter_tags=job['filter_tags'],
		keywords=job['keywords'],
		detail_bullets=job['detail_bullets'],
	)


def _candidate_to_recommended(cand: dict, score: int, reasons: list[str]) -> RecommendedCandidate:
	return RecommendedCandidate(
		id=cand['id'],
		name=cand['name'],
		gender=cand['gender'],
		age=cand['age'],
		degree=cand['degree'],
		years=cand['years'],
		current=cand['current'],
		salary=cand['salary'],
		match_score=score,
		match_reasons=reasons,
		tags=cand['tags'],
		reason=cand['reason'],
		avatar=cand['avatar'],
		filter_tags=cand['filter_tags'],
		keywords=cand['keywords'],
		target_roles=cand['target_roles'],
		city=cand['city'],
	)


@api_router.post('/recommend/jobs', response_model=JobRecommendResponse)
def recommend_jobs(req: JobRecommendRequest) -> JobRecommendResponse:
	scored = match_jobs_to_candidate(req.profile, req.raw_query)
	page = req.options.page
	page_size = req.options.page_size
	page_items, total_pages = paginate(scored, page, page_size)

	items = [_job_to_recommended(job, score, reasons) for job, score, reasons in page_items]

	return JobRecommendResponse(
		items=items,
		pagination=Pagination(
			page=page,
			page_size=page_size,
			total=len(scored),
			total_pages=total_pages,
		),
	)


@api_router.post('/recommend/candidates', response_model=CandidateRecommendResponse)
def recommend_candidates(req: CandidateRecommendRequest) -> CandidateRecommendResponse:
	scored = match_candidates_to_job(req.profile, req.raw_query)
	page = req.options.page
	page_size = req.options.page_size
	page_items, total_pages = paginate(scored, page, page_size)

	items = [_candidate_to_recommended(cand, score, reasons) for cand, score, reasons in page_items]

	return CandidateRecommendResponse(
		items=items,
		pagination=Pagination(
			page=page,
			page_size=page_size,
			total=len(scored),
			total_pages=total_pages,
		),
	)
