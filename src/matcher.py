from __future__ import annotations

import json
import random
import re
from pathlib import Path

from src.schemas import CandidateProfile, JobProfile

MOCK_DIR = Path(__file__).parent / 'mock_data'


def _load_json(filename: str) -> list[dict]:
	with open(MOCK_DIR / filename, encoding='utf-8') as f:
		return json.load(f)


def _is_profile_empty(profile: CandidateProfile | JobProfile) -> bool:
	for val in profile.model_dump().values():
		if val is not None:
			return False
	return True


def _parse_salary_range(salary_str: str) -> tuple[int, int] | None:
	"""将 K 薪资范围解析为以元为单位的闭区间。"""
	m = re.match(r'(\d+)\s*K?\s*-\s*(\d+)\s*K?', salary_str, re.IGNORECASE)
	if m:
		return int(m.group(1)) * 1000, int(m.group(2)) * 1000
	m = re.match(r'(\d+)\s*K?', salary_str, re.IGNORECASE)
	if m:
		v = int(m.group(1)) * 1000
		return v, v
	return None


def _salaries_overlap(sal1: str, sal2: str) -> bool:
	r1 = _parse_salary_range(sal1)
	r2 = _parse_salary_range(sal2)
	if r1 is None or r2 is None:
		return False
	return r1[0] <= r2[1] and r2[0] <= r1[1]


def match_jobs_to_candidate(
	profile: CandidateProfile,
	raw_query: str | None = None,
) -> list[tuple[dict, int, list[str]]]:
	"""返回原始岗位、匹配分和匹配原因，并按匹配分降序排列。"""
	jobs = _load_json('jobs.json')
	profile_empty = _is_profile_empty(profile)

	scored: list[tuple[dict, int, list[str]]] = []

	for job in jobs:
		if raw_query and profile_empty:
			# 画像为空时使用原始查询兜底，保留演示搜索能力。
			text = json.dumps(job, ensure_ascii=False).lower()
			if raw_query.lower() in text:
				score = 70 + random.randint(0, 20)
				reasons = [f'匹配查询: {raw_query}']
				scored.append((job, min(score, 100), reasons))
			else:
				scored.append((job, 0, []))
			continue

		score = 0
		reasons: list[str] = []
		all_text = json.dumps(job, ensure_ascii=False).lower()
		kw_text = ' '.join(job.get('keywords', [])).lower()

		if profile.skills:
			for skill in profile.skills:
				sl = skill.lower()
				if sl in kw_text or sl in all_text:
					score += 30
					reasons.append(f'技能匹配: {skill}')

		cities = set()
		if profile.desired_cities:
			for c in profile.desired_cities:
				cities.add(c.lower())
		if profile.current_city:
			cities.add(profile.current_city.lower())
		if cities:
			job_city = job.get('city', '').lower()
			if any(c == job_city or c in job_city or job_city in c for c in cities):
				score += 20
				reasons.append('城市符合')

		if profile.desired_roles:
			for role in profile.desired_roles:
				rl = role.lower()
				if rl in job.get('title', '').lower() or rl in kw_text:
					score += 20
					reasons.append(f'岗位匹配: {role}')
					break

		if profile.salary_expectation and job.get('salary'):
			if _salaries_overlap(profile.salary_expectation, job['salary']):
				score += 10
				reasons.append('薪资匹配')

		if profile.years_experience is not None and '经验' in all_text:
			score += 10
			if not any('经验' in r for r in reasons):
				reasons.append('经验吻合')
		if profile.degree and profile.degree in all_text:
			score += 5
			if not any('学历' in r for r in reasons):
				reasons.append('学历符合')

		scored.append((job, min(score, 100), reasons))

	scored.sort(key=lambda x: x[1], reverse=True)
	return scored


def match_candidates_to_job(
	profile: JobProfile,
	raw_query: str | None = None,
) -> list[tuple[dict, int, list[str]]]:
	"""返回原始候选人、匹配分和匹配原因，并按匹配分降序排列。"""
	candidates = _load_json('candidates.json')
	profile_empty = _is_profile_empty(profile)

	scored: list[tuple[dict, int, list[str]]] = []

	for cand in candidates:
		if raw_query and profile_empty:
			text = json.dumps(cand, ensure_ascii=False).lower()
			if raw_query.lower() in text:
				score = 70 + random.randint(0, 20)
				reasons = [f'匹配查询: {raw_query}']
				scored.append((cand, min(score, 100), reasons))
			else:
				scored.append((cand, 0, []))
			continue

		score = 0
		reasons: list[str] = []
		all_text = json.dumps(cand, ensure_ascii=False).lower()
		kw_text = ' '.join(cand.get('keywords', [])).lower()

		if profile.required_skills:
			for skill in profile.required_skills:
				sl = skill.lower()
				if sl in kw_text or sl in all_text:
					score += 30
					reasons.append(f'技能匹配: {skill}')

		if profile.city:
			cand_city = cand.get('city', '').lower()
			if profile.city.lower() == cand_city:
				score += 20
				reasons.append('城市符合')

		if profile.title:
			tl = profile.title.lower()
			for role in cand.get('target_roles', []):
				if tl in role.lower() or role.lower() in tl:
					score += 20
					reasons.append(f'岗位匹配: {profile.title}')
					break

		if profile.salary_range and cand.get('salary'):
			if _salaries_overlap(profile.salary_range, cand['salary']):
				score += 10
				reasons.append('薪资匹配')

		if profile.degree_requirement and profile.degree_requirement in all_text:
			score += 5
			if not any('学历' in r for r in reasons):
				reasons.append('学历符合')
		if profile.years_requirement and '经验' in all_text:
			score += 10
			if not any('经验' in r for r in reasons):
				reasons.append('经验吻合')

		scored.append((cand, min(score, 100), reasons))

	scored.sort(key=lambda x: x[1], reverse=True)
	return scored


def get_hot_jobs() -> list[tuple[dict, int, list[str]]]:
	"""返回全量 mock 岗位数据，每条附带默认热门分与固定推荐原因。"""
	jobs = _load_json('jobs.json')
	return [(job, job.get('match', 85), ['热门推荐']) for job in jobs]


def get_hot_candidates() -> list[tuple[dict, int, list[str]]]:
	"""返回全量 mock 候选人数据，每条附带默认热门分与固定推荐原因。"""
	candidates = _load_json('candidates.json')
	return [(cand, cand.get('match', 85), ['热门推荐']) for cand in candidates]


def paginate(
	items: list[tuple[dict, int, list[str]]],
	page: int,
	page_size: int,
) -> tuple[list[tuple[dict, int, list[str]]], int]:
	total = len(items)
	total_pages = max(1, (total + page_size - 1) // page_size)
	start = (page - 1) * page_size
	end = start + page_size
	return items[start:end], total_pages
