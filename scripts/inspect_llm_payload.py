"""Inspect the exact LLM request payload built from mock match evidence.

This script does not call a real LLM. It monkeypatches ``urllib.request.urlopen``
and captures the request that ``GroundedExplainer`` would send to an
OpenAI-compatible chat completions endpoint.

Run:
    python scripts/inspect_llm_payload.py

Default output:
    experiments/runs/llm_payload_inspection.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from jobmatch_gnn.explanation.kg_evidence import MatchEvidence, SkillWeight, build_local_subgraph  # noqa: E402
from jobmatch_gnn.explanation.llm_explainer import GroundedExplainer, set_global_skill_vocab  # noqa: E402


def build_mock_evidence() -> MatchEvidence:
	"""Return deterministic evidence that resembles one ranked job result."""

	user_id = 'mock_user_001'
	job_id = 'mock_job_9001'
	job_title = '推荐算法工程师'
	score = 2.7183
	matched_skills = [
		SkillWeight('python', 0.42),
		SkillWeight('机器学习', 0.31),
		SkillWeight('mysql', 0.16),
	]
	missing_skills = ['召回排序', '特征工程', 'docker']
	graph_paths = [
		'Candidate:mock_user_001 -HAS_SKILL(0.42)-> Skill:python <-REQUIRES_SKILL- Job:mock_job_9001',
		'Candidate:mock_user_001 -HAS_SKILL(0.31)-> Skill:机器学习 <-REQUIRES_SKILL- Job:mock_job_9001',
		'Job:mock_job_9001 -REQUIRES_SKILL-> Skill:召回排序 (missing_for_candidate)',
	]
	reasons = [
		'技能覆盖率 50%(3/6 项要求技能匹配)',
		'工作城市符合期望',
		'工作年限满足岗位要求',
	]
	local_subgraph = build_local_subgraph(
		user_id=user_id,
		job_id=job_id,
		job_title=job_title,
		score=score,
		matched_skills=matched_skills,
		missing_skills=missing_skills,
		graph_paths=graph_paths,
		reasons=reasons,
		candidate_context={
			'live_city': '上海',
			'desired_cities': ['上海', '杭州'],
			'desired_types': ['算法工程师', '后端开发'],
			'degree_rank': 4,
			'years': 3,
		},
		job_context={
			'city': '上海',
			'job_type': '算法工程师',
			'min_degree_rank': 4,
			'min_years': 2,
		},
	)
	return MatchEvidence(
		user_id=user_id,
		job_id=job_id,
		job_title=job_title,
		score=score,
		matched_skills=matched_skills,
		missing_skills=missing_skills,
		graph_paths=graph_paths,
		reasons=reasons,
		local_subgraph=local_subgraph,
	)


class FakeResponse:
	"""Small context-manager response object for urllib.request.urlopen."""

	def __enter__(self) -> 'FakeResponse':
		return self

	def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
		return False

	def read(self) -> bytes:
		payload = {
			'choices': [
				{
					'message': {
						'content': (
							'推荐岗位「推荐算法工程师」:你的 python、机器学习 和 mysql 与岗位要求匹配。建议继续补充 召回排序、特征工程 和 docker。'
						)
					}
				}
			]
		}
		return json.dumps(payload, ensure_ascii=False).encode('utf-8')


def capture_request(req: Request, timeout: float | None) -> dict[str, Any]:
	"""Extract URL, headers, raw body, parsed body, and nested evidence JSON."""

	raw_body = (req.data or b'').decode('utf-8')
	parsed_body = json.loads(raw_body) if raw_body else {}
	evidence_json = parsed_body.get('messages', [{}, {}])[1].get('content', '{}')
	return {
		'url': req.full_url,
		'timeout': timeout,
		'headers': dict(req.header_items()),
		'raw_body': raw_body,
		'body': parsed_body,
		'user_evidence_json': json.loads(evidence_json),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--base-url', default='https://mock-llm.local/v1')
	parser.add_argument('--api-key', default='test-key')
	parser.add_argument('--model', default='deepseek-chat')
	parser.add_argument('--timeout', type=float, default=20.0)
	parser.add_argument(
		'--output',
		type=Path,
		default=Path('experiments/runs/llm_payload_inspection.json'),
		help='Path to write captured JSON.',
	)
	args = parser.parse_args()

	evidence = build_mock_evidence()
	set_global_skill_vocab(['python', '机器学习', 'mysql', '召回排序', '特征工程', 'docker', 'kafka'])

	captured: dict[str, Any] = {}

	def fake_urlopen(req: Request, timeout: float | None = None) -> FakeResponse:
		captured.update(capture_request(req, timeout))
		return FakeResponse()

	explainer = GroundedExplainer(
		base_url=args.base_url,
		api_key=args.api_key,
		model=args.model,
		timeout=args.timeout,
	)
	with patch('urllib.request.urlopen', fake_urlopen):
		result = explainer.explain(evidence)

	output = {
		'mock_evidence': json.loads(evidence.to_json()),
		'llm_request': captured,
		'explainer_result': result,
	}
	text = json.dumps(output, ensure_ascii=False, indent=2)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(text + '\n', encoding='utf-8')
	print(f'wrote {args.output}')
	print(text)


if __name__ == '__main__':
	main()
