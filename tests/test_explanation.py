"""Tests for the faithfulness guard and template fallback (docs/16 §5)."""

import json

from jobmatch_gnn.explanation.kg_evidence import MatchEvidence, SkillWeight, build_local_subgraph
from jobmatch_gnn.explanation.llm_explainer import (
	GroundedExplainer,
	_faithful,
	render_html,
	render_template,
	set_global_skill_vocab,
)


def _evidence() -> MatchEvidence:
	matched_skills = [SkillWeight('python', 0.6), SkillWeight('mysql', 0.4)]
	missing_skills = ['redis', 'docker']
	graph_paths = ['Candidate:u1 -HAS_SKILL(0.6)-> Skill:python <-REQUIRES_SKILL- Job:j1']
	reasons = ['技能覆盖率 50%']
	return MatchEvidence(
		user_id='u1',
		job_id='j1',
		job_title='python 后端工程师',
		score=1.0,
		matched_skills=matched_skills,
		missing_skills=missing_skills,
		graph_paths=graph_paths,
		reasons=reasons,
		local_subgraph=build_local_subgraph(
			user_id='u1',
			job_id='j1',
			job_title='python 后端工程师',
			score=1.0,
			matched_skills=matched_skills,
			missing_skills=missing_skills,
			graph_paths=graph_paths,
			reasons=reasons,
		),
	)


def test_template_is_faithful_and_nonempty():
	ev = _evidence()
	text = render_template(ev)
	assert 'python' in text
	assert 'redis' in text


def test_guard_accepts_evidence_skills():
	ev = _evidence()
	set_global_skill_vocab(['python', 'mysql', 'redis', 'docker', 'kafka', 'spark'])
	assert _faithful('你的 python 和 mysql 很匹配,建议学习 redis 和 docker。', ev) is True


def test_guard_rejects_fabricated_skill():
	ev = _evidence()
	# kafka/spark are real skills in the global vocab but NOT in this pair's evidence
	set_global_skill_vocab(['python', 'mysql', 'redis', 'docker', 'kafka', 'spark'])
	assert _faithful('这个岗位还需要 kafka 和 spark 经验。', ev) is False


def test_guard_allows_plain_prose():
	ev = _evidence()
	set_global_skill_vocab(['python', 'mysql', 'redis', 'docker'])
	assert _faithful('这是一个很适合你的岗位,匹配度高。', ev) is True


def test_html_renderer_escapes_untrusted_text():
	html = render_html('<script>alert(1)</script> python')
	assert '<script>' not in html
	assert '&lt;script&gt;alert(1)&lt;/script&gt; python' in html
	assert html.startswith('<div class="llm-explanation">')


def test_explainer_returns_html_fragment_with_template_fallback():
	out = GroundedExplainer().explain(_evidence())
	assert out['source'] == 'template'
	assert out['faithful'] is True
	assert 'python' in str(out['text'])
	assert str(out['html']).startswith('<div class="llm-explanation">')
	assert '<p>' in str(out['html'])


def test_llm_payload_contains_local_subgraph():
	class CapturingExplainer(GroundedExplainer):
		def __init__(self):
			super().__init__(base_url='https://mock.local/v1', api_key='test-key')
			self.payload = {}

		def _call(self, evidence_json: str) -> str | None:
			self.payload = json.loads(evidence_json)
			return '你的 python 和 mysql 很匹配,建议学习 redis 和 docker。'

	ev = _evidence()
	set_global_skill_vocab(['python', 'mysql', 'redis', 'docker'])
	explainer = CapturingExplainer()
	out = explainer.explain(ev)
	assert out['source'] == 'llm'
	assert explainer.payload['matched_skills'][0] == {'skill': 'python', 'weight': 0.6}
	assert explainer.payload['graph_paths'] == ev.graph_paths
	assert explainer.payload['local_subgraph']['schema_version'] == 'local_kg_v1'
	assert explainer.payload['local_subgraph']['nodes']
	assert explainer.payload['local_subgraph']['edges']
