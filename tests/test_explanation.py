"""Tests for the faithfulness guard and template fallback (docs/16 §5)."""

from jobmatch_gnn.explanation.kg_evidence import MatchEvidence, SkillWeight
from jobmatch_gnn.explanation.llm_explainer import (
	GroundedExplainer,
	_faithful,
	render_html,
	render_template,
	set_global_skill_vocab,
)


def _evidence() -> MatchEvidence:
	return MatchEvidence(
		user_id='u1',
		job_id='j1',
		job_title='python 后端工程师',
		score=1.0,
		matched_skills=[SkillWeight('python', 0.6), SkillWeight('mysql', 0.4)],
		missing_skills=['redis', 'docker'],
		reasons=['技能覆盖率 50%'],
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
