"""Optional LLM expression layer, strictly grounded on KG evidence.

The LLM never invents facts: its prompt contains only the structured evidence,
and its output passes a whitelist faithfulness check (every skill it mentions
must be in the evidence). On any failure (no API key, network error, check
fails) it falls back to a deterministic template. This implements the
"KG = fact source, LLM = replaceable expression layer" design (docs/16 §3).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from html import escape

from jobmatch_gnn.explanation.kg_evidence import MatchEvidence

SYSTEM_PROMPT = (
	'你是招聘推荐系统的解释助手。你只能基于给定的结构化证据(JSON)生成解释,'
	'严禁编造证据中不存在的技能、岗位或事实。'
	'输出 2-4 句中文:先说明为什么推荐这个岗位(引用匹配技能与匹配理由),'
	'再客观指出候选人需要补充的关键技能(来自 missing_skills)。'
	'不要输出 JSON,不要列表,只输出自然语言段落。'
)


def render_template(ev: MatchEvidence) -> str:
	"""Deterministic, fully faithful fallback explanation (no LLM)."""

	top = '、'.join(sw.skill for sw in ev.matched_skills[:4]) or '(无直接技能重叠)'
	parts = [f'推荐岗位「{ev.job_title}」:你的技能 {top} 与该岗位要求匹配。']
	if ev.reasons:
		parts.append('匹配依据:' + ';'.join(ev.reasons[:3]) + '。')
	if ev.missing_skills:
		parts.append('建议补充技能:' + '、'.join(ev.missing_skills[:5]) + '。')
	return ''.join(parts)


def render_html(text: str) -> str:
	"""Render a safe HTML fragment for frontend display."""

	paragraphs = [p.strip() for p in re.split(r'\r?\n+', text) if p.strip()]
	body = ''.join(f'<p>{escape(p)}</p>' for p in paragraphs)
	return f'<div class="llm-explanation">{body}</div>'


def _faithful(text: str, ev: MatchEvidence) -> bool:
	"""Every skill-like token the LLM emits must exist in the evidence whitelist.

	We only police skills that appear in the evidence vocabulary; generic prose
	is allowed. The guard rejects fabricated *skills* (the documented LLM
	failure mode), not ordinary words.
	"""

	whitelist = {s.lower() for s in ev.skill_whitelist()}
	all_skill_terms = whitelist | {ev.job_title.lower()}
	# any evidence skill mentioned must of course be allowed; the real risk is a
	# skill from the global vocab that is NOT in this pair's evidence.
	for term in re.findall(r'[A-Za-z][A-Za-z0-9+.#]{2,}', text):
		t = term.lower()
		if t in all_skill_terms:
			continue
		# unknown ascii tech token that looks like a skill but isn't in evidence
		if t in _GLOBAL_SKILLS and t not in whitelist:
			return False
	return True


_GLOBAL_SKILLS: set[str] = set()


def set_global_skill_vocab(skills: list[str]) -> None:
	"""Register the full skill vocabulary so the guard can spot out-of-evidence skills."""

	global _GLOBAL_SKILLS
	_GLOBAL_SKILLS = {s.lower() for s in skills if re.fullmatch(r'[A-Za-z0-9+.#]+', s)}


class GroundedExplainer:
	"""Calls an OpenAI-compatible chat API; degrades gracefully to template."""

	def __init__(
		self,
		base_url: str | None = None,
		api_key: str | None = None,
		model: str = 'deepseek-chat',
		timeout: float = 20.0,
	) -> None:
		self.base_url = (base_url or os.environ.get('LLM_BASE_URL', '')).rstrip('/')
		self.api_key = api_key or os.environ.get('LLM_API_KEY', '')
		self.model = os.environ.get('LLM_MODEL', model)
		self.timeout = timeout

	@property
	def enabled(self) -> bool:
		return bool(self.base_url and self.api_key)

	def _call(self, evidence_json: str) -> str | None:
		payload = json.dumps(
			{
				'model': self.model,
				'messages': [
					{'role': 'system', 'content': SYSTEM_PROMPT},
					{'role': 'user', 'content': evidence_json},
				],
				'temperature': 0.3,
				'max_tokens': 300,
			}
		).encode('utf-8')
		req = urllib.request.Request(
			f'{self.base_url}/chat/completions',
			data=payload,
			headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
		)
		try:
			with urllib.request.urlopen(req, timeout=self.timeout) as resp:
				data = json.loads(resp.read().decode('utf-8'))
			return data['choices'][0]['message']['content'].strip()
		except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError):
			return None

	def explain(self, ev: MatchEvidence) -> dict[str, object]:
		"""Return explanation with provenance and faithfulness status."""

		evidence_payload = json.dumps(
			{
				'job_title': ev.job_title,
				'matched_skills': [sw.skill for sw in ev.matched_skills],
				'missing_skills': ev.missing_skills,
				'reasons': ev.reasons,
			},
			ensure_ascii=False,
		)
		if self.enabled:
			text = self._call(evidence_payload)
			if text and _faithful(text, ev):
				return {'text': text, 'html': render_html(text), 'source': 'llm', 'faithful': True}
			if text:
				fallback = render_template(ev)
				return {
					'text': fallback,
					'html': render_html(fallback),
					'source': 'template_fallback_unfaithful',
					'faithful': True,
					'rejected_llm': text,
				}
		fallback = render_template(ev)
		return {'text': fallback, 'html': render_html(fallback), 'source': 'template', 'faithful': True}
