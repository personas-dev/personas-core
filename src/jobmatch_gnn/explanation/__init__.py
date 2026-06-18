"""Faithful KG evidence extraction + optional LLM natural-language explanation."""

from jobmatch_gnn.explanation.kg_evidence import MatchEvidence, EvidenceExtractor
from jobmatch_gnn.explanation.llm_explainer import GroundedExplainer, render_html, render_template

__all__ = [
	'MatchEvidence',
	'EvidenceExtractor',
	'GroundedExplainer',
	'render_template',
	'render_html',
]
