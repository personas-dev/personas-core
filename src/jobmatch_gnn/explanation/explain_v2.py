"""CLI for faithful SPC-HGT v2 match explanations.

Layer 1 is deterministic KG evidence: matched skills, missing skills, graph
paths, and structured reasons. Layer 2 is optional grounded LLM narration; it is
allowed to verbalize only the KG evidence and falls back to a safe template when
the LLM is unavailable or unfaithful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from jobmatch_gnn.data.bundle_v2 import load_bundle_v2
from jobmatch_gnn.explanation.kg_evidence import EvidenceExtractor, MatchEvidence
from jobmatch_gnn.explanation.llm_explainer import GroundedExplainer, set_global_skill_vocab


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Explain SPC-HGT v2 recommendations.')
	parser.add_argument('--processed-dir', default='data/processed', help='Processed v2 data directory.')
	parser.add_argument('--checkpoint', required=True, help='SPC-HGT v2 checkpoint path.')
	parser.add_argument('--user-idx', type=int, required=True, help='Internal user row index.')
	parser.add_argument('--job-idx', type=int, default=None, help='Internal job row index. If omitted, rank top-k first.')
	parser.add_argument('--k', type=int, default=5, help='Number of ranked jobs to explain when --job-idx is omitted.')
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--llm', action='store_true', help='Attach grounded LLM/template explanation with HTML output.')
	parser.add_argument('--output', default='', help='Optional JSON output path. Prints to stdout when omitted.')
	return parser.parse_args()


def _evidence_dict(ev: MatchEvidence) -> dict[str, Any]:
	return json.loads(ev.to_json())


def _build_item(ev: MatchEvidence, explainer: GroundedExplainer | None) -> dict[str, Any]:
	item = _evidence_dict(ev)
	if explainer is not None:
		item['llm'] = explainer.explain(ev)
		item['llm_explanation_html'] = item['llm']['html']
	return item


def main() -> None:
	args = _parse_args()
	bundle = load_bundle_v2(args.processed_dir)
	extractor = EvidenceExtractor(bundle, args.checkpoint, device=args.device)
	explainer = GroundedExplainer() if args.llm else None
	if explainer is not None:
		set_global_skill_vocab(extractor.skills)

	job_indices = [args.job_idx] if args.job_idx is not None else extractor.rank(args.user_idx, topn=args.k)
	items = [_build_item(extractor.evidence(args.user_idx, int(job_idx)), explainer) for job_idx in job_indices]
	payload = {
		'user_idx': args.user_idx,
		'job_indices': [int(job_idx) for job_idx in job_indices],
		'items': items,
	}
	text = json.dumps(payload, ensure_ascii=False, indent=2)
	if args.output:
		Path(args.output).write_text(text + '\n', encoding='utf-8')
	else:
		print(text)


if __name__ == '__main__':
	main()
