"""Demo: produce faithful KG explanations (+ optional LLM expression) for users.

    CUDA_VISIBLE_DEVICES=0 python -m jobmatch_gnn.explanation.explain_demo \
        --checkpoint experiments/runs/v2_spc_hgt/spc_hgt_v2.pt --num-users 3 --topk 3

Set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL to enable the natural-language layer;
without them the demo prints the deterministic, fully faithful template output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jobmatch_gnn.data.bundle_v2 import load_bundle_v2
from jobmatch_gnn.explanation.kg_evidence import EvidenceExtractor
from jobmatch_gnn.explanation.llm_explainer import GroundedExplainer, set_global_skill_vocab


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--processed-dir', default='data/processed')
	parser.add_argument('--device', default='cuda')
	parser.add_argument('--num-users', type=int, default=3)
	parser.add_argument('--topk', type=int, default=3)
	parser.add_argument('--out', type=Path, default=Path('experiments/runs/explanations_demo.json'))
	args = parser.parse_args()

	bundle = load_bundle_v2(args.processed_dir)
	extractor = EvidenceExtractor(bundle, args.checkpoint, device=args.device)
	set_global_skill_vocab(extractor.skills)
	explainer = GroundedExplainer()
	print(f'LLM expression layer: {"ENABLED" if explainer.enabled else "disabled (template only)"}', flush=True)

	test_users = sorted(bundle.test_pos)[: args.num_users]
	results = []
	for user_idx in test_users:
		for job_idx in extractor.rank(user_idx, topn=args.topk):
			ev = extractor.evidence(user_idx, job_idx)
			expl = explainer.explain(ev)
			results.append({'evidence': json.loads(ev.to_json()), 'explanation': expl})
			print('=' * 70)
			print(f'user={ev.user_id}  job={ev.job_title}  score={ev.score}')
			print('  matched:', ', '.join(f'{s.skill}({s.weight})' for s in ev.matched_skills[:5]) or '-')
			print('  missing:', ', '.join(ev.missing_skills[:6]) or '-')
			print('  reasons:', '; '.join(ev.reasons))
			print(f'  [{expl["source"]}] {expl["text"]}')

	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
	print(f'\nwrote {args.out}')


if __name__ == '__main__':
	main()
