"""Extract faithful, hallucination-free match evidence for a (user, job) pair.

Everything here is computed directly from the data and the trained model's
skill-path attention; no text is generated. This is the *fact layer* that the
optional LLM expression layer (llm_explainer) is constrained to.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from jobmatch_gnn.data.bundle_v2 import BundleV2
from jobmatch_gnn.models.spc_hgt_v2 import MAX_PATH, SPCHGTv2, build_graph_tensors
from jobmatch_gnn.training.train_spc_hgt_v2 import PairFeaturizer, _to_dev


@dataclass
class SkillWeight:
	skill: str
	weight: float


@dataclass
class MatchEvidence:
	"""Structured, fully faithful evidence for one candidate-job match."""

	user_id: str
	job_id: str
	job_title: str
	score: float
	matched_skills: list[SkillWeight] = field(default_factory=list)
	missing_skills: list[str] = field(default_factory=list)
	graph_paths: list[str] = field(default_factory=list)
	reasons: list[str] = field(default_factory=list)

	def skill_whitelist(self) -> set[str]:
		"""All skill strings the LLM is allowed to mention (faithfulness guard)."""

		return {sw.skill for sw in self.matched_skills} | set(self.missing_skills)

	def to_json(self) -> str:
		return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class EvidenceExtractor:
	"""Load a trained SPC-HGT v2 checkpoint and emit per-pair evidence."""

	def __init__(self, bundle: BundleV2, checkpoint_path: str | Path, device: str = 'cuda') -> None:
		self.bundle = bundle
		self.device = device
		self.featurizer = PairFeaturizer(bundle)
		self.skills = [
			tag
			for tag, _ in sorted(
				json.loads((bundle.processed_dir / 'skill_vocab.json').read_text(encoding='utf-8')).items(),
				key=lambda kv: kv[1],
			)
		]
		ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
		meta = ckpt['meta']
		self.model = SPCHGTv2(
			ckpt['sbert_dim'],
			ckpt['dim'],
			meta['num_cities'],
			meta['num_types'],
			num_users=bundle.num_users,
			num_jobs=bundle.num_jobs,
			use_path=True,
			use_text=True,
			use_id=ckpt['config'].get('use_id', True),
		).to(device)
		self.model.load_state_dict(ckpt['state_dict'])
		self.model.eval()
		self.feats, self.edges, _ = build_graph_tensors(bundle, device)
		with torch.no_grad():
			self.z = self.model.encode(self.feats, self.edges)

	def rank(self, user_idx: int, topn: int = 10) -> list[int]:
		"""Return top-N job indices for a user (dot-product recall, for demo)."""

		with torch.no_grad():
			scores = (self.z['c'][user_idx] @ self.z['j'].T).cpu().numpy()
		exclude = self.bundle.train_pos.get(user_idx, set()) | self.bundle.valid_pos.get(user_idx, set())
		if exclude:
			scores[np.fromiter(exclude, dtype=np.int64)] = -np.inf
		return np.argsort(-scores)[:topn].tolist()

	def evidence(self, user_idx: int, job_idx: int) -> MatchEvidence:
		"""Compute faithful evidence with skill-path attention weights."""

		bundle = self.bundle
		u_arr = np.array([user_idx], dtype=np.int64)
		j_arr = np.array([job_idx], dtype=np.int64)
		feats = _to_dev(self.featurizer.build(u_arr, j_arr), self.device)
		with torch.no_grad():
			score, alpha = self.model.score_pairs(
				self.z,
				torch.tensor(u_arr, device=self.device),
				torch.tensor(j_arr, device=self.device),
				feats[4],
				feats[0],
				feats[1],
				feats[2],
				feats[3],
				return_alpha=True,
			)
		score_val = float(score.item())

		user_skills = set(bundle.users.iloc[user_idx]['skill_ids'])
		job_skills = list(bundle.jobs.iloc[job_idx]['skill_ids'])
		shared = [s for s in job_skills if s in user_skills][:MAX_PATH]
		missing = [s for s in job_skills if s not in user_skills]

		matched: list[SkillWeight] = []
		if alpha is not None and shared:
			weights = alpha[0].cpu().numpy()[: len(shared)]
			for sid, w in sorted(zip(shared, weights, strict=False), key=lambda kv: -kv[1]):
				matched.append(SkillWeight(skill=self.skills[sid], weight=round(float(w), 4)))
		else:
			matched = [SkillWeight(skill=self.skills[s], weight=0.0) for s in shared]

		job = bundle.jobs.iloc[job_idx]
		user = bundle.users.iloc[user_idx]
		reasons: list[str] = []
		coverage = len(shared) / max(1, len(job_skills))
		reasons.append(f'技能覆盖率 {coverage:.0%}({len(shared)}/{len(job_skills)} 项要求技能匹配)')
		if job['city'] in (set(user['desired_cities']) | {user['live_city']}):
			reasons.append('工作城市符合期望')
		if user['degree_rank'] >= job['min_degree_rank'] and job['min_degree_rank'] > 0:
			reasons.append('学历满足岗位要求')
		if user['years'] >= job['min_years'] and job['min_years'] > 0:
			reasons.append('工作年限满足岗位要求')
		if job['job_type'] in set(user['desired_types']):
			reasons.append('岗位类型符合求职意向')

		paths = [f'候选人 → {sw.skill} ← 岗位' for sw in matched[:5]]
		return MatchEvidence(
			user_id=bundle.users.iloc[user_idx]['user_id'],
			job_id=bundle.jobs.iloc[job_idx]['job_id'],
			job_title=str(job['title']),
			score=round(score_val, 4),
			matched_skills=matched,
			missing_skills=[self.skills[s] for s in missing[:12]],
			graph_paths=paths,
			reasons=reasons,
		)
