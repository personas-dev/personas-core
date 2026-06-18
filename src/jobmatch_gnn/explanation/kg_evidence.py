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


def _json_safe(value: object) -> object:
	"""Convert numpy/pandas-ish values into JSON-friendly primitives."""

	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return [_json_safe(item) for item in value.tolist()]
	if isinstance(value, (list, tuple, set)):
		return [_json_safe(item) for item in value if item is not None and str(item) != '']
	if value is None:
		return None
	return value


def _attrs(**attributes: object) -> dict[str, object]:
	return {key: _json_safe(value) for key, value in attributes.items() if value is not None and str(value) != ''}


def _node(node_id: str, node_type: str, label: str, role: str, **attributes: object) -> dict[str, object]:
	return {'id': node_id, 'type': node_type, 'label': label, 'role': role, 'attributes': _attrs(**attributes)}


def _edge(source: str, target: str, relation: str, **attributes: object) -> dict[str, object]:
	return {'source': source, 'target': target, 'relation': relation, 'attributes': _attrs(**attributes)}


def _as_list(value: object) -> list[object]:
	if value is None:
		return []
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, (list, tuple, set)):
		return list(value)
	return [value]


def build_local_subgraph(
	user_id: str,
	job_id: str,
	job_title: str,
	score: float,
	matched_skills: list[SkillWeight],
	missing_skills: list[str],
	graph_paths: list[str],
	reasons: list[str],
	candidate_context: dict[str, object] | None = None,
	job_context: dict[str, object] | None = None,
	max_missing: int = 8,
) -> dict[str, object]:
	"""Build a compact local KG neighborhood around one candidate-job pair."""

	candidate_context = candidate_context or {}
	job_context = job_context or {}
	candidate_node = f'candidate:{user_id}'
	job_node = f'job:{job_id}'
	nodes = [
		_node(candidate_node, 'Candidate', user_id, 'query_candidate', **candidate_context),
		_node(job_node, 'Job', job_title, 'recommended_job', job_id=job_id, score=round(float(score), 4), **job_context),
	]
	edges = [_edge(candidate_node, job_node, 'MATCHED_TO', score=round(float(score), 4))]
	structured_paths: list[dict[str, object]] = []

	for sw in matched_skills[:10]:
		skill_node = f'skill:{sw.skill}'
		nodes.append(_node(skill_node, 'Skill', sw.skill, 'matched_skill', attention=sw.weight))
		edges.append(_edge(candidate_node, skill_node, 'HAS_SKILL', attention=sw.weight, status='matched'))
		edges.append(_edge(job_node, skill_node, 'REQUIRES_SKILL', attention=sw.weight, status='matched'))
		structured_paths.append(
			{
				'nodes': [candidate_node, skill_node, job_node],
				'relations': ['HAS_SKILL', 'REQUIRES_SKILL'],
				'support': 'matched_skill_attention',
				'weight': sw.weight,
			}
		)

	for skill in missing_skills[:max_missing]:
		skill_node = f'skill:{skill}'
		nodes.append(_node(skill_node, 'Skill', skill, 'missing_required_skill'))
		edges.append(_edge(job_node, skill_node, 'REQUIRES_SKILL', status='missing_for_candidate'))

	live_city = candidate_context.get('live_city')
	desired_cities = _as_list(candidate_context.get('desired_cities'))
	job_city = job_context.get('city')
	for city in {str(city) for city in [live_city, job_city, *desired_cities] if city}:
		city_node = f'city:{city}'
		nodes.append(_node(city_node, 'City', city, 'context_city'))
		if city == live_city:
			edges.append(_edge(candidate_node, city_node, 'LIVES_IN'))
		if city in desired_cities:
			edges.append(_edge(candidate_node, city_node, 'DESIRES_CITY'))
		if city == job_city:
			edges.append(_edge(job_node, city_node, 'LOCATED_IN'))

	desired_types = _as_list(candidate_context.get('desired_types'))
	job_type = job_context.get('job_type')
	for job_type_value in {str(value) for value in [job_type, *desired_types] if value}:
		type_node = f'job_type:{job_type_value}'
		nodes.append(_node(type_node, 'JobType', job_type_value, 'context_job_type'))
		if job_type_value in desired_types:
			edges.append(_edge(candidate_node, type_node, 'DESIRES_TYPE'))
		if job_type_value == job_type:
			edges.append(_edge(job_node, type_node, 'OF_TYPE'))

	return {
		'schema_version': 'local_kg_v1',
		'focus_pair': {'candidate_id': user_id, 'job_id': job_id, 'job_title': job_title, 'score': round(float(score), 4)},
		'nodes': nodes,
		'edges': edges,
		'graph_paths': graph_paths,
		'structured_paths': structured_paths,
		'reason_evidence': reasons,
	}


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
	local_subgraph: dict[str, object] = field(default_factory=dict)

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

		user_id = str(bundle.users.iloc[user_idx]['user_id'])
		job_id = str(bundle.jobs.iloc[job_idx]['job_id'])
		paths = [f'Candidate:{user_id} -HAS_SKILL({sw.weight})-> Skill:{sw.skill} <-REQUIRES_SKILL- Job:{job_id}' for sw in matched[:5]]
		missing_skills = [self.skills[s] for s in missing[:12]]
		candidate_context = {
			'live_city': user['live_city'],
			'desired_cities': list(user['desired_cities']),
			'desired_types': list(user['desired_types']),
			'degree_rank': user['degree_rank'],
			'years': user['years'],
		}
		job_context = {
			'city': job['city'],
			'job_type': job['job_type'],
			'min_degree_rank': job['min_degree_rank'],
			'min_years': job['min_years'],
		}
		local_subgraph = build_local_subgraph(
			user_id=str(user_id),
			job_id=str(job_id),
			job_title=str(job['title']),
			score=round(score_val, 4),
			matched_skills=matched,
			missing_skills=missing_skills,
			graph_paths=paths,
			reasons=reasons,
			candidate_context=candidate_context,
			job_context=job_context,
		)
		return MatchEvidence(
			user_id=user_id,
			job_id=job_id,
			job_title=str(job['title']),
			score=round(score_val, 4),
			matched_skills=matched,
			missing_skills=missing_skills,
			graph_paths=paths,
			reasons=reasons,
			local_subgraph=local_subgraph,
		)
