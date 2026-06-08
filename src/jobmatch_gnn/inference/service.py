"""Frontend-facing recommendation service wrapper."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from jobmatch_gnn.baselines import BM25Matcher, HashSemanticMatcher, RuleMatcher
from jobmatch_gnn.data.dataset import CandidateRecord, DatasetBundle, JobRecord, load_dataset_bundle, pair_features, split_multi_value, tokenize_text
from jobmatch_gnn.training.train import load_config

SUPPORTED_MODELS = {"rule", "bm25", "semantic_hash", "spc_hgt"}


@dataclass(frozen=True)
class CandidateProfileInput:
    """Candidate profile payload accepted by the recommendation API."""

    candidate_id: str = "anonymous"
    skills: list[str] = field(default_factory=list)
    experience_text: str = ""
    desired_city_ids: list[str] = field(default_factory=list)
    current_city_ids: list[str] = field(default_factory=list)
    industries: list[str] = field(default_factory=list)
    desired_job_types: list[str] = field(default_factory=list)
    degree: str = ""
    years_experience: float = 0.0


@dataclass(frozen=True)
class RecommendationRequest:
    """Recommendation request used by HTTP and direct Python callers."""

    candidate_id: str | None = None
    candidate_profile: CandidateProfileInput | None = None
    model: str = "spc_hgt"
    top_k: int = 10
    include_explanations: bool = True
    request_id: str | None = None


@dataclass(frozen=True)
class RecommendationItem:
    """One ranked job recommendation with structured explanation evidence."""

    rank: int
    job_id: str
    title: str
    score: float
    city_id: str
    job_type: str
    matched_skills: list[str]
    missing_skills: list[str]
    graph_paths: list[str]
    reasons: list[str]
    feature_evidence: dict[str, float]


@dataclass(frozen=True)
class RecommendationResponse:
    """Recommendation response returned to frontend callers."""

    request_id: str
    model: str
    candidate_id: str
    generated_at: str
    top_k: int
    recommendations: list[RecommendationItem]
    warnings: list[str] = field(default_factory=list)


class RecommendationError(ValueError):
    """Typed error raised for invalid recommendation requests."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class HRKGRecommender:
    """Load algorithm artifacts and serve Top-K person-job recommendations."""

    def __init__(self, bundle: DatasetBundle, prediction_orders: dict[str, list[str]] | None = None) -> None:
        self.bundle = bundle
        self.prediction_orders = prediction_orders or {}
        self.rule_matcher = RuleMatcher()
        self.bm25_matcher = BM25Matcher(bundle.jobs)
        self.semantic_matcher = HashSemanticMatcher(bundle.jobs)

    @classmethod
    def from_config(cls, config_path: Path, run_dir: Path | None = None, model_name: str = "spc_hgt") -> "HRKGRecommender":
        """Build a recommender from a training config and optional run prediction directory."""

        config = load_config(config_path)
        bundle = load_dataset_bundle(dict(config.get("data", {})))
        prediction_orders = load_prediction_orders(run_dir, model_name) if run_dir else {}
        return cls(bundle=bundle, prediction_orders=prediction_orders)

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """Return Top-K job recommendations for one candidate request."""

        model = request.model.strip().lower()
        if model not in SUPPORTED_MODELS:
            raise RecommendationError("UNSUPPORTED_MODEL", f"Unsupported model '{request.model}'. Supported: {sorted(SUPPORTED_MODELS)}")
        top_k = min(max(int(request.top_k), 1), 100)
        candidate = self._resolve_candidate(request)
        warnings: list[str] = []
        ranked_jobs, actual_model = self._rank_jobs(candidate, model, warnings)
        items = [
            self._build_item(candidate, job_id, rank, actual_model, request.include_explanations)
            for rank, job_id in enumerate(ranked_jobs[:top_k], start=1)
        ]
        return RecommendationResponse(
            request_id=request.request_id or str(uuid4()),
            model=actual_model,
            candidate_id=candidate.user_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            top_k=top_k,
            recommendations=items,
            warnings=warnings,
        )

    def to_dict(self, response: RecommendationResponse) -> dict[str, Any]:
        """Convert a response dataclass to JSON-serializable dictionaries."""

        return asdict(response)

    def _resolve_candidate(self, request: RecommendationRequest) -> CandidateRecord:
        if request.candidate_profile is not None:
            return candidate_input_to_record(request.candidate_profile)
        if request.candidate_id is None:
            raise RecommendationError("MISSING_CANDIDATE", "Provide either candidate_id or candidate_profile.")
        candidate = self.bundle.candidates.get(request.candidate_id)
        if candidate is None:
            raise RecommendationError("CANDIDATE_NOT_FOUND", f"Candidate '{request.candidate_id}' was not found in the loaded dataset sample.")
        return candidate

    def _rank_jobs(self, candidate: CandidateRecord, model: str, warnings: list[str]) -> tuple[list[str], str]:
        if model == "spc_hgt" and candidate.user_id in self.prediction_orders:
            known_jobs = [job_id for job_id in self.prediction_orders[candidate.user_id] if job_id in self.bundle.jobs]
            if known_jobs:
                return known_jobs, "spc_hgt"
        if model == "spc_hgt":
            warnings.append("SPC-HGT prediction order is unavailable for this candidate; falling back to rule ranking.")
            model = "rule"
        matcher = {"rule": self.rule_matcher, "bm25": self.bm25_matcher, "semantic_hash": self.semantic_matcher}[model]
        scored = [(job_id, float(matcher.score(candidate, job))) for job_id, job in self.bundle.jobs.items()]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [job_id for job_id, _ in scored], model

    def _build_item(self, candidate: CandidateRecord, job_id: str, rank: int, model: str, include_explanations: bool) -> RecommendationItem:
        job = self.bundle.jobs[job_id]
        score = self._display_score(candidate, job, rank, model)
        evidence = pair_feature_dict(candidate, job)
        if include_explanations:
            matched, missing, paths, reasons = explain_pair(candidate, job, evidence)
        else:
            matched, missing, paths, reasons = [], [], [], []
        return RecommendationItem(
            rank=rank,
            job_id=job.job_id,
            title=job.title,
            score=score,
            city_id=job.city_id,
            job_type=job.job_type,
            matched_skills=matched,
            missing_skills=missing,
            graph_paths=paths,
            reasons=reasons,
            feature_evidence=evidence,
        )

    def _display_score(self, candidate: CandidateRecord, job: JobRecord, rank: int, model: str) -> float:
        if model == "spc_hgt" and candidate.user_id in self.prediction_orders:
            return round(1.0 / max(rank, 1), 6)
        if model == "bm25":
            return float(self.bm25_matcher.score(candidate, job))
        if model == "semantic_hash":
            return float(self.semantic_matcher.score(candidate, job))
        return float(self.rule_matcher.score(candidate, job))


def candidate_input_to_record(payload: CandidateProfileInput) -> CandidateRecord:
    """Convert API candidate profile input into the internal candidate record."""

    text = " ".join([payload.experience_text, " ".join(payload.skills), " ".join(payload.industries), " ".join(payload.desired_job_types)]).strip()
    explicit_skills = [token for value in payload.skills for token in split_multi_value(value)]
    text_skills = tokenize_text(payload.experience_text, max_tokens=160)
    seen: set[str] = set()
    skills = []
    for token in [*explicit_skills, *text_skills]:
        if token not in seen:
            seen.add(token)
            skills.append(token)
    return CandidateRecord(
        user_id=payload.candidate_id or "anonymous",
        text=text,
        skills=tuple(skills),
        city_ids=tuple(payload.current_city_ids),
        desired_city_ids=tuple(payload.desired_city_ids),
        industries=tuple(payload.industries),
        desired_job_types=tuple(payload.desired_job_types),
        degree=payload.degree,
        years_experience=float(payload.years_experience),
    )


def pair_feature_dict(candidate: CandidateRecord, job: JobRecord) -> dict[str, float]:
    """Return named feature evidence for one candidate-job pair."""

    values = pair_features(candidate, job).tolist()
    keys = [
        "required_skill_coverage",
        "missing_skill_ratio",
        "city_match",
        "education_match",
        "experience_match",
        "job_type_match",
        "skill_path_count_log1p",
    ]
    return {key: round(float(value), 6) for key, value in zip(keys, values, strict=False)}


def explain_pair(candidate: CandidateRecord, job: JobRecord, evidence: dict[str, float]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Generate structured explanation fields for a ranked pair."""

    candidate_skills = set(candidate.skills)
    job_skills = set(job.skills)
    matched = sorted(candidate_skills & job_skills)[:20]
    missing = sorted(job_skills - candidate_skills)[:20]
    paths = [f"Candidate:{candidate.user_id} -> Skill:{skill} <- Job:{job.job_id}" for skill in matched[:5]]
    reasons: list[str] = []
    if evidence.get("required_skill_coverage", 0.0) > 0:
        reasons.append(f"技能覆盖度 {evidence['required_skill_coverage']:.2f}")
    if evidence.get("city_match", 0.0) >= 1.0:
        reasons.append("城市匹配")
    if evidence.get("education_match", 0.0) >= 1.0:
        reasons.append("学历满足岗位要求")
    if evidence.get("experience_match", 0.0) >= 1.0:
        reasons.append("经验满足岗位要求")
    if evidence.get("job_type_match", 0.0) >= 1.0:
        reasons.append("岗位类型符合期望")
    if not reasons:
        reasons.append("根据当前模型排序进入 Top-K")
    return matched, missing, paths, reasons


def load_prediction_orders(run_dir: Path | None, model_name: str) -> dict[str, list[str]]:
    """Load optional prediction order JSON emitted by training."""

    if run_dir is None:
        return {}
    path = run_dir / "predictions" / f"{model_name}.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return {str(user_id): [str(job_id) for job_id in job_ids] for user_id, job_ids in raw.items() if isinstance(job_ids, list)}
