"""Skill text quality gates guided by Chinese occupational standards."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any

from jobmatch_gnn.data.dataset import clean_token

ASCII_TOKEN_RE = re.compile(r"^[a-z0-9+.#-]+$")
ASCII_BOUNDARY_RE = re.compile(r"[a-z0-9+#.]")

DEFAULT_SHORT_EN_ALLOWLIST = frozenset(
	{
		"ai",
		"bi",
		"c",
		"c#",
		"c++",
		"cv",
		"go",
		"hr",
		"js",
		"ml",
		"nlp",
		"oa",
		"os",
		"pm",
		"qa",
		"qc",
		"r",
		"ts",
		"ui",
		"ux",
	}
)

DEFAULT_DROP_TERMS = frozenset(
	{
		"or",
		"ex",
		"cc",
		"em",
		"自我评价",
		"协助",
		"客户",
		"材料",
		"资料",
		"学历",
		"经理",
		"员工",
		"福利",
		"管理",
		"管理工作",
		"工作",
		"岗位",
		"职位",
		"职责",
		"任职",
		"要求",
		"经验",
		"相关",
		"以上",
		"优先",
		"熟悉",
		"负责",
		"参与",
		"完成",
	}
)

DEFAULT_TRANSVERSAL_TERMS = frozenset(
	{
		"沟通",
		"沟通能力",
		"协调",
		"协调能力",
		"组织",
		"组织能力",
		"计划",
		"计划能力",
		"团队",
		"团队合作",
		"团队协作",
		"团队领导",
		"领导",
		"执行",
		"执行力",
		"责任心",
		"服务意识",
		"学习能力",
		"抗压能力",
		"表达能力",
		"办公",
	}
)


@dataclass(frozen=True)
class SkillDecision:
	"""Decision made for a normalized skill surface form."""

	token: str
	keep: bool
	skill_type: str
	weight: float
	reasons: tuple[str, ...] = ()


@dataclass
class SkillReference:
	"""Reference terms extracted from OSTA career and skill-standard systems."""

	occupations: set[str] = field(default_factory=set)
	categories: set[str] = field(default_factory=set)
	work_types: set[str] = field(default_factory=set)
	standards: set[str] = field(default_factory=set)
	task_terms: set[str] = field(default_factory=set)

	@property
	def occupation_like_terms(self) -> set[str]:
		"""Terms that describe occupations or job families rather than skills."""

		return self.occupations | self.categories | self.work_types


@dataclass
class SkillQualityConfig:
	"""Configurable rules for the skill ontology quality gate."""

	enabled: bool = True
	english_min_len: int = 3
	strict_ascii_boundaries: bool = True
	drop_occupation_terms: bool = True
	reference_path: Path | None = None
	short_english_allowlist: frozenset[str] = field(default_factory=lambda: DEFAULT_SHORT_EN_ALLOWLIST)
	drop_terms: frozenset[str] = field(default_factory=lambda: DEFAULT_DROP_TERMS)
	transversal_terms: frozenset[str] = field(default_factory=lambda: DEFAULT_TRANSVERSAL_TERMS)
	type_weights: dict[str, float] = field(
		default_factory=lambda: {
			"hard_or_knowledge": 1.0,
			"reference_task": 1.0,
			"transversal": 0.35,
			"unknown": 0.7,
		}
	)

	@classmethod
	def from_mapping(cls, raw: dict[str, Any] | None) -> "SkillQualityConfig":
		"""Create config from YAML-compatible mapping."""

		if not raw:
			return cls(enabled=False)
		reference_path = raw.get("reference_path")
		return cls(
			enabled=bool(raw.get("enabled", True)),
			english_min_len=int(raw.get("english_min_len", 3)),
			strict_ascii_boundaries=bool(raw.get("strict_ascii_boundaries", True)),
			drop_occupation_terms=bool(raw.get("drop_occupation_terms", True)),
			reference_path=Path(reference_path) if reference_path else None,
			short_english_allowlist=frozenset(
				clean_token(item).lower() for item in raw.get("short_english_allowlist", DEFAULT_SHORT_EN_ALLOWLIST)
			),
			drop_terms=frozenset(clean_token(item).lower() for item in raw.get("drop_terms", DEFAULT_DROP_TERMS)),
			transversal_terms=frozenset(
				clean_token(item).lower() for item in raw.get("transversal_terms", DEFAULT_TRANSVERSAL_TERMS)
			),
			type_weights={**cls().type_weights, **raw.get("type_weights", {})},
		)


class SkillQualityGate:
	"""Classify, filter, and weight skill tokens before KG construction."""

	def __init__(self, config: SkillQualityConfig | None = None, reference: SkillReference | None = None):
		self.config = config or SkillQualityConfig(enabled=reference is not None)
		self.reference = reference or load_reference(self.config.reference_path)
		self._decision_cache: dict[str, SkillDecision] = {}

	@classmethod
	def from_config(cls, raw: dict[str, Any] | None) -> "SkillQualityGate":
		"""Build a quality gate from the `skill_quality` config section."""

		return cls(SkillQualityConfig.from_mapping(raw))

	def decide(self, token: str) -> SkillDecision:
		"""Return the filtering/type decision for one token."""

		normalized = clean_token(token).lower()
		if normalized in self._decision_cache:
			return self._decision_cache[normalized]
		decision = self._decide_uncached(normalized)
		self._decision_cache[normalized] = decision
		return decision

	def keep_vocab_token(self, token: str) -> bool:
		"""Whether a token should be allowed into the skill vocabulary."""

		return self.decide(token).keep

	def match_weight(self, token: str) -> float:
		"""Weight multiplier used when a token is matched in job text."""

		return self.decide(token).weight

	def valid_text_match(self, text: str, start: int, end: int, token: str) -> bool:
		"""Validate an AC hit, especially English boundary matches."""

		if not self.config.enabled or not self.config.strict_ascii_boundaries:
			return True
		normalized = clean_token(token).lower()
		if not ASCII_TOKEN_RE.fullmatch(normalized):
			return True
		before = text[start - 1] if start > 0 else ""
		after = text[end + 1] if end + 1 < len(text) else ""
		return not ASCII_BOUNDARY_RE.fullmatch(before) and not ASCII_BOUNDARY_RE.fullmatch(after)

	def summary(self) -> dict[str, Any]:
		"""Compact metadata for preprocess meta.json."""

		return {
			"enabled": self.config.enabled,
			"reference_path": str(self.config.reference_path) if self.config.reference_path else "",
			"reference_occupations": len(self.reference.occupations),
			"reference_categories": len(self.reference.categories),
			"reference_work_types": len(self.reference.work_types),
			"reference_standards": len(self.reference.standards),
			"drop_terms": len(self.config.drop_terms),
			"transversal_terms": len(self.config.transversal_terms),
			"english_min_len": self.config.english_min_len,
		}

	def _decide_uncached(self, token: str) -> SkillDecision:
		if not self.config.enabled:
			return SkillDecision(token=token, keep=True, skill_type="unknown", weight=1.0)
		if not token:
			return SkillDecision(token=token, keep=False, skill_type="empty", weight=0.0, reasons=("empty",))
		if ASCII_TOKEN_RE.fullmatch(token) and len(token) < self.config.english_min_len and token not in self.config.short_english_allowlist:
			return SkillDecision(token=token, keep=False, skill_type="english_noise", weight=0.0, reasons=("short_english",))
		if token in self.config.drop_terms:
			return SkillDecision(token=token, keep=False, skill_type="generic_non_skill", weight=0.0, reasons=("drop_term",))
		if self.config.drop_occupation_terms and token in self.reference.occupation_like_terms:
			return SkillDecision(
				token=token,
				keep=False,
				skill_type="occupation_or_job_type",
				weight=0.0,
				reasons=("osta_occupation_reference",),
			)
		if token in self.config.transversal_terms:
			return SkillDecision(
				token=token,
				keep=True,
				skill_type="transversal",
				weight=float(self.config.type_weights.get("transversal", 0.35)),
				reasons=("transversal_downweight",),
			)
		if token in self.reference.task_terms or token in self.reference.standards:
			return SkillDecision(
				token=token,
				keep=True,
				skill_type="reference_task",
				weight=float(self.config.type_weights.get("reference_task", 1.0)),
				reasons=("osta_standard_reference",),
			)
		return SkillDecision(
			token=token,
			keep=True,
			skill_type="hard_or_knowledge",
			weight=float(self.config.type_weights.get("hard_or_knowledge", 1.0)),
		)


def load_reference(path: Path | None) -> SkillReference:
	"""Load an optional OSTA reference term JSON file."""

	reference = SkillReference()
	if path is None or not path.exists():
		return reference
	raw = json.loads(path.read_text(encoding="utf-8"))
	_add_terms(reference.categories, raw.get("career_categories", ()))
	_add_terms(reference.occupations, raw.get("occupation_terms", ()))
	_add_terms(reference.work_types, raw.get("work_type_terms", ()))
	_add_terms(reference.standards, raw.get("standard_terms", ()))
	_add_terms(reference.task_terms, raw.get("task_terms", ()))
	for item in raw.get("occupations", ()):
		if isinstance(item, dict):
			_add_terms(reference.occupations, (item.get("name"), item.get("careerName"), item.get("career_name")))
			_add_terms(reference.work_types, (work.get("name") for work in item.get("works", ()) if isinstance(work, dict)))
			_add_task_text(reference.task_terms, item.get("task", ""))
		else:
			_add_terms(reference.occupations, (item,))
	for item in raw.get("standards", ()):
		if isinstance(item, dict):
			_add_terms(reference.standards, (_standard_name(item.get("name", "")), _standard_name(item.get("standardInfoName", ""))))
		else:
			_add_terms(reference.standards, (_standard_name(str(item)),))
	return reference


def _add_terms(target: set[str], terms: Any) -> None:
	if not terms:
		return
	for term in terms:
		if not term:
			continue
		token = clean_token(str(term)).lower()
		if token:
			target.add(token)


def _add_task_text(target: set[str], text: str) -> None:
	for part in re.split(r"[;；,，、。.\n\r\t]+", text or ""):
		token = clean_token(part).lower()
		if 2 <= len(token) <= 32:
			target.add(token)


def _standard_name(value: str) -> str:
	text = re.sub(r"\([^)]*\)|（[^）]*）", "", value)
	text = re.sub(r"[-_]?(\d{4}年版|\d{6}|更新说明|\.pdf)$", "", text, flags=re.IGNORECASE)
	text = re.sub(r"^\d-\d{2}-\d{2}-\d{2}-?", "", text)
	return clean_token(text).lower()
