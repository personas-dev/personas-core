"""Dataset loading and feature extraction for personas-core HR data."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
from datetime import date
import json
import math
import re
import zipfile

import numpy as np

MISSING_MARKERS = {"", "-", "-1", "\\N", "null", "none", "nan"}
POSITIVE_COLUMNS = ("browsed", "delivered", "satisfied")


@dataclass(frozen=True)
class CandidateRecord:
    """Candidate profile normalized from table1_user."""

    user_id: str
    text: str
    skills: tuple[str, ...]
    city_ids: tuple[str, ...]
    desired_city_ids: tuple[str, ...]
    industries: tuple[str, ...]
    desired_job_types: tuple[str, ...]
    degree: str
    years_experience: float


@dataclass(frozen=True)
class JobRecord:
    """Job profile normalized from table2_jd."""

    job_id: str
    title: str
    text: str
    skills: tuple[str, ...]
    city_id: str
    job_type: str
    min_degree: str
    min_years: float


@dataclass(frozen=True)
class InteractionRecord:
    """Candidate-job interaction with a weighted behavior label."""

    user_id: str
    job_id: str
    label: float
    row_index: int


@dataclass(frozen=True)
class DatasetBundle:
    """Loaded data plus deterministic train/valid/test splits."""

    candidates: dict[str, CandidateRecord]
    jobs: dict[str, JobRecord]
    interactions: list[InteractionRecord]
    train_by_user: dict[str, set[str]]
    valid_by_user: dict[str, set[str]]
    test_by_user: dict[str, set[str]]
    positive_by_user: dict[str, set[str]]
    kg_stats: dict[str, int]
    stats: dict[str, int | float | str]


def normalize_value(value: object) -> str:
    """Normalize a raw table value into a clean string."""

    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in MISSING_MARKERS:
        return ""
    return text


def split_multi_value(value: object) -> list[str]:
    """Split a semi-structured field into unique values."""

    text = normalize_value(value)
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for part in re.split(r"[|,，、;；/\n\r\t]+", text):
        token = clean_token(part)
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def clean_token(token: str) -> str:
    """Trim punctuation and normalize obvious English tokens."""

    text = normalize_value(token)
    text = text.strip().strip(" ：:;；,，。.!！？?()（）[]【】{}<>《》\"'“”‘’")
    text = re.sub(r"\s+", "", text)
    if re.fullmatch(r"[A-Za-z0-9+.#\-]+", text):
        text = text.lower()
    return text


def tokenize_text(value: object, max_tokens: int | None = None) -> list[str]:
    """Tokenize mixed Chinese/English HR text without external segmenters."""

    text = normalize_value(value).lower()
    if not text:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    coarse_parts = re.split(r"[|,，、;；/\n\r\t\s]+", text)
    for part in coarse_parts:
        part = clean_token(part)
        if not part:
            continue
        ascii_terms = re.findall(r"[a-z0-9+.#\-]{2,}", part)
        chinese_runs = re.findall(r"[\u4e00-\u9fff]{2,}", part)
        candidates = list(ascii_terms)
        for run in chinese_runs:
            if len(run) <= 6:
                candidates.append(run)
            else:
                candidates.extend(run[i : i + 2] for i in range(len(run) - 1))
                candidates.extend(run[i : i + 3] for i in range(len(run) - 2))
        if not candidates and 2 <= len(part) <= 32:
            candidates.append(part)
        for token in candidates:
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if max_tokens is not None and len(tokens) >= max_tokens:
                return tokens
    return tokens


def degree_rank(value: str) -> int:
    """Map common Chinese degree labels to an ordinal rank."""

    text = normalize_value(value)
    if not text:
        return 0
    order = ["初中", "高中", "中专", "大专", "本科", "硕士", "MBA", "博士"]
    for idx, label in enumerate(order, start=1):
        if label.lower() in text.lower():
            return idx
    return 0


def parse_years(value: str) -> float:
    """Parse work-year fields used by the source data."""

    text = normalize_value(value)
    if not text:
        return 0.0
    number = re.search(r"\d+", text)
    if not number:
        return 0.0
    raw = int(number.group(0))
    if 1900 <= raw <= 2100:
        return float(max(0, date.today().year - raw))
    if raw > 100:
        return max(0.0, raw / 100.0)
    return float(raw)


def stable_sample(items: list[str], limit: int | None) -> list[str]:
    """Return a deterministic prefix when a positive limit is supplied."""

    if limit is None or limit <= 0:
        return items
    return items[:limit]


def _read_tsv_from_zip(zip_path: Path, member_name: str, max_rows: int | None = None) -> list[dict[str, str]]:
    """Read a TSV member from a zip file with lenient row repair."""

    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as raw:
            header_line = raw.readline().decode("utf-8-sig", errors="replace").rstrip("\n\r")
            columns = header_line.split("\t")
            expected = len(columns)
            for index, line in enumerate(raw):
                if max_rows is not None and len(rows) >= max_rows:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n\r")
                parts = text.split("\t", expected - 1)
                if len(parts) < expected:
                    parts.extend([""] * (expected - len(parts)))
                elif len(parts) > expected:
                    parts = parts[: expected - 1] + ["\t".join(parts[expected - 1 :])]
                row = dict(zip(columns, parts, strict=False))
                row["__row_index__"] = str(index)
                rows.append(row)
    return rows


def _read_filtered_jobs(zip_path: Path, wanted_job_ids: set[str], max_jobs: int | None) -> dict[str, JobRecord]:
    """Read jobs that appear in sampled interactions, with deterministic filler jobs."""

    jobs: dict[str, JobRecord] = {}
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("datasets/table2_jd.txt") as raw:
            header_line = raw.readline().decode("utf-8-sig", errors="replace").rstrip("\n\r")
            columns = header_line.split("\t")
            expected = len(columns)
            for line in raw:
                if max_jobs is not None and len(jobs) >= max_jobs:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n\r")
                parts = text.split("\t", expected - 1)
                if len(parts) < expected:
                    parts.extend([""] * (expected - len(parts)))
                row = dict(zip(columns, parts, strict=False))
                job_id = normalize_value(row.get("jd_no"))
                if not job_id:
                    continue
                should_take = job_id in wanted_job_ids or (max_jobs is not None and len(jobs) < min(max_jobs, 500))
                if not should_take:
                    continue
                jobs[job_id] = make_job(row)
    return jobs


def read_kg_stats(kg_zip_path: Path | None) -> dict[str, int]:
    """Read graph statistics from kg_output.zip when available."""

    if kg_zip_path is None or not kg_zip_path.exists():
        return {}
    with zipfile.ZipFile(kg_zip_path) as archive:
        with archive.open("kg_output/graph_stats.csv") as raw:
            text = raw.read().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(text.splitlines())
    return {normalize_value(row.get("item")): int(float(normalize_value(row.get("count")) or 0)) for row in reader}


def make_candidate(row: dict[str, str]) -> CandidateRecord:
    """Create a normalized candidate profile from one source row."""

    user_id = normalize_value(row.get("user_id"))
    experience = normalize_value(row.get("experience"))
    desired_job_types = split_multi_value(row.get("desire_jd_type_id"))
    industries = split_multi_value(row.get("desire_jd_industry_id")) + split_multi_value(row.get("cur_industry_id"))
    skills = tuple(tokenize_text(experience, max_tokens=160))
    text_parts = [experience, " ".join(desired_job_types), " ".join(industries), normalize_value(row.get("cur_jd_type"))]
    return CandidateRecord(
        user_id=user_id,
        text=" ".join(part for part in text_parts if part),
        skills=skills,
        city_ids=tuple(split_multi_value(row.get("live_city_id"))),
        desired_city_ids=tuple(split_multi_value(row.get("desire_jd_city_id"))),
        industries=tuple(industries),
        desired_job_types=tuple(desired_job_types),
        degree=normalize_value(row.get("cur_degree_id")),
        years_experience=parse_years(normalize_value(row.get("start_work_date"))),
    )


def make_job(row: dict[str, str]) -> JobRecord:
    """Create a normalized job profile from one source row."""

    title = normalize_value(row.get("jd_title"))
    description = normalize_value(row.get("job_description"))
    job_type = normalize_value(row.get("jd_sub_type"))
    text = " ".join(part for part in (title, job_type, description) if part)
    return JobRecord(
        job_id=normalize_value(row.get("jd_no")),
        title=title,
        text=text,
        skills=tuple(tokenize_text(text, max_tokens=220)),
        city_id=normalize_value(row.get("city")),
        job_type=job_type,
        min_degree=normalize_value(row.get("min_edu_level")),
        min_years=parse_years(normalize_value(row.get("min_years"))),
    )


def interaction_label(row: dict[str, str]) -> float:
    """Convert browsed/delivered/satisfied flags into an ordered positive label."""

    browsed = normalize_value(row.get("browsed")) in {"1", "1.0", "true", "True"}
    delivered = normalize_value(row.get("delivered")) in {"1", "1.0", "true", "True"}
    satisfied = normalize_value(row.get("satisfied")) in {"1", "1.0", "true", "True"}
    if satisfied:
        return 3.0
    if delivered:
        return 2.0
    if browsed:
        return 1.0
    return 0.0


def split_positives(interactions: list[InteractionRecord]) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """Create deterministic per-user train/valid/test splits from positive rows."""

    grouped: dict[str, list[InteractionRecord]] = defaultdict(list)
    for item in interactions:
        if item.label > 0:
            grouped[item.user_id].append(item)
    train: dict[str, set[str]] = defaultdict(set)
    valid: dict[str, set[str]] = defaultdict(set)
    test: dict[str, set[str]] = defaultdict(set)
    for user_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row.row_index)
        if len(rows) >= 3:
            for row in rows[:-2]:
                train[user_id].add(row.job_id)
            valid[user_id].add(rows[-2].job_id)
            test[user_id].add(rows[-1].job_id)
        elif len(rows) == 2:
            train[user_id].add(rows[0].job_id)
            test[user_id].add(rows[1].job_id)
        elif rows:
            train[user_id].add(rows[0].job_id)
    return dict(train), dict(valid), dict(test)


def pair_features(candidate: CandidateRecord, job: JobRecord) -> np.ndarray:
    """Compute skill-path and rule features for one candidate-job pair."""

    candidate_skills = set(candidate.skills)
    job_skills = set(job.skills)
    shared = candidate_skills & job_skills
    coverage = len(shared) / max(1, len(job_skills))
    missing = max(0, len(job_skills) - len(shared)) / max(1, len(job_skills))
    city_match = float(job.city_id in set(candidate.city_ids) or job.city_id in set(candidate.desired_city_ids))
    education_match = float(degree_rank(candidate.degree) >= degree_rank(job.min_degree)) if job.min_degree else 0.0
    experience_match = float(candidate.years_experience >= job.min_years) if job.min_years else 0.0
    job_type_match = float(job.job_type in set(candidate.desired_job_types)) if job.job_type else 0.0
    path_count = math.log1p(len(shared))
    return np.array([coverage, missing, city_match, education_match, experience_match, job_type_match, path_count], dtype=np.float32)


def load_dataset_bundle(config: dict[str, object]) -> DatasetBundle:
    """Load a sampled training bundle from datasets.zip and kg_output.zip."""

    data_zip_path = Path(str(config.get("datasets_zip", "data/datasets.zip")))
    kg_zip_raw = config.get("kg_output_zip", "data/kg_output.zip")
    kg_zip_path = Path(str(kg_zip_raw)) if kg_zip_raw else None
    max_candidates = int(config.get("max_candidates", 0) or 0) or None
    max_jobs = int(config.get("max_jobs", 3000) or 3000)
    max_interactions = int(config.get("max_interactions", 120000) or 120000)

    action_rows = _read_tsv_from_zip(data_zip_path, "datasets/table3_action.txt", max_rows=max_interactions)
    raw_interactions = [
        InteractionRecord(
            user_id=normalize_value(row.get("user_id")),
            job_id=normalize_value(row.get("jd_no")),
            label=interaction_label(row),
            row_index=int(row.get("__row_index__", "0")),
        )
        for row in action_rows
    ]
    raw_interactions = [row for row in raw_interactions if row.user_id and row.job_id]
    wanted_user_ids = {row.user_id for row in raw_interactions}
    wanted_job_ids = {row.job_id for row in raw_interactions}

    candidate_rows = _read_tsv_from_zip(data_zip_path, "datasets/table1_user.txt", max_rows=None)
    candidates: dict[str, CandidateRecord] = {}
    for row in candidate_rows:
        user_id = normalize_value(row.get("user_id"))
        if not user_id or user_id not in wanted_user_ids:
            continue
        candidates[user_id] = make_candidate(row)
        if max_candidates is not None and len(candidates) >= max_candidates:
            break

    jobs = _read_filtered_jobs(data_zip_path, wanted_job_ids, max_jobs=max_jobs)
    filtered = [row for row in raw_interactions if row.user_id in candidates and row.job_id in jobs]
    train_by_user, valid_by_user, test_by_user = split_positives(filtered)
    positive_by_user: dict[str, set[str]] = defaultdict(set)
    for row in filtered:
        if row.label > 0:
            positive_by_user[row.user_id].add(row.job_id)

    kg_stats = read_kg_stats(kg_zip_path)
    stats: dict[str, int | float | str] = {
        "candidate_count": len(candidates),
        "job_count": len(jobs),
        "interaction_count": len(filtered),
        "positive_interaction_count": sum(1 for row in filtered if row.label > 0),
        "train_positive_count": sum(len(items) for items in train_by_user.values()),
        "valid_positive_count": sum(len(items) for items in valid_by_user.values()),
        "test_positive_count": sum(len(items) for items in test_by_user.values()),
        "kg_stats_source": str(kg_zip_path) if kg_zip_path else "",
    }
    for key, value in kg_stats.items():
        stats[f"kg_{key}"] = value
    return DatasetBundle(
        candidates=candidates,
        jobs=jobs,
        interactions=filtered,
        train_by_user=train_by_user,
        valid_by_user=valid_by_user,
        test_by_user=test_by_user,
        positive_by_user=dict(positive_by_user),
        kg_stats=kg_stats,
        stats=stats,
    )


def write_json(path: Path, data: object) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
