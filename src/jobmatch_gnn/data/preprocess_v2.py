"""Full-data preprocessing v2: clean skill vocab, aligned job skills, splits.

Run:
    python -m jobmatch_gnn.data.preprocess_v2 --config configs/v2_data.yaml

Outputs under ``data/processed/`` (see docs/11_data_kg_v2.md):
    users.parquet, jobs.parquet, interactions.parquet, skill_vocab.json, meta.json
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick
import pandas as pd
import yaml

from jobmatch_gnn.data.dataset import clean_token, degree_rank, normalize_value, parse_years

MISSING = {"", "-", "-1", "\\n", "null", "none", "nan"}


def _read_rows(zip_path: Path, member: str):
    """Yield dict rows from a TSV member inside the dataset zip."""

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as raw:
            columns = raw.readline().decode("utf-8-sig", errors="replace").rstrip("\n\r").split("\t")
            expected = len(columns)
            for index, line in enumerate(raw):
                parts = line.decode("utf-8", errors="replace").rstrip("\n\r").split("\t", expected - 1)
                if len(parts) < expected:
                    parts.extend([""] * (expected - len(parts)))
                row = dict(zip(columns, parts, strict=False))
                row["__row_index__"] = index
                yield row


def _split_tags(value: str) -> list[str]:
    """Split a pipe/comma separated tag field into clean unique tokens."""

    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[|,，、;；/]+", value):
        token = clean_token(part)
        if len(token) >= 2 and token.lower() not in MISSING and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def build_skill_vocab(user_rows: list[dict], min_count: int) -> dict[str, int]:
    """Build the skill vocabulary from user experience tags (docs/11 S1-S2)."""

    counter: Counter[str] = Counter()
    for row in user_rows:
        counter.update(_split_tags(normalize_value(row.get("experience"))))
    kept = sorted(tag for tag, count in counter.items() if count >= min_count)
    return {tag: idx for idx, tag in enumerate(kept)}


def build_automaton(vocab: dict[str, int]) -> ahocorasick.Automaton:
    """Compile the skill vocabulary into an Aho-Corasick automaton."""

    automaton = ahocorasick.Automaton()
    for tag, idx in vocab.items():
        automaton.add_word(tag.lower(), (idx, len(tag)))
    automaton.make_automaton()
    return automaton


def match_job_skills(
    automaton: ahocorasick.Automaton,
    title: str,
    description: str,
    max_skills: int,
    title_weight: float,
) -> list[tuple[int, float]]:
    """Extract weighted vocabulary skills from job text (docs/11 S3-S4)."""

    scores: dict[int, float] = defaultdict(float)
    for text, weight in ((title.lower(), title_weight), (description.lower(), 1.0)):
        if not text:
            continue
        for _, (skill_idx, length) in automaton.iter(text):
            scores[skill_idx] += weight * (1.0 + 0.1 * length)
    top = sorted(scores.items(), key=lambda item: -item[1])[:max_skills]
    return [(idx, round(score, 3)) for idx, score in top]


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_zip = Path(config["datasets_zip"])
    out_dir = Path(config["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    min_count = int(config.get("skill_min_count", 2))
    max_job_skills = int(config.get("max_job_skills", 30))
    max_user_skills = int(config.get("max_user_skills", 64))
    title_weight = float(config.get("title_weight", 2.0))
    max_desc_chars = int(config.get("max_desc_chars", 1200))

    print("[1/5] reading users ...", flush=True)
    user_rows = list(_read_rows(data_zip, "datasets/table1_user.txt"))
    vocab = build_skill_vocab(user_rows, min_count=min_count)
    automaton = build_automaton(vocab)
    print(f"  users={len(user_rows)} skill_vocab={len(vocab)}", flush=True)

    users = []
    for row in user_rows:
        user_id = normalize_value(row.get("user_id"))
        if not user_id:
            continue
        tags = _split_tags(normalize_value(row.get("experience")))
        skill_ids = [vocab[t] for t in tags if t in vocab][:max_user_skills]
        desired_types = _split_tags(normalize_value(row.get("desire_jd_type_id")))
        industries = _split_tags(normalize_value(row.get("desire_jd_industry_id"))) + _split_tags(
            normalize_value(row.get("cur_industry_id"))
        )
        text = " ".join(
            part
            for part in (
                " ".join(tags[:48]),
                " ".join(desired_types),
                " ".join(industries[:6]),
                normalize_value(row.get("cur_jd_type")),
                normalize_value(row.get("cur_degree_id")),
            )
            if part
        )
        users.append(
            {
                "user_id": user_id,
                "skill_ids": skill_ids,
                "live_city": normalize_value(row.get("live_city_id")),
                "desired_cities": _split_tags(normalize_value(row.get("desire_jd_city_id"))),
                "desired_types": desired_types,
                "industries": industries,
                "degree_rank": degree_rank(normalize_value(row.get("cur_degree_id"))),
                "years": parse_years(normalize_value(row.get("start_work_date"))),
                "text": text,
            }
        )
    users_df = pd.DataFrame(users)

    print("[2/5] reading actions ...", flush=True)
    inter = []
    wanted_jobs: set[str] = set()
    for row in _read_rows(data_zip, "datasets/table3_action.txt"):
        user_id = normalize_value(row.get("user_id"))
        job_id = normalize_value(row.get("jd_no"))
        if not user_id or not job_id:
            continue
        satisfied = row.get("satisfied") == "1"
        delivered = row.get("delivered") == "1"
        browsed = row.get("browsed") == "1"
        level = 3 if satisfied else 2 if delivered else 1 if browsed else 0
        if level == 0:
            continue
        wanted_jobs.add(job_id)
        inter.append({"user_id": user_id, "job_id": job_id, "level": level, "row_index": row["__row_index__"]})
    inter_df = pd.DataFrame(inter)
    print(f"  interactions(level>0)={len(inter_df)} jobs_seen={len(wanted_jobs)}", flush=True)

    print("[3/5] reading + skill-matching jobs ...", flush=True)
    jobs = []
    for row in _read_rows(data_zip, "datasets/table2_jd.txt"):
        job_id = normalize_value(row.get("jd_no"))
        if not job_id or job_id not in wanted_jobs:
            continue
        title = normalize_value(row.get("jd_title"))
        description = normalize_value(row.get("job_description"))[:max_desc_chars]
        skills = match_job_skills(automaton, title, description, max_job_skills, title_weight)
        jobs.append(
            {
                "job_id": job_id,
                "title": title,
                "city": normalize_value(row.get("city")),
                "job_type": normalize_value(row.get("jd_sub_type")),
                "min_degree_rank": degree_rank(normalize_value(row.get("min_edu_level"))),
                "min_years": parse_years(normalize_value(row.get("min_years"))),
                "skill_ids": [s for s, _ in skills],
                "skill_weights": [w for _, w in skills],
                "text": " ".join(p for p in (title, normalize_value(row.get("jd_sub_type")), description[:480]) if p),
            }
        )
        if len(jobs) % 50000 == 0:
            print(f"  matched {len(jobs)} jobs", flush=True)
    jobs_df = pd.DataFrame(jobs)

    print("[4/5] splitting ...", flush=True)
    known_users = set(users_df["user_id"])
    known_jobs = set(jobs_df["job_id"])
    inter_df = inter_df[inter_df["user_id"].isin(known_users) & inter_df["job_id"].isin(known_jobs)].copy()
    inter_df["split"] = "train"
    positives = inter_df[inter_df["level"] >= 2].sort_values("row_index")
    for _, group in positives.groupby("user_id", sort=False):
        rows = group.index.to_list()
        if len(rows) >= 3:
            inter_df.loc[rows[-1], "split"] = "test"
            inter_df.loc[rows[-2], "split"] = "valid"
        elif len(rows) == 2:
            inter_df.loc[rows[-1], "split"] = "test"

    print("[5/5] writing parquet ...", flush=True)
    users_df.to_parquet(out_dir / "users.parquet", index=False)
    jobs_df.to_parquet(out_dir / "jobs.parquet", index=False)
    inter_df.to_parquet(out_dir / "interactions.parquet", index=False)
    (out_dir / "skill_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    meta = {
        "users": len(users_df),
        "jobs": len(jobs_df),
        "interactions": len(inter_df),
        "positives": int((inter_df["level"] >= 2).sum()),
        "train_pos": int(((inter_df["level"] >= 2) & (inter_df["split"] == "train")).sum()),
        "valid_pos": int((inter_df["split"] == "valid").sum()),
        "test_pos": int((inter_df["split"] == "test").sum()),
        "skill_vocab": len(vocab),
        "avg_job_skills": float(jobs_df["skill_ids"].map(len).mean()),
        "avg_user_skills": float(users_df["skill_ids"].map(len).mean()),
        "config": config,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v2_data.yaml"))
    run(parser.parse_args().config)
