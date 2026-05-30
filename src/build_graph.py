"""Build a recruitment knowledge graph from personas-core/datasets.

This script reads:

* table1_user.txt
* table2_jd.txt
* table3_action.txt

It then exports a lightweight heterogeneous graph as CSV files:

* nodes.csv
* edges.csv
* graph_stats.csv

The implementation is dependency-light and uses rule-based extraction only.

Usage:

      python build_graph.py --data-dir datasets --output-dir kg_output
      python build_graph.py --max-rows 2000
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


MISSING_MARKERS = {"", "-", "-1", "\\N", "null", "none", "nan"}
ACTION_WEIGHTS = {
      "browsed": 1,
      "delivered": 5,
      "satisfied": 10,
}

SKILL_STOPWORDS = {
      "岗位职责",
      "任职要求",
      "任职资格",
      "职位要求",
      "岗位要求",
      "职责",
      "要求",
      "其他",
      "工作",
      "岗位",
      "能力",
      "经验",
      "相关",
      "以上",
      "良好",
      "熟练",
      "较强",
      "具有",
      "具备",
      "优先",
      "优先考虑",
      "优先录用",
      "完成",
      "负责",
      "协助",
      "支持",
      "配合",
      "管理",
      "处理",
      "执行",
      "学习",
      "学习能力",
      "自我评价",
      "团队",
      "团队领导",
      "团队协作",
      "团队合作",
}


def normalize_value(value: object) -> str:
      """Normalize raw cell values into clean strings."""

      if pd.isna(value):
            return ""
      text = str(value).strip()
      if not text:
            return ""
      if text.lower() in MISSING_MARKERS:
            return ""
      return text


def clean_token(token: str) -> str:
      """Remove leading/trailing punctuation and whitespace from a token."""

      text = normalize_value(token)
      if not text:
            return ""
      text = text.strip().strip(" ：:;；,，。.!！？?()（）[]【】{}<>《》\"'“”‘’")
      text = re.sub(r"\s+", "", text)
      return text


def split_multi_value(value: object) -> List[str]:
      """Split comma/pipe separated fields into unique cleaned values."""

      text = normalize_value(value)
      if not text:
            return []

      parts = re.split(r"[|,，、;；/\n\r\t]+", text)
      values: List[str] = []
      seen = set()
      for part in parts:
            token = clean_token(part)
            if not token or token in seen:
                  continue
            seen.add(token)
            values.append(token)
      return values


def extract_skill_tokens(value: object, max_tokens: int | None = None) -> List[str]:
      """Extract skill-like tokens from a text field.

      The data is already semi-structured. We split on common separators and
      keep short, meaningful tokens. English technical words are normalized to
      lower case.
      """

      text = normalize_value(value)
      if not text:
            return []

      parts = re.split(r"[|,，、;；\n\r\t]+", text)
      tokens: List[str] = []
      seen = set()

      for part in parts:
            sub_parts = re.split(r"[/／]+", part)
            for sub_part in sub_parts:
                  token = clean_token(sub_part)
                  if not token:
                        continue
                  if token in SKILL_STOPWORDS:
                        continue
                  if len(token) < 2 and not re.search(r"[A-Za-z0-9]", token):
                        continue
                  if len(token) > 32:
                        continue
                  if re.fullmatch(r"[A-Za-z0-9+.#\-]+", token):
                        token = token.lower()
                  if token in seen:
                        continue
                  seen.add(token)
                  tokens.append(token)
                  if max_tokens is not None and len(tokens) >= max_tokens:
                        return tokens

      return tokens


def make_node_id(node_type: str, value: str) -> str:
      """Create a stable node identifier."""

      return f"{node_type.upper()}::{value}"


def make_node_rows(node_type: str, values: Iterable[str]) -> List[dict]:
      """Convert a value collection into node rows."""

      rows: List[dict] = []
      for value in values:
            cleaned = clean_token(value)
            if not cleaned:
                  continue
            rows.append(
                  {
                        "node_id": make_node_id(node_type, cleaned),
                        "node_type": node_type,
                        "node_name": cleaned,
                        "attributes_json": json.dumps({"name": cleaned}, ensure_ascii=False),
                  }
            )
      return rows


def read_table(path: Path) -> pd.DataFrame:
      """Read a tab-separated file as strings."""

      with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = [line.rstrip("\n\r") for line in handle]

      if not lines:
            return pd.DataFrame()

      header = lines[0].split("\t")
      expected_cols = len(header)
      rows = []

      for line in lines[1:]:
            parts = line.split("\t", expected_cols - 1)
            if len(parts) < expected_cols:
                  parts.extend([""] * (expected_cols - len(parts)))
            elif len(parts) > expected_cols:
                  parts = parts[: expected_cols - 1] + ["\t".join(parts[expected_cols - 1 :])]
            rows.append(parts)

      return pd.DataFrame(rows, columns=header, dtype=str)


def limit_frame(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
      if max_rows is None:
            return df
      return df.head(max_rows).copy()


def dedupe_rows(rows: Sequence[dict], key_fields: Sequence[str]) -> List[dict]:
      """Deduplicate rows while keeping first occurrence order."""

      seen = set()
      result: List[dict] = []
      for row in rows:
            key = tuple(row[field] for field in key_fields)
            if key in seen:
                  continue
            seen.add(key)
            result.append(row)
      return result


def add_edge(
      edges: List[dict],
      source_type: str,
      source_value: str,
      relation: str,
      target_type: str,
      target_value: str,
      weight: float = 1.0,
      evidence: str = "",
) -> None:
      """Append a normalized edge row."""

      source_value = clean_token(source_value)
      target_value = clean_token(target_value)
      if not source_value or not target_value:
            return

      edges.append(
            {
                  "source_id": make_node_id(source_type, source_value),
                  "source_type": source_type,
                  "relation": relation,
                  "target_id": make_node_id(target_type, target_value),
                  "target_type": target_type,
                  "weight": weight,
                  "evidence": evidence,
            }
      )


def build_graph(data_dir: Path, output_dir: Path, max_rows: int | None = None, min_skill_freq: int = 1) -> None:
      """Build the graph and export CSV files."""

      user_path = data_dir / "table1_user.txt"
      job_path = data_dir / "table2_jd.txt"
      action_path = data_dir / "table3_action.txt"

      user_df = limit_frame(read_table(user_path), max_rows)
      job_df = limit_frame(read_table(job_path), max_rows)
      action_df = limit_frame(read_table(action_path), max_rows)

      edges: List[dict] = []
      node_rows: List[dict] = []

      # User nodes and user-side relations.
      user_rows: List[dict] = []
      user_skill_counter: Counter[str] = Counter()
      job_skill_counter: Counter[str] = Counter()

      for row in user_df.itertuples(index=False):
            user_id = normalize_value(getattr(row, "user_id", ""))
            if not user_id:
                  continue

            attributes = {
                  "live_city_id": normalize_value(getattr(row, "live_city_id", "")),
                  "desire_jd_city_id": normalize_value(getattr(row, "desire_jd_city_id", "")),
                  "desire_jd_industry_id": normalize_value(getattr(row, "desire_jd_industry_id", "")),
                  "desire_jd_type_id": normalize_value(getattr(row, "desire_jd_type_id", "")),
                  "desire_jd_salary_id": normalize_value(getattr(row, "desire_jd_salary_id", "")),
                  "cur_industry_id": normalize_value(getattr(row, "cur_industry_id", "")),
                  "cur_jd_type": normalize_value(getattr(row, "cur_jd_type", "")),
                  "cur_salary_id": normalize_value(getattr(row, "cur_salary_id", "")),
                  "cur_degree_id": normalize_value(getattr(row, "cur_degree_id", "")),
                  "birthday": normalize_value(getattr(row, "birthday", "")),
                  "start_work_date": normalize_value(getattr(row, "start_work_date", "")),
                  "experience": normalize_value(getattr(row, "experience", "")),
            }
            user_rows.append(
                  {
                        "node_id": make_node_id("user", user_id),
                        "node_type": "user",
                        "node_name": user_id,
                        "attributes_json": json.dumps(attributes, ensure_ascii=False, sort_keys=True),
                  }
            )

            for city_id in split_multi_value(attributes["live_city_id"]):
                  add_edge(edges, "user", user_id, "LIVES_IN_CITY", "city", city_id, evidence="table1_user.live_city_id")

            for city_id in split_multi_value(attributes["desire_jd_city_id"]):
                  add_edge(edges, "user", user_id, "DESIRES_CITY", "city", city_id, evidence="table1_user.desire_jd_city_id")

            for industry in split_multi_value(attributes["desire_jd_industry_id"]):
                  add_edge(edges, "user", user_id, "DESIRES_INDUSTRY", "industry", industry, evidence="table1_user.desire_jd_industry_id")

            for job_type in split_multi_value(attributes["desire_jd_type_id"]):
                  add_edge(edges, "user", user_id, "DESIRES_JOB_TYPE", "job_type", job_type, evidence="table1_user.desire_jd_type_id")

            for industry in split_multi_value(attributes["cur_industry_id"]):
                  add_edge(edges, "user", user_id, "CURRENT_INDUSTRY", "industry", industry, evidence="table1_user.cur_industry_id")

            for job_type in split_multi_value(attributes["cur_jd_type"]):
                  add_edge(edges, "user", user_id, "CURRENT_JOB_TYPE", "job_type", job_type, evidence="table1_user.cur_jd_type")

            for degree in split_multi_value(attributes["cur_degree_id"]):
                  add_edge(edges, "user", user_id, "HAS_DEGREE", "degree", degree, evidence="table1_user.cur_degree_id")

            user_skills = extract_skill_tokens(attributes["experience"], max_tokens=60)
            for skill in user_skills:
                  user_skill_counter[skill] += 1
                  add_edge(edges, "user", user_id, "HAS_SKILL", "skill", skill, evidence="table1_user.experience")

      # Job nodes and job-side relations.
      job_rows: List[dict] = []
      for row in job_df.itertuples(index=False):
            job_id = normalize_value(getattr(row, "jd_no", ""))
            if not job_id:
                  continue

            company_name = normalize_value(getattr(row, "company_name", ""))
            attributes = {
                  "jd_title": normalize_value(getattr(row, "jd_title", "")),
                  "company_name": company_name,
                  "city": normalize_value(getattr(row, "city", "")),
                  "jd_sub_type": normalize_value(getattr(row, "jd_sub_type", "")),
                  "require_nums": normalize_value(getattr(row, "require_nums", "")),
                  "max_salary": normalize_value(getattr(row, "max_salary", "")),
                  "min_salary": normalize_value(getattr(row, "min_salary", "")),
                  "start_date": normalize_value(getattr(row, "start_date", "")),
                  "end_date": normalize_value(getattr(row, "end_date", "")),
                  "is_travel": normalize_value(getattr(row, "is_travel", "")),
                  "min_years": normalize_value(getattr(row, "min_years", "")),
                  "key": normalize_value(getattr(row, "key", "")),
                  "min_edu_level": normalize_value(getattr(row, "min_edu_level", "")),
                  "max_edu_level": normalize_value(getattr(row, "max_edu_level", "")),
                  "is_mangerial": normalize_value(getattr(row, "is_mangerial", "")),
                  "resume_language_required": normalize_value(getattr(row, "resume_language_required", "")),
                  "job_description": normalize_value(getattr(row, "job_description", "")),
            }
            job_rows.append(
                  {
                        "node_id": make_node_id("job", job_id),
                        "node_type": "job",
                        "node_name": attributes["jd_title"] or job_id,
                        "attributes_json": json.dumps(attributes, ensure_ascii=False, sort_keys=True),
                  }
            )

            for city_id in split_multi_value(attributes["city"]):
                  add_edge(edges, "job", job_id, "LOCATED_IN_CITY", "city", city_id, evidence="table2_jd.city")

            for job_type in split_multi_value(attributes["jd_sub_type"]):
                  add_edge(edges, "job", job_id, "BELONGS_TO_JOB_TYPE", "job_type", job_type, evidence="table2_jd.jd_sub_type")

            for degree in split_multi_value(attributes["min_edu_level"]):
                  add_edge(edges, "job", job_id, "REQUIRES_DEGREE", "degree", degree, evidence="table2_jd.min_edu_level")

            # Skill extraction is rule-based. We combine title and description,
            # then keep short tokens that survive the separator split.
            job_skill_text = f"{attributes['jd_title']}\n{attributes['job_description']}"
            job_skills = extract_skill_tokens(job_skill_text, max_tokens=40)
            for skill in job_skills:
                  job_skill_counter[skill] += 1
                  add_edge(edges, "job", job_id, "REQUIRES_SKILL", "skill", skill, evidence="table2_jd.jd_title+job_description")

      # Actions: user-job behavior edges.
      for row in action_df.itertuples(index=False):
            user_id = normalize_value(getattr(row, "user_id", ""))
            job_id = normalize_value(getattr(row, "jd_no", ""))
            if not user_id or not job_id:
                  continue

            behaviors = {
                  "browsed": normalize_value(getattr(row, "browsed", "")),
                  "delivered": normalize_value(getattr(row, "delivered", "")),
                  "satisfied": normalize_value(getattr(row, "satisfied", "")),
            }
            for behavior_name, flag in behaviors.items():
                  if flag not in {"1", "1.0", "true", "True"}:
                        continue
                  add_edge(
                        edges,
                        "user",
                        user_id,
                        f"{behavior_name.upper()}_JOB",
                        "job",
                        job_id,
                        weight=ACTION_WEIGHTS[behavior_name],
                        evidence="table3_action",
                  )

      # Build node tables from observed values.
      city_values = set()
      industry_values = set()
      job_type_values = set()
      degree_values = set()
      company_values = set()

      for row in user_df.itertuples(index=False):
            city_values.update(split_multi_value(getattr(row, "live_city_id", "")))
            city_values.update(split_multi_value(getattr(row, "desire_jd_city_id", "")))
            industry_values.update(split_multi_value(getattr(row, "desire_jd_industry_id", "")))
            industry_values.update(split_multi_value(getattr(row, "cur_industry_id", "")))
            job_type_values.update(split_multi_value(getattr(row, "desire_jd_type_id", "")))
            job_type_values.update(split_multi_value(getattr(row, "cur_jd_type", "")))
            degree_values.update(split_multi_value(getattr(row, "cur_degree_id", "")))

      for row in job_df.itertuples(index=False):
            city_values.update(split_multi_value(getattr(row, "city", "")))
            job_type_values.update(split_multi_value(getattr(row, "jd_sub_type", "")))
            degree_values.update(split_multi_value(getattr(row, "min_edu_level", "")))
            company_name = normalize_value(getattr(row, "company_name", ""))
            if company_name:
                  company_values.add(company_name)

      skill_values = set()
      for skill, count in user_skill_counter.items():
            if count >= min_skill_freq:
                  skill_values.add(skill)
      for skill, count in job_skill_counter.items():
            if count >= min_skill_freq:
                  skill_values.add(skill)

      node_rows.extend(dedupe_rows(user_rows, ["node_id"]))
      node_rows.extend(dedupe_rows(job_rows, ["node_id"]))
      node_rows.extend(make_node_rows("city", sorted(city_values)))
      node_rows.extend(make_node_rows("industry", sorted(industry_values)))
      node_rows.extend(make_node_rows("job_type", sorted(job_type_values)))
      node_rows.extend(make_node_rows("degree", sorted(degree_values)))
      node_rows.extend(make_node_rows("company", sorted(company_values)))
      node_rows.extend(make_node_rows("skill", sorted(skill_values)))

      edges = dedupe_rows(
            edges,
            ["source_id", "relation", "target_id", "weight", "evidence"],
      )

      output_dir.mkdir(parents=True, exist_ok=True)
      nodes_df = pd.DataFrame(node_rows)
      edges_df = pd.DataFrame(edges)

      nodes_df.to_csv(output_dir / "nodes.csv", index=False, encoding="utf-8-sig")
      edges_df.to_csv(output_dir / "edges.csv", index=False, encoding="utf-8-sig")

      stats_df = pd.DataFrame(
            [
                  {"item": "users", "count": len(user_rows)},
                  {"item": "jobs", "count": len(job_rows)},
                  {"item": "cities", "count": len(city_values)},
                  {"item": "industries", "count": len(industry_values)},
                  {"item": "job_types", "count": len(job_type_values)},
                  {"item": "degrees", "count": len(degree_values)},
                  {"item": "companies", "count": len(company_values)},
                  {"item": "skills", "count": len(skill_values)},
                  {"item": "edges", "count": len(edges)},
            ]
      )
      stats_df.to_csv(output_dir / "graph_stats.csv", index=False, encoding="utf-8-sig")

      print(f"Graph exported to: {output_dir}")
      print(stats_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
      parser = argparse.ArgumentParser(description="Build a knowledge graph from personas-core/datasets")
      parser.add_argument(
            "--data-dir",
            type=Path,
            default=Path(__file__).resolve().parent / "../datasets",
            help="Directory containing table1_user.txt, table2_jd.txt, table3_action.txt",
      )
      parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path(__file__).resolve().parent / "../kg_output",
            help="Directory for exported CSV files",
      )
      parser.add_argument(
            "--max-rows",
            type=int,
            default=None,
            help="Optional row cap for quick validation or sampling",
      )
      parser.add_argument(
            "--min-skill-freq",
            type=int,
            default=1,
            help="Keep skills that appear at least this many times",
      )
      return parser.parse_args()


def main() -> None:
      args = parse_args()
      build_graph(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_rows=args.max_rows,
            min_skill_freq=args.min_skill_freq,
      )


if __name__ == "__main__":
      main()