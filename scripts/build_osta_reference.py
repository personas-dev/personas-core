"""Build local OSTA reference terms for skill cleaning.

The output is intentionally written under data/external/ by default, which is
ignored by git in this repository.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

BASE_URL = "https://www.osta.org.cn/api"


def fetch_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Fetch an OSTA JSON endpoint."""

	url = f"{BASE_URL}{path}"
	if params:
		url = f"{url}?{urlencode(params)}"
	with urlopen(url, timeout=30) as response:  # noqa: S310 - trusted official HTTPS endpoint.
		return json.loads(response.read().decode("utf-8"))


def flatten_tree(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Flatten career classification tree nodes."""

	out: list[dict[str, Any]] = []

	def walk(items: list[dict[str, Any]], parent_path: list[str]) -> None:
		for item in items:
			children = item.get("children") or []
			out.append(
				{
					"career_code": item.get("careerCode", ""),
					"name": item.get("careerName", ""),
					"version_id": item.get("versionId"),
					"parent_code": item.get("parentCode"),
					"path": parent_path + [item.get("careerName", "")],
					"has_children": bool(children),
				}
			)
			walk(children, parent_path + [item.get("careerName", "")])

	walk(nodes, [])
	return out


def fetch_standards(page_size: int) -> list[dict[str, Any]]:
	"""Fetch all national occupational skill-standard metadata."""

	first = fetch_json("/public/skillStandardList", {"pageSize": page_size, "pageNum": 1, "nameCode": "", "status": 1})
	body = first.get("body", {})
	pages = int(body.get("pages") or 1)
	items = list(body.get("list") or [])
	for page_num in range(2, pages + 1):
		page = fetch_json(
			"/public/skillStandardList",
			{"pageSize": page_size, "pageNum": page_num, "nameCode": "", "status": 1},
		)
		items.extend(page.get("body", {}).get("list") or [])
	return items


def fetch_occupations(categories: list[dict[str, Any]], max_subordinate_requests: int) -> list[dict[str, Any]]:
	"""Fetch occupation rows under career classification nodes."""

	occupations: dict[tuple[str, int], dict[str, Any]] = {}
	requests = 0
	for category in categories:
		if requests >= max_subordinate_requests:
			break
		career_code = category.get("career_code")
		version_id = category.get("version_id")
		if not career_code or not version_id:
			continue
		requests += 1
		response = fetch_json("/client/subordinate/data", {"careerCode": career_code, "versionId": version_id})
		for item in response.get("body", []) or []:
			code = item.get("careerCode")
			version = item.get("versionId")
			if not code or not version:
				continue
			occupations[(code, int(version))] = {
				"career_code": code,
				"name": item.get("name") or item.get("careerName") or "",
				"version_id": version,
				"work_num": item.get("workNum"),
				"career_num": item.get("careerNum"),
			}
	return sorted(occupations.values(), key=lambda item: (item["career_code"], item["name"]))


def fetch_details(occupations: list[dict[str, Any]], max_details: int) -> list[dict[str, Any]]:
	"""Fetch detailed definition, task, and work-type data for occupations."""

	details: list[dict[str, Any]] = []
	for item in occupations[:max_details]:
		response = fetch_json(
			"/client/career/detail",
			{"careerCode": item["career_code"], "versionId": item["version_id"]},
		)
		body = response.get("body") or {}
		details.append(
			{
				"career_code": body.get("careerCode", item["career_code"]),
				"name": body.get("name", item["name"]),
				"definition": body.get("text", ""),
				"task": body.get("task", ""),
				"big_name": body.get("bigName", ""),
				"centre_name": body.get("centreName", ""),
				"small_name": body.get("smallName", ""),
				"works": body.get("works") or [],
				"version_id": body.get("versionId", item["version_id"]),
			}
		)
	return details


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output", type=Path, default=Path("data/external/osta_reference_terms.json"))
	parser.add_argument("--page-size", type=int, default=200)
	parser.add_argument("--max-subordinate-requests", type=int, default=600)
	parser.add_argument("--max-details", type=int, default=0, help="Fetch detailed task/work data for first N occupations.")
	args = parser.parse_args()

	tree = fetch_json("/client/get/tree").get("body") or []
	categories = flatten_tree(tree)
	standards = fetch_standards(args.page_size)
	occupations = fetch_occupations(categories, args.max_subordinate_requests)
	details = fetch_details(occupations, args.max_details) if args.max_details > 0 else []
	output = {
		"source": {
			"name": "技能人才评价工作网 OSTA 职业分类系统/国家职业标准查询系统",
			"base_url": "https://www.osta.org.cn/",
			"fetched_at": datetime.now(timezone.utc).isoformat(),
		},
		"career_categories": [item["name"] for item in categories if item.get("name")],
		"occupation_terms": [item["name"] for item in occupations if item.get("name")],
		"occupations": details,
		"standard_terms": [item.get("name", "") for item in standards if item.get("name")],
		"standards": standards,
	}
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
	print(json.dumps({"output": str(args.output), "categories": len(categories), "occupations": len(occupations), "standards": len(standards)}, ensure_ascii=False))


if __name__ == "__main__":
	main()
