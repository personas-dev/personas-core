"""Training CLI for HR KG-GNN baselines and GNN models."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from jobmatch_gnn.baselines import BM25Matcher, HashSemanticMatcher, RuleMatcher
from jobmatch_gnn.data.dataset import DatasetBundle, load_dataset_bundle, write_json
from jobmatch_gnn.evaluation.metrics import evaluate_rankings


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def make_run_dir(output_dir: Path, run_name: str | None = None) -> Path:
    """Create a timestamped experiment run directory."""

    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def rank_with_matcher(bundle: DatasetBundle, matcher: object, mask_train: bool = True) -> dict[str, list[str]]:
    """Rank all sampled jobs for every test candidate using a baseline matcher."""

    rankings: dict[str, list[str]] = {}
    job_items = list(bundle.jobs.items())
    for user_id, relevant in bundle.test_by_user.items():
        if not relevant or user_id not in bundle.candidates:
            continue
        candidate = bundle.candidates[user_id]
        scored: list[tuple[str, float]] = []
        for job_id, job in job_items:
            score = float(matcher.score(candidate, job))  # type: ignore[attr-defined]
            if mask_train and job_id in bundle.train_by_user.get(user_id, set()):
                score = -1.0e9
            scored.append((job_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        rankings[user_id] = [job_id for job_id, _ in scored]
    return rankings


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact metrics comparison CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_sample(path: Path, rankings: dict[str, list[str]], limit_users: int = 5, limit_jobs: int = 10) -> None:
    """Write a small JSON prediction sample for manual inspection."""

    sample = {user_id: jobs[:limit_jobs] for user_id, jobs in list(rankings.items())[:limit_users]}
    write_json(path, sample)


def evaluate_model(bundle: DatasetBundle, rankings: dict[str, list[str]], k: int) -> dict[str, float]:
    """Evaluate rankings against the test split."""

    return evaluate_rankings(bundle.test_by_user, rankings, k)


def run(config_path: Path) -> dict[str, Any]:
    """Run configured baselines and optional torch GNN models."""

    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    run_dir = make_run_dir(Path(str(config.get("output_dir", "experiments/runs"))), config.get("run_name"))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    bundle = load_dataset_bundle(dict(config.get("data", {})))
    write_json(run_dir / "dataset_stats.json", bundle.stats)
    k = int(config.get("eval", {}).get("k", 10))
    requested_models = [str(item).strip() for item in config.get("models", ["rule", "bm25", "semantic_hash", "lightgcn", "spc_hgt"])]
    results: list[dict[str, Any]] = []
    prediction_dir = run_dir / "predictions"

    baseline_factories = {
        "rule": lambda: RuleMatcher(),
        "bm25": lambda: BM25Matcher(bundle.jobs),
        "semantic_hash": lambda: HashSemanticMatcher(bundle.jobs, dim=int(config.get("semantic_hash", {}).get("dim", 512))),
    }
    for model_name in requested_models:
        if model_name not in baseline_factories:
            continue
        matcher = baseline_factories[model_name]()
        rankings = rank_with_matcher(bundle, matcher)
        metrics = evaluate_model(bundle, rankings, k)
        row: dict[str, Any] = {"model": model_name, **metrics}
        results.append(row)
        write_prediction_sample(prediction_dir / f"{model_name}.json", rankings)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))

    torch_models = [model for model in requested_models if model in {"lightgcn", "spc_hgt"}]
    if torch_models:
        try:
            from jobmatch_gnn.models import train_torch_matcher

            torch_config = dict(config.get("gnn", {}))
            torch_config.setdefault("seed", seed)
            for model_name in torch_models:
                result = train_torch_matcher(bundle, model_name, torch_config, lambda relevant, rankings: evaluate_rankings(relevant, rankings, k))
                row = {"model": model_name, **result.metrics}
                results.append(row)
                write_prediction_sample(prediction_dir / f"{model_name}.json", result.rankings)
                checkpoint_dir = run_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                try:
                    import torch

                    torch.save({"model_state_dict": result.model.state_dict(), "user_ids": result.user_ids, "job_ids": result.job_ids}, checkpoint_dir / f"{model_name}.pt")
                except Exception as exc:  # pragma: no cover
                    row["checkpoint_error"] = str(exc)
                print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        except Exception as exc:
            skipped = {"model": "torch_gnn", "status": "skipped", "reason": str(exc)}
            results.append(skipped)
            print(json.dumps(skipped, ensure_ascii=False, sort_keys=True))

    write_json(run_dir / "metrics.json", results)
    write_metrics_csv(run_dir / "metrics.csv", results)
    return {"run_dir": str(run_dir), "metrics": results, "dataset_stats": bundle.stats}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train HR KG-GNN baselines and optional GNN models.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_gnn.yaml"), help="YAML config path")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    result = run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
