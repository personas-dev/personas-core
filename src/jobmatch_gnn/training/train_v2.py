"""v2 experiment orchestrator: run configured models on the shared full split.

Run:
    python -m jobmatch_gnn.training.train_v2 --config configs/v2_baselines.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from jobmatch_gnn.data.bundle_v2 import load_bundle_v2
from jobmatch_gnn.evaluation.rank_eval import evaluate_rankings


def run_model(name: str, bundle, model_cfg: dict, eval_users: list[int]):
    from jobmatch_gnn.baselines import v2_baselines as bl

    if name == "popularity":
        return bl.popularity_rankings(bundle, eval_users), {}
    if name == "rule":
        return bl.rule_rankings(bundle, eval_users, weights=model_cfg.get("weights")), {}
    if name == "bm25":
        return bl.bm25_rankings(bundle, eval_users), {}
    if name == "sbert":
        return bl.sbert_rankings(bundle, eval_users, device=model_cfg.get("device", "cuda")), {}
    if name == "lightgcn":
        from jobmatch_gnn.models.lightgcn_v2 import train_lightgcn_v2

        return train_lightgcn_v2(bundle, model_cfg, eval_users)
    if name.startswith("spc_hgt"):
        from jobmatch_gnn.training.train_spc_hgt_v2 import train_spc_hgt_v2

        return train_spc_hgt_v2(bundle, model_cfg)
    raise ValueError(f"unknown model {name}")


def main(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bundle = load_bundle_v2(config.get("processed_dir", "data/processed"))
    out_dir = Path(config.get("output_dir", "experiments/runs")) / config.get("run_name", "v2_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_users = sorted(bundle.test_pos)
    print(f"bundle: users={bundle.num_users} jobs={bundle.num_jobs} test_users={len(eval_users)}", flush=True)

    rows = []
    for spec in config["models"]:
        name = spec["name"] if isinstance(spec, dict) else spec
        model_cfg = dict(spec.get("config", {})) if isinstance(spec, dict) else {}
        label = spec.get("label", name) if isinstance(spec, dict) else name
        seeds = model_cfg.pop("seeds", [model_cfg.get("seed", 42)])
        seed_metrics = []
        for seed in seeds:
            cfg = dict(model_cfg, seed=seed)
            start = time.time()
            print(f"=== {label} (seed={seed}) ===", flush=True)
            rankings, info = run_model(name, bundle, cfg, eval_users)
            result = evaluate_rankings(rankings, bundle.test_pos)
            metrics = dict(result.metrics)
            metrics.update({f"info_{k}": v for k, v in info.items()})
            metrics["runtime_s"] = round(time.time() - start, 1)
            metrics["seed"] = seed
            seed_metrics.append(metrics)
            np.save(out_dir / f"ndcg_user_{label}_s{seed}.npy", np.array(sorted(result.per_user_ndcg.items()), dtype=np.float64))
            print(json.dumps({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}, indent=None), flush=True)
        agg = {"model": label, "n_seeds": len(seed_metrics)}
        keys = [k for k in seed_metrics[0] if isinstance(seed_metrics[0][k], (int, float)) and k != "seed"]
        for key in keys:
            values = [m[key] for m in seed_metrics]
            agg[key] = float(np.mean(values))
            if len(values) > 1:
                agg[f"{key}_std"] = float(np.std(values))
        rows.append(agg)
        pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)
        (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    print(f"wrote {out_dir}/metrics.csv", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    main(parser.parse_args().config)
