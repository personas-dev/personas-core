"""Encode user/job/skill texts with a Chinese sentence-embedding model.

Run:
    HF_ENDPOINT=https://hf-mirror.com python -m jobmatch_gnn.text.encode_sbert --config configs/v2_data.yaml

Outputs ``emb_user.npy``, ``emb_job.npy``, ``emb_skill.npy`` (L2-normalized,
row order matches the parquet row order / skill id order) under processed_dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def run(config_path: Path) -> None:
    from sentence_transformers import SentenceTransformer

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir = Path(config["processed_dir"])
    sbert_cfg = config.get("sbert", {})
    model = SentenceTransformer(sbert_cfg.get("model", "BAAI/bge-small-zh-v1.5"), device=sbert_cfg.get("device", "cuda"))
    model.max_seq_length = int(sbert_cfg.get("max_seq_length", 256))
    batch_size = int(sbert_cfg.get("batch_size", 512))
    query_prefix = sbert_cfg.get("query_prefix", "")

    users = pd.read_parquet(out_dir / "users.parquet")
    jobs = pd.read_parquet(out_dir / "jobs.parquet")
    vocab = json.loads((out_dir / "skill_vocab.json").read_text(encoding="utf-8"))
    skills = [tag for tag, _ in sorted(vocab.items(), key=lambda item: item[1])]

    def encode(texts: list[str], prefix: str = "") -> np.ndarray:
        payload = [prefix + (t or " ") for t in texts]
        return model.encode(payload, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    print(f"encoding {len(users)} users / {len(jobs)} jobs / {len(skills)} skills", flush=True)
    np.save(out_dir / "emb_user.npy", encode(users["text"].tolist(), prefix=query_prefix))
    np.save(out_dir / "emb_job.npy", encode(jobs["text"].tolist()))
    np.save(out_dir / "emb_skill.npy", encode(skills))
    print("done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v2_data.yaml"))
    run(parser.parse_args().config)
