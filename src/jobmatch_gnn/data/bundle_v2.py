"""Shared loader for the processed v2 dataset (docs/11)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class BundleV2:
    """Processed full dataset with index maps and split structures."""

    users: pd.DataFrame
    jobs: pd.DataFrame
    interactions: pd.DataFrame
    skill_count: int
    user_index: dict[str, int]
    job_index: dict[str, int]
    train_pos: dict[int, set[int]]
    valid_pos: dict[int, set[int]]
    test_pos: dict[int, set[int]]
    browsed: dict[int, set[int]]
    user_skill: sp.csr_matrix
    job_skill: sp.csr_matrix
    processed_dir: Path

    @property
    def num_users(self) -> int:
        return len(self.users)

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    def load_embeddings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (user, job, skill) SBERT embeddings."""

        return (
            np.load(self.processed_dir / "emb_user.npy"),
            np.load(self.processed_dir / "emb_job.npy"),
            np.load(self.processed_dir / "emb_skill.npy"),
        )


def _skill_matrix(frame: pd.DataFrame, column: str, rows: int, skill_count: int, weights: str | None = None) -> sp.csr_matrix:
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    weight_col = frame[weights].tolist() if weights else None
    for row_idx, skill_ids in enumerate(frame[column].tolist()):
        ids = list(skill_ids)
        indices.extend(ids)
        data.extend(list(weight_col[row_idx])[: len(ids)] if weight_col else [1.0] * len(ids))
        indptr.append(len(indices))
    return sp.csr_matrix((np.array(data, dtype=np.float32), np.array(indices), np.array(indptr)), shape=(rows, skill_count))


def load_bundle_v2(processed_dir: str | Path = "data/processed") -> BundleV2:
    """Load processed parquet artifacts into an indexed bundle."""

    processed = Path(processed_dir)
    users = pd.read_parquet(processed / "users.parquet")
    jobs = pd.read_parquet(processed / "jobs.parquet")
    interactions = pd.read_parquet(processed / "interactions.parquet")
    vocab = json.loads((processed / "skill_vocab.json").read_text(encoding="utf-8"))
    user_index = {user_id: idx for idx, user_id in enumerate(users["user_id"].tolist())}
    job_index = {job_id: idx for idx, job_id in enumerate(jobs["job_id"].tolist())}

    interactions = interactions.assign(
        u=interactions["user_id"].map(user_index),
        j=interactions["job_id"].map(job_index),
    ).dropna(subset=["u", "j"])
    interactions["u"] = interactions["u"].astype(int)
    interactions["j"] = interactions["j"].astype(int)

    def group(mask: pd.Series) -> dict[int, set[int]]:
        out: dict[int, set[int]] = {}
        for u, j in interactions.loc[mask, ["u", "j"]].itertuples(index=False):
            out.setdefault(u, set()).add(j)
        return out

    positive = interactions["level"] >= 2
    bundle = BundleV2(
        users=users,
        jobs=jobs,
        interactions=interactions,
        skill_count=len(vocab),
        user_index=user_index,
        job_index=job_index,
        train_pos=group(positive & (interactions["split"] == "train")),
        valid_pos=group(interactions["split"] == "valid"),
        test_pos=group(interactions["split"] == "test"),
        browsed=group(interactions["level"] == 1),
        user_skill=_skill_matrix(users, "skill_ids", len(users), len(vocab)),
        job_skill=_skill_matrix(jobs, "skill_ids", len(jobs), len(vocab), weights="skill_weights"),
        processed_dir=processed,
    )
    return bundle
