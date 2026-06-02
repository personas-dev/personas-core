"""PyTorch LightGCN and SPC-HGT-lite training helpers."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np

from jobmatch_gnn.data.dataset import DatasetBundle, pair_features

try:  # pragma: no cover - exercised only when torch is installed.
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None

    class _MissingNN:
        Module = object

    nn = _MissingNN()
    F = None


@dataclass(frozen=True)
class TorchTrainResult:
    """Result object from a torch training run."""

    model: Any
    user_ids: list[str]
    job_ids: list[str]
    metrics: dict[str, float | str]
    rankings: dict[str, list[str]]


class LightGCN(nn.Module):  # type: ignore[misc]
    """Minimal bipartite LightGCN for candidate-job interactions."""

    def __init__(self, num_users: int, num_jobs: int, embedding_dim: int, num_layers: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.job_embedding = nn.Embedding(num_jobs, embedding_dim)
        self.num_layers = num_layers
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.job_embedding.weight, std=0.05)

    def embeddings(self, edge_index: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return propagated user and job embeddings."""

        user = self.user_embedding.weight
        job = self.job_embedding.weight
        users = [user]
        jobs = [job]
        if edge_index.numel() == 0:
            return user, job
        src_user = edge_index[0]
        dst_job = edge_index[1]
        ones = torch.ones(src_user.shape[0], device=user.device)
        deg_user = torch.zeros(user.shape[0], device=user.device).index_add(0, src_user, ones).clamp_min(1.0)
        deg_job = torch.zeros(job.shape[0], device=job.device).index_add(0, dst_job, ones).clamp_min(1.0)
        norm = torch.rsqrt(deg_user[src_user] * deg_job[dst_job]).unsqueeze(1)
        for _ in range(self.num_layers):
            next_job = torch.zeros_like(job).index_add(0, dst_job, user[src_user] * norm)
            next_user = torch.zeros_like(user).index_add(0, src_user, job[dst_job] * norm)
            user, job = next_user, next_job
            users.append(user)
            jobs.append(job)
        return torch.stack(users, dim=0).mean(dim=0), torch.stack(jobs, dim=0).mean(dim=0)

    def score(self, user_idx: "torch.Tensor", job_idx: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
        """Score indexed user-job pairs."""

        user_emb, job_emb = self.embeddings(edge_index)
        return (user_emb[user_idx] * job_emb[job_idx]).sum(dim=1)


class SPCHGTLite(LightGCN):
    """Skill-path enhanced LightGCN used as a dependency-light SPC-HGT prototype."""

    def __init__(self, num_users: int, num_jobs: int, embedding_dim: int, num_layers: int, feature_dim: int) -> None:
        super().__init__(num_users, num_jobs, embedding_dim, num_layers)
        self.feature_head = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 1),
        )

    def score_with_features(
        self,
        user_idx: "torch.Tensor",
        job_idx: "torch.Tensor",
        features: "torch.Tensor",
        edge_index: "torch.Tensor",
    ) -> "torch.Tensor":
        """Score pairs with graph embeddings and path/rule features."""

        graph_score = self.score(user_idx, job_idx, edge_index)
        feature_score = self.feature_head(features).squeeze(1)
        return graph_score + feature_score


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is not installed; install torch to train GPU GNN models.")


def _build_indices(bundle: DatasetBundle) -> tuple[list[str], list[str], dict[str, int], dict[str, int]]:
    user_ids = sorted(bundle.candidates)
    job_ids = sorted(bundle.jobs)
    return user_ids, job_ids, {user_id: idx for idx, user_id in enumerate(user_ids)}, {job_id: idx for idx, job_id in enumerate(job_ids)}


def _train_pairs(bundle: DatasetBundle, user_to_idx: dict[str, int], job_to_idx: dict[str, int]) -> list[tuple[int, int, str, str]]:
    pairs: list[tuple[int, int, str, str]] = []
    for user_id, job_set in bundle.train_by_user.items():
        if user_id not in user_to_idx:
            continue
        for job_id in job_set:
            if job_id in job_to_idx:
                pairs.append((user_to_idx[user_id], job_to_idx[job_id], user_id, job_id))
    return pairs


def _edge_index_from_pairs(pairs: list[tuple[int, int, str, str]], device: str) -> "torch.Tensor":
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    return torch.tensor([[row[0] for row in pairs], [row[1] for row in pairs]], dtype=torch.long, device=device)


def _sample_negatives(
    pairs: list[tuple[int, int, str, str]],
    job_ids: list[str],
    job_to_idx: dict[str, int],
    positive_by_user: dict[str, set[str]],
    rng: random.Random,
) -> list[int]:
    negatives: list[int] = []
    for _, _, user_id, _ in pairs:
        positives = positive_by_user.get(user_id, set())
        for _ in range(100):
            job_id = rng.choice(job_ids)
            if job_id not in positives:
                negatives.append(job_to_idx[job_id])
                break
        else:
            negatives.append(job_to_idx[rng.choice(job_ids)])
    return negatives


def _feature_batch(bundle: DatasetBundle, user_ids: list[str], job_ids: list[str], users: list[int], jobs: list[int], device: str) -> "torch.Tensor":
    features = [pair_features(bundle.candidates[user_ids[user_idx]], bundle.jobs[job_ids[job_idx]]) for user_idx, job_idx in zip(users, jobs, strict=False)]
    return torch.tensor(np.stack(features), dtype=torch.float32, device=device)


def _rank_torch_model(
    model: Any,
    bundle: DatasetBundle,
    user_ids: list[str],
    job_ids: list[str],
    user_to_idx: dict[str, int],
    edge_index: "torch.Tensor",
    model_name: str,
    device: str,
    batch_jobs: int = 1024,
) -> dict[str, list[str]]:
    model.eval()
    rankings: dict[str, list[str]] = {}
    all_job_idx = list(range(len(job_ids)))
    with torch.no_grad():
        for user_id, relevant in bundle.test_by_user.items():
            if not relevant or user_id not in user_to_idx:
                continue
            user_idx = user_to_idx[user_id]
            scores: list[np.ndarray] = []
            for start in range(0, len(all_job_idx), batch_jobs):
                chunk = all_job_idx[start : start + batch_jobs]
                users = [user_idx] * len(chunk)
                user_tensor = torch.tensor(users, dtype=torch.long, device=device)
                job_tensor = torch.tensor(chunk, dtype=torch.long, device=device)
                if model_name == "spc_hgt":
                    feats = _feature_batch(bundle, user_ids, job_ids, users, chunk, device)
                    chunk_scores = model.score_with_features(user_tensor, job_tensor, feats, edge_index)
                else:
                    chunk_scores = model.score(user_tensor, job_tensor, edge_index)
                scores.append(chunk_scores.detach().cpu().numpy())
            score_arr = np.concatenate(scores)
            for train_job in bundle.train_by_user.get(user_id, set()):
                if train_job in job_ids:
                    score_arr[job_ids.index(train_job)] = -1.0e9
            order = np.argsort(-score_arr)
            rankings[user_id] = [job_ids[index] for index in order]
    return rankings


def train_torch_matcher(bundle: DatasetBundle, model_name: str, config: dict[str, object], evaluate_fn: Any) -> TorchTrainResult:
    """Train a LightGCN or SPC-HGT-lite model and return rankings."""

    _require_torch()
    requested_device = str(config.get("device", "cuda"))
    device = requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu"
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    rng = random.Random(seed)
    user_ids, job_ids, user_to_idx, job_to_idx = _build_indices(bundle)
    pairs = _train_pairs(bundle, user_to_idx, job_to_idx)
    if not pairs:
        raise RuntimeError("No positive train interactions are available for GNN training.")
    edge_index = _edge_index_from_pairs(pairs, device)
    embedding_dim = int(config.get("embedding_dim", 64))
    num_layers = int(config.get("num_layers", 2))
    if model_name == "spc_hgt":
        feature_dim = len(pair_features(next(iter(bundle.candidates.values())), next(iter(bundle.jobs.values()))))
        model = SPCHGTLite(len(user_ids), len(job_ids), embedding_dim, num_layers, feature_dim).to(device)
    else:
        model = LightGCN(len(user_ids), len(job_ids), embedding_dim, num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 0.003)), weight_decay=float(config.get("weight_decay", 1.0e-5)))
    batch_size = int(config.get("batch_size", 512))
    epochs = int(config.get("epochs", 20))
    last_loss = 0.0
    for _ in range(epochs):
        rng.shuffle(pairs)
        negatives = _sample_negatives(pairs, job_ids, job_to_idx, bundle.positive_by_user, rng)
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            neg_batch = negatives[start : start + batch_size]
            users = [row[0] for row in batch]
            pos_jobs = [row[1] for row in batch]
            user_tensor = torch.tensor(users, dtype=torch.long, device=device)
            pos_tensor = torch.tensor(pos_jobs, dtype=torch.long, device=device)
            neg_tensor = torch.tensor(neg_batch, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if model_name == "spc_hgt":
                pos_features = _feature_batch(bundle, user_ids, job_ids, users, pos_jobs, device)
                neg_features = _feature_batch(bundle, user_ids, job_ids, users, neg_batch, device)
                pos_score = model.score_with_features(user_tensor, pos_tensor, pos_features, edge_index)
                neg_score = model.score_with_features(user_tensor, neg_tensor, neg_features, edge_index)
                bce_scores = torch.cat([pos_score, neg_score])
                bce_labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
                bce = F.binary_cross_entropy_with_logits(bce_scores, bce_labels)
            else:
                pos_score = model.score(user_tensor, pos_tensor, edge_index)
                neg_score = model.score(user_tensor, neg_tensor, edge_index)
                bce = torch.tensor(0.0, device=device)
            bpr = -F.logsigmoid(pos_score - neg_score).mean()
            loss = bpr + 0.2 * bce
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    rankings = _rank_torch_model(model, bundle, user_ids, job_ids, user_to_idx, edge_index, model_name, device)
    metrics = evaluate_fn(bundle.test_by_user, rankings)
    metrics.update({"train_loss": last_loss, "device": device, "train_positive_count": float(len(pairs))})
    return TorchTrainResult(model=model, user_ids=user_ids, job_ids=job_ids, metrics=metrics, rankings=rankings)
