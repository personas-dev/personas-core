"""PJFCANN-style baseline adapted to the v2 full-catalog protocol.

The official PJFCANN implementation combines local resume/job text encoders
with global recruitment-history graph representations, then classifies a
candidate-job pair from ``[h_c, h_j, |h_c-h_j|, h_c*h_j]``. This module keeps
that architecture family but replaces the original dataset-specific InferSent
and session graph inputs with the processed v2 BGE text embeddings and the
train-only candidate-job interaction graph.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from jobmatch_gnn.data.bundle_v2 import BundleV2
from jobmatch_gnn.evaluation.rank_eval import evaluate_rankings, topn_from_scores


class PJFCANNV2(nn.Module):
    """Local semantic encoder + global interaction graph encoder."""

    def __init__(self, text_dim: int, num_users: int, num_jobs: int, dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = layers
        self.dim = dim
        self.user_text = nn.Sequential(nn.Linear(text_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.job_text = nn.Sequential(nn.Linear(text_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.user_emb = nn.Embedding(num_users, dim)
        self.job_emb = nn.Embedding(num_jobs, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.job_emb.weight, std=0.1)
        # PJFCANN concatenates local and global states before the pair classifier.
        pair_dim = 4 * (2 * dim)
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def propagate(self, edge_u: torch.Tensor, edge_j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """LightGCN-style propagation over train-only recruitment history."""

        user, job = self.user_emb.weight, self.job_emb.weight
        deg_u = torch.zeros(user.shape[0], device=user.device).index_add_(
            0, edge_u, torch.ones_like(edge_u, dtype=torch.float)
        ).clamp_min(1)
        deg_j = torch.zeros(job.shape[0], device=job.device).index_add_(
            0, edge_j, torch.ones_like(edge_j, dtype=torch.float)
        ).clamp_min(1)
        norm = torch.rsqrt(deg_u[edge_u] * deg_j[edge_j]).unsqueeze(1)
        users, jobs = [user], [job]
        for _ in range(self.layers):
            next_job = torch.zeros_like(job).index_add_(0, edge_j, user[edge_u] * norm)
            next_user = torch.zeros_like(user).index_add_(0, edge_u, job[edge_j] * norm)
            user, job = next_user, next_job
            users.append(user)
            jobs.append(job)
        return torch.stack(users).mean(0), torch.stack(jobs).mean(0)

    def encode(
        self,
        user_text: torch.Tensor,
        job_text: torch.Tensor,
        edge_u: torch.Tensor,
        edge_j: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return local and global representations for all users/jobs."""

        user_global, job_global = self.propagate(edge_u, edge_j)
        return {
            "u_local": F.normalize(self.user_text(user_text), dim=1),
            "j_local": F.normalize(self.job_text(job_text), dim=1),
            "u_global": F.normalize(user_global, dim=1),
            "j_global": F.normalize(job_global, dim=1),
        }

    def score_pairs(self, enc: dict[str, torch.Tensor], users: torch.Tensor, jobs: torch.Tensor) -> torch.Tensor:
        """Score candidate-job pairs with the PJFCANN pair classifier."""

        u_local, j_local = enc["u_local"][users], enc["j_local"][jobs]
        u_global, j_global = enc["u_global"][users], enc["j_global"][jobs]
        h_user = torch.cat([u_local, u_global], dim=1)
        h_job = torch.cat([j_local, j_global], dim=1)
        features = torch.cat([h_user, h_job, torch.abs(h_user - h_job), h_user * h_job], dim=1)
        retrieval_score = ((u_local * j_local).sum(1) + (u_global * j_global).sum(1)) / math.sqrt(2.0)
        return retrieval_score + self.classifier(features).squeeze(1)

    def base_scores(self, enc: dict[str, torch.Tensor], users: torch.Tensor) -> torch.Tensor:
        """Fast dot-product score for first-stage full-catalog retrieval."""

        local = enc["u_local"][users] @ enc["j_local"].T
        global_score = enc["u_global"][users] @ enc["j_global"].T
        return (local + global_score) / math.sqrt(2.0)


def _train_pairs(bundle: BundleV2) -> np.ndarray:
    pairs = [(u, j) for u, jobs in bundle.train_pos.items() for j in jobs]
    return np.array(pairs, dtype=np.int64)


def _sample_negatives(
    users: np.ndarray,
    num_jobs: int,
    positives: dict[int, set[int]],
    rng: np.random.Generator,
    neg_per_pos: int,
) -> np.ndarray:
    neg = rng.integers(0, num_jobs, size=(len(users), neg_per_pos), dtype=np.int64)
    for row, user in enumerate(users):
        seen = positives.get(int(user), set())
        for col in range(neg_per_pos):
            while int(neg[row, col]) in seen:
                neg[row, col] = rng.integers(0, num_jobs)
    return neg


def _rankings(
    model: PJFCANNV2,
    enc: dict[str, torch.Tensor],
    bundle: BundleV2,
    users: list[int],
    device: str,
    topn: int,
    recall_k: int,
    exclude_valid: bool,
    pair_batch: int,
) -> dict[int, np.ndarray]:
    rankings: dict[int, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(users), 256):
            chunk = users[start : start + 256]
            base = model.base_scores(enc, torch.tensor(chunk, device=device)).cpu().numpy()
            for row, user in enumerate(chunk):
                exclude = set(bundle.train_pos.get(user, set()))
                if exclude_valid:
                    exclude |= bundle.valid_pos.get(user, set())
                candidates = topn_from_scores(base[row], exclude, topn)
                head = candidates[:recall_k]
                if len(head) == 0:
                    rankings[user] = candidates
                    continue
                u_arr = np.full(len(head), user, dtype=np.int64)
                scores = []
                for batch_start in range(0, len(head), pair_batch):
                    part = head[batch_start : batch_start + pair_batch]
                    scores.append(
                        model.score_pairs(
                            enc,
                            torch.tensor(u_arr[batch_start : batch_start + len(part)], device=device),
                            torch.tensor(part, device=device),
                        ).detach().cpu()
                    )
                rerank_scores = torch.cat(scores).numpy()
                order = np.argsort(-rerank_scores)
                rankings[user] = np.concatenate([head[order], candidates[recall_k:]])
    return rankings


def train_pjfcann_v2(
    bundle: BundleV2,
    config: dict,
    eval_users: list[int],
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """Train the PJFCANN-style baseline and return full-catalog rankings."""

    device = config.get("device", "cuda")
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    emb_user, emb_job, _ = bundle.load_embeddings()
    user_text = torch.tensor(emb_user, dtype=torch.float32, device=device)
    job_text = torch.tensor(emb_job, dtype=torch.float32, device=device)

    pairs = _train_pairs(bundle)
    edge_u = torch.tensor(pairs[:, 0], device=device)
    edge_j = torch.tensor(pairs[:, 1], device=device)
    model = PJFCANNV2(
        text_dim=emb_user.shape[1],
        num_users=bundle.num_users,
        num_jobs=bundle.num_jobs,
        dim=int(config.get("dim", 128)),
        layers=int(config.get("layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    batch = int(config.get("batch", 4096))
    epochs = int(config.get("epochs", 40))
    patience = int(config.get("patience", 6))
    neg_per_pos = int(config.get("neg_per_pos", 2))
    lambda_bpr = float(config.get("lambda_bpr", 0.2))
    positives = {u: jobs | bundle.valid_pos.get(u, set()) | bundle.test_pos.get(u, set()) for u, jobs in bundle.train_pos.items()}
    valid_users = sorted(bundle.valid_pos)
    valid_sample = valid_users[:: max(1, len(valid_users) // int(config.get("valid_users", 1500)))]
    valid_topn = int(config.get("valid_topn", 200))
    valid_recall_k = int(config.get("valid_recall_k", 200))
    topn = int(config.get("topn", 1000))
    recall_k = int(config.get("recall_k", 1000))
    pair_batch = int(config.get("pair_batch", 8192))

    best_ndcg, best_state, bad = -1.0, None, 0
    last_loss = 0.0
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(pairs))
        epoch_loss, steps = 0.0, 0
        for start in range(0, len(perm), batch):
            idx = perm[start : start + batch]
            users = pairs[idx, 0]
            pos_jobs = pairs[idx, 1]
            neg_jobs = _sample_negatives(users, bundle.num_jobs, positives, rng, neg_per_pos)
            enc = model.encode(user_text, job_text, edge_u, edge_j)
            u_t = torch.tensor(users, device=device)
            s_pos = model.score_pairs(enc, u_t, torch.tensor(pos_jobs, device=device))
            neg_flat = neg_jobs.reshape(-1)
            s_neg = model.score_pairs(
                enc,
                torch.tensor(np.repeat(users, neg_per_pos), device=device),
                torch.tensor(neg_flat, device=device),
            ).view(len(users), neg_per_pos)
            labels = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg.reshape(-1))])
            logits = torch.cat([s_pos, s_neg.reshape(-1)])
            bce = F.binary_cross_entropy_with_logits(logits, labels)
            bpr = -F.logsigmoid(s_pos.unsqueeze(1) - s_neg).mean()
            loss = bce + lambda_bpr * bpr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            steps += 1
        last_loss = epoch_loss / max(1, steps)
        model.eval()
        with torch.no_grad():
            enc = model.encode(user_text, job_text, edge_u, edge_j)
            valid_rank = _rankings(
                model,
                enc,
                bundle,
                valid_sample,
                device,
                topn=valid_topn,
                recall_k=valid_recall_k,
                exclude_valid=False,
                pair_batch=pair_batch,
            )
        ndcg = evaluate_rankings(valid_rank, {u: bundle.valid_pos[u] for u in valid_sample}).metrics.get("ndcg@10", 0.0)
        print(f"  pjfcann epoch {epoch + 1}/{epochs} loss={last_loss:.4f} valid_ndcg@10={ndcg:.4f}", flush=True)
        if ndcg > best_ndcg:
            best_ndcg, bad = ndcg, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("  early stop", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        enc = model.encode(user_text, job_text, edge_u, edge_j)
        rankings = _rankings(
            model,
            enc,
            bundle,
            eval_users,
            device,
            topn=topn,
            recall_k=recall_k,
            exclude_valid=True,
            pair_batch=pair_batch,
        )
    return rankings, {"train_loss": last_loss, "best_valid_ndcg10": best_ndcg, "train_pairs": float(len(pairs))}
