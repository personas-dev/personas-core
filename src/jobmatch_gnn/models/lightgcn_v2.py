"""LightGCN baseline on the full interaction graph (docs/13 B4)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from jobmatch_gnn.data.bundle_v2 import BundleV2
from jobmatch_gnn.evaluation.rank_eval import topn_from_scores


class LightGCNV2(nn.Module):
    """Standard LightGCN over the candidate-job bipartite train graph."""

    def __init__(self, num_users: int, num_jobs: int, dim: int, layers: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.job_emb = nn.Embedding(num_jobs, dim)
        self.layers = layers
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.job_emb.weight, std=0.1)

    def propagate(self, edge_u: torch.Tensor, edge_j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        user, job = self.user_emb.weight, self.job_emb.weight
        deg_u = torch.zeros(user.shape[0], device=user.device).index_add_(0, edge_u, torch.ones_like(edge_u, dtype=torch.float)).clamp_min(1)
        deg_j = torch.zeros(job.shape[0], device=job.device).index_add_(0, edge_j, torch.ones_like(edge_j, dtype=torch.float)).clamp_min(1)
        norm = torch.rsqrt(deg_u[edge_u] * deg_j[edge_j]).unsqueeze(1)
        users, jobs = [user], [job]
        for _ in range(self.layers):
            nxt_j = torch.zeros_like(job).index_add_(0, edge_j, user[edge_u] * norm)
            nxt_u = torch.zeros_like(user).index_add_(0, edge_u, job[edge_j] * norm)
            user, job = nxt_u, nxt_j
            users.append(user)
            jobs.append(job)
        return torch.stack(users).mean(0), torch.stack(jobs).mean(0)


def train_lightgcn_v2(
    bundle: BundleV2,
    config: dict,
    eval_users: list[int],
    topn: int = 1000,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """Train with BPR on full train positives; return test rankings."""

    device = config.get("device", "cuda")
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pairs = np.array([(u, j) for u, jobs in bundle.train_pos.items() for j in jobs], dtype=np.int64)
    edge_u = torch.tensor(pairs[:, 0], device=device)
    edge_j = torch.tensor(pairs[:, 1], device=device)
    model = LightGCNV2(bundle.num_users, bundle.num_jobs, int(config.get("dim", 128)), int(config.get("layers", 3))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config.get("lr", 1e-3)))
    batch = int(config.get("batch", 8192))
    epochs = int(config.get("epochs", 200))
    pos_sets = {u: jobs | bundle.valid_pos.get(u, set()) | bundle.test_pos.get(u, set()) for u, jobs in bundle.train_pos.items()}

    last_loss = 0.0
    for epoch in range(epochs):
        perm = rng.permutation(len(pairs))
        epoch_loss, steps = 0.0, 0
        for start in range(0, len(perm), batch):
            idx = perm[start : start + batch]
            u = pairs[idx, 0]
            j_pos = pairs[idx, 1]
            j_neg = rng.integers(0, bundle.num_jobs, size=len(idx))
            for row in range(len(idx)):  # resample collisions with user positives
                while j_neg[row] in pos_sets.get(u[row], set()):
                    j_neg[row] = rng.integers(0, bundle.num_jobs)
            ue, je = model.propagate(edge_u, edge_j)
            u_t = torch.tensor(u, device=device)
            score_pos = (ue[u_t] * je[torch.tensor(j_pos, device=device)]).sum(1)
            score_neg = (ue[u_t] * je[torch.tensor(j_neg, device=device)]).sum(1)
            reg = float(config.get("l2", 1e-5)) * (
                model.user_emb(u_t).norm(2).pow(2)
                + model.job_emb(torch.tensor(j_pos, device=device)).norm(2).pow(2)
                + model.job_emb(torch.tensor(j_neg, device=device)).norm(2).pow(2)
            ) / len(idx)
            loss = -F.logsigmoid(score_pos - score_neg).mean() + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            steps += 1
        last_loss = epoch_loss / max(1, steps)
        if (epoch + 1) % 20 == 0:
            print(f"  lightgcn epoch {epoch + 1}/{epochs} loss={last_loss:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        ue, je = model.propagate(edge_u, edge_j)
        rankings: dict[int, np.ndarray] = {}
        for start in range(0, len(eval_users), 512):
            chunk = eval_users[start : start + 512]
            scores = (ue[torch.tensor(chunk, device=device)] @ je.T).cpu().numpy()
            for row, u in enumerate(chunk):
                exclude = bundle.train_pos.get(u, set()) | bundle.valid_pos.get(u, set())
                rankings[u] = topn_from_scores(scores[row], exclude, topn)
    return rankings, {"train_loss": last_loss, "train_pairs": float(len(pairs))}
