"""Training loop for SPC-HGT v2 (docs/12): pair sampling, losses, two-stage eval."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from jobmatch_gnn.data.bundle_v2 import BundleV2
from jobmatch_gnn.evaluation.rank_eval import evaluate_rankings, topn_from_scores
from jobmatch_gnn.models.spc_hgt_v2 import MAX_PATH, SPCHGTv2, build_graph_tensors


class PairFeaturizer:
    """CPU-side per-pair features: shared/gap skill ids + rule features."""

    def __init__(self, bundle: BundleV2) -> None:
        self.user_sets = [set(ids) for ids in bundle.users["skill_ids"]]
        self.job_lists = [list(ids) for ids in bundle.jobs["skill_ids"]]
        users, jobs = bundle.users, bundle.jobs
        self.u_city = [set(d) | {c} for d, c in zip(users["desired_cities"], users["live_city"], strict=False)]
        self.u_types = [set(t) for t in users["desired_types"]]
        self.u_degree = users["degree_rank"].to_numpy(np.float32)
        self.u_years = users["years"].to_numpy(np.float32)
        self.j_city = jobs["city"].tolist()
        self.j_type = jobs["job_type"].tolist()
        self.j_degree = jobs["min_degree_rank"].to_numpy(np.float32)
        self.j_years = jobs["min_years"].to_numpy(np.float32)

    def build(self, users: np.ndarray, jobs: np.ndarray):
        n = len(users)
        shared = np.full((n, MAX_PATH), -1, dtype=np.int64)
        gap = np.full((n, MAX_PATH), -1, dtype=np.int64)
        rule = np.zeros((n, 7), dtype=np.float32)
        for row in range(n):
            u, j = users[row], jobs[row]
            uset = self.user_sets[u]
            jlist = self.job_lists[j]
            sh = [s for s in jlist if s in uset][:MAX_PATH]
            ms = [s for s in jlist if s not in uset][:MAX_PATH]
            shared[row, : len(sh)] = sh
            gap[row, : len(ms)] = ms
            n_job = max(1, len(jlist))
            n_sh = sum(1 for s in jlist if s in uset)
            rule[row] = (
                n_sh / n_job,
                (n_job - n_sh) / n_job,
                float(self.j_city[j] in self.u_city[u]),
                float(self.u_degree[u] >= self.j_degree[j]),
                float(self.u_years[u] >= self.j_years[j]),
                float(self.j_type[j] in self.u_types[u]),
                np.log1p(n_sh),
            )
        return shared, gap, rule


def _to_dev(arrs, device):
    shared, gap, rule = arrs
    return (
        torch.tensor(shared, device=device),
        torch.tensor(shared >= 0, device=device),
        torch.tensor(gap, device=device),
        torch.tensor(gap >= 0, device=device),
        torch.tensor(rule, device=device),
    )


def build_hard_negatives(bundle: BundleV2, device: str, top: int = 200) -> dict[int, np.ndarray]:
    """SBERT-similar jobs the user never touched (semantic hard negatives)."""

    emb_user, emb_job, _ = bundle.load_embeddings()
    job_t = torch.tensor(emb_job, device=device)
    seen: dict[int, set[int]] = {}
    for source in (bundle.train_pos, bundle.valid_pos, bundle.test_pos, bundle.browsed):
        for u, jobs in source.items():
            seen.setdefault(u, set()).update(jobs)
    hard: dict[int, np.ndarray] = {}
    users = list(bundle.train_pos)
    for start in range(0, len(users), 512):
        chunk = users[start : start + 512]
        sims = torch.tensor(emb_user[chunk], device=device) @ job_t.T
        topk = torch.topk(sims, top + 50, dim=1).indices.cpu().numpy()
        for row, u in enumerate(chunk):
            cand = [j for j in topk[row] if j not in seen.get(u, set())][:top]
            hard[u] = np.array(cand, dtype=np.int64)
    return hard


def build_struct_negatives(bundle: BundleV2) -> dict[tuple[str, str], np.ndarray]:
    """Bucket job indices by (job_type, city) for structural negatives."""

    buckets: dict[tuple[str, str], list[int]] = {}
    for j, (jt, city) in enumerate(zip(bundle.jobs["job_type"], bundle.jobs["city"], strict=False)):
        buckets.setdefault((jt, city), []).append(j)
    return {key: np.array(val, dtype=np.int64) for key, val in buckets.items()}


def two_stage_rankings(
    model: SPCHGTv2,
    z: dict[str, torch.Tensor],
    bundle: BundleV2,
    featurizer: PairFeaturizer,
    users: list[int],
    device: str,
    recall_k: int = 500,
    topn: int = 1000,
    pair_batch: int = 65536,
    exclude_valid: bool = True,
) -> dict[int, np.ndarray]:
    """Dot-product recall to top-500, rerank with the full scorer, fill to top-N.

    ``exclude_valid`` must be False when ranking for the valid users themselves,
    otherwise their held-out positives would be removed from the candidate list.
    """

    rankings: dict[int, np.ndarray] = {}
    z_j_t = z["j"].T.contiguous()
    with torch.no_grad():
        for start in range(0, len(users), 256):
            chunk = users[start : start + 256]
            scores = (z["c"][torch.tensor(chunk, device=device)] @ z_j_t).cpu().numpy()
            for row, u in enumerate(chunk):
                exclude = set(bundle.train_pos.get(u, set()))
                if exclude_valid:
                    exclude |= bundle.valid_pos.get(u, set())
                base = topn_from_scores(scores[row], exclude, topn)
                head = base[:recall_k]
                u_arr = np.full(len(head), u, dtype=np.int64)
                feats = _to_dev(featurizer.build(u_arr, head), device)
                s = model.score_pairs(
                    z,
                    torch.tensor(u_arr, device=device),
                    torch.tensor(head, device=device),
                    feats[4],
                    feats[0],
                    feats[1],
                    feats[2],
                    feats[3],
                )
                order = torch.argsort(-s).cpu().numpy()
                rankings[u] = np.concatenate([head[order], base[recall_k:]])
    return rankings


def train_spc_hgt_v2(bundle: BundleV2, config: dict) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """Train the full model; returns test rankings + training info."""

    device = config.get("device", "cuda")
    seed = int(config.get("seed", 42))
    use_path = bool(config.get("use_path", True))
    use_nce = bool(config.get("use_nce", True))
    use_hard = bool(config.get("use_hard_neg", True))
    use_text = bool(config.get("use_text", True))
    dim = int(config.get("dim", 128))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    feats, edges, meta = build_graph_tensors(bundle, device)
    if not use_text:  # ablation: replace SBERT features with random vectors
        for key in ("c_text", "j_text", "s_text"):
            g = torch.Generator(device="cpu").manual_seed(seed)
            feats[key] = torch.randn(feats[key].shape, generator=g).to(device) * 0.1
    sbert_dim = feats["c_text"].shape[1]
    model = SPCHGTv2(
        sbert_dim,
        dim,
        meta["num_cities"],
        meta["num_types"],
        num_users=bundle.num_users,
        num_jobs=bundle.num_jobs,
        use_path=use_path,
        use_text=True,
        use_id=bool(config.get("use_id", True)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-5)))

    featurizer = PairFeaturizer(bundle)
    pairs = np.array(
        [(u, j, lvl) for (u, jobs) in bundle.train_pos.items() for j in jobs for lvl in (2,)],
        dtype=np.int64,
    )
    # behavior level per train pair (3 = satisfied)
    level_lookup = {}
    pos_mask = (bundle.interactions["level"] >= 2) & (bundle.interactions["split"] == "train")
    for u, j, lvl in bundle.interactions.loc[pos_mask, ["u", "j", "level"]].itertuples(index=False):
        level_lookup[(u, j)] = lvl
    pairs[:, 2] = [level_lookup.get((u, j), 2) for u, j, _ in pairs]

    pos_sets = {u: set(jobs) | bundle.valid_pos.get(u, set()) | bundle.test_pos.get(u, set()) for u, jobs in bundle.train_pos.items()}
    hard_sem = build_hard_negatives(bundle, device) if use_hard else {}
    struct_buckets = build_struct_negatives(bundle) if use_hard else {}
    j_type, j_city = bundle.jobs["job_type"].tolist(), bundle.jobs["city"].tolist()

    batch = int(config.get("batch", 2048))
    epochs = int(config.get("epochs", 30))
    warm = int(config.get("warmup_epochs", 3))
    neg_per_pos = int(config.get("neg_per_pos", 4))
    lam_bce = float(config.get("lambda_bce", 0.2))
    lam_nce = float(config.get("lambda_nce", 0.1))
    tau = {3: 0.07, 2: 0.10}
    patience = int(config.get("patience", 5))
    valid_users = sorted(bundle.valid_pos)
    valid_sample = valid_users[:: max(1, len(valid_users) // 1500)]

    def sample_negative(u: int, j_pos: int, epoch: int) -> int:
        roll = rng.random()
        if use_hard and epoch >= warm and roll < 0.5 and u in hard_sem and len(hard_sem[u]):
            return int(rng.choice(hard_sem[u]))
        if use_hard and epoch >= warm and roll < 0.7:
            bucket = struct_buckets.get((j_type[j_pos], j_city[j_pos]))
            if bucket is not None and len(bucket) > 1:
                for _ in range(5):
                    j = int(rng.choice(bucket))
                    if j not in pos_sets.get(u, set()):
                        return j
        while True:
            j = int(rng.integers(0, bundle.num_jobs))
            if j not in pos_sets.get(u, set()):
                return j

    best_ndcg, best_state, bad = -1.0, None, 0
    info: dict[str, float] = {}
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(pairs))
        epoch_loss, steps = 0.0, 0
        for start in range(0, len(perm), batch):
            idx = perm[start : start + batch]
            u = pairs[idx, 0]
            j_pos = pairs[idx, 1]
            levels = pairs[idx, 2]
            j_neg = np.array([[sample_negative(int(uu), int(jj), epoch) for _ in range(neg_per_pos)] for uu, jj in zip(u, j_pos, strict=False)])

            z = model.encode(feats, edges)
            u_t = torch.tensor(u, device=device)
            sp_feats = _to_dev(featurizer.build(u, j_pos), device)
            s_pos = model.score_pairs(z, u_t, torch.tensor(j_pos, device=device), sp_feats[4], sp_feats[0], sp_feats[1], sp_feats[2], sp_feats[3])
            neg_flat = j_neg.reshape(-1)
            u_rep = np.repeat(u, neg_per_pos)
            sn_feats = _to_dev(featurizer.build(u_rep, neg_flat), device)
            s_neg = model.score_pairs(z, torch.tensor(u_rep, device=device), torch.tensor(neg_flat, device=device), sn_feats[4], sn_feats[0], sn_feats[1], sn_feats[2], sn_feats[3]).view(len(idx), neg_per_pos)

            bpr = -F.logsigmoid(s_pos.unsqueeze(1) - s_neg).mean()
            bce = F.binary_cross_entropy_with_logits(
                torch.cat([s_pos, s_neg.reshape(-1)]),
                torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg.reshape(-1))]),
            )
            loss = bpr + lam_bce * bce
            if use_nce:
                z_c = F.normalize(z["c"][u_t], dim=1)
                z_jp = F.normalize(z["j"][torch.tensor(j_pos, device=device)], dim=1)
                tau_t = torch.tensor([tau.get(int(lv), 0.1) for lv in levels], device=device)
                logits = (z_c @ z_jp.T) / tau_t.unsqueeze(1)
                loss = loss + lam_nce * F.cross_entropy(logits, torch.arange(len(idx), device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            steps += 1
        model.eval()
        with torch.no_grad():
            z = model.encode(feats, edges)
        valid_rank = two_stage_rankings(model, z, bundle, featurizer, valid_sample, device, topn=200, exclude_valid=False)
        ndcg = evaluate_rankings(valid_rank, {u: bundle.valid_pos[u] for u in valid_sample}).metrics.get("ndcg@10", 0.0)
        print(f"  spc_hgt epoch {epoch + 1}/{epochs} loss={epoch_loss / max(1, steps):.4f} valid_ndcg@10={ndcg:.4f}", flush=True)
        if ndcg > best_ndcg:
            best_ndcg, bad = ndcg, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("  early stop", flush=True)
                break
        info["train_loss"] = epoch_loss / max(1, steps)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model.encode(feats, edges)
    test_users = sorted(bundle.test_pos)
    rankings = two_stage_rankings(model, z, bundle, featurizer, test_users, device)
    info.update({"best_valid_ndcg10": best_ndcg, "train_pairs": float(len(pairs))})
    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))},
                "meta": meta,
                "sbert_dim": sbert_dim,
                "dim": dim,
            },
            checkpoint_path,
        )
        print(f"  saved checkpoint to {checkpoint_path}", flush=True)
    return rankings, info
