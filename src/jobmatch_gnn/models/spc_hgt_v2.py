"""SPC-HGT v2: skill-path contrastive heterogeneous GNN (docs/12).

Components:
  - Relation-aware 2-layer message passing over Candidate/Job/Skill nodes
    (scatter-mean per relation), SBERT-initialized node features, structured
    fields (city / job type / degree / years) embedded into the input layer.
  - Skill-path attention scoring: attention over shared Candidate-Skill-Job
    paths; weights double as the explanation output.
  - Losses: BPR + BCE + behavior-level-tempered InfoNCE.
  - Hard negatives: SBERT-similar non-interacted jobs + same-type/city jobs,
    mixed in after a warmup (curriculum).
Ablation flags: use_path / use_nce / use_hard_neg / use_text.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from jobmatch_gnn.data.bundle_v2 import BundleV2

MAX_PATH = 10
RULE_DIM = 7


def _codes(values, vocab: dict[str, int]) -> np.ndarray:
    return np.array([vocab.setdefault(v, len(vocab)) for v in values], dtype=np.int64)


class HeteroEncoder(nn.Module):
    """Two relation-aware scatter-mean layers over C/J/S nodes."""

    RELATIONS = ("cs", "sc", "js", "sj", "cj", "jc")

    def __init__(self, dim: int, layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.layers = layers
        self.rel_lin = nn.ModuleList(
            [nn.ModuleDict({rel: nn.Linear(dim, dim) for rel in self.RELATIONS}) for _ in range(layers)]
        )
        self.norms = nn.ModuleList(
            [nn.ModuleDict({t: nn.LayerNorm(dim) for t in ("c", "j", "s")}) for _ in range(layers)]
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _agg(src: torch.Tensor, index_src: torch.Tensor, index_dst: torch.Tensor, dst_size: int) -> torch.Tensor:
        out = torch.zeros(dst_size, src.shape[1], device=src.device)
        count = torch.zeros(dst_size, device=src.device)
        out.index_add_(0, index_dst, src[index_src])
        count.index_add_(0, index_dst, torch.ones_like(index_dst, dtype=torch.float))
        return out / count.clamp_min(1.0).unsqueeze(1)

    def forward(
        self,
        h: dict[str, torch.Tensor],
        edges: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        for layer in range(self.layers):
            lin = self.rel_lin[layer]
            msg: dict[str, torch.Tensor] = {}
            spec = {
                "c": [("sc", "s"), ("jc", "j")],
                "j": [("sj", "s"), ("cj", "c")],
                "s": [("cs", "c"), ("js", "j")],
            }
            for dst, rels in spec.items():
                acc = torch.zeros_like(h[dst])
                for rel, src_type in rels:
                    if rel not in edges:
                        continue
                    src_idx, dst_idx = edges[rel]
                    acc = acc + self._agg(lin[rel](h[src_type]), src_idx, dst_idx, h[dst].shape[0])
                msg[dst] = acc
            h = {t: self.norms[layer][t](h[t] + self.dropout(F.gelu(msg[t]))) for t in h}
        return h


class SPCHGTv2(nn.Module):
    """Full model: encoder + skill-path attention matching head."""

    def __init__(
        self,
        sbert_dim: int,
        dim: int,
        num_cities: int,
        num_types: int,
        num_users: int = 0,
        num_jobs: int = 0,
        use_path: bool = True,
        use_text: bool = True,
        use_id: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_path = use_path
        self.use_text = use_text
        self.use_id = use_id and num_users > 0
        if self.use_id:  # collaborative ID embeddings fused with content features
            self.user_id_emb = nn.Embedding(num_users, dim)
            self.job_id_emb = nn.Embedding(num_jobs, dim)
            nn.init.normal_(self.user_id_emb.weight, std=0.1)
            nn.init.normal_(self.job_id_emb.weight, std=0.1)
        self.proj = nn.ModuleDict(
            {t: nn.Linear(sbert_dim if use_text else dim, dim) for t in ("c", "j", "s")}
        )
        self.city_emb = nn.Embedding(num_cities + 1, dim)
        self.type_emb = nn.Embedding(num_types + 1, dim)
        self.struct_mlp = nn.Sequential(nn.Linear(2, dim), nn.GELU())
        self.encoder = HeteroEncoder(dim, layers=2, dropout=dropout)
        self.path_att = nn.Sequential(nn.Linear(3 * dim, dim), nn.Tanh(), nn.Linear(dim, 1))
        head_in = 2 * dim + (2 * dim if use_path else 0) + RULE_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.dim = dim

    def encode(self, feats: dict[str, torch.Tensor], edges) -> dict[str, torch.Tensor]:
        h = {
            "c": self.proj["c"](feats["c_text"]) + self.city_emb(feats["c_city"]) + self.type_emb(feats["c_type"]) + self.struct_mlp(feats["c_struct"]),
            "j": self.proj["j"](feats["j_text"]) + self.city_emb(feats["j_city"]) + self.type_emb(feats["j_type"]) + self.struct_mlp(feats["j_struct"]),
            "s": self.proj["s"](feats["s_text"]),
        }
        if self.use_id:
            h["c"] = h["c"] + self.user_id_emb.weight
            h["j"] = h["j"] + self.job_id_emb.weight
        return self.encoder(h, edges)

    def path_repr(
        self,
        z_s: torch.Tensor,
        z_c_pair: torch.Tensor,
        z_j_pair: torch.Tensor,
        shared: torch.Tensor,
        shared_mask: torch.Tensor,
        gap: torch.Tensor,
        gap_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attention over shared skills; mean over missing skills. Returns (p, g, alpha)."""

        zs_shared = z_s[shared.clamp_min(0)]  # B x P x d
        ctx = torch.cat([zs_shared, z_c_pair.unsqueeze(1).expand(-1, MAX_PATH, -1), z_j_pair.unsqueeze(1).expand(-1, MAX_PATH, -1)], dim=2)
        logits = self.path_att(ctx).squeeze(2).masked_fill(~shared_mask, -1e9)
        alpha = torch.softmax(logits, dim=1) * shared_mask.float()
        p = (alpha.unsqueeze(2) * zs_shared).sum(1)
        zs_gap = z_s[gap.clamp_min(0)] * gap_mask.unsqueeze(2).float()
        g = zs_gap.sum(1) / gap_mask.sum(1, keepdim=True).clamp_min(1).float()
        has_path = shared_mask.any(dim=1, keepdim=True).float()
        return p * has_path, g, alpha

    def score_pairs(
        self,
        z: dict[str, torch.Tensor],
        users: torch.Tensor,
        jobs: torch.Tensor,
        rule_feats: torch.Tensor,
        shared: torch.Tensor,
        shared_mask: torch.Tensor,
        gap: torch.Tensor,
        gap_mask: torch.Tensor,
        return_alpha: bool = False,
    ):
        z_c, z_j = z["c"][users], z["j"][jobs]
        dot = (z_c * z_j).sum(1) / self.dim**0.5
        parts = [z_c, z_j]
        alpha = None
        if self.use_path:
            p, g, alpha = self.path_repr(z["s"], z_c, z_j, shared, shared_mask, gap, gap_mask)
            parts.extend([p, g])
        parts.append(rule_feats)
        score = dot + self.head(torch.cat(parts, dim=1)).squeeze(1)
        return (score, alpha) if return_alpha else score


def build_graph_tensors(bundle: BundleV2, device: str, include_browsed: bool = False):
    """Assemble node features and edge index tensors (train edges only)."""

    emb_user, emb_job, emb_skill = bundle.load_embeddings()
    city_vocab: dict[str, int] = {"<unk>": 0}
    type_vocab: dict[str, int] = {"<unk>": 0}
    c_city = _codes(bundle.users["live_city"].fillna(""), city_vocab)
    j_city = _codes(bundle.jobs["city"].fillna(""), city_vocab)
    c_type = _codes([t[0] if len(t) else "" for t in bundle.users["desired_types"]], type_vocab)
    j_type = _codes(bundle.jobs["job_type"].fillna(""), type_vocab)
    c_struct = np.stack([bundle.users["degree_rank"].to_numpy(np.float32) / 8.0, np.log1p(bundle.users["years"].to_numpy(np.float32)) / 4.0], axis=1)
    j_struct = np.stack([bundle.jobs["min_degree_rank"].to_numpy(np.float32) / 8.0, np.log1p(bundle.jobs["min_years"].to_numpy(np.float32)) / 4.0], axis=1)

    feats = {
        "c_text": torch.tensor(emb_user, device=device),
        "j_text": torch.tensor(emb_job, device=device),
        "s_text": torch.tensor(emb_skill, device=device),
        "c_city": torch.tensor(c_city, device=device),
        "j_city": torch.tensor(j_city, device=device),
        "c_type": torch.tensor(c_type, device=device),
        "j_type": torch.tensor(j_type, device=device),
        "c_struct": torch.tensor(c_struct, device=device),
        "j_struct": torch.tensor(j_struct, device=device),
    }

    cs = bundle.user_skill.tocoo()
    js = bundle.job_skill.tocoo()
    cj = np.array([(u, j) for u, jobs in bundle.train_pos.items() for j in jobs], dtype=np.int64)
    if include_browsed:
        weak = np.array([(u, j) for u, jobs in bundle.browsed.items() for j in jobs], dtype=np.int64)
        cj = np.concatenate([cj, weak]) if len(weak) else cj
    def t(arr) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.long, device=device)
    edges = {
        "cs": (t(cs.row), t(cs.col)),
        "sc": (t(cs.col), t(cs.row)),
        "js": (t(js.row), t(js.col)),
        "sj": (t(js.col), t(js.row)),
        "cj": (t(cj[:, 0]), t(cj[:, 1])),
        "jc": (t(cj[:, 1]), t(cj[:, 0])),
    }
    meta = {"num_cities": len(city_vocab), "num_types": len(type_vocab)}
    return feats, edges, meta
