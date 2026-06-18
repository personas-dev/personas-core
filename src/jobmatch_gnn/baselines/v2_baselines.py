"""Strong baselines on the full v2 split: Popularity / Rule / BM25 / SBERT (docs/13)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import scipy.sparse as sp

from jobmatch_gnn.data.bundle_v2 import BundleV2
from jobmatch_gnn.evaluation.rank_eval import topn_from_scores

TOPN = 1000


def _exclusions(bundle: BundleV2, user: int) -> set[int]:
    """Jobs to hide from the test ranking: the user's train + valid positives."""

    return bundle.train_pos.get(user, set()) | bundle.valid_pos.get(user, set())


def popularity_rankings(bundle: BundleV2, users: list[int], topn: int = TOPN) -> dict[int, np.ndarray]:
    """Rank jobs by train-positive frequency (non-personalized lower bound)."""

    scores = np.zeros(bundle.num_jobs, dtype=np.float32)
    for jobs in bundle.train_pos.values():
        for j in jobs:
            scores[j] += 1.0
    return {u: topn_from_scores(scores, _exclusions(bundle, u), topn) for u in users}


def rule_rankings(
    bundle: BundleV2,
    users: list[int],
    weights: dict[str, float] | None = None,
    topn: int = TOPN,
    batch: int = 256,
) -> dict[int, np.ndarray]:
    """Skill-coverage + structured matching score over the clean vocabulary."""

    w = {"coverage": 3.0, "city": 1.0, "type": 1.5, "degree": 0.3, "years": 0.3}
    w.update(weights or {})
    jobs = bundle.jobs
    job_city = jobs["city"].to_numpy()
    job_type = jobs["job_type"].to_numpy()
    job_degree = jobs["min_degree_rank"].to_numpy(dtype=np.float32)
    job_years = jobs["min_years"].to_numpy(dtype=np.float32)
    # binarized skill matrices; coverage = |shared| / |job skills|
    user_bin = bundle.user_skill.sign().tocsr()
    job_bin = bundle.job_skill.sign().tocsr()
    job_skill_count = np.asarray(job_bin.sum(axis=1)).ravel().clip(min=1.0)
    job_bin_t = job_bin.T.tocsr()

    rankings: dict[int, np.ndarray] = {}
    user_rows = bundle.users
    for start in range(0, len(users), batch):
        chunk = users[start : start + batch]
        shared = (user_bin[chunk] @ job_bin_t).toarray().astype(np.float32)
        coverage = shared / job_skill_count[None, :]
        for row, u in enumerate(chunk):
            rec = user_rows.iloc[u]
            cities = set(rec["desired_cities"]) | {rec["live_city"]}
            types = set(rec["desired_types"])
            score = w["coverage"] * coverage[row]
            score += w["city"] * np.isin(job_city, list(cities)).astype(np.float32)
            if types:
                score += w["type"] * np.isin(job_type, list(types)).astype(np.float32)
            score += w["degree"] * (job_degree <= rec["degree_rank"]).astype(np.float32)
            score += w["years"] * (job_years <= rec["years"]).astype(np.float32)
            rankings[u] = topn_from_scores(score, _exclusions(bundle, u), topn)
    return rankings


def _bm25_matrix(token_lists: list[list[str]], vocab: dict[str, int], k1: float = 1.5, b: float = 0.75) -> sp.csr_matrix:
    """Build a BM25-weighted doc-term CSR matrix."""

    indptr, indices, data = [0], [], []
    lengths = np.array([max(1, len(tokens)) for tokens in token_lists], dtype=np.float32)
    avg_len = float(lengths.mean())
    df: Counter[int] = Counter()
    rows: list[Counter[int]] = []
    for tokens in token_lists:
        counts = Counter(vocab[t] for t in tokens if t in vocab)
        rows.append(counts)
        df.update(counts.keys())
    n_docs = len(token_lists)
    idf = np.zeros(len(vocab), dtype=np.float32)
    for term, count in df.items():
        idf[term] = math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))
    for doc_idx, counts in enumerate(rows):
        norm = k1 * (1.0 - b + b * lengths[doc_idx] / avg_len)
        for term, tf in counts.items():
            indices.append(term)
            data.append(idf[term] * tf * (k1 + 1.0) / (tf + norm))
        indptr.append(len(indices))
    return sp.csr_matrix((np.array(data, dtype=np.float32), np.array(indices), np.array(indptr)), shape=(n_docs, len(vocab)))


def bm25_rankings(bundle: BundleV2, users: list[int], topn: int = TOPN, batch: int = 256) -> dict[int, np.ndarray]:
    """BM25 retrieval with jieba tokenization over title*2 + description."""

    import jieba

    job_tokens = [
        [t for t in jieba.lcut_for_search((title + " ") * 2 + text) if len(t.strip()) >= 2]
        for title, text in zip(bundle.jobs["title"].tolist(), bundle.jobs["text"].tolist(), strict=False)
    ]
    vocab: dict[str, int] = {}
    for tokens in job_tokens:
        for token in tokens:
            vocab.setdefault(token, len(vocab))
    doc_matrix = _bm25_matrix(job_tokens, vocab).T.tocsr()  # term x doc

    rankings: dict[int, np.ndarray] = {}
    user_texts = bundle.users["text"].tolist()
    for start in range(0, len(users), batch):
        chunk = users[start : start + batch]
        indptr, indices, data = [0], [], []
        for u in chunk:
            counts = Counter(vocab[t] for t in jieba.lcut_for_search(user_texts[u]) if len(t.strip()) >= 2 and t in vocab)
            for term, tf in counts.items():
                indices.append(term)
                data.append(float(tf))
            indptr.append(len(indices))
        queries = sp.csr_matrix((np.array(data, dtype=np.float32), np.array(indices), np.array(indptr)), shape=(len(chunk), len(vocab)))
        scores = (queries @ doc_matrix).toarray()
        for row, u in enumerate(chunk):
            rankings[u] = topn_from_scores(scores[row], _exclusions(bundle, u), topn)
    return rankings


def sbert_rankings(bundle: BundleV2, users: list[int], topn: int = TOPN, batch: int = 512, device: str = "cuda") -> dict[int, np.ndarray]:
    """Zero-shot dense retrieval with cached bge embeddings (cosine)."""

    import torch

    emb_user, emb_job, _ = bundle.load_embeddings()
    job_tensor = torch.tensor(emb_job, device=device)
    rankings: dict[int, np.ndarray] = {}
    for start in range(0, len(users), batch):
        chunk = users[start : start + batch]
        user_tensor = torch.tensor(emb_user[chunk], device=device)
        scores = (user_tensor @ job_tensor.T).cpu().numpy()
        for row, u in enumerate(chunk):
            rankings[u] = topn_from_scores(scores[row], _exclusions(bundle, u), topn)
    return rankings
