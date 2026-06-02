"""Deterministic recommendation baselines."""
from __future__ import annotations

from collections import Counter
import hashlib
import math

import numpy as np

from jobmatch_gnn.data.dataset import CandidateRecord, JobRecord, pair_features, tokenize_text


class RuleMatcher:
    """Rule baseline using skill path and structured feature overlap."""

    name = "rule"

    def score(self, candidate: CandidateRecord, job: JobRecord) -> float:
        """Score a candidate-job pair."""

        features = pair_features(candidate, job)
        coverage, missing, city, education, experience, job_type, path_count = features.tolist()
        return float(0.50 * coverage + 0.10 * (1.0 - missing) + 0.15 * city + 0.10 * education + 0.10 * experience + 0.05 * job_type + 0.03 * path_count)


class BM25Matcher:
    """Sparse lexical BM25 matcher over job text."""

    name = "bm25"

    def __init__(self, jobs: dict[str, JobRecord], k1: float = 1.5, b: float = 0.75) -> None:
        self.jobs = jobs
        self.k1 = k1
        self.b = b
        self.doc_tokens: dict[str, list[str]] = {job_id: tokenize_text(job.text, max_tokens=500) for job_id, job in jobs.items()}
        self.doc_tf: dict[str, Counter[str]] = {job_id: Counter(tokens) for job_id, tokens in self.doc_tokens.items()}
        self.doc_len: dict[str, int] = {job_id: max(1, len(tokens)) for job_id, tokens in self.doc_tokens.items()}
        self.avg_doc_len = float(np.mean(list(self.doc_len.values()))) if self.doc_len else 1.0
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens.values():
            df.update(set(tokens))
        doc_count = max(1, len(self.doc_tokens))
        self.idf = {term: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        self.query_cache: dict[str, list[str]] = {}

    def score(self, candidate: CandidateRecord, job: JobRecord) -> float:
        """Score one candidate-job pair with BM25."""

        query_terms = self.query_cache.setdefault(candidate.user_id, tokenize_text(candidate.text, max_tokens=220))
        tf = self.doc_tf.get(job.job_id, Counter())
        doc_len = self.doc_len.get(job.job_id, 1)
        score = 0.0
        for term in query_terms:
            term_freq = tf.get(term, 0)
            if term_freq <= 0:
                continue
            denom = term_freq + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len)
            score += self.idf.get(term, 0.0) * term_freq * (self.k1 + 1.0) / denom
        return float(score)


class HashSemanticMatcher:
    """Dependency-free semantic fallback using hashed TF-IDF vectors."""

    name = "semantic_hash"

    def __init__(self, jobs: dict[str, JobRecord], dim: int = 512) -> None:
        self.jobs = jobs
        self.dim = dim
        doc_tokens = {job_id: tokenize_text(job.text, max_tokens=500) for job_id, job in jobs.items()}
        df: Counter[str] = Counter()
        for tokens in doc_tokens.values():
            df.update(set(tokens))
        doc_count = max(1, len(doc_tokens))
        self.idf = {term: math.log(1.0 + doc_count / (1.0 + freq)) for term, freq in df.items()}
        self.job_vectors = {job_id: self._vectorize(tokens) for job_id, tokens in doc_tokens.items()}
        self.query_cache: dict[str, np.ndarray] = {}

    def _index(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        sign = 1.0 if value % 2 == 0 else -1.0
        return value % self.dim, sign

    def _vectorize(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts = Counter(tokens)
        for token, count in counts.items():
            idx, sign = self._index(token)
            vec[idx] += sign * float(count) * float(self.idf.get(token, 1.0))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def score(self, candidate: CandidateRecord, job: JobRecord) -> float:
        """Score one candidate-job pair with hashed cosine similarity."""

        query = self.query_cache.setdefault(candidate.user_id, self._vectorize(tokenize_text(candidate.text, max_tokens=300)))
        return float(np.dot(query, self.job_vectors.get(job.job_id, np.zeros(self.dim, dtype=np.float32))))
