"""BM25 baseline skeleton."""
from __future__ import annotations


class BM25Matcher:
    """Sparse retrieval baseline using BM25."""

    def fit(self, job_texts: list[str]) -> None:
        """Build BM25 corpus."""
        raise NotImplementedError

    def search(self, candidate_text: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return top-k job indices and scores."""
        raise NotImplementedError
