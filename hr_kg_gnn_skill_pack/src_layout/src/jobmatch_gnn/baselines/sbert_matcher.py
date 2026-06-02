"""SBERT baseline skeleton."""
from __future__ import annotations


class SBERTMatcher:
    """Dense semantic retrieval baseline."""

    def fit(self, job_texts: list[str]) -> None:
        """Encode job texts and build ANN index."""
        raise NotImplementedError

    def search(self, candidate_text: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return top-k job indices and cosine scores."""
        raise NotImplementedError
