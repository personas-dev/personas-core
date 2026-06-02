"""Preprocess candidate/job raw records."""
from __future__ import annotations


def normalize_text(text: str | None) -> str:
    """Normalize raw text for retrieval and encoding."""
    if text is None:
        return ""
    return " ".join(text.strip().split())
