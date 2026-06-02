"""Train/valid/test splitting utilities."""
from __future__ import annotations


def time_split(records: list[dict], timestamp_key: str = "timestamp", valid_ratio: float = 0.1, test_ratio: float = 0.2) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records by timestamp to avoid future leakage."""
    ordered = sorted(records, key=lambda x: x[timestamp_key])
    n = len(ordered)
    test_start = int(n * (1 - test_ratio))
    valid_start = int(test_start * (1 - valid_ratio))
    return ordered[:valid_start], ordered[valid_start:test_start], ordered[test_start:]
