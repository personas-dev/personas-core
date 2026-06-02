"""IO utilities."""
from __future__ import annotations

import json
from pathlib import Path


def write_json(path: str | Path, data: object) -> None:
    """Write JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
