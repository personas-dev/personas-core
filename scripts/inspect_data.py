"""Print schema and sample rows of the three raw tables (tutorial step 2)."""
from __future__ import annotations

import zipfile

ZIP = "data/datasets.zip"
TABLES = ["datasets/table1_user.txt", "datasets/table2_jd.txt", "datasets/table3_action.txt"]

with zipfile.ZipFile(ZIP) as archive:
    for name in TABLES:
        with archive.open(name) as raw:
            header = raw.readline().decode("utf-8-sig").rstrip("\n").split("\t")
            sample = raw.readline().decode("utf-8", "ignore").rstrip("\n").split("\t")
        print(f"\n===== {name} =====")
        for col, val in zip(header, sample, strict=False):
            print(f"  {col:28s} {val[:80]}")
