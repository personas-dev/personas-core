"""Build heterogeneous KG for person-job matching."""
from __future__ import annotations

from .schema import CandidateProfile, JobProfile


def build_hetero_graph(candidates: list[CandidateProfile], jobs: list[JobProfile]) -> object:
    """Build PyG HeteroData or DGL heterograph.

    TODO: Implement with torch_geometric.data.HeteroData or dgl.heterograph.
    """
    raise NotImplementedError
