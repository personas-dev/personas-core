"""Data loading utilities for HR person-job matching."""

from jobmatch_gnn.data.dataset import CandidateRecord, DatasetBundle, InteractionRecord, JobRecord, load_dataset_bundle

__all__ = [
    "CandidateRecord",
    "DatasetBundle",
    "InteractionRecord",
    "JobRecord",
    "load_dataset_bundle",
]
