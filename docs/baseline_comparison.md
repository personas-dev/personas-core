# Baseline Comparison: HR KG-GNN Sample Run

Date: 2026-06-02
Run directory: `experiments/runs/baseline_sample_gpu`
Config: `configs/train_gnn.yaml`

## Dataset Slice

This run uses both local datasets under `data/`:

- `data/datasets.zip`: source candidate, job, and interaction tables.
- `data/kg_output.zip`: prebuilt KG statistics used to record the graph snapshot scale.

Sample and split:

| Item | Value |
|---|---:|
| Candidates loaded | 784 |
| Jobs loaded | 3,000 |
| Filtered interactions | 4,396 |
| Positive interactions | 950 |
| Train positives | 583 |
| Valid positives | 122 |
| Test positives | 219 |
| Test users | 219 |
| Full KG users | 4,500 |
| Full KG jobs | 269,534 |
| Full KG skills | 2,596,614 |
| Full KG edges | 8,679,483 |

The source action table has no explicit timestamp in the sampled fields, so the split is deterministic row-order per user: earlier positives are train, the last positives are validation/test. This is weaker than a true time split and should be replaced when event timestamps are available.

## Models Compared

| Model | Status | Notes |
|---|---|---|
| Rule | completed | Skill-path and structured feature overlap baseline. |
| BM25 | completed | Dependency-free sparse lexical retrieval over job text. |
| semantic_hash | completed | Dependency-free hashed TF-IDF cosine fallback for SBERT. This is not a real SBERT model. |
| LightGCN | implemented, not trained | Training code exists, but PyTorch is not installed in the current Python environment. |
| SPC-HGT-lite | implemented, not trained | Skill-path feature-enhanced GNN prototype exists, but PyTorch is not installed. |

## Metrics

| Model | Recall@10 | Precision@10 | NDCG@10 | MRR | HitRate@10 |
|---|---:|---:|---:|---:|---:|
| Rule | 0.2192 | 0.0219 | 0.1211 | 0.1055 | 0.2192 |
| BM25 | 0.0639 | 0.0064 | 0.0393 | 0.0402 | 0.0639 |
| semantic_hash | 0.0137 | 0.0014 | 0.0048 | 0.0046 | 0.0137 |
| LightGCN | skipped | skipped | skipped | skipped | skipped |
| SPC-HGT-lite | skipped | skipped | skipped | skipped | skipped |

Best completed baseline: **Rule**, with `NDCG@10 = 0.1211`.

## GPU / GNN Training Status

Available GPUs were checked with `nvidia-smi`; several GPUs were effectively idle. GPU training could not start because the current Python 3.13 environment did not have PyTorch installed.

A PyTorch install was attempted with:

```bash
python -m pip install 'torch>=2.7'
```

The install began downloading the CUDA 13 `torch-2.12.0` wheel but stalled for a long period with no progress, so the process was terminated to avoid blocking the workspace. The training CLI now records this condition explicitly:

```json
{"model": "torch_gnn", "status": "skipped", "reason": "PyTorch is not installed; install torch to train GPU GNN models."}
```

## Conclusion

The current runnable baseline is complete, but the main GNN model has **not** been trained yet. Therefore, the project must not claim SPC-HGT effectiveness or improvement over baseline at this point.

On the sampled split, Rule is the strongest completed method. This suggests the current data slice has strong structured matching signals and sparse text matching is weak without Chinese segmentation or pretrained sentence embeddings.

## Next Steps

1. Install a working CUDA PyTorch build for Python 3.13, or create a Python 3.12/3.11 environment with a known stable PyTorch CUDA wheel.
2. Run `PYTHONPATH=src python -m jobmatch_gnn.training.train --config configs/train_gnn.yaml` again; LightGCN and SPC-HGT-lite will train automatically once `torch` is importable.
3. Replace `semantic_hash` with real SBERT/SentenceTransformer embeddings when dependencies and model weights are available.
4. Improve Chinese tokenization and skill normalization before interpreting BM25 or semantic baseline quality.
5. Move from row-order split to true time split if timestamped behavior logs become available.
