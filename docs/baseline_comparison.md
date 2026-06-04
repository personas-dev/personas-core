# Baseline Comparison: HR KG-GNN GPU Run

Date: 2026-06-04
Run directory: `experiments/runs/baseline_sample_gpu`
Config: `configs/train_gnn.yaml`
Environment: `.venv` created by `uv`, Python 3.12.13, `torch==2.12.0+cu130`
Device: CUDA GPU 6, NVIDIA RTX PRO 6000 Blackwell Server Edition

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
| LightGCN | completed on CUDA | Candidate-job interaction graph only. |
| SPC-HGT-lite | completed on CUDA | LightGCN backbone plus skill-path/rule feature MLP. |

## Metrics

| Model | Recall@10 | Precision@10 | NDCG@10 | MRR | HitRate@10 | Train loss | Device |
|---|---:|---:|---:|---:|---:|---:|---|
| Rule | 0.2192 | 0.0219 | 0.1211 | 0.1055 | 0.2192 |  | CPU |
| BM25 | 0.0639 | 0.0064 | 0.0393 | 0.0402 | 0.0639 |  | CPU |
| semantic_hash | 0.0137 | 0.0014 | 0.0048 | 0.0046 | 0.0137 |  | CPU |
| LightGCN | 0.0046 | 0.0005 | 0.0046 | 0.0060 | 0.0046 | 0.4217 | CUDA |
| SPC-HGT-lite | 0.2603 | 0.0260 | 0.1310 | 0.1065 | 0.2603 | 0.2507 | CUDA |

Best completed baseline before GNN: **Rule**, with `NDCG@10 = 0.1211`.

Best completed model after GPU training: **SPC-HGT-lite**, with `NDCG@10 = 0.1310`.

Absolute gain over Rule: `+0.0099 NDCG@10`. Relative gain: about `+8.2%`.

## Interpretation

SPC-HGT-lite beats the strongest completed baseline on this sampled split. The gain is modest but meaningful because all models use the same candidate/job sample and held-out positives.

LightGCN alone performs poorly. This indicates that the sampled interaction graph is too sparse for pure collaborative filtering. The useful signal comes from combining graph embeddings with explicit skill-path and structured matching features.

The current result supports the direction of skill-path enhanced graph ranking, but it is still a prototype rather than the full SPC-HGT design. A stronger claim requires:

- multiple seeds,
- true timestamp splits,
- larger job universes,
- hard-negative mining,
- real SBERT baseline,
- full heterogeneous message passing over Candidate, Job, Skill, City, Industry, Education, and Experience nodes.

## Training Command

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend cu130 -e . pytest ruff 'torch>=2.12'
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m jobmatch_gnn.training.train --config configs/train_gnn.yaml
```

## Conclusion

The GPU training path is now functional. On the current sampled dataset, SPC-HGT-lite is the best completed model and exceeds the strongest baseline. The next engineering step is to make the comparison more robust with multi-seed runs and stronger semantic/hard-negative baselines.
