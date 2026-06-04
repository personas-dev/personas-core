# HR KG-GNN Algorithm and Training Strategy

Date: 2026-06-04

## Goal

The core task is person-job matching:

```text
Input: candidate profile
Output: Top-K job recommendations with structured explanations
```

The system treats matching as a recommendation and link-prediction problem over a recruitment knowledge graph. The most important path is:

```text
Candidate -> HAS_SKILL -> Skill <- REQUIRES_SKILL <- Job
```

This path lets the model combine behavior signals with interpretable skill evidence.

## Data Strategy

The current implementation reads the two local dataset archives directly:

- `data/datasets.zip`: raw candidate, job, and interaction tables.
- `data/kg_output.zip`: prebuilt knowledge graph statistics and graph snapshot metadata.

The training loader samples from the large raw tables by configuration instead of hard-coding scale. The default config in `configs/train_gnn.yaml` uses:

| Field | Default |
|---|---:|
| max_jobs | 3,000 |
| max_interactions | 120,000 |
| eval@k | 10 |
| GNN layers | 2 |
| GNN embedding dim | 64 |
| GNN epochs | 20 |

The action table does not expose a timestamp in the sampled fields, so the split is deterministic row-order per user. For each user, earlier positive interactions become train examples and the latest held-out positives become validation/test examples. This is weaker than a true time split, but it preserves a no-duplicate, future-like holdout within each user sequence.

## Feature Strategy

The feature layer intentionally starts with dependency-light features so that the baseline is reproducible before large model dependencies are available.

Candidate features:

- Skill tokens extracted from experience text.
- Desired city, industry, and job type.
- Current degree and rough years of experience.
- Mixed Chinese/English text tokens for lexical matching.

Job features:

- Title, job type, and job description tokens.
- Required skill-like tokens from title and JD text.
- City, minimum degree, and minimum years.

Pair features:

- Required skill coverage.
- Missing skill ratio.
- City match.
- Education match.
- Experience match.
- Desired job-type match.
- Skill-path count via shared candidate/job skills.

These features support both the rule baseline and the SPC-HGT-lite matching head.

## Baseline Strategy

The project follows a staged baseline policy: simple baselines must run first, and the GNN model is only considered useful if it beats the strongest completed baseline on the same split.

Implemented baselines:

| Model | Role |
|---|---|
| Rule | Interpretable lower bound using skill-path and structured matching features. |
| BM25 | Sparse lexical retrieval baseline over job text. |
| semantic_hash | Dependency-free hashed TF-IDF cosine fallback for SBERT. |
| LightGCN | Collaborative-filtering graph baseline over candidate-job positive interactions. |
| SPC-HGT-lite | Skill-path enhanced GNN prototype that combines LightGCN propagation and pair features. |

`semantic_hash` is not a replacement for real SBERT. It is only a deterministic fallback until SentenceTransformer dependencies and model weights are available.

## GNN Strategy

The current GNN implementation has two trainable variants.

### LightGCN

LightGCN uses the candidate-job positive interaction graph. It learns candidate and job embeddings and performs 2-hop normalized bipartite message passing by default. The loss is BPR:

```text
L_bpr = -log sigmoid(score(candidate, positive_job) - score(candidate, negative_job))
```

Negatives are sampled from jobs not known as positives for the same candidate.

### SPC-HGT-lite

SPC-HGT-lite is the current runnable prototype for the full SPC-HGT direction. It keeps the LightGCN interaction backbone and adds an MLP over explicit skill-path/rule features:

```text
score(c, j) = dot(z_c, z_j) + MLP(path_features(c, j))
```

It trains with:

```text
L = L_bpr + 0.2 * L_bce
```

The BCE term teaches the feature-enhanced score to distinguish positive and sampled negative pairs. This is a smaller implementation of the intended SPC-HGT MatchNet strategy and is designed to be extended to full heterogeneous node/edge types later.

## Evaluation Strategy

All models are evaluated on the same held-out test positives and same sampled job universe. The main metric is `NDCG@10`, with supporting metrics:

- Recall@10
- Precision@10
- MRR
- HitRate@10

The current reporting rule is strict: if the GNN does not beat the strongest completed baseline, the document must say so. No effectiveness claim is allowed without a fair comparison.

## Explanation Strategy

The system should expose structured recommendation evidence rather than only natural language. The current feature layer already computes the evidence needed for:

```json
{
  "matched_skills": ["shared candidate/job skills"],
  "missing_skills": ["job skills absent from candidate"],
  "graph_paths": ["Candidate -> Skill <- Job"],
  "reasons": ["skill coverage", "city match", "education match", "experience match"]
}
```

The next inference iteration should turn the pair-feature calculations into a formal explanation API.

## Training Strategy

1. Build `.venv` with `uv` and Python 3.12.
2. Install project dependencies plus CUDA PyTorch using `uv pip install --torch-backend cu130`.
3. Run compile and tests before training.
4. Train all configured models with:

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m jobmatch_gnn.training.train --config configs/train_gnn.yaml
```

5. Write metrics under `experiments/runs/<run_name>/` and keep those artifacts out of git.
6. Update `docs/baseline_comparison.md` after every meaningful run.

## Current Risks

- The full KG has millions of skill nodes and edges; full-graph training needs sampling or neighbor loaders before scaling beyond the current prototype.
- Row-order split is not a substitute for true timestamp-based evaluation.
- Rule features are currently strong on the sample, so GNN gains require better hard negatives, skill normalization, and richer graph structure.
- BM25 and semantic fallback quality are limited by rule-based Chinese tokenization.
- Real SBERT, PyG HGT/HeteroConv, and hard-negative mining remain planned improvements.


## 2026-06-04 GPU Training Result

After creating `.venv` with `uv` and installing `torch==2.12.0+cu130`, CUDA training ran successfully on GPU 6.

| Model | Recall@10 | Precision@10 | NDCG@10 | MRR | HitRate@10 | Device |
|---|---:|---:|---:|---:|---:|---|
| Rule | 0.2192 | 0.0219 | 0.1211 | 0.1055 | 0.2192 | CPU |
| BM25 | 0.0639 | 0.0064 | 0.0393 | 0.0402 | 0.0639 | CPU |
| semantic_hash | 0.0137 | 0.0014 | 0.0048 | 0.0046 | 0.0137 | CPU |
| LightGCN | 0.0046 | 0.0005 | 0.0046 | 0.0060 | 0.0046 | CUDA |
| SPC-HGT-lite | 0.2603 | 0.0260 | 0.1310 | 0.1065 | 0.2603 | CUDA |

SPC-HGT-lite improves over the strongest completed baseline, Rule, by `+0.0099 NDCG@10` on the sampled split. The result supports the skill-path enhanced ranking strategy, while LightGCN alone shows that sparse interactions are not enough without structured skill/path features.
