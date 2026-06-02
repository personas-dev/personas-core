# Model Card: SPC-HGT MatchNet

## Model Details

- Model name: SPC-HGT MatchNet / SPC-HGT-lite prototype
- Model version: 0.2.0
- Schema version: Candidate/User - Skill - Job KG schema
- Graph snapshot id: `data/kg_output.zip` (`8,679,483` edges in graph stats)
- Training date: 2026-06-02
- Owner: personas-core algorithm workspace

## Intended Use

Input a candidate profile and output Top-K job recommendations with structured evidence fields such as matched skills, missing skills, graph paths, and reasons.

## Training Data

- Candidate data: `data/datasets.zip::datasets/table1_user.txt`
- Job data: `data/datasets.zip::datasets/table2_jd.txt`
- Interaction data: `data/datasets.zip::datasets/table3_action.txt`
- KG stats: `data/kg_output.zip::kg_output/graph_stats.csv`
- Split: deterministic per-user row-order split because timestamp fields are not available in the sampled interaction table.

## Evaluation

Current completed run: `experiments/runs/baseline_sample_gpu`.

| Model | Recall@10 | NDCG@10 | MRR | HitRate@10 | Status |
|---|---:|---:|---:|---:|---|
| Rule | 0.2192 | 0.1211 | 0.1055 | 0.2192 | completed |
| BM25 | 0.0639 | 0.0393 | 0.0402 | 0.0639 | completed |
| semantic_hash | 0.0137 | 0.0048 | 0.0046 | 0.0137 | completed SBERT fallback |
| LightGCN |  |  |  |  | skipped: PyTorch not installed |
| SPC-HGT-lite |  |  |  |  | skipped: PyTorch not installed |

## Baseline Comparison

The strongest completed baseline is Rule (`NDCG@10 = 0.1211`). SPC-HGT has not been trained, so there is no valid claim that it improves over baseline yet.

## Limitations

- PyTorch is not installed in the current Python 3.13 environment, so GPU GNN training did not start.
- The semantic baseline is a hashed TF-IDF fallback, not pretrained SBERT.
- Chinese tokenization and skill normalization are rule-based and should be improved.
- The split is row-order based rather than timestamp based.
- The current run uses a sampled job universe of 3,000 jobs, not the full 269,534-job graph.

## Explainability

The implemented feature layer computes matched skill coverage, missing skill ratio, city match, education match, experience match, job-type match, and skill-path count. The next inference layer should expose these as structured `matched_skills`, `missing_skills`, `graph_paths`, and `reasons`.
