# Model Card: SPC-HGT MatchNet

## Model Details

- Model name: SPC-HGT MatchNet / SPC-HGT-lite prototype
- Model version: 0.3.0
- Schema version: Candidate/User - Skill - Job KG schema
- Graph snapshot id: `data/kg_output.zip` (`8,679,483` edges in graph stats)
- Training date: 2026-06-04
- Owner: personas-core algorithm workspace
- Runtime: `.venv`, Python 3.12.13, `torch==2.12.0+cu130`

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
| LightGCN | 0.0046 | 0.0046 | 0.0060 | 0.0046 | completed on CUDA |
| SPC-HGT-lite | 0.2603 | 0.1310 | 0.1065 | 0.2603 | completed on CUDA |

## Baseline Comparison

The strongest pre-GNN baseline is Rule (`NDCG@10 = 0.1211`). SPC-HGT-lite improves to `NDCG@10 = 0.1310`, an absolute gain of `+0.0099` and a relative gain of about `+8.2%` on this sampled split.

## Limitations

- The current model is SPC-HGT-lite, not the full heterogeneous transformer.
- The semantic baseline is a hashed TF-IDF fallback, not pretrained SBERT.
- Chinese tokenization and skill normalization are rule-based and should be improved.
- The split is row-order based rather than timestamp based.
- The current run uses a sampled job universe of 3,000 jobs, not the full 269,534-job graph.
- Only one seed has been run so far.

## Explainability

The implemented feature layer computes matched skill coverage, missing skill ratio, city match, education match, experience match, job-type match, and skill-path count. The next inference layer should expose these as structured `matched_skills`, `missing_skills`, `graph_paths`, and `reasons`.
