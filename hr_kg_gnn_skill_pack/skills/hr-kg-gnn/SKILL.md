---
name: hr-kg-gnn
description: Build, train, evaluate, and document HR person-job matching systems that combine Candidate-Skill-Job knowledge graphs, recommendation baselines, and GNN models. Use when Codex works on recruitment/job matching data, KG schema construction, Rule/BM25/SBERT/LightGCN/HGT/SPC-HGT baselines, hard-negative ranking, explanation outputs, ablations, or baseline comparison reports for personas-core style HR datasets.
---

# HR KG-GNN

## Operating Mode

Act as the core algorithm agent for an HR person-job matching module:

```text
input: CandidateProfile
output: Top-K job recommendations plus structured explanations
```

Prefer a working, measurable baseline before adding complexity. Do not claim the main model is effective unless it beats the strongest comparable baseline on the same split.

## Load References

Use bundled references only when needed:

- Read `references/reading_priority.md` before choosing papers or implementation priorities.
- Read `references/references.md` or `references/references.csv` when citing model lineage, baselines, or adoption decisions.
- Read `references/uploaded_source_notes.md` for project constraints from uploaded materials.
- Read `docs/00_path_function_map.md` when adding or reorganizing code.
- Read `docs/02_kg_schema.md`, `docs/03_model_design.md`, `docs/04_training_evaluation.md`, and `docs/05_baseline_and_ablation.md` before changing graph, model, training, or evaluation behavior.

## Minimal Roadmap

Implement in this order:

```text
P0: Rule / BM25 / SBERT baseline
P1: Candidate-Skill-Job knowledge graph
P2: Node2Vec or LightGCN graph baseline
P3: SPC-HGT MatchNet main model
P4: hard negative mining and contrastive learning
P5: explanation module
P6: ablation and baseline comparison report
P7: inference optimization and deployment API
```

Keep the first runnable loop small enough to train and evaluate locally; make scale controls configurable instead of hard-coding samples.

## Graph Schema

Use these canonical node types when data supports them:

```text
Candidate, Job, Skill, Company, City, Industry, Education, Experience
```

For personas-core tables, map `User` to `Candidate` and `Degree` to `Education` in public APIs and reports if the raw graph exporter uses the original table names.

Core relation types:

```text
Candidate - HAS_SKILL - Skill
Job - REQUIRES_SKILL - Skill
Job - PREFERS_SKILL - Skill
Candidate - APPLIED_TO / CLICKED / MATCHES - Job
Job - LOCATED_IN - City
Job - BELONGS_TO - Industry
Company - PUBLISHES - Job
Skill - SIMILAR_TO / PREREQUISITE_OF - Skill
```

Add reverse edges for message passing whenever using a GNN.

## Model Requirements

Main model name:

```text
SPC-HGT MatchNet
Skill-Path Contrastive Heterogeneous Graph Transformer
```

Include these ideas where the data supports them:

- Heterogeneous Candidate-Job-Skill graph encoding.
- Explicit skill path features for `Candidate -> Skill <- Job`.
- Hard negatives from high lexical or semantic similarity without positive behavior.
- Pairwise ranking with BPR and binary BCE signals.
- Contrastive alignment between candidate embeddings and positive job embeddings.
- Structured explanations containing `matched_skills`, `missing_skills`, `graph_paths`, and `reasons`.

Default to 2 graph layers unless experiments justify deeper message passing.

## Baselines

Implement and compare on the same train/valid/test split:

```text
Rule-based
BM25
SBERT or TF-IDF/SVD semantic fallback when SBERT is unavailable
Node2Vec or LightGCN
KGCN or KGAT when KG neighbor sampling is available
HGT/HeteroConv without skill path features
SPC-HGT MatchNet
```

If a dependency is unavailable, implement a deterministic fallback and document the difference.

## Metrics

For behavior labels, report:

```text
Recall@K, Precision@K, NDCG@K, MRR, HitRate@K, AUC, F1
```

Use time splits when timestamps exist. If timestamps are absent, use deterministic row-order or grouped splits and state the limitation.

## Code Rules

- Use Python >= 3.10.
- Add type annotations and docstrings to public functions.
- Validate external input with dataclasses or pydantic.
- Put paths, data column names, sample sizes, seeds, and model hyperparameters in YAML/config objects.
- Write metrics to JSON/CSV.
- Keep checkpoints, extracted datasets, run artifacts, and large generated files out of git.
- Run the available subset of `ruff`, formatter, type checks, and tests before reporting completion.

## Documentation Rules

When algorithm behavior changes, update the relevant docs:

```text
docs/01_algorithm_api.md
docs/02_kg_schema.md
docs/03_model_design.md
docs/04_training_evaluation.md
docs/05_baseline_and_ablation.md
docs/06_innovation_and_advantage.md
docs/08_versioning_and_workflow.md
CHANGELOG.md
MODEL_CARD.md
references/references.md
```

For experiment summaries, create a focused comparison document under `docs/` and include:

- Dataset slice and split strategy.
- Models compared.
- Metrics table.
- Whether the main model beats the strongest baseline.
- Known limitations and next steps.
