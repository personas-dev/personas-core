# Model Card：SPC-HGT MatchNet

## Model Details

- Model name:
- Model version:
- Schema version:
- Graph snapshot id:
- Training date:
- Owner:

## Intended Use

输入候选人画像，输出 Top-K 岗位推荐和结构化解释。

## Training Data

- Candidate data:
- Job data:
- Interaction data:
- Time split:

## Evaluation

| Metric | Value |
|---|---:|
| Recall@10 |  |
| NDCG@10 |  |
| MRR |  |
| HitRate@10 |  |
| AUC |  |

## Baseline Comparison

| Baseline | NDCG@10 | Delta |
|---|---:|---:|
| Rule |  |  |
| BM25 |  |  |
| SBERT |  |  |
| LightGCN |  |  |
| KGAT/KGCN |  |  |

## Limitations

- 数据稀疏：
- 冷启动：
- 技能标准化误差：
- 行为偏差：

## Explainability

解释字段包括：matched_skills、missing_skills、graph_paths、reasons。
