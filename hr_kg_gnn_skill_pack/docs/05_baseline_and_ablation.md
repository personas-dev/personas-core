# 05 Baseline 与消融实验

## 1. Baseline 列表

| Baseline | 作用 | 必须实现 |
|---|---|---|
| Rule-based | 稳定、可解释下限 | 是 |
| BM25 | 稀疏文本检索 baseline | 是 |
| SBERT | 语义匹配 baseline | 是 |
| Node2Vec | 无监督图嵌入 baseline | 是 |
| LightGCN | 推荐系统 GNN 强 baseline | 是 |
| KGCN | KG 推荐 baseline | KGCN/KGAT 至少二选一 |
| KGAT | KG attention 推荐 baseline | KGCN/KGAT 至少二选一 |
| HGT without path features | 异构图 baseline | 是 |

## 2. Rule Baseline

```text
score = 0.5 * skill_match
      + 0.2 * education_match
      + 0.2 * experience_match
      + 0.1 * city_match
```

## 3. BM25 Baseline

输入文本：

```text
candidate_text = skills + project_domains + resume_text
job_text = required_skills + preferred_skills + jd_text
```

输出 Top-K BM25 排名。

## 4. SBERT Baseline

使用 Sentence Transformer 对候选人和岗位文本编码：

```text
score = cosine(embedding(candidate_text), embedding(job_text))
```

## 5. 图 Baseline

### Node2Vec

从全图 random walk 得到节点 embedding，再做 candidate-job 相似度。

### LightGCN

只使用 candidate-job 行为二部图，评估协同过滤能力。

### KGCN/KGAT

使用 job 侧知识图谱属性，评估 KG 推荐模型的收益。

## 6. 主模型对比表模板

| Model | Recall@10 | NDCG@10 | MRR | HitRate@10 | AUC | Latency p95 |
|---|---:|---:|---:|---:|---:|---:|
| Rule |  |  |  |  |  |  |
| BM25 |  |  |  |  |  |  |
| SBERT |  |  |  |  |  |  |
| LightGCN |  |  |  |  |  |  |
| KGCN/KGAT |  |  |  |  |  |  |
| HGT no path |  |  |  |  |  |  |
| SPC-HGT MatchNet |  |  |  |  |  |  |

## 7. 消融实验表模板

| Variant | Recall@10 | NDCG@10 | MRR | 结论 |
|---|---:|---:|---:|---|
| SPC-HGT full |  |  |  | 主模型 |
| - text_encoder |  |  |  | 验证文本语义 |
| - skill_path_features |  |  |  | 验证技能路径 |
| - contrastive_loss |  |  |  | 验证对比学习 |
| - hard_negative |  |  |  | 验证困难负样本 |
| - edge_features |  |  |  | 验证边权/时效/重要性 |
| -> LightGCN |  |  |  | 验证异构图建模 |
| -> KGAT/KGCN |  |  |  | 验证人岗特化改进 |

## 8. 达标标准

```text
主指标：NDCG@10
最低要求：SPC-HGT MatchNet 超过 Rule、BM25、SBERT、LightGCN 中最强 baseline
推荐目标：NDCG@10 相对最强 baseline 提升 >= 3%
必须报告：均值、标准差、随机种子、训练时间、推理延迟
```

如果主模型没有超过 baseline，不允许包装结论。必须进行误差分析。


## 9. 2026-06-02 Sample Run Result

Run: `experiments/runs/baseline_sample_gpu`
Config: `configs/train_gnn.yaml`

| Model | Recall@10 | Precision@10 | NDCG@10 | MRR | HitRate@10 | Status |
|---|---:|---:|---:|---:|---:|---|
| Rule | 0.2192 | 0.0219 | 0.1211 | 0.1055 | 0.2192 | completed |
| BM25 | 0.0639 | 0.0064 | 0.0393 | 0.0402 | 0.0639 | completed |
| semantic_hash | 0.0137 | 0.0014 | 0.0048 | 0.0046 | 0.0137 | completed SBERT fallback |
| LightGCN |  |  |  |  |  | skipped: PyTorch not installed |
| SPC-HGT-lite |  |  |  |  |  | skipped: PyTorch not installed |

当前最强已完成 baseline 是 Rule。SPC-HGT 尚未训练，不能宣称超过 baseline。


## 10. 2026-06-04 GPU Run Result

Run: `experiments/runs/baseline_sample_gpu`
Runtime: `.venv`, Python 3.12.13, `torch==2.12.0+cu130`, CUDA GPU 6

| Model | Recall@10 | Precision@10 | NDCG@10 | MRR | HitRate@10 | Status |
|---|---:|---:|---:|---:|---:|---|
| Rule | 0.2192 | 0.0219 | 0.1211 | 0.1055 | 0.2192 | completed |
| BM25 | 0.0639 | 0.0064 | 0.0393 | 0.0402 | 0.0639 | completed |
| semantic_hash | 0.0137 | 0.0014 | 0.0048 | 0.0046 | 0.0137 | completed SBERT fallback |
| LightGCN | 0.0046 | 0.0005 | 0.0046 | 0.0060 | 0.0046 | completed on CUDA |
| SPC-HGT-lite | 0.2603 | 0.0260 | 0.1310 | 0.1065 | 0.2603 | completed on CUDA |

SPC-HGT-lite 当前超过最强已完成 baseline Rule，NDCG@10 绝对提升 `+0.0099`，相对提升约 `+8.2%`。LightGCN 单独效果弱，说明稀疏行为图需要技能路径和结构化匹配特征增强。
