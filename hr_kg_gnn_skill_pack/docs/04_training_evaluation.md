# 04 训练与评估流程

## 1. 训练流水线

```text
Step 1: 数据校验
Step 2: 技能标准化
Step 3: 构建候选人画像、岗位画像
Step 4: 构建异构知识图谱
Step 5: 生成 train/valid/test 切分
Step 6: 训练 Rule/BM25/SBERT baseline
Step 7: 训练 Node2Vec/LightGCN/KGCN/KGAT baseline
Step 8: 训练 SPC-HGT MatchNet
Step 9: 评估主指标和消融实验
Step 10: 生成 baseline_comparison.md
Step 11: 构建 ANN 向量索引
Step 12: 推理并输出解释
```

## 2. 数据切分

优先采用时间切分：

```text
train: 历史行为前 70%
valid: 之后 10%-15%
test: 最新 15%-20%
```

必须避免：

```text
同一个候选人的未来点击泄漏到训练集
同一个岗位的未来录用标签泄漏到训练集
同一条投递记录重复出现在 train 和 test
```

## 3. 标签设计

正样本：

```text
录用 > 面试 > 投递 > 收藏 > 长点击
```

负样本：

```text
曝光未点击
点击后快速退出
投递被拒
同批推荐未反馈
采样负样本
```

不同信号建议加权：

```yaml
label_weights:
  hire: 1.0
  interview: 0.8
  apply: 0.6
  save: 0.4
  click: 0.2
  reject: 0.0
```

## 4. 评价指标

推荐系统指标：

```text
Recall@K
Precision@K
NDCG@K
MRR
HitRate@K
AUC
F1
```

解释指标：

```text
matched_skill_precision
missing_skill_precision
path_explanation_coverage
reason_consistency_score
```

性能指标：

```text
training_time_per_epoch
inference_latency_p50
inference_latency_p95
memory_usage
index_build_time
```

## 5. 报告格式

每次实验必须写入：

```text
experiments/runs/{run_id}/config.yaml
experiments/runs/{run_id}/metrics.json
experiments/runs/{run_id}/predictions.parquet
experiments/runs/{run_id}/error_cases.md
experiments/runs/{run_id}/ablation.csv
```

## 6. 随机种子

至少运行 3 个随机种子：

```yaml
seeds: [42, 2025, 3407]
```

报告必须写均值和标准差，不允许只报最好一次。
