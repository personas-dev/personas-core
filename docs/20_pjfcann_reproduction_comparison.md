# 20 PJFCANN 复现适配与全库排序比较

日期:2026-06-20

关联: `docs/15_results_v2.md`, `docs/18_osta_kg_pipeline_results.md`, `docs/19_strategy_innovation_sota_assessment.md`

## 1. 复现来源

PJFCANN 对应论文为 *Person-job fit estimation from candidate profile and related recruitment history with co-attention neural networks*。论文页面给出的官方代码仓库为:

- GitHub: https://github.com/CCIIPLab/PJFCANN
- 本次临时拉取路径: `/tmp/PJFCANN`
- 拉取 commit: `5c0f72b Add files via upload`
- License: Apache-2.0

官方 README 依赖为 Python 3, PyTorch >= 1.2.0, jieba, tqdm;训练入口为:

```bash
python run_InsightModel.py
```

## 2. 为什么不能直接用官方脚本产出可比指标

官方实现是 pair classification 管线:

```text
s1/s2 文本对
  + s1_graph/s2_graph 相似历史序列
  -> InsightModel
  -> classification accuracy
```

而本项目的主评估协议是:

```text
给定 candidate,在 91,479 个岗位全库中排序
  -> Recall@10 / NDCG@10 / Recall@50 / MRR
```

直接运行官方脚本会遇到三个不可比点:

1. 官方数据格式依赖预处理好的 `s1`, `s2`, `s1_graph`, `s2_graph`, `labels` 文件,与 personas-core 的 `users/jobs/interactions` 不同。
2. 官方评估是分类 accuracy,不是全库 top-K 推荐。
3. 官方文本编码依赖 Tencent AILab ChineseEmbedding + InferSent/BLSTM,本项目当前统一使用 BGE 文本向量和 `data/processed` split。

因此本次没有把官方仓库代码直接拷入项目,而是基于其结构实现了一个同口径适配版 baseline。

## 3. 本项目适配版实现

新增文件:

- `src/jobmatch_gnn/models/pjfcann_v2.py`
- `configs/v2_pjfcann.yaml`

接入入口:

- `src/jobmatch_gnn/training/train_v2.py -> model name: pjfcann`

适配版保留 PJFCANN 的核心结构:

```text
Local semantic representation:
  BGE(candidate text), BGE(job text) -> projection

Global experience representation:
  train-only Candidate-Job graph -> LightGCN-style propagation

Pair classifier:
  H_candidate = [local_candidate, global_candidate]
  H_job       = [local_job, global_job]
  score       = retrieval_dot + MLP([Hc, Hj, |Hc-Hj|, Hc*Hj])
```

其中 `retrieval_dot` 是为了适配全库排序新增的 residual score。没有这个分数时,模型只会优化 pair classifier,但第一阶段从 91,479 个岗位召回 top-1000 的 dot score 没有被直接训练,会导致全库 NDCG 明显偏低。

训练目标:

```text
loss = BCE(pos/neg pair classification) + 1.0 * BPR(pos > neg)
```

评估方式与现有 v2 管线一致:

1. 用 dot score 从全库召回 top-1000。
2. 用 PJFCANN pair classifier rerank top-1000。
3. 对 4,095 个 test users 计算全库排序指标。
4. 排除 train positives 和 valid positives,不排除 test positives。

## 4. 数据与配置

本次使用当前 `data/processed` 产物,即 OSTA job20 KG:

| 项 | 值 |
|---|---:|
| users | 4,500 |
| jobs | 91,479 |
| positives | 61,297 |
| train / valid / test positives | 53,408 / 3,794 / 4,095 |
| test users | 4,095 |
| skill_vocab | 4,634 |
| avg_user_skills | 32.09 |
| avg_job_skills | 18.94 |

最终运行命令:

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m jobmatch_gnn.training.train_v2 --config configs/v2_pjfcann.yaml
```

最终配置摘要:

```yaml
dim: 128
layers: 2
epochs: 50
patience: 8
batch: 4096
neg_per_pos: 4
lambda_bpr: 1.0
topn: 1000
recall_k: 1000
seed: 42
```

运行目录:

- `experiments/runs/v2_pjfcann`

## 5. 结果

PJFCANN rank-adapted 单 seed 结果:

| 模型 | NDCG@10 | Recall@10 | Recall@50 | NDCG@50 | MRR | runtime |
|---|---:|---:|---:|---:|---:|---:|
| PJFCANN adapted | 0.0063 | 0.0110 | 0.0261 | 0.0097 | 0.0059 | 179.6s |

训练信息:

| 项 | 值 |
|---|---:|
| best valid NDCG@10 | 0.0083 |
| train pairs | 53,404 |
| final train loss | 0.0002 |

与 OSTA job20 同切分结果比较:

| 模型 | NDCG@10 | Recall@10 | Recall@50 | MRR |
|---|---:|---:|---:|---:|
| Popularity | 0.0016 | 0.0039 | 0.0164 | 0.0019 |
| SBERT | 0.0028 | 0.0049 | 0.0149 | 0.0031 |
| BM25 | 0.0049 | 0.0090 | 0.0278 | 0.0050 |
| **PJFCANN adapted** | **0.0063** | **0.0110** | **0.0261** | **0.0059** |
| Rule | 0.0116 | 0.0256 | 0.1001 | 0.0126 |
| LightGCN | 0.0224 | 0.0425 | 0.1021 | 0.0199 |
| SPC-HGT | 0.0267 | 0.0508 | 0.1206 | 0.0248 |

## 6. 判断

PJFCANN adapted 强于纯文本零样本 SBERT 和 BM25 的 NDCG@10,说明“文本语义 + 招聘历史图 + pair interaction classifier”确实有增益。但它没有超过 Rule,更没有接近 LightGCN 和 SPC-HGT:

| 对比 | NDCG@10 差距 |
|---|---:|
| PJFCANN vs BM25 | +29.9% |
| PJFCANN vs Rule | -45.4% |
| PJFCANN vs LightGCN | -71.9% |
| PJFCANN vs SPC-HGT | -76.4% |

原因分析:

1. **原始 PJFCANN 更偏 pair-fit classification**,不是大规模候选池召回。即使加入 residual dot 和 BPR,它的第一阶段检索仍弱于专门为协同召回训练的 LightGCN。
2. **本项目行为信号比文本信号更关键**。LightGCN 只用候选人-岗位行为图即可达到 0.0224,而 PJFCANN 的局部文本分支没有带来足够召回增益。
3. **PJFCANN 没有显式 skill-path 通道**。它不能像 SPC-HGT 一样利用 `Candidate -> Skill <- Job` 路径注意力,也不能输出同等粒度的技能解释。
4. **官方 graph history 输入没有完全等价复现**。官方代码使用 `s1_graph/s2_graph` 相似历史序列;本次用 train-only Candidate-Job 图传播替代,保证无泄漏和同协议,但不等价于原论文数据构造。

## 7. 对 SOTA 判断的影响

加入 PJFCANN 后,当前内部排序仍为:

```text
SPC-HGT > LightGCN > Rule > PJFCANN adapted > BM25 > SBERT > Popularity
```

因此 `docs/19_strategy_innovation_sota_assessment.md` 的结论不变:

- personas-core 当前协议下,SPC-HGT 仍是内部最优。
- PJFCANN adapted 是一个有价值的公开方法对照,但没有动摇 SPC-HGT 的主模型地位。
- 公开 SOTA 仍不能宣称,因为 PJFCANN 是适配复现,还不是在其原始公开数据和指标上的完全复现。

## 8. 下一步

1. 若要更忠实复现 PJFCANN,需要从 personas-core 构造 `s1_graph/s2_graph` 相似历史序列,而不是只用全局 C-J 图。
2. 若要更强公开 baseline,建议优先复现 ConFit v2,因为它直接面向 resume-job matching 的文本对比召回。
3. 若继续优化 PJFCANN,应加入 full-catalog sampled-softmax 或 ANN retrieval 训练,否则 pair classifier 很难转化为强 top-K 召回。

## 9. Sources

- PJFCANN paper: https://arxiv.org/abs/2206.09116
- PJFCANN official code: https://github.com/CCIIPLab/PJFCANN
- Evaluation Metrics for Item Recommendation under Sampling: https://arxiv.org/abs/1912.02263
