# 13 评估协议与 Baseline 体系 v2

日期:2026-06-12
代码:`src/jobmatch_gnn/baselines/`、`src/jobmatch_gnn/evaluation/metrics.py`

## 1. 统一评估协议(所有模型必须遵守)

```text
数据      :全量 4,500 用户、全量正样本(delivered ∪ satisfied)
划分      :leave-last-out(见 docs/11 §5),所有模型用同一份 split 文件
候选池    :全部出现在 action 表中的 job(265,825 个)——全库排序,不做 sampled metrics
排除      :对每个用户,把其 train/valid 正例从排序中剔除(不剔除其它用户的)
指标      :Recall@K, Precision@K, NDCG@K, MRR, HitRate@K (K=10,50),AUC 可选
报告      :metrics.json + metrics.csv 落盘到 experiments/runs/<run_name>/
随机性    :可训练模型 ≥3 seeds 报均值±标准差;确定性模型跑 1 次
```

不允许 sampled evaluation(如 1 正 + 99 随机负):在 26.9 万库上会系统性高估,且不同模型间不可比。

## 2. Baseline 体系

| # | 模型 | 类型 | 实现要点 |
|---|---|---|---|
| B0 | Popularity | 频次下界 | 按 train 中 job 被投递次数排序,所有用户同一列表 |
| B1 | Rule v2 | 可解释规则 | 干净词表上的技能覆盖率 + 城市/学历/年限/岗位类型匹配的加权和;权重在 valid 上网格搜索 |
| B2 | BM25 | 稀疏检索 | jieba 分词;query = 用户技能标签+期望岗位类型+行业;doc = jd_title×2 + description;scipy 稀疏矩阵全库打分 |
| B3 | SBERT 双塔 | 稠密检索 | bge-small-zh-v1.5 零样本;query/passage 前缀按 bge 规范;余弦全库检索 |
| B4 | LightGCN | 协同过滤 | 全量 train 交互;d=128, 3 层, BPR, 负采样 4:1, ≥200 epoch 早停 |
| 主 | SPC-HGT v2 | 异构 GNN | docs/12 |

判定规则(沿用 skill pack):主模型只有在 NDCG@10 上超过 **全部** baseline 才允许声称有效;最强 baseline 是谁、差多少,必须写进 docs/15。

## 3. 指标定义

对用户 u,R_u 为其 test 正例集合(本协议下 |R_u|=1),L_u 为模型排序:

```text
Recall@K   = |L_u[:K] ∩ R_u| / |R_u|
Precision@K= |L_u[:K] ∩ R_u| / K
NDCG@K     = DCG@K / IDCG@K,  DCG = Σ_i rel_i / log2(i+1)
MRR        = 1 / rank(第一个命中)
HitRate@K  = 1[L_u[:K] 命中]      (|R_u|=1 时 = Recall@K)
```

全部按用户宏平均。实现见 `evaluation/metrics.py`,有 pytest 单测锚定数值。

## 4. 显著性与公平性

- 可训练模型多 seed,报均值±std;主模型 vs 最强 baseline 做配对 t 检验(按用户的 NDCG)。
- 所有模型共享同一 split、同一候选池、同一剔除规则;任何模型不得见到 valid/test 正例(图边、统计特征均只来自 train)。
- 调参只看 valid,test 只在最终报告时评一次。
