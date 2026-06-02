# 03 模型设计：SPC-HGT MatchNet

## 1. 目标

学习候选人和岗位在异构知识图谱中的表示，并融合文本语义、结构化字段、技能路径和行为反馈，输出高质量 Top-K 推荐。

## 2. 模型结构

```text
Text Encoder
  resume_text / jd_text -> text_embedding

Structured Encoder
  education / experience / city / industry -> structured_embedding

KG Builder
  Candidate / Job / Skill / Company / City / Industry / Education / Experience -> Heterogeneous Graph

SPC-HGT Encoder
  type-aware attention + relation-aware message passing -> node embeddings

Path Feature Encoder
  Candidate -> Skill <- Job paths -> path features

Matching Head
  recall scorer + rank scorer
```

## 3. SPC-HGT Encoder

推荐消息传递公式：

```text
m_{u -> v, r}^{l} = alpha_{u,v,r}^{l} * W_r^{l} h_u^{l}

h_v^{l+1} = LayerNorm(
    h_v^{l} + Dropout(
        sigma(sum_{r in R} sum_{u in N_r(v)} m_{u -> v, r}^{l})
    )
)
```

其中：

```text
r: 边类型
alpha: 关系感知注意力
W_r: 关系特定参数
h_v: 节点表示
```

默认层数：2 层。没有强证据时不要堆到 4 层以上，避免 over-smoothing。

## 4. 技能路径增强

核心路径：

```text
Candidate -> HAS_SKILL -> Skill <- REQUIRES_SKILL <- Job
```

路径特征：

```text
shared_required_skill_count
shared_preferred_skill_count
missing_required_skill_count
weighted_skill_coverage
skill_path_count
skill_taxonomy_similarity
required_skill_importance_sum
candidate_skill_recency_score
```

技能覆盖率：

```text
weighted_skill_coverage(c, j) =
  sum_{s in skills(c) intersect required_skills(j)} importance(j, s) * proficiency(c, s)
  / sum_{s in required_skills(j)} importance(j, s)
```

## 5. 打分函数

### 5.1 召回

```text
score_recall(c, j) = cosine(z_c, z_j)
```

### 5.2 精排

```text
score_rank(c, j) = MLP([
  z_c,
  z_j,
  z_c * z_j,
  abs(z_c - z_j),
  path_features(c, j),
  rule_features(c, j),
  semantic_features(c, j)
])
```

## 6. 多目标损失

```text
L = lambda_1 * L_bpr
  + lambda_2 * L_bce
  + lambda_3 * L_contrastive
  + lambda_4 * L_kg_reconstruction
  + lambda_5 * L_reg
```

建议默认权重：

```yaml
loss_weights:
  bpr: 1.0
  bce: 0.5
  contrastive: 0.2
  kg_reconstruction: 0.1
  reg: 1.0e-5
```

## 7. Hard Negative Mining

困难负样本来源：

```text
BM25 hard negative：文本关键词高度相似但无正反馈
SBERT hard negative：语义高度相似但无正反馈
same_city_negative：城市相同但技能不满足
same_industry_negative：行业相同但技能缺口大
recent_job_negative：近期热门岗位但候选人无交互
```

## 8. 冷启动策略

新候选人：

```text
简历文本 embedding + 技能路径临时子图 + 规则特征
```

新岗位：

```text
JD 文本 embedding + required_skills 边 + ANN index 增量更新
```

无行为数据时，降级为：

```text
Rule + BM25 + SBERT + KG path score
```

## 9. 可解释性

解释模块不能只输出自然语言，必须返回结构化证据：

```json
{
  "matched_skills": ["Python", "PyTorch"],
  "missing_skills": ["Docker"],
  "graph_paths": ["Candidate:C001 -> Skill:Python <- Job:J001"],
  "reasons": ["技能覆盖度高", "城市匹配"]
}
```
