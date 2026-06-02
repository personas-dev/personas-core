# 07 推理与部署

## 1. 两阶段推荐

```text
Stage 1: Retriever
  candidate embedding -> ANN index -> Top-N jobs

Stage 2: Reranker
  SPC-HGT + path/rule/semantic features -> Top-K jobs

Stage 3: Explanation
  matched skills + missing skills + graph paths + reasons
```

## 2. 缓存对象

必须缓存：

```text
candidate_embedding
job_embedding
skill_embedding
job_ann_index
graph_snapshot
path_feature_cache
skill_normalization_cache
```

## 3. 新岗位上线流程

```text
1. 解析 JD
2. 标准化 required_skills / preferred_skills
3. 添加 Job 节点
4. 添加 REQUIRES_SKILL / PREFERS_SKILL 边
5. 生成 job embedding
6. 写入 ANN index
7. 更新 graph_snapshot_id
```

## 4. 新候选人推理流程

```text
1. 解析简历
2. 标准化技能
3. 构建临时 Candidate 子图
4. 计算 candidate embedding
5. ANN 召回 Top-N
6. 精排 Top-K
7. 输出解释
```

## 5. 服务接口

推荐 FastAPI：

```text
POST /v1/jobmatch/recommend
POST /v1/jobmatch/rerank
POST /v1/jobmatch/explain
GET  /v1/jobmatch/health
GET  /v1/jobmatch/model-card
```

## 6. 延迟目标

| 阶段 | p95 目标 |
|---|---:|
| ANN 召回 | < 50 ms |
| 精排 Top-N | < 200 ms |
| 解释生成 | < 100 ms |
| 总接口 | < 500 ms |

原型阶段可以放宽，但必须记录 p50/p95。

## 7. 降级策略

```text
GNN 模型不可用 -> SBERT + Rule
ANN 索引不可用 -> BM25 + Rule
图谱路径查询失败 -> 推荐结果保留，解释字段降级为空
技能标准化失败 -> 使用原始技能文本
```
