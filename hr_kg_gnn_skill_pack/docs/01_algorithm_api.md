# 01 算法调用接口文档

## 1. Python API

```python
from jobmatch_gnn.inference.ranker import rank_jobs_for_candidate
from jobmatch_gnn.data.schema import CandidateProfile, JobProfile

response = rank_jobs_for_candidate(
    candidate=candidate,
    jobs=jobs,
    top_k=10,
    mode="full",
)
```

## 2. 输入 Schema

### CandidateProfile

```json
{
  "candidate_id": "C001",
  "education": "本科",
  "major": "计算机科学与技术",
  "experience_years": 1.0,
  "skills": ["Python", "PyTorch", "SQL", "Linux"],
  "project_domains": ["推荐系统", "深度学习"],
  "expected_city": "广州",
  "expected_position": "算法工程师",
  "resume_text": "..."
}
```

### JobProfile

```json
{
  "job_id": "J001",
  "job_title": "算法工程师",
  "required_skills": ["Python", "机器学习", "PyTorch"],
  "preferred_skills": ["推荐系统", "Docker"],
  "education_requirement": "本科",
  "experience_requirement": 1.0,
  "city": "广州",
  "industry": "互联网",
  "jd_text": "..."
}
```

## 3. 输出 Schema

```json
{
  "candidate_id": "C001",
  "model_version": "0.3.0",
  "items": [
    {
      "job_id": "J001",
      "score": 0.873,
      "matched_skills": ["Python", "PyTorch"],
      "missing_skills": ["Docker"],
      "graph_paths": [
        "Candidate:C001 -> Skill:Python <- Job:J001",
        "Candidate:C001 -> Skill:PyTorch <- Job:J001"
      ],
      "reasons": ["技能覆盖度高", "城市匹配", "经验年限满足要求"]
    }
  ]
}
```

## 4. 推理模式

| 模式 | 作用 | 适用场景 |
|---|---|---|
| `recall` | 只做向量召回 | 大规模候选岗位初筛 |
| `rerank` | 对给定候选岗位精排 | 前端已给候选集合 |
| `full` | 召回 + 精排 + 解释 | 标准线上推荐 |

## 5. 错误码

| 错误码 | 含义 | 处理方式 |
|---|---|---|
| `E_INPUT_EMPTY` | 候选人或岗位输入为空 | 返回 400，提示补充数据 |
| `E_SCHEMA_INVALID` | 输入不符合 Schema | 返回校验错误字段 |
| `E_SKILL_EMPTY` | 技能字段为空 | 降级使用文本语义匹配 |
| `E_MODEL_NOT_READY` | 模型或索引未加载 | 返回 503，触发重载 |
| `E_INDEX_STALE` | ANN 索引版本落后 | 继续服务但记录 warning |
| `E_EXPLAIN_FAILED` | 解释生成失败 | 推荐结果正常返回，解释字段为空并记录日志 |

## 6. 时间复杂度

设：

```text
N = 岗位数量
d = embedding 维度
R = 边类型数量
E_b = batch 子图边数
L = GNN 层数
K = 推荐返回数量
```

复杂度：

```text
向量召回：O(log N) 或近似 O(sqrt(N))，取决于 ANN 索引
GNN 编码：O(L * E_b * d)
精排：O(N_recall * d + N_recall * path_feature_cost)
解释：O(K * path_query_cost)
```

## 7. 版本兼容

每个响应必须携带：

```text
model_version
schema_version
graph_snapshot_id
index_version
```

只要 `schema_version` 主版本变化，旧模型不可直接加载。
