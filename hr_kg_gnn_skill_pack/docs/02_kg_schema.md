# 02 知识图谱 Schema

## 1. 设计原则

人岗匹配不是单纯文本匹配，而是“候选人能力、岗位要求、技能语义、行业城市、行为反馈”的异构关系匹配。因此图谱必须采用异构 Schema。

最小可行图谱围绕：

```text
Candidate -> Skill <- Job
```

扩展图谱再加入公司、城市、行业、学历、经验、技能同义词和技能层级。

## 2. 节点类型

| 节点类型 | 必需字段 | 可选字段 |
|---|---|---|
| `Candidate` | `candidate_id`, `skills`, `text_embedding` | `education`, `major`, `experience_years`, `expected_city`, `salary_expectation` |
| `Job` | `job_id`, `job_title`, `required_skills`, `text_embedding` | `preferred_skills`, `salary_range`, `publish_time`, `company_id` |
| `Skill` | `skill_id`, `skill_name`, `normalized_name` | `aliases`, `category`, `level`, `is_tool`, `is_domain` |
| `Company` | `company_id`, `company_name` | `industry`, `size`, `financing_stage` |
| `City` | `city_id`, `city_name` | `province`, `country`, `lat`, `lon` |
| `Industry` | `industry_id`, `industry_name` | `parent_industry` |
| `Education` | `education_id`, `level` | `rank` |
| `Experience` | `experience_id`, `years_bucket` | `domain` |

## 3. 边类型

| 边类型 | 源节点 | 目标节点 | 边特征 |
|---|---|---|---|
| `HAS_SKILL` | Candidate | Skill | `proficiency`, `years`, `recency`, `confidence` |
| `REQUIRES_SKILL` | Job | Skill | `importance`, `min_years`, `is_required` |
| `PREFERS_SKILL` | Job | Skill | `importance`, `is_bonus` |
| `APPLIED_TO` | Candidate | Job | `timestamp`, `source`, `label` |
| `CLICKED` | Candidate | Job | `timestamp`, `dwell_time` |
| `MATCHES` | Candidate | Job | `label`, `score_source` |
| `LOCATED_IN` | Job | City | `distance_level` |
| `BELONGS_TO` | Job | Industry | 无 |
| `PUBLISHES` | Company | Job | `timestamp` |
| `SIMILAR_TO` | Skill | Skill | `similarity`, `source` |
| `PREREQUISITE_OF` | Skill | Skill | `confidence` |

## 4. Reverse Edge 规则

所有边都必须生成反向边：

```text
HAS_SKILL        -> REV_HAS_SKILL
REQUIRES_SKILL   -> REV_REQUIRES_SKILL
PREFERS_SKILL    -> REV_PREFERS_SKILL
APPLIED_TO       -> REV_APPLIED_TO
CLICKED          -> REV_CLICKED
MATCHES          -> REV_MATCHES
LOCATED_IN       -> REV_LOCATED_IN
BELONGS_TO       -> REV_BELONGS_TO
PUBLISHES        -> REV_PUBLISHES
SIMILAR_TO       -> SIMILAR_TO，若语义对称可不另建 rev
PREREQUISITE_OF  -> REV_PREREQUISITE_OF
```

## 5. 技能标准化

技能标准化必须处理：

```text
大小写：python -> Python
中英文别名：机器学习 -> Machine Learning
框架别名：PyTorch Lightning -> PyTorch
同义技能：NLP -> Natural Language Processing
技能层级：Deep Learning 是 Machine Learning 的子类
工具与能力区分：Docker 是工具，推荐系统是领域能力
```

标准化输出：

```json
{
  "raw_skill": "机器学习",
  "normalized_skill": "Machine Learning",
  "skill_id": "S000123",
  "confidence": 0.94,
  "aliases": ["ML", "机器学习"],
  "category": "AI"
}
```

## 6. PyG 映射

```python
from torch_geometric.data import HeteroData

data = HeteroData()
data["candidate"].x = candidate_x
data["job"].x = job_x
data["skill"].x = skill_x

data[("candidate", "has_skill", "skill")].edge_index = edge_index_candidate_skill
data[("job", "requires_skill", "skill")].edge_index = edge_index_job_skill
```

## 7. DGL 映射

```python
import dgl

g = dgl.heterograph({
    ("candidate", "has_skill", "skill"): (candidate_ids, skill_ids),
    ("job", "requires_skill", "skill"): (job_ids, skill_ids),
})
```
