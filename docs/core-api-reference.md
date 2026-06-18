# Core API Reference v2.1

更新时间：2026-06-18

本文定义前端调用 Core 推荐算法服务的字段协议。当前算法主线为 SPC-HGT v2，输入依赖候选人/岗位文本、规范化技能、城市、岗位类型、学历和年限等结构化特征；解释层由 KG evidence 提供事实依据，LLM 只负责把事实依据表达成可展示文案。

## 通用约定

- 基础路径：`/api/v1`
- 编码：UTF-8，请求/响应均为 `application/json`
- 字段命名：snake_case
- 日期时间：ISO 8601
- 匹配度 `match`：0~100 整数，Core 在响应前完成单位转换
- 分页：页码从 1 开始，`page_size` 默认 20，建议最大 50
- 业务标识符：使用 string 类型
- 请求追踪：通过 `X-Request-ID` 请求头传入，Core 日志应记录该值
- 错误响应：使用标准 HTTP 状态码，响应体包含 `detail` 字段（string 或校验错误列表），不得包含密钥、堆栈、真实文件路径
- `id` 字段统一使用 string 类型，Backend 负责与前端 integer 的映射
- Core 必须先完成召回、过滤和排序，再根据 `page` / `page_size` 截取结果并计算 `pagination.total`
- `items` 必须已按 `options.sort_by` 排序返回，Backend 不做二次排序或分页

## LLM HTML 解释约定

当 `options.include_llm_explanation=true` 时，每条推荐结果必须返回 `explanation.llm.html`。该字段是服务端生成并转义后的 HTML fragment，用于前端直接展示解释结果。

- 推荐前端优先读取：`item.explanation.llm.html`
- 兼容便捷字段：`item.llm_explanation_html`，内容等同于 `item.explanation.llm.html`
- HTML 允许标签：`div`、`p`、`strong`、`span`、`ul`、`li`、`br`
- HTML 禁止内容：`script`、`style`、`iframe`、内联事件属性（如 `onclick`）、外链脚本、未转义的用户原文
- 前端若要展示可解释 LLM 结果，必须在 `options` 中传 `include_explanations=true` 和 `include_llm_explanation=true`
- 前端不需要自行拼装解释文案；应展示 Core 返回的 `highlight`、`reason` 或 `explanation.llm.html`，并保留 `matched_skills`、`missing_skills` 用于可视化展开
- LLM 不可编造事实；如果 LLM 不可用或未通过忠实性检查，Core 必须返回模板兜底解释，`source` 标记为 `template` 或 `template_fallback_unfaithful`
- LLM 解释必须从 `matched_skills`、`missing_skills`、`graph_paths`、`reasons` 生成；不得引入 evidence 中不存在的技能、公司、薪资或岗位要求

## 接口一：POST /api/v1/recommend/jobs

人找岗推荐。接收求职者画像，返回岗位推荐结果。

### 请求字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| profile | CandidateProfile | 是 | 求职者画像 |
| options | RecommendOptions | 否 | 推荐控制参数 |

### 响应字段（JobRecommendationData）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| items | JobItem[] | 是 | 推荐结果列表 |
| pagination | Pagination | 是 | 分页信息 |
| request_id | string | 否 | Core 回传的请求追踪 ID |

**JobItem 字段：**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| id | string | 是 | 岗位标识 |
| title | string | 是 | 岗位标题 |
| company | string | 是 | 公司名称；未知时返回空字符串 |
| category | string | 是 | 岗位类别展示名 |
| salary | string | 是 | 薪资展示文本；未知时返回空字符串 |
| city | string | 是 | 城市展示名 |
| district | string | 否 | 区域展示名 |
| education | string | 否 | 学历要求展示文本 |
| experience | string | 否 | 经验要求展示文本 |
| match | integer | 是 | 匹配度 0~100 |
| level | string | 是 | 匹配等级，如 `高匹配` / `中匹配` / `可尝试` |
| highlight | string | 是 | 匹配亮点一句话，适合卡片摘要 |
| duty | string | 否 | 岗位职责摘要 |
| filter_tags | string[] | 是 | 筛选标签 |
| keywords | string[] | 是 | 技能/关键词 |
| detail_bullets | string[] | 否 | 岗位详情要点 |
| match_details | object | 否 | 兼容字段：分项匹配证据，建议新前端读取 `explanation` |
| matched_keywords | string[] | 否 | 兼容字段：命中的技能/关键词 |
| algorithm | string | 否 | 产生该条结果的算法标识 |
| explanation | ExplanationPayload | 否 | KG 事实证据和 LLM 展示解释 |
| llm_explanation_html | string | 否 | 兼容便捷字段，等同 `explanation.llm.html` |

说明：`include_explanations=false` 时可以省略 `explanation`、`match_details`、`matched_keywords`、`algorithm`，但不得省略 `level`、`highlight` 等卡片展示必填字段。

## 接口二：POST /api/v1/recommend/candidates

岗找人推荐。接收岗位画像，返回候选人推荐结果。

### 请求字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| profile | JobProfile | 是 | 岗位画像 |
| options | RecommendOptions | 否 | 推荐控制参数 |

### 响应字段（CandidateRecommendationData）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| items | CandidateItem[] | 是 | 推荐结果列表 |
| pagination | Pagination | 是 | 分页信息 |
| request_id | string | 否 | Core 回传的请求追踪 ID |

**CandidateItem 字段：**

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| id | string | 是 | 候选人标识 |
| name | string | 是 | 姓名；匿名场景可返回脱敏名 |
| gender | string | 否 | 性别；未知时返回空字符串 |
| age | integer | 否 | 年龄；未知时省略 |
| degree | string | 否 | 学历展示文本 |
| years | integer | 否 | 工作年限 |
| current | string | 否 | 当前职位或经历摘要 |
| salary | string | 否 | 薪资展示文本 |
| match | integer | 是 | 匹配度 0~100 |
| tags | string[] | 是 | 匹配标签 |
| reason | string | 是 | 卡片级匹配理由 |
| avatar | string | 否 | 头像标识，如姓氏或头像 key |
| filter_tags | string[] | 是 | 筛选标签 |
| keywords | string[] | 是 | 技能/关键词 |
| target_roles | string[] | 否 | 期望岗位或职类 |
| city | string | 否 | 城市展示名 |
| match_details | object | 否 | 兼容字段：分项匹配证据，建议新前端读取 `explanation` |
| matched_keywords | string[] | 否 | 兼容字段：命中的技能/关键词 |
| algorithm | string | 否 | 产生该条结果的算法标识 |
| explanation | ExplanationPayload | 否 | KG 事实证据和 LLM 展示解释 |
| llm_explanation_html | string | 否 | 兼容便捷字段，等同 `explanation.llm.html` |

说明：`include_explanations=false` 时可以省略 `explanation`、`match_details`、`matched_keywords`、`algorithm`，但不得省略 `reason`、`tags` 等卡片展示必填字段。

## DTO 定义

### CandidateProfile（求职者画像）

旧协议中的 `skills`、`experience_text`、`desired_cities`、`desired_roles`、`degree`、`years_experience` 可以支撑基础规则排序和部分 v2 特征，但要稳定复现当前 SPC-HGT v2 输入，前端应同时提供规范化技能、城市 ID、岗位类型、学历等级和工作年限。城市字段优先传 `desired_city_ids`、`current_city_id`；岗位方向字段建议由旧 `desired_roles` 迁移为 `desired_type_ids` + `desired_job_types`；技能字段优先使用 Core 技能词表中的规范化标签。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| candidate_id | string | 否 | 候选人标识；临时画像可为空 |
| skills | string[] | 条件必填 | 规范化技能标签，优先使用 Core 技能词表中的原词；与 `experience_text` 至少提供一项，推荐最多 64 个 |
| experience_text | string | 条件必填 | 经历摘要、项目经历或简历片段；与 `skills` 至少提供一项 |
| desired_city_ids | string[] | 否 | 期望城市 ID，推荐提供；用于城市结构化特征 |
| desired_cities | string[] | 否 | 期望城市展示名；仅有展示名时 Core 会尝试映射，映射失败则城市特征降级 |
| current_city_id | string | 否 | 当前城市 ID；用于 live city 特征 |
| current_city | string | 否 | 当前城市展示名 |
| desired_type_ids | string[] | 否 | 期望岗位类型 ID，推荐提供；对应数据中的 `desire_jd_type_id` |
| desired_job_types | string[] | 否 | 期望岗位类型展示名；替代旧字段 `desired_roles` |
| desired_roles | string[] | 否 | 兼容旧前端字段；Core 会按岗位类型或标题关键词尝试映射 |
| industries | string[] | 否 | 期望或当前行业；有 ID 时可传 ID 字符串 |
| age | integer | 否 | 年龄；当前算法不强依赖，仅用于展示或规则过滤 |
| degree | string | 否 | 学历展示名，如 `本科`、`硕士` |
| degree_rank | integer | 否 | 归一化学历等级；若前端可提供，优先传该字段 |
| years_experience | integer | 否 | 工作年限；用于年限匹配特征 |
| salary_expectation | string | 否 | 期望薪资展示文本；当前主模型不直接使用 |

### JobProfile（岗位画像）

旧协议中的 `description`、`required_skills` 可以支撑基础规则排序和部分 v2 特征，但岗找人和临时岗位召回需要岗位文本、规范化技能、城市 ID、岗位类型、最低学历和最低年限。`description` 与 `required_skills` 至少提供一项，推荐都提供；岗位侧建议传 `city_id`、`job_type_id`、`min_degree_rank`、`min_years`，仅传展示文本时 Core 会尝试解析或映射。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| job_id | string | 否 | 岗位标识；临时画像可为空 |
| title | string | 是 | 岗位标题 |
| description | string | 条件必填 | 岗位职责与要求摘要；与 `required_skills` 至少提供一项 |
| city_id | string | 否 | 工作城市 ID，推荐提供；用于城市结构化特征 |
| city | string | 否 | 工作城市展示名；仅有展示名时 Core 会尝试映射 |
| industry_id | string | 否 | 行业 ID |
| industry | string | 否 | 行业展示名 |
| job_type_id | string | 否 | 岗位类型 ID，推荐提供；对应数据中的 `jd_sub_type` 或映射后的类型 |
| job_type | string | 否 | 岗位类别展示名 |
| required_skills | string[] | 条件必填 | 规范化技能要求，优先使用 Core 技能词表中的原词；与 `description` 至少提供一项，推荐最多 30 个 |
| skill_weights | object | 否 | 技能权重，键为技能名，值为 0~1；未传时 Core 使用默认权重 |
| degree_requirement | string | 否 | 学历要求展示名 |
| min_degree_rank | integer | 否 | 归一化最低学历等级；若前端可提供，优先传该字段 |
| years_requirement | string | 否 | 经验要求展示文本 |
| min_years | integer | 否 | 最低工作年限；若前端可提供，优先传该字段 |
| salary_range | string | 否 | 薪资展示文本 |
| require_nums | integer | 否 | 招聘人数；当前主模型不直接使用 |
| is_travel | boolean | 否 | 是否要求出差；当前主模型不直接使用 |
| is_managerial | boolean | 否 | 是否要求管理经验；当前主模型不直接使用 |
| resume_language_required | string | 否 | 简历语言要求；当前主模型不直接使用 |

### RecommendOptions（推荐控制参数）

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| page | integer | 否 | 1 | 页码，从 1 开始 |
| page_size | integer | 否 | 20 | 每页数量，建议最大 50 |
| sort_by | string | 否 | match | `match` / `salary` / `experience` |
| model | string | 否 | auto | `auto` / `spc_hgt_v2` / `lightgcn_v2` / `rule_v2` / `bm25` / `sbert` / `popularity` |
| include_explanations | boolean | 否 | true | 是否返回 `explanation`、`match_details`、`matched_keywords`、`algorithm` |
| include_llm_explanation | boolean | 否 | false | 是否返回 `explanation.llm` 与 `llm_explanation_html`；涉及额外 LLM 成本，默认关闭 |
| llm_output_format | string | 否 | html | `html` / `text` / `both`；前端展示推荐使用 `html` |
| explanation_language | string | 否 | zh-CN | 解释语言，目前支持 `zh-CN` |
| include_debug_evidence | boolean | 否 | false | 是否返回调试级特征；生产环境默认关闭 |

排序方向：`match` 按匹配度降序，`salary` 按薪资上界降序，`experience` 按经验年限降序。`include_llm_explanation=false` 时必须保留结构化解释字段（若 `include_explanations=true`），但可以省略 `explanation.llm` 和 `llm_explanation_html`。

### ExplanationPayload（解释结果）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| matched_skills | WeightedSkill[] | 是 | 候选人与岗位共同命中的技能，按注意力或权重降序 |
| missing_skills | string[] | 是 | 岗位需要但候选人画像中未命中的技能 |
| graph_paths | string[] | 是 | 可解释 KG 路径，如 `候选人 -> python <- 岗位` |
| reasons | string[] | 是 | 规则或结构化匹配理由，如城市、学历、年限、技能覆盖率 |
| rule_features | object | 否 | 调试用规则特征，`include_debug_evidence=true` 时返回 |
| llm | LLMExplanation | 否 | LLM 或模板生成的展示解释 |

### WeightedSkill

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| skill | string | 是 | 技能名 |
| weight | number | 是 | 0~1 权重或注意力分数 |

### LLMExplanation

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| html | string | 是 | 安全 HTML fragment；前端推荐直接使用该字段展示 |
| text | string | 否 | 纯文本解释；`llm_output_format=text` 或 `both` 时返回 |
| source | string | 是 | `llm` / `template` / `template_fallback_unfaithful` |
| faithful | boolean | 是 | 是否通过忠实性检查；返回给前端的解释必须为 true |
| rejected_llm | string | 否 | 调试字段；仅在非生产调试环境、且 LLM 未通过检查时返回 |

### Pagination（分页信息）

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| page | integer | 是 | 当前页码 |
| page_size | integer | 是 | 每页数量 |
| total | integer | 是 | 总记录数 |
| total_pages | integer | 是 | 总页数 |

## 请求示例

```json
{
  "profile": {
    "candidate_id": "u_1001",
    "skills": ["python", "mysql", "机器学习"],
    "experience_text": "3 年 Python 后端和推荐系统经验，熟悉 MySQL 与特征工程。",
    "desired_city_ids": ["310000"],
    "desired_cities": ["上海"],
    "current_city_id": "310000",
    "desired_type_ids": ["backend"],
    "desired_job_types": ["后端开发"],
    "degree": "本科",
    "degree_rank": 4,
    "years_experience": 3
  },
  "options": {
    "page": 1,
    "page_size": 10,
    "model": "spc_hgt_v2",
    "include_explanations": true,
    "include_llm_explanation": true,
    "llm_output_format": "html"
  }
}
```

## 响应片段示例

```json
{
  "id": "j_9001",
  "title": "推荐算法工程师",
  "match": 86,
  "level": "高匹配",
  "highlight": "Python、机器学习与城市偏好匹配度较高",
  "algorithm": "spc_hgt_v2",
  "explanation": {
    "matched_skills": [
      {"skill": "python", "weight": 0.42},
      {"skill": "机器学习", "weight": 0.31}
    ],
    "missing_skills": ["召回排序", "特征工程"],
    "graph_paths": ["候选人 -> python <- 岗位"],
    "reasons": ["技能覆盖率 50%", "工作城市符合期望城市"],
    "llm": {
      "html": "<div class=\"llm-explanation\"><p>推荐岗位「推荐算法工程师」：你的技能 python、机器学习与该岗位要求匹配。</p></div>",
      "source": "llm",
      "faithful": true
    }
  },
  "llm_explanation_html": "<div class=\"llm-explanation\"><p>推荐岗位「推荐算法工程师」：你的技能 python、机器学习与该岗位要求匹配。</p></div>"
}
```
