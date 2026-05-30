# 知识图谱 Schema 设计

本文档对应 `Lab_last/datasets1` 的招聘知识图谱构建方案。图谱以“用户-岗位-技能-行业-职位类型-城市-学历”为核心，适合做岗位推荐、用户画像补全和行为分析。

## 1. 实体设计

| 实体类型 | 实体ID规则 | 主要来源字段 | 主要属性 | 说明 |
| --- | --- | --- | --- | --- |
| User | `USER::<user_id>` | `table1_user.user_id` | `live_city_id`, `desire_jd_city_id`, `desire_jd_industry_id`, `desire_jd_type_id`, `desire_jd_salary_id`, `cur_industry_id`, `cur_jd_type`, `cur_salary_id`, `cur_degree_id`, `birthday`, `start_work_date`, `experience` | 求职者主体节点，用于表示用户画像 |
| Job | `JOB::<jd_no>` | `table2_jd.jd_no` | `jd_title`, `company_name`, `city`, `jd_sub_type`, `require_nums`, `max_salary`, `min_salary`, `start_date`, `end_date`, `is_travel`, `min_years`, `key`, `min_edu_level`, `max_edu_level`, `is_mangerial`, `resume_language_required`, `job_description` | 岗位主体节点，包含岗位基本信息和文本描述 |
| Skill | `SKILL::<skill>` | `table1_user.experience`, `table2_jd.jd_title + job_description` | `node_name` | 从用户经验和岗位描述中抽取的技能/关键词节点 |
| Industry | `INDUSTRY::<industry>` | `table1_user.desire_jd_industry_id`, `table1_user.cur_industry_id` | `node_name` | 行业节点，保留原始行业名称 |
| JobType | `JOB_TYPE::<job_type>` | `table1_user.desire_jd_type_id`, `table1_user.cur_jd_type`, `table2_jd.jd_sub_type` | `node_name` | 职位类别/岗位类型节点 |
| City | `CITY::<city_id>` | `table1_user.live_city_id`, `table1_user.desire_jd_city_id`, `table2_jd.city` | `node_name` | 城市节点，数据集中以城市ID为主 |
| Degree | `DEGREE::<degree>` | `table1_user.cur_degree_id`, `table2_jd.min_edu_level` | `node_name` | 学历节点，如“大专”“本科”“MBA” |
| Company | `COMPANY::<company_name>` | `table2_jd.company_name` | `node_name` | 公司节点，适合扩展企业维度分析，若为空可忽略 |

## 2. 关系设计

| 关系类型 | 起点实体 | 终点实体 | 来源字段 | 权重/含义 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `LIVES_IN_CITY` | User | City | `table1_user.live_city_id` | 1 | 用户当前所在城市 |
| `DESIRES_CITY` | User | City | `table1_user.desire_jd_city_id` | 1 | 用户期望岗位城市 |
| `DESIRES_INDUSTRY` | User | Industry | `table1_user.desire_jd_industry_id` | 1 | 用户期望行业 |
| `DESIRES_JOB_TYPE` | User | JobType | `table1_user.desire_jd_type_id` | 1 | 用户期望职位类型 |
| `CURRENT_INDUSTRY` | User | Industry | `table1_user.cur_industry_id` | 1 | 用户当前行业 |
| `CURRENT_JOB_TYPE` | User | JobType | `table1_user.cur_jd_type` | 1 | 用户当前职位类型 |
| `HAS_DEGREE` | User | Degree | `table1_user.cur_degree_id` | 1 | 用户当前学历 |
| `HAS_SKILL` | User | Skill | `table1_user.experience` | 1 | 用户经验中出现的技能或关键词 |
| `LOCATED_IN_CITY` | Job | City | `table2_jd.city` | 1 | 岗位所在城市 |
| `BELONGS_TO_JOB_TYPE` | Job | JobType | `table2_jd.jd_sub_type` | 1 | 岗位所属职位类型 |
| `REQUIRES_DEGREE` | Job | Degree | `table2_jd.min_edu_level` | 1 | 岗位最低学历要求 |
| `REQUIRES_SKILL` | Job | Skill | `table2_jd.jd_title + job_description` | 1 | 从岗位标题和描述抽取的技能要求 |
| `BROWSED_JOB` | User | Job | `table3_action.browsed` | 1 | 浏览行为，表示弱兴趣 |
| `DELIVERED_JOB` | User | Job | `table3_action.delivered` | 5 | 投递行为，表示明确求职意图 |
| `SATISFIED_JOB` | User | Job | `table3_action.satisfied` | 10 | 满意/成功反馈，表示更强的正样本 |

## 3. 导出文件

脚本会在输出目录生成以下 CSV：

| 文件名 | 内容 |
| --- | --- |
| `nodes.csv` | 全部实体节点，包含 `node_id`, `node_type`, `node_name`, `attributes_json` |
| `edges.csv` | 全部关系边，包含 `source_id`, `source_type`, `relation`, `target_id`, `target_type`, `weight`, `evidence` |
| `graph_stats.csv` | 节点、边数量统计 |

## 4. 建图说明

1. 结构化字段直接建边，保证图谱的主干稳定。
2. 文本字段用规则抽取技能节点，作为岗位匹配的重要补充。
3. 行为表中的 `browsed`、`delivered`、`satisfied` 可作为边权重或监督信号，用于推荐与链路预测任务。
4. 如果后续需要做更强的语义抽取，可以在 `HAS_SKILL` 和 `REQUIRES_SKILL` 的基础上叠加分词、词典和实体对齐模块。
