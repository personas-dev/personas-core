# 11 数据管线与知识图谱 v2

日期:2026-06-12
上游:`docs/10_optimization_plan.md` 阶段 A
代码:`src/jobmatch_gnn/data/full_loader.py`、`scripts/preprocess_v2.py`

## 1. 源数据

`data/datasets.zip`(智联招聘人岗匹配数据):

| 表 | 行数 | 关键字段 |
|---|---:|---|
| table1_user | 4,500 | user_id, live_city_id, desire_jd_city_id, desire_jd_industry_id, desire_jd_type_id, cur_degree_id, start_work_date, **experience**(竖线分隔的预抽取技能标签) |
| table2_jd | 269,534 | jd_no, jd_title, city, jd_sub_type, min_years, min_edu_level, **job_description**(全文) |
| table3_action | 700,938 | user_id, jd_no, browsed, delivered, satisfied |

行为统计:browsed=150,930;delivered=66,285;satisfied=31,889。全部 4,500 个用户都有 ≥1 个 delivered/satisfied,人均正样本中位数 10。

`data/kg_output.zip` 为 v1 时代的产物(260 万噪声技能节点),**v2 不再使用**,仅在文档中保留对照。

## 2. 标签定义

```text
正样本   = delivered=1 或 satisfied=1   (用户主动投递/双方满意,强意图)
弱正样本 = browsed=1 且未投递           (只用于对比学习的弱正对与图的弱边,不进 eval 正例)
负样本   = 训练时采样(随机 + 难负),eval 时为全库排除已知正例
```

理由:browsed 噪声大(曝光偏差),delivered 是用户的真实选择行为;以 delivered∪satisfied 为 eval 正例与多数 person-job fit 文献一致。

## 3. 技能词表 v2(核心修复)

v1 失败原因:对中文做 2/3-gram 开放切分 → 260 万碎片节点。

v2 流程:

```text
S1 种子词表:收集全部 user.experience 标签(竖线分隔,已是人工/上游系统抽取的技能短语)
   → 去重、clean_token 规范化(全角/空白/英文小写)
S2 频次过滤:保留出现次数 >= min_user_count(默认 2)或长度>=2 的英文技术词
S3 job 侧对齐:用词表构建 Aho-Corasick 自动机,对 jd_title + job_description 扫描,
   命中即建立 Job -REQUIRES_SKILL-> Skill 边(标题命中权重 2.0,正文 1.0)
S4 截断:每个 job 最多保留 top-30 命中技能(按权重×词长),每个 user 最多 64 个技能
```

预期规模:词表数万级;Job-Skill 边千万级以内。所有阈值进 config,不许硬编码。

## 4. 异构图 Schema(v2)

节点(6 类):

| 类型 | 来源 | 规模(约) |
|---|---|---:|
| Candidate | table1_user | 4,500 |
| Job | table2_jd(只保留 action 中出现过的 job + 训练采样需要的) | 265,825 |
| Skill | 技能词表 v2 | 数万 |
| City | user.live/desire 城市 ∪ jd.city | ~500 |
| Industry | desire_jd_industry_id ∪ cur_industry_id | ~130 |
| JobType | desire_jd_type_id ∪ jd_sub_type | ~1,200 |

边(全部加反向边供消息传递):

```text
Candidate -HAS_SKILL->     Skill     (experience 标签)
Job       -REQUIRES_SKILL-> Skill    (AC 匹配,带权)
Candidate -INTERACTED->    Job       (仅 train 正样本,杜绝泄漏;browsed 弱边可选,权重 0.3)
Candidate -LIVES_IN/DESIRES_CITY-> City
Job       -LOCATED_IN->    City
Candidate -DESIRES_TYPE->  JobType
Job       -OF_TYPE->       JobType
Candidate -IN_INDUSTRY->   Industry
```

注意:**test/valid 的交互边绝不能进图**,否则信息泄漏。

## 5. 划分协议

action 表无时间戳。采用确定性 leave-last-out(按源文件行序近似时间序):

```text
对每个用户的正样本序列(按 row_index 升序):
  最后 1 个   → test
  倒数第 2 个 → valid(正样本数 >= 3 时)
  其余        → train
```

局限(必须在所有报告声明):行序≠真实时间;若未来拿到时间戳,切换为全局时间切分。

## 6. 产出缓存(data/processed/)

| 文件 | 内容 |
|---|---|
| users.parquet | 规范化用户画像(含技能 id 列表) |
| jobs.parquet | 规范化岗位(含技能 id 列表与权重) |
| interactions.parquet | user_idx, job_idx, level(1/2/3), split(train/valid/test) |
| skill_vocab.json | skill → id、频次 |
| graph_v2.pt | 异构图张量(edge_index 字典) |
| emb_user.npy / emb_job.npy / emb_skill.npy | SBERT 嵌入(阶段 B 产出) |

缓存带版本号与生成参数指纹,源参数变化时自动重建。大文件全部在 .gitignore 内。

## 7. 与 v1 的对照

| 维度 | v1 | v2 |
|---|---|---|
| 用户覆盖 | 784(17%) | 4,500(100%) |
| 正样本 | 950 | ~98,000 |
| 技能节点 | 2,596,614(噪声) | 数万(词表对齐) |
| job 技能来源 | 文本 n-gram | 与 user 同词表的 AC 匹配 |
| 图 | 二部交互图 | 6 类节点异构图 |
