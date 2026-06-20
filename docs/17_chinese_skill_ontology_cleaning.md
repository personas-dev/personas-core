# 17 中文职业技能参照库与技能清洗策略

日期:2026-06-20

关联: `docs/11_data_kg_v2.md`, `docs/12_spc_hgt_v2_design.md`, `docs/15_results_v2.md`

## 1. 结论

可以找到中文侧可作为 ESCO/O*NET 参照的权威资源,但它不是单一的“技能节点库”:

1. **技能人才评价工作网(OSTA)** 提供“职业分类系统”和“职业标准系统”,网站标注政策指导单位为人力资源和社会保障部职业能力建设司,版权所有方为中国就业培训技术指导中心/人社部职业技能鉴定中心。它适合作为中文官方参照库。
2. OSTA 职业分类系统可以用于识别“职业、工种、职业小类”,这些词应映射到 JobType/Occupation,不应进入 Skill 节点。
3. OSTA 国家职业标准查询系统可用于抽取“职业功能、工作内容、技能要求、相关知识”等候选术语,这些内容才适合作为技能本体的强参照。
4. Chinese-SkillSpan 2026 提供中文招聘文本上的 ESCO 对齐标注框架,其 LSKT 维度(knowledge, skill, transversal competence, language competence)适合指导技能类型分类。

因此 v3 清洗策略改为:

```text
用户/岗位文本候选词
  -> 词法质量门: 短英文、边界、泛词
  -> OSTA 参照门: 职业/工种/职业类目剔除或转 JobType, 职业标准术语增强保留
  -> 技能类型门: hard_or_knowledge / transversal / language / occupation_or_job_type / generic_non_skill
  -> 图谱建边: hard/knowledge 保留高权重, transversal 降权限额, generic/occupation 不进 Skill
```

## 2. 本次检索到的中文参照资源

### OSTA 技能人才评价工作网

已验证可访问系统:

- 首页列出“职业分类系统”和“职业标准系统”。
- `/api/client/get/tree` 返回职业分类树,字段包括 `careerCode`, `careerName`, `versionId`, `children`。
- `/api/client/career/detail` 返回职业定义、所属大中小类、工种、主要工作任务。例如“计算机程序设计员S”返回“编写、修改程序代码”等任务。
- `/api/public/skillStandardList` 返回国家职业标准列表。2026-06-20 实测 `total=697`,包含标准名称、职业编号、颁布时间、发文号和 PDF 文件名。

使用方式:

- `careerName`, `bigName`, `centreName`, `smallName`, `works.name` 默认归为职业/工种/岗位方向,用于过滤 Skill 误入。
- `task` 和标准 PDF 中的“技能要求/相关知识”可抽取为参考技能短语。
- 职业标准 PDF 不直接提交到 git,通过脚本在本地 `data/external/` 生成参照 JSON。

### Chinese-SkillSpan

该数据集是中文招聘广告的 span-level JobSkillNER 数据集,声明与 ESCO 技能标准四类维度对齐: knowledge, skill, transversal competence, language competence。它适合用来训练或校验中文技能类型分类器。

### China's First Workforce Skill Taxonomy

该工作提出中国劳动力技能 taxonomy 并映射到 O*NET。它适合作为研究背景和后续跨国标准映射依据,但当前优先级低于 OSTA,因为 OSTA 有官方职业分类和职业标准查询接口。

## 3. 当前 v2 问题复盘

v2 已把 v1 的 260 万碎片技能节点压到约 4,765 个,解决了“节点爆炸”。但高频技能仍有语义噪声:

| 来源 | 噪声例子 | 问题类型 |
|---|---|---|
| 用户侧 | 自我评价, 协助, 客户, 材料, 责任心 | 泛化词/软素质/上下文词 |
| 岗位侧 | 管理, 学历, 沟通, 经理, 员工 | 职业属性/招聘条件/泛词 |
| 英文 | or, ex, cc, em | 短 token 与英文边界误命中 |

根因:

- `build_skill_vocab` 主要依赖用户标签频次,没有外部本体参照。
- `match_job_skills` 对岗位标题/描述做 Aho-Corasick 裸匹配,英文没有词边界。
- Skill 节点没有类型,导致 hard skill、soft skill、岗位属性、职业名混在同一个消息传递通道。

## 4. 已落地的清洗入口

新增文件:

- `src/jobmatch_gnn/data/skill_quality.py`
- `scripts/build_osta_reference.py`
- `tests/test_skill_quality.py`

配置入口:

- `configs/v2_data.yaml -> skill_quality`

默认策略:

1. 删除明确非技能泛词: `自我评价`, `客户`, `学历`, `管理`, `经理`, `员工`, `材料`, `资料`, `or`, `ex`, `cc`, `em` 等。
2. 英文 token 长度 `<3` 时默认删除,但保留技术白名单: `c`, `c++`, `c#`, `r`, `go`, `bi`, `ui`, `ux`, `hr` 等。
3. 对 ASCII 技能启用边界匹配,避免 `go` 命中 `golang`, `or` 命中普通英文片段。
4. 将 OSTA 职业、工种、职业类目识别为 `occupation_or_job_type`,不进入 Skill 词表。
5. 将 `沟通能力`, `责任心`, `团队合作`, `执行力` 等归为 `transversal`,仍可保留,但岗位侧匹配权重降到 `0.35`,避免支配图结构。

## 5. OSTA 参照词表构建

本地生成命令:

```bash
python scripts/build_osta_reference.py \
  --output data/external/osta_reference_terms.json \
  --max-subordinate-requests 600 \
  --max-details 500
```

输出在 `data/external/` 下,被 `.gitignore` 忽略。核心字段:

| 字段 | 用途 |
|---|---|
| `career_categories` | 职业大/中/小类,用于过滤职业类目误入 Skill |
| `occupation_terms` | 职业名称,用于转 JobType/Occupation |
| `occupations[].works` | 工种名称,用于转 JobType/Occupation |
| `occupations[].task` | 主要工作任务,用于抽取任务型技能候选 |
| `standard_terms` / `standards` | 国家职业标准名称和 PDF 元数据 |

后续增强:

- 下载标准 PDF 后只抽取“技能要求”“相关知识”章节。
- 把这些术语写成 `reference_task` 或 `hard_or_knowledge` 类型。
- 对职位/职业名称只做岗位方向特征,不做 Skill 节点。

## 6. v3 技能类型 Schema

建议 `skill_vocab.json` 从 `skill -> id` 升级为:

```json
{
  "python": {
    "id": 1,
    "canonical": "python",
    "skill_type": "hard_or_knowledge",
    "source": ["user_tag", "job_text"],
    "quality_flag": "keep",
    "user_df": 0,
    "job_df": 0,
    "osta_refs": []
  }
}
```

类型定义:

| 类型 | 处理 |
|---|---|
| `hard_or_knowledge` | 进入 Skill 节点,正常权重 |
| `reference_task` | 进入 Skill 节点,来自 OSTA 任务/标准章节 |
| `language` | 进入 Skill 节点,如 英语, 日语 |
| `transversal` | 可进入 Skill,但降权并限制每个 job 最多 3 个 |
| `occupation_or_job_type` | 不进 Skill,转 JobType/Occupation |
| `generic_non_skill` | 删除 |
| `english_noise` | 删除 |

## 7. 评估协议

每次重建 KG 后必须报告:

| 指标 | 目标 |
|---|---|
| skill_vocab size | 不追求越小越好,重点看有效率 |
| top-50 高频 Skill 人审有效率 | >= 85% |
| 短英文噪声数 | `or/ex/cc/em` 等为 0 |
| occupation/job_type 误入 Skill 数 | top 高频中为 0 |
| avg job skills | 10-20 |
| zero-skill job ratio | < 2% |
| SPC-HGT NDCG@10 | 不低于 v2,最好提升 |
| 解释证据质量 | `matched_skills` 中职业名/泛词显著减少 |

对比实验:

```text
Rule v2 vs Rule v3
LightGCN v2 vs LightGCN v3
SPC-HGT v2 vs SPC-HGT v3
SPC-HGT v3 w/o OSTA reference
SPC-HGT v3 w/o transversal downweight
```

## 8. 风险与边界

- OSTA 是官方职业分类和职业标准体系,不是完整招聘市场技能词典;互联网/软件工具名仍需从招聘语料和技术白名单补充。
- OSTA 职业名有时也可表示资质或能力,例如“电工”。短期先作为职业/工种处理,后续可通过上下文判断映射为 certification 或 occupation。
- PDF 章节抽取需要做版式解析和人审抽样,不能直接把 PDF 全文切词塞回 Skill。
- `transversal` 不是无用信息,但它不应该在 Skill 路径中与 hard skill 等权。

## 9. 来源

- OSTA 技能人才评价工作网: https://www.osta.org.cn/
- OSTA 国家职业标准查询接口: https://www.osta.org.cn/api/public/skillStandardList
- OSTA 职业分类树接口: https://www.osta.org.cn/api/client/get/tree
- Chinese-SkillSpan: https://arxiv.org/abs/2604.23009
- China's First Workforce Skill Taxonomy: https://arxiv.org/abs/2001.02863
