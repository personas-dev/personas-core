# 19 策略创新点与 SOTA 判断

日期:2026-06-20

关联: `docs/12_spc_hgt_v2_design.md`, `docs/15_results_v2.md`, `docs/16_llm_explainability.md`, `docs/17_chinese_skill_ontology_cleaning.md`, `docs/18_osta_kg_pipeline_results.md`

## 1. 结论摘要

当前方案可以概括为:

```text
OSTA-aware 中文技能本体清洗
  + Candidate-Skill-Job 异构知识图谱
  + Skill-Path Contrastive HGT 排序模型
  + 全库排序评估
  + KG 接地 LLM 可解释性
```

在本项目当前全库排序协议下,SPC-HGT 是**内部最优方案**:

- v2 final: NDCG@10 = 0.0266 ± 0.0005,超过最强 baseline LightGCN 的 0.0233 ± 0.0011。
- v2 final 相对 LightGCN:NDCG@10 +14.2%,Recall@10 +15.0%,Recall@50 +19.9%。
- seed 42 用户级配对 t 检验:p = 0.018,达到当前文档定义的有效性门槛。
- OSTA job20 单 seed:NDCG@10 = 0.0267,在更稀疏 KG 上保持并略高于 OSTA job30 的 0.0258。

但是,**目前不能严谨宣称公开领域 SOTA**。原因不是模型无效,而是缺少与公开方法在同一数据、同一候选池、同一指标、同一切分上的直接对比。更准确的表述是:

> 当前策略在 personas-core 内部全库排序协议上达到项目内 SOTA;在公开招聘推荐/人岗匹配领域属于 SOTA-aligned 的方法组合,但尚未证明为 public SOTA。

## 2. 当前策略的核心创新点

### 2.1 面向中文招聘的 OSTA-aware 技能本体清洗

v2 已经把 v1 的 260 万碎片技能节点压缩到约 4,765 个,解决了“节点爆炸”。后续 OSTA 策略进一步引入中文官方职业分类/职业标准作为参照,把职业名、工种、岗位方向与技能节点拆开:

```text
职业/工种/岗位方向 -> JobType / Occupation
硬技能/知识/任务能力 -> Skill
泛词/招聘条件/短英文噪声 -> drop 或降权
transversal competence -> 保留但低权重、限额
```

创新性不在于“用了一个词表”,而在于把中文招聘文本中的三类常见混淆拆开:

| 混淆项 | 旧问题 | 当前策略 |
|---|---|---|
| 职业名 vs 技能 | `经理`, `员工`, `工程` 等容易进入 Skill | OSTA 职业/工种参照门,转 JobType 或剔除 |
| 软素质 vs 硬技能 | `沟通`, `责任心`, `团队领导` 支配高频边 | transversal 降权,后续按 job 限额 |
| 英文短 token | `or`, `ex`, `cc`, `em` 误命中 | 英文最短长度 + 技术白名单 + 边界匹配 |

job20 复验显示,降低 `max_job_skills` 从 30 到 20 后:

| 指标 | OSTA job30 | OSTA job20 |
|---|---:|---:|
| avg_job_skills | 26.60 | 18.94 |
| job-skill edges | 约 2.43M | 1.73M |
| SPC-HGT NDCG@10 | 0.0258 | 0.0267 |

这说明降密度没有破坏排序信号,反而让主模型在单 seed 下略有收益。

### 2.2 Skill-Path Contrastive HGT:把“可解释路径”纳入排序

模型不是单纯的文本相似度,也不是纯协同过滤。核心路径是:

```text
Candidate -> HAS_SKILL -> Skill <- REQUIRES_SKILL <- Job
```

SPC-HGT 的创新点是把这条路径同时用于三件事:

1. 排序:路径注意力作为候选人-岗位匹配证据。
2. 训练:行为分级 InfoNCE 对齐候选人与岗位表示空间。
3. 解释:输出 `matched_skills`, `missing_skills`, `graph_paths`, `local_subgraph`。

消融结果支持该设计:

| 去掉组件 | NDCG@10 | 相对掉幅 |
|---|---:|---:|
| 行为分级 InfoNCE | 0.0183 | -33.0% |
| 技能路径注意力 | 0.0249 | -8.8% |
| ID 协同嵌入 | 0.0252 | -7.7% |
| SBERT 文本特征 | 0.0260 | -4.8% |

因此当前收益主要来自“对比学习训练策略 + 路径结构 + 协同 ID 记忆”的组合,而不是单一模块。

### 2.3 暴露偏差下的难负样本克制

很多文本匹配工作会使用“语义相似但未交互”的样本作为 hard negative。当前项目实验证明这在本数据上是负贡献:

- 开启语义难负样本课程后,NDCG@10 从 0.0273 降到 0.0239。
- 原因是招聘行为是曝光受限的:未投递不等于不匹配,很可能只是没看到。

这形成一个重要策略:

```text
没有 impression/exposure log 时,不把“语义相似但未交互”直接当真负。
```

这个负结果本身很有价值。它把模型策略从“盲目追难负”改成“曝光感知负采样”,避免把潜在好岗位误惩罚。

### 2.4 全库排序评估,避免 sampled metrics 高估

当前主结果在 91,479 个岗位上做全库排序,每个 test user 只有一个 held-out positive。这比 sampled evaluation 更难,绝对指标也会更低,但更接近真实召回场景。

这点是方法学创新/严谨性优势:

- 不用 1 正 + 100 负的小池子抬高指标。
- 同一 split 上比较 Rule/BM25/SBERT/LightGCN/SPC-HGT。
- 可训练模型做 3 seeds 和用户级显著性检验。

因此,NDCG@10 = 0.0266 不能直接和 sampled 或小候选池论文中的 0.3/0.5 指标比较。

### 2.5 KG 接地 LLM 解释:LLM 只做表达层

当前可解释性策略不是让 LLM 自由编写推荐理由,而是:

```text
KG / 模型证据
  -> matched_skills, missing_skills, graph_paths, local_subgraph, reasons
  -> LLM 只基于证据生成 HTML 解释
  -> 证据白名单校验
  -> 不通过则回退结构化解释
```

创新点在于把 LLM 从“事实来源”降级为“表达层”。这和 G-Refer、PEARLM 等图接地解释方向一致,适合招聘这种不能编造技能、薪资、岗位要求的场景。

## 3. 与公开 SOTA 坐标的关系

| 公开方向 | 代表工作 | 对我们的启发 | 是否已直接超过 |
|---|---|---|---|
| 文本双塔/对比学习人岗匹配 | ConFit v2 | 假设简历嵌入、runner-up hard negative、强文本编码器 | 否,未同数据同指标复现 |
| 招聘历史 + GNN | PJFCANN | 局部语义 + 全局招聘历史图 | 否,需要同口径复现 |
| 中文技能抽取 | Chinese-SkillSpan | ESCO LSKT 类型:knowledge/skill/transversal/language | 否,但已用于指导清洗策略 |
| 图接地 LLM 推荐解释 | G-Refer,PEARLM | 图检索、路径约束、忠实解释 | 否,当前只完成工程闭环,缺解释指标 |
| 生产级人岗解释 | LANTERN | LLM 蒸馏、多目标 fit + explanation | 否,当前没有在线 A/B 或蒸馏实验 |

当前方案与这些工作是同方向、同趋势,但还不能说已经超过它们。尤其 ConFit v2 使用更强的文本增强和 hard-negative 框架,但其有效前提与我们的曝光受限数据不同;只有在同一协议下复现,才知道谁更强。

## 4. SOTA 判断

### 4.1 内部 SOTA:可以成立

在 personas-core 当前协议中,可以说:

```text
SPC-HGT v2 是当前内部 SOTA。
```

依据:

- 同一数据切分。
- 同一 91,479 岗位全库候选池。
- 与 Rule/BM25/SBERT/LightGCN 全部对比。
- 主模型超过最强 baseline LightGCN。
- 3 seeds 均值和显著性检验已完成。

OSTA job20 当前只能说“单 seed 复验结果达到或略高于 v2 final 均值”,还不能替代主结果,因为它还缺 3 seeds。

### 4.2 公开领域 SOTA:不能成立

不能宣称 public SOTA,原因如下:

1. 数据集是内部 personas-core 数据,没有公开 leaderboard。
2. 公开论文常用不同数据、不同正负构造、不同 K、不同候选池规模。
3. 我们当前是全库排序,公开结果常是 sampled metrics 或小候选池,绝对值不可比。
4. OSTA job20 还只是 single seed。
5. 解释模块缺少自动/人工解释质量指标,不能与 G-Refer、PEARLM、LANTERN 的 explainability 指标直接比较。
6. 还没有复现 ConFit v2/PJFCANN 这类强公开 baseline。

### 4.3 最合适的对外表述

建议使用:

```text
我们提出了一套面向中文招聘场景的 OSTA-aware Skill-Path Contrastive HGT 方法。
在 personas-core 全库排序协议上,该方法超过 Rule/BM25/SBERT/LightGCN 等 baseline,
达到当前项目内最佳效果,并提供 KG 接地的可解释推荐证据。
```

不建议使用:

```text
本方法已经达到招聘推荐领域 SOTA。
```

可以使用但要加限定:

```text
该方法是 SOTA-aligned 的中文人岗匹配 KG-GNN 策略,但 public SOTA 仍需同口径公开基准验证。
```

## 5. 要把“内部 SOTA”推进到“可声明 public SOTA”的最短路径

| 优先级 | 任务 | 验收标准 |
|---|---|---|
| P0 | OSTA job20 跑满 3 seeds | NDCG@10 均值和标准差不低于 v2 final |
| P1 | 实现 sampled(1+100,1+1000)评估开关 | 同时报告 full-catalog 和 sampled,解释不可比性 |
| P2 | 复现 ConFit v2/PJFCANN 关键 baseline | 在 personas-core 同 split 上比较 |
| P3 | 加强文本编码器 | bge-large/e5-large 或 ConFit 式 hypothetical resume,证明文本消融贡献提升 |
| P4 | 引入 exposure-aware negative | 有 browsed/impression 后重启 hard negative,验证不再伤害排序 |
| P5 | 解释质量评测 | 忠实性、覆盖率、幻觉率、人工有用性评分 |
| P6 | 公开可复现实验包 | 配置、脚本、数据脱敏或公开替代数据集 |

完成 P0-P2 后,可以较强地声明“在 personas-core 协议上优于强公开 baseline”。完成 P3-P6 后,才有机会支撑更广义的 SOTA 论证。

## 6. Sources

- ConFit v2: https://arxiv.org/abs/2502.12361
- PJFCANN: https://arxiv.org/abs/2206.09116
- Chinese-SkillSpan: https://arxiv.org/abs/2604.23009
- China's First Workforce Skill Taxonomy: https://arxiv.org/abs/2001.02863
- G-Refer: https://arxiv.org/abs/2502.12586
- PEARLM / Faithful Path Language Modeling: https://arxiv.org/abs/2310.16452
- LANTERN: https://arxiv.org/abs/2510.05490
- JobRecoGPT: https://arxiv.org/abs/2309.11805
- Evaluation Metrics for Item Recommendation under Sampling: https://arxiv.org/abs/1912.02263
