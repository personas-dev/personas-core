# 16 知识图谱 + LLM 可解释性分析与增强策略

日期:2026-06-12
关联:`docs/12`(模型)、`docs/15`(结果)、`src/jobmatch_gnn/explanation/`

本文回答四个问题,并给出可落地的增强策略。所有 SOTA 引用见文末 Sources。

---

## 0. 检索到的 SOTA 坐标(用于对标)

| 工作 | 任务/数据 | 关键结论 | 与我们的关系 |
|---|---|---|---|
| ConFit v2 (ACL 2025 Findings) | resume-job matching,Intellipro/AliYun | hypothetical resume embedding + runner-up 难负;E5-base 上 Recall@100 84.4%、nDCG@100 48.4% | 双塔文本编码 SOTA;**注意它是 sampled/小候选池排序,数字与全库不可比** |
| Fine-Grained Semantics GNN (Entropy 2025, MDPI) | person-job fit,**智联(Zhaopin)** | 把技能/经历做节点、PMI+滑窗建边,较前法提升 4.08%–46.69% | 与我们同数据源、同 GNN 方向,是最直接对标 |
| G-Refer (WWW 2025) | 可解释推荐 | 图检索增强 LLM,把协同路径转成自然语言解释 | LLM 解释的代表范式 |
| Faithful Path Language Modeling (2023+) / "KG reduce hallucination" survey | KG 上可解释推荐 | LLM 在路径推理中会"幻觉"出 KG 中不存在的关系;KG 接地可显著降低幻觉 | 我们 LLM 方案的安全护栏依据 |
| LANTERN (2025) | job-person fit + explanation | 把大 LLM 蒸馏成小模型同时产出匹配与解释 | 生产化路径参考 |
| JobRecoGPT (2023) | 可解释岗位推荐 | LLM 直接生成解释,存在 fabrication/few-shot 问题 | 反面教材:为何要 KG 接地 |
| "On Sampled Metrics" (KDD 2020) / TOIS 2022 revisiting | 评估方法 | sampled metrics 系统性高估且改变模型排序;全库排序更可信但绝对值低得多 | 解释我们 NDCG 绝对值为何低 |

---

## 1. 我们目前的方案是否优于大多数论文结果?

**不能直接比较绝对数字,但在方法学严谨性上我们处于较好水平;在原始效果上属于"合理但不靠前"。** 分三点说明:

### 1.1 绝对指标不可直接对比(关键)
我们的 NDCG@10≈0.027 看起来远低于 ConFit v2 的 nDCG@100≈0.48、或 MovieLens 上常见的 0.3–0.5,但这是**评估口径差异**,不是效果差 20 倍:

- 我们用**全库排序**(91,479 个岗位里排 1 个正例),论文多数用 **sampled metrics**(1 正 + 100/1000 负)或**小候选池**。"On Sampled Metrics"(KDD 2020)明确指出 sampled 会系统性高估且绝对值高得多;全库设定下 NDCG@10 在 0.01–0.05 量级是正常的。
- 我们是 **leave-last-out 单正例**;ConFit v2 的 nDCG@100 分母是多正例、且 K=100。
- 因此把 0.027 和 0.48 并列是错误对比。要对比必须统一口径。

### 1.2 同口径下我们的相对提升与论文相当
- 我们主模型相对最强 baseline:**NDCG@10 +14.2%、Recall@50 +19.9%**。
- ConFit v2 相对其 baseline:recall +13.8%、nDCG +17.5%。
- 智联 GNN(MDPI 2025)相对前法:4.08%–46.69%。
- **结论:我们的相对增益落在主流论文区间内,方法是站得住的;但我们尚未与论文做同口径直接对拼。**

### 1.3 我们更严谨的地方,也是论文常被诟病的地方
- 全库排序(无 sampled 高估)、严格防泄漏(test/valid 边不进图)、多 seed + 配对 t 检验(p=0.018)、并诚实报告了一个**负结果**(难负样本课程)。多数论文不会报负结果。

> 一句话:**我们的方案在"同口径相对提升"上不输大多数论文,且评估更可信;但绝对 SOTA 效果(尤其文本侧)还有差距,主要因为我们用的是 bge-small + 轻量异构图,而非 ConFit v2 那样的大编码器 + 假设简历增强。**

---

## 2. 我们的优势主要来自哪里?

消融(docs/15 §3,基准 NDCG@10=0.0273)给出**定量归因**:

| 去掉的组件 | NDCG@10 | 相对掉幅 | 性质 |
|---|---:|---:|---|
| 行为分级对比损失 (InfoNCE) | 0.0183 | **−33.0%** | 训练策略 |
| 技能路径注意力 | 0.0249 | −8.8% | 模型设计 |
| ID 协同嵌入 | 0.0252 | −7.7% | 模型设计 |
| SBERT 文本特征 | 0.0260 | −4.8% | 特征/数据 |

**优势是"多策略融合",但权重不均:训练策略(对比学习)贡献最大,模型设计(路径注意力 + ID 嵌入)次之,文本特征最小。**

具体拆解:
- **最大头是训练策略**:行为分级 InfoNCE 把候选人/岗位嵌入对齐到同一空间,直接决定了召回阶段质量。去掉它掉 1/3,说明"怎么训"比"用什么结构"更关键。
- **模型设计贡献稳定但中等**:异构图上的技能路径注意力(+8.8%)和 ID 协同记忆(+7.7%)各自有效,二者叠加约 16%,是结构带来的净收益。
- **数据/特征是基础但边际小**:干净技能词表(从 260 万→4,765)是让路径注意力"能工作"的前提;但 SBERT 文本特征本身只贡献 4.8%,说明在这个行为稠密的数据上,协同信号 > 文本语义。
- **关键洞察**:把单点都不强的信号(文本弱、LightGCN 中等)通过**对比对齐 + 异构融合**组合起来,才超过任一单一信号——这正是融合的价值。

> 结论:**优势 ≈ 训练策略(对比学习)为主导 + 模型设计(路径/ID)为骨架 + 干净数据为地基的融合**,而非单一来源。

---

## 3. 结合 LLM API 做可解释性增强,可行性高不高?与"直接用 KG 作为解释"相比哪个更好?

### 3.1 两种解释的本质区别

| 维度 | 直接用 KG(我们已有) | LLM API 生成 |
|---|---|---|
| 内容 | 结构化:matched_skills + 注意力权重、missing_skills、graph_paths、规则命中 | 自然语言段落:"推荐这个岗位是因为你的 X/Y 技能匹配,但需补充 Z" |
| 忠实性 | **100% 忠实**(直接来自模型计算,无幻觉) | 有幻觉风险(文献明确:LLM 会编造 KG 中不存在的关系) |
| 可读性 | 对开发者/HR 友好,对求职者偏生硬 | 对终端用户友好,流畅可读 |
| 成本/延迟 | 近 0(本地计算) | 每条请求一次 API 调用,有延迟和费用 |
| 可验证 | 是 | 需额外校验 |

### 3.2 可行性结论:**高,但必须是"KG 接地的 LLM",不是"自由生成的 LLM"**

文献高度一致(G-Refer WWW 2025、Faithful Path LM、"KG reduce hallucination" survey):
- **直接让 LLM 自由解释(如 JobRecoGPT)会 fabrication**,把不存在的技能/关系写进解释,在招聘这种高风险场景不可接受。
- **正确范式是 RAG/接地**:把我们 KG 模块已经算出的结构化证据(matched/missing skills + 注意力权重 + 真实图路径)作为**唯一事实来源**喂给 LLM,LLM 只负责"把结构化证据翻译成通顺的中文",并禁止引入证据外的内容。这样幻觉被 KG 钳制,流畅度由 LLM 提供。

### 3.3 哪种更好?——不是二选一,是分层

**最佳方案 = KG 解释做"事实层 + 兜底",LLM 做"表达层(可选增强)":**

```text
事实层(必须,本地,100% 忠实):
  matched_skills[带权重] / missing_skills / graph_paths / rule_reasons
        │
        ├──→ 直接渲染为结构化卡片(默认输出,无 LLM 也完整可用)
        │
        └──→ (可选)作为 grounded context 调用 LLM API
                 ↓ 严格约束:只能转述,不能新增事实
              自然语言解释段落 + 技能提升建议
                 ↓
              忠实性校验:LLM 输出里提到的每个技能必须 ∈ 事实层集合,
              否则丢弃该句并回退到结构化卡片
```

**判断:**
- 若用户/场景是开发或 B 端 HR 系统 → **KG 结构化解释已足够且更可靠**,LLM 非必需。
- 若是 C 端求职者产品、需要"像人一样解释 + 给提升建议" → **KG 接地 + LLM 表达层** 明显更好,且可行性高(只需一个 OpenAI 兼容 API)。
- **任何情况下都不应抛弃 KG 解释而纯靠 LLM**:那会牺牲忠实性,文献已反复证明其失败模式。

> 结论:可行性高;最优是**分层方案**——KG 做忠实事实与兜底,LLM 做可选的自然语言表达层,中间加一道"证据集合内白名单"忠实性校验。本仓库已落地这套架构(见 §5)。

---

## 4. 进一步增强策略(总纲)

按"投入产出比 + 与创新性/有效性目标对齐"排序:

| 优先级 | 策略 | 攻击的短板 | 预期收益 | 类型 |
|---|---|---|---|---|
| P0 | **解释闭环落地**(KG 证据抽取 + LLM 接地表达 + 忠实性校验) | 可解释性只停在设计 | 可解释性从"有数据"到"有产品级输出" | 已在本轮实现 |
| P1 | **更强文本编码器**(bge-large / E5-large,或 ConFit 式假设简历嵌入) | 文本贡献仅 4.8%,是最弱环节 | 直接拉高召回上限 | 有效性 |
| P2 | **曝光感知的负样本(IPS / 去伪负)** | 难负样本因伪负失败(docs/15 §4) | 让难负真正可用 | 创新性+有效性 |
| P3 | **同口径对标实验**(在 sampled 协议下复跑,与 ConFit/GNN 论文对拼) | 无法与论文直接比 | 把"相对提升"变成"可比的 SOTA 论证" | 论证力 |
| P4 | **LLM-as-judge 解释质量评测** | 解释好坏无量化 | 给可解释性一个可报告指标 | 创新性 |

本轮先落地 **P0**(端到端可解释闭环),并把 P1–P4 写成带验收标准的路线(§6),供后续执行。

---

## 5. P0 已落地内容(本轮)

代码:`src/jobmatch_gnn/explanation/`

- `kg_evidence.py`:从训练好的 SPC-HGT v2 checkpoint 为任意 (user, job) 抽取**忠实证据**——共享技能及其路径注意力权重 α、缺口技能、真实图路径 `Candidate→Skill←Job`、规则命中项。纯本地计算,无幻觉。
- `llm_explainer.py`:OpenAI 兼容 API 客户端。把证据序列化为受控 prompt,要求 LLM **只转述不新增**;返回后做**白名单忠实性校验**(解释中出现的技能必须在证据集合内),不通过则回退到模板化结构解释。无 API key 时自动走纯模板,保证可用。
- `explain_demo.py`:命令行 demo,对若干测试用户打印"结构化卡片 + (可选)LLM 段落 + 忠实性校验结果"。

设计原则(与文献对齐):**KG 是事实唯一来源,LLM 只是可替换的表达层,忠实性可校验、可回退。**

## 6. P1–P4 路线(带验收)

- **P1 强编码器**:把 `encode_sbert.py` 的模型换 `bge-large-zh-v1.5`/`e5-large`,重算嵌入并重训;验收:SPC-HGT v2 NDCG@10 相对当前 +≥5%,且 SBERT 消融掉幅扩大(证明文本贡献上升)。
- **P2 曝光感知负样本**:用 browsed 作为"曝光未投递"的真负代理,或对随机负做 IPS 加权;验收:重新开启难负后 NDCG@10 不低于关闭时(扭转 §4 的负结果)。
- **P3 同口径对标**:实现 sampled(1+100)评估开关,在该口径下报 SPC-HGT v2 与 LightGCN/SBERT,并引 ConFit v2/MDPI-GNN 数字做定性对位;验收:产出 `docs/17_sota_comparison.md`。
- **P4 LLM-as-judge**:用 LLM 对生成解释打分(忠实性/有用性/流畅性),与纯模板解释对比;验收:报告平均分与幻觉率(应 ≈0,因有白名单校验)。

---

## Sources

- ConFit v2 (ACL 2025 Findings): https://aclanthology.org/2025.findings-acl.661/ , https://arxiv.org/abs/2502.12361
- Fine-Grained Semantics-Enhanced GNN for Person-Job Fit (Entropy/MDPI 2025): https://www.mdpi.com/1099-4300/27/7/703 , https://pmc.ncbi.nlm.nih.gov/articles/PMC12294162/
- G-Refer: Graph RAG LLM for Explainable Recommendation (WWW 2025): https://dl.acm.org/doi/10.1145/3696410.3714727
- Faithful Path Language Modeling over KG: https://arxiv.org/pdf/2310.16452
- Can Knowledge Graphs Reduce Hallucinations in LLMs? A Survey: https://arxiv.org/html/2311.07914v2
- Path-based summary explanations for graph recommenders: https://arxiv.org/pdf/2410.22020
- LANTERN: Distillation of LLMs for Job-Person Fit and Explanation: https://arxiv.org/pdf/2510.05490
- JobRecoGPT: Explainable job recommendations using LLMs: https://arxiv.org/pdf/2309.11805
- On Sampled Metrics for Item Recommendation (KDD 2020): https://dl.acm.org/doi/pdf/10.1145/3394486.3403226
- A Revisiting Study of Appropriate Offline Evaluation for Top-N Recommendation (TOIS 2022): https://dl.acm.org/doi/full/10.1145/3545796
