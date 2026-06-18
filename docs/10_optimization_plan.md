# 10 项目优化总体方案(v2 Master Plan)

日期:2026-06-12
状态:已批准,按此执行
关联文档:`11_data_kg_v2.md`、`12_spc_hgt_v2_design.md`、`13_evaluation_protocol.md`、`14_tutorial.md`、`15_results_v2.md`(结果产出后补充)

## 1. 现状审计(为什么说完整性低)

对 2026-06-04 的 `baseline_sample_gpu` 运行与现有代码(`src/jobmatch_gnn/`)审计后,确认四类问题:

### 问题 1:缺乏强有力的 baseline

| 现有 baseline | 缺陷 |
|---|---|
| BM25 | 中文分词用字符 n-gram 近似,NDCG@10 仅 0.039,远低于合理水平,不构成有效对照 |
| semantic_hash | 哈希 TF-IDF 占位实现,NDCG@10 = 0.005,接近随机 |
| LightGCN | 只用了 583 条训练正样本(全量约 7 万),NDCG@10 = 0.005,等于没训 |
| (缺失) | 没有真实 SBERT、没有 Popularity 下界、没有异构 GNN 对照(HGT/KGAT) |

结论:目前唯一有效的对照是 Rule(0.121),"主模型超过最强 baseline"的结论建立在对照普遍失效的基础上,不可信。

### 问题 2:算法/模型简陋

现有 `SPC-HGT-lite`(`models/torch_gnn.py`)实质是 **LightGCN + 7 维规则特征 MLP**:

- 没有异构图:不存在 Skill/City/Industry 节点,消息传递只在 Candidate-Job 二部图上;
- 没有文本语义:未使用任何预训练文本编码器;
- 没有对比学习、没有难负样本(负样本为全局均匀采样);
- "skill-path" 只是 `log(1+共享token数)` 一个标量,不是图上的路径建模。

与 skill pack 中 SPC-HGT MatchNet 的设计(异构图 Transformer + 技能路径 + 对比学习 + 难负样本)差距很大。

### 问题 3:数据集/知识图谱简陋

- 训练只读取 action 表前 12 万行 → 实际只覆盖 784/4500 个用户、950/98174 条正样本(<1%);
- 词表污染:`tokenize_text` 把中文切成 2/3-gram,导致 kg_output 中出现 **260 万个"技能"节点**,绝大部分是无意义碎片("的工作"、"负责人"之类),图谱不可用;
- job 侧技能完全来自 JD 文本的 n-gram,与 user 侧技能词表没有对齐,Candidate→Skill←Job 路径命中率低且噪声大。

### 问题 4:缺乏可学习的文档

`docs/` 下只有 3 份文件(策略、一次实验报告、前端协议),没有从数据→图谱→模型→训练→评估的逐步教程,新人无法复现。

## 2. 优化目标与验收标准

总目标:**算法的创新性与有效性**。

| # | 目标 | 验收标准 |
|---|---|---|
| G1 | 强 baseline 体系 | Popularity / Rule v2 / BM25(jieba) / SBERT(真实) / LightGCN(全量) 全部在同一全量划分上跑通,且 BM25、SBERT、LightGCN 显著高于随机 |
| G2 | 主模型有效 | SPC-HGT v2 在 NDCG@10 上超过最强 baseline,并给出消融证明每个创新组件有贡献 |
| G3 | 数据/图谱 v2 | 技能词表从 260 万降到数万级的干净词表;训练覆盖全部 4500 用户与全量正样本;Candidate-Skill-Job 异构图可加载、可训练 |
| G4 | 文档完整 | docs/10–15 六份文档 + 教程,新人按 `14_tutorial.md` 可端到端复现 |

## 3. 技术路线总览

```text
阶段 A 数据与图谱 v2(docs/11)
  A1 全量加载 70 万 action,正样本 = delivered ∪ satisfied(browsed 作弱信号)
  A2 技能词表 v2:以 user.experience 的预抽取标签为种子词表(频次过滤)
  A3 job 技能抽取:用词表对 jd_title+description 做 Aho-Corasick 多模式匹配
  A4 leave-last-out 按行序划分 train/valid/test(每用户最后 1 个正样本为 test)
  A5 产出缓存:data/processed/*.parquet + 异构图张量

阶段 B 文本语义层(docs/12 §3)
  B1 BAAI/bge-small-zh-v1.5(经 hf-mirror)GPU 批量编码 26.9 万 JD、4500 用户画像、技能词表
  B2 嵌入缓存为 npz,供 SBERT baseline 与 GNN 节点初始化共用

阶段 C 强 baseline(docs/13)
  C1 Popularity(频次下界)
  C2 Rule v2(干净技能词表上的覆盖率 + 结构化匹配)
  C3 BM25(jieba 分词,稀疏矩阵实现,全库检索)
  C4 SBERT 双塔零样本(余弦检索)
  C5 LightGCN(全量交互、调参、多 epoch)

阶段 D 主模型 SPC-HGT v2(docs/12)
  D1 异构图:Candidate/Job/Skill/City/Industry/JobType 六类节点
  D2 关系感知消息传递(2 层),节点特征用 SBERT 嵌入初始化
  D3 技能路径注意力:对 (c,j) 的共享技能集合做注意力聚合 → 可解释路径得分
  D4 损失:BPR + BCE + InfoNCE 对比对齐
  D5 难负样本:SBERT 相似但无正反馈的 job + 同城同类岗位负样本
  D6 解释输出:matched_skills(带注意力权重)/ missing_skills / graph_paths / reasons

阶段 E 消融与报告(docs/15)
  E1 消融:-路径注意力 / -对比损失 / -难负样本 / -文本特征 / -异构图(退化为 LightGCN)
  E2 多 seed(≥3)报告均值
  E3 docs/15_results_v2.md + CHANGELOG + MODEL_CARD 更新
```

## 4. 创新点(相对已有工作)

1. **技能路径注意力评分(Skill-Path Attention)**:不是把路径数当标量特征,而是对 `Candidate→Skill←Job` 的每条路径,用技能嵌入 + 候选人/岗位上下文计算注意力权重并聚合成路径表征;注意力权重直接作为解释输出(哪个技能、贡献多大),实现"评分即解释"。
2. **行为分级对比对齐**:利用 browsed < delivered < satisfied 的天然偏序,satisfied 对作为强正对、delivered 为正对、browsed 为弱正对参与 InfoNCE 温度加权,而不是简单二值化。
3. **语义难负样本课程**:用 SBERT 检索"语义最像但无投递行为"的岗位作为难负样本,训练中按课程(先易后难)混入,直接攻击人岗匹配中"看起来匹配但实际不投"的核心难点。
4. **词表对齐的轻量 KG 构建**:user 侧人工标签作为技能本体种子,job 侧用多模式匹配对齐到同一词表,避免开放抽取造成的百万级噪声节点——图谱质量优先于规模。

## 5. 里程碑与产出物

| 里程碑 | 产出 |
|---|---|
| M1 方案文档 | docs/10–14(本批) |
| M2 数据 v2 | `src/jobmatch_gnn/data/full_loader.py`、`scripts/preprocess_v2.py`、`data/processed/` 缓存 |
| M3 嵌入 | `scripts/encode_sbert.py`、`data/processed/emb_*.npy` |
| M4 baseline | `experiments/runs/v2_baselines/metrics.csv` |
| M5 主模型 | `models/spc_hgt_v2.py`、`experiments/runs/v2_spc_hgt/` |
| M6 报告 | `docs/15_results_v2.md`、CHANGELOG、MODEL_CARD |

## 6. 风险与对策

- **PyG 与 torch 2.12 兼容**:torch_geometric 纯 Python 安装即可用(HGTConv 无需编译扩展);若不兼容则自实现关系感知卷积(设计已不依赖 PyG 专有算子)。
- **26.9 万 job 全库排序开销**:嵌入类模型用矩阵乘(GPU 上毫秒级);BM25 用 scipy 稀疏矩阵;Rule 只对候选池(技能倒排召回)排序。
- **无时间戳**:保持行序 leave-last-out 并在所有报告中声明该局限。
- **HF 直连被墙**:统一 `HF_ENDPOINT=https://hf-mirror.com`。
