# 00 路径与功能拆分

## 1. 顶层目录

```text
project_root/
  configs/             # 训练、评估、推理、模型和 baseline 配置
  src/jobmatch_gnn/    # 核心算法代码
  tests/               # 单元测试和集成测试
  docs/                # 技术文档和接口文档
  references/          # 单独维护参考资料
  experiments/         # 实验记录、指标表、消融结果
  skills/hr_kg_gnn/    # Agent Skill
```

## 2. `src/jobmatch_gnn/` 模块拆分

| 路径 | 功能 | 关键产物 |
|---|---|---|
| `data/schema.py` | 输入输出数据结构 | `CandidateProfile`, `JobProfile`, `RecommendationResponse` |
| `data/validate.py` | 数据校验 | 空字段、类型、枚举、ID 唯一性校验 |
| `data/preprocess.py` | 文本和结构化字段预处理 | 清洗、去重、归一化 |
| `data/skill_normalizer.py` | 技能标准化 | 同义词、别名、大小写、中文英文映射 |
| `data/build_graph.py` | 构建异构图 | PyG `HeteroData` 或 DGL heterograph |
| `data/negative_sampling.py` | 负样本构造 | random、same_city、same_industry、BM25/SBERT hard negatives |
| `data/split.py` | 数据切分 | 时间切分、按候选人分组切分、冷启动切分 |
| `features/text_encoder.py` | 文本语义特征 | SBERT/BERT embedding，缓存机制 |
| `features/structured_encoder.py` | 结构化特征 | 学历、经验、城市、行业 embedding |
| `features/path_features.py` | 图谱路径特征 | 技能路径数、技能覆盖、缺失技能、技能相似度 |
| `features/rule_features.py` | 规则特征 | 技能/学历/经验/城市匹配分 |
| `models/spc_hgt.py` | 主模型 | SPC-HGT Encoder |
| `models/scorer.py` | 打分层 | recall scorer、rank scorer |
| `models/losses.py` | 损失函数 | BPR、BCE、InfoNCE、KG reconstruction |
| `models/lightgcn.py` | 图 baseline | LightGCN |
| `models/kgcn.py` | KG baseline | KGCN |
| `models/kgat.py` | KG attention baseline | KGAT |
| `baselines/rule_matcher.py` | 规则 baseline | Rule score |
| `baselines/bm25_matcher.py` | BM25 baseline | 稀疏文本召回 |
| `baselines/sbert_matcher.py` | SBERT baseline | 语义召回 |
| `baselines/node2vec_matcher.py` | Node2Vec baseline | 图嵌入召回 |
| `training/train.py` | 训练入口 | CLI/API |
| `training/trainer.py` | 训练循环 | checkpoint、early stopping、AMP |
| `evaluation/metrics.py` | 指标 | Recall@K、NDCG@K、MRR、AUC |
| `evaluation/evaluate.py` | 评估入口 | 对比所有模型 |
| `evaluation/ablation.py` | 消融实验 | 自动生成 ablation 表 |
| `inference/index_builder.py` | 向量索引 | FAISS index |
| `inference/ranker.py` | Top-K 排序 | 召回 + 精排 |
| `inference/service.py` | 服务入口 | FastAPI/Streamlit 调用 |
| `explanation/explain.py` | 解释生成 | matched_skills、missing_skills、graph_paths |
| `utils/` | 通用工具 | seed、logging、config、IO |

## 3. `docs/` 文档拆分

| 文档 | 必须回答的问题 |
|---|---|
| `01_algorithm_api.md` | 算法怎么调用？输入输出是什么？错误码是什么？ |
| `02_kg_schema.md` | 图谱有哪些实体、关系和边特征？ |
| `03_model_design.md` | SPC-HGT 如何建模？损失函数是什么？ |
| `04_training_evaluation.md` | 怎么训练、切分、评估？ |
| `05_baseline_and_ablation.md` | baseline 怎么比？消融怎么做？ |
| `06_innovation_and_advantage.md` | 创新性和优势在哪里？ |
| `07_deployment.md` | 怎么部署，如何缓存，延迟如何控制？ |
| `08_versioning_and_workflow.md` | 版本怎么维护？PR 怎么合并？ |
| `09_document_maintenance.md` | 文档如何更新？ |

## 4. `references/` 拆分

| 文件 | 用途 |
|---|---|
| `references.md` | 人岗匹配、GNN、KG 推荐、工程博客的主参考库 |
| `references.csv` | 机器可读版本，便于 Agent 检索和筛选 |
| `reading_priority.md` | 阅读优先级和采用建议 |
| `search_queries.md` | 后续持续更新 references 的检索式 |
| `uploaded_source_notes.md` | 本轮上传材料中的可复用结论 |
