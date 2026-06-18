# Changelog

## v2.0.0 - 2026-06-12

### 数据/图谱
- 新增全量预处理管线 `jobmatch_gnn.data.preprocess_v2`:覆盖全部 4,500 用户、61,297 条正样本(v1 仅 784 用户/950 正样本)。
- 技能词表 v2:以用户 experience 标签为种子 + Aho-Corasick 对齐 JD 文本,4,765 个干净技能(v1 的 kg_output 为 260 万噪声节点,已弃用)。
- leave-last-out 划分写入 `data/processed/interactions.parquet`,所有模型共享。

### 模型
- 新增主模型 `SPC-HGT v2`(`models/spc_hgt_v2.py`):C/J/S 异构消息传递 + SBERT 文本特征 + ID 嵌入融合 + 技能路径注意力 + 行为分级 InfoNCE;两阶段(召回+重排)推理。
- 新增 `LightGCN v2`(全量交互,修复 v1 仅训 583 对导致接近随机的问题)。
- 新增强 baseline:Popularity、Rule v2、BM25(jieba)、SBERT(bge-small-zh-v1.5,本地模型目录加载)。
- 负结果:语义难负样本课程在本数据集为负贡献(曝光受限导致伪负样本),已从最终模型移除,保留 `use_hard_neg` 开关(docs/15 §4)。

### 评估
- 新增统一全库排序评估 `evaluation/rank_eval.py`(91,479 岗位候选池,Recall/Precision/NDCG/HitRate@{10,50} + MRR),含 pytest 锚定单测。
- 修复验证集评估将自身 held-out 正例剔除导致 valid 指标恒 0 的 bug(`exclude_valid`)。
- 结果:SPC-HGT v2 三 seed NDCG@10 = 0.0266±0.0005,超最强 baseline LightGCN(0.0233±0.0011)+14.2%,配对 t 检验 p=0.018。详见 `docs/15_results_v2.md`。

### 可解释性(KG + LLM 两层)
- 新增 `explanation/kg_evidence.py`:从 SPC-HGT v2 checkpoint 抽取忠实证据(匹配技能 + 路径注意力权重 / 缺口技能 / 图路径 / 规则理由),纯本地计算无幻觉。
- 新增 `explanation/llm_explainer.py`:OpenAI 兼容 API 表达层 + 技能白名单忠实性校验 + 模板回退;无 API key 自动走确定性模板。
- 新增 `explanation/explain_demo.py` 端到端 demo;`tests/test_explanation.py` 验证护栏能拒绝"真实但不在证据内"的幻觉技能。
- 设计与 SOTA 对标见 `docs/16_llm_explainability.md`。

### 文档
- 新增 docs/10(优化总方案)、11(数据/图谱 v2)、12(模型设计)、13(评估协议)、14(端到端教程)、15(结果报告)、16(KG+LLM 可解释性分析与增强策略)。

## v1 - 2026-06-04
- 初始原型:采样数据上的 Rule/BM25/semantic_hash/LightGCN/SPC-HGT-lite,见 `docs/algorithm_strategy.md`。
