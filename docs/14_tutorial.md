# 14 端到端教程:从原始数据到人岗匹配模型

> 面向第一次接触本项目的同学。跟着本文从零跑通整个 v2 管线,并理解每一步在做什么、为什么这么做。
> 前置:Linux、Python 基础;一块 NVIDIA GPU(无 GPU 也能跑 baseline)。

## 0. 项目是做什么的

输入一个候选人画像(技能、期望城市/岗位、学历、经验),输出 Top-K 推荐岗位 + 结构化解释(匹配了哪些技能、缺哪些技能、为什么推荐)。本质是 **知识图谱上的推荐/链接预测** 问题,核心路径:

```text
候选人 --掌握--> 技能 <--要求-- 岗位
```

## 1. 环境准备

```bash
cd /home/ubuntu/data/personas-core
uv venv --python 3.12 .venv                       # 创建虚拟环境
uv pip install --python .venv/bin/python --torch-backend cu130 \
    -e . torch jieba sentence-transformers torch_geometric \
    scipy scikit-learn pandas pyarrow pyahocorasick pytest ruff
export HF_ENDPOINT=https://hf-mirror.com           # 国内访问 HuggingFace 模型
```

## 2. 认识数据(5 分钟)

```bash
.venv/bin/python scripts/inspect_data.py   # 打印三张表的字段与样例
```

三张表的关系:

```text
table1_user (4,500 候选人)  ──user_id──┐
                                        ├── table3_action (70 万行为: browsed/delivered/satisfied)
table2_jd  (26.9 万岗位)   ──jd_no────┘
```

关键认知:
- `user.experience` 是竖线分隔的**已抽取技能标签**(如 `预算编制|标书制作|工程造价`),这是构建技能词表的金矿;
- `delivered`(投递)是强意图信号,`browsed`(浏览)噪声大 —— 所以正样本定义为 delivered∪satisfied;
- 行为表没有时间戳 → 只能用行序近似时间做 leave-last-out 划分(局限,要诚实写进报告)。

## 3. 数据预处理与建图(阶段 A)

```bash
.venv/bin/python -m jobmatch_gnn.data.preprocess_v2 --config configs/v2_data.yaml
```

做了四件事(详见 `docs/11_data_kg_v2.md`):
1. 全量解析三张表,规范化字段;
2. 用 user 技能标签建**技能词表**(频次过滤),再用 Aho-Corasick 自动机在 JD 文本里匹配同一词表 → user/job 技能对齐到同一套 id;
3. leave-last-out 划分,写入 `interactions.parquet` 的 `split` 列;
4. 组装 6 类节点异构图,缓存到 `data/processed/`。

> 思考题:为什么 job 的技能不直接对 JD 分词?(答案:开放分词产生百万级噪声节点,v1 就是这么失败的——词表对齐保证 Candidate→Skill←Job 路径两端说的是同一个技能。)

## 4. 文本嵌入(阶段 B)

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m jobmatch_gnn.text.encode_sbert --config configs/v2_data.yaml
```

用中文句向量模型 `BAAI/bge-small-zh-v1.5` 把用户画像、JD、技能短语编码为 512 维向量,缓存为 `.npy`。它们有两个用途:SBERT baseline 直接做余弦检索;GNN 用它做节点初始特征(这叫"文本增强的图学习")。

## 5. 跑 baseline(阶段 C)

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m jobmatch_gnn.training.train_v2 \
    --config configs/v2_baselines.yaml
cat experiments/runs/v2_baselines/metrics.csv
```

为什么要这么多 baseline?**没有强对照,主模型的任何数字都没有意义。** 五个 baseline 各代表一类方法的上限:流行度(无个性化)、规则(纯先验)、BM25(词面匹配)、SBERT(语义匹配)、LightGCN(协同过滤)。主模型必须全部打败它们(`docs/13`)。

## 6. 训练主模型 SPC-HGT v2(阶段 D)

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m jobmatch_gnn.training.train_v2 \
    --config configs/v2_spc_hgt.yaml
```

模型结构看 `docs/12_spc_hgt_v2_design.md`,三个关键创新按重要性:
1. **技能路径注意力**:对共享技能算注意力 → 既提分又可解释;
2. **行为分级对比学习**:satisfied/delivered/browsed 用不同温度做 InfoNCE;
3. **语义难负样本课程**:专门学会区分"看起来匹配但用户不投"的岗位。

训练日志会打印每个 epoch 的 valid NDCG@10,早停后自动在 test 上评一次并落盘。

## 7. 消融与报告(阶段 E)

```bash
.venv/bin/python -m jobmatch_gnn.training.train_v2 --config configs/v2_ablations.yaml
```

读结果:`docs/15_results_v2.md`。读法:先看主表(所有模型同协议对比),再看消融表(每行减一个组件,掉多少分 = 该组件贡献多大)。

## 7.5 可解释性:KG 证据 + 可选 LLM 表达

```bash
# 训练时存 checkpoint(配置里加 checkpoint_path),然后:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m jobmatch_gnn.explanation.explain_demo \
    --checkpoint experiments/runs/v2_spc_hgt/spc_hgt_v2.pt --num-users 3 --topk 3
# 想开 LLM 自然语言层(OpenAI 兼容 API):
export LLM_BASE_URL=https://api.deepseek.com LLM_API_KEY=sk-xxx LLM_MODEL=deepseek-chat
```

两层设计(`docs/16_llm_explainability.md`):
1. **事实层**(`kg_evidence.py`):从训练好的模型抽取匹配技能(带路径注意力权重)、缺口技能、图路径、规则理由 —— 纯计算,100% 忠实,无 LLM 也完整可用。
2. **表达层**(`llm_explainer.py`,可选):把事实层 JSON 喂给 LLM 转成通顺中文,输出后做**技能白名单忠实性校验**(LLM 提到的技能必须在证据内,否则丢弃并回退模板)。无 API key 时自动走确定性模板。

核心原则:KG 是唯一事实来源,LLM 只是可替换、可校验、可回退的表达层 —— 避免文献中 LLM 自由生成的幻觉问题。

## 8. 常见坑

| 坑 | 症状 | 解法 |
|---|---|---|
| HF 下载超时 | `MaxRetryError huggingface.co` | `export HF_ENDPOINT=https://hf-mirror.com` |
| Python 端 SSL 握手超时 | hf-mirror 也反复 Retry | 用 curl 把模型文件手动下到本地目录(如 `/home/ubuntu/data/models/bge-small-zh-v1.5`),把 `configs/v2_data.yaml` 的 `sbert.model` 改成该路径 |
| 验证集指标恒为 0 | 早停失效,停在第 1 个 epoch | 检查评估时是否把该用户自己的 held-out 正例也剔除了(本项目曾踩过:`two_stage_rankings` 的 `exclude_valid` 必须在验证时设为 False) |
| 信息泄漏 | test 指标异常高 | 检查图里是否混入 valid/test 交互边 |
| LightGCN 不收敛 | NDCG≈0 | 检查是否用了全量 train 正样本、epoch 是否够(v1 就是只喂了 583 条) |
| sampled metrics | 不同机器结果不可比 | 永远全库排序(docs/13 §1) |

## 9. 下一步去读什么

- 图谱 schema:`docs/11_data_kg_v2.md`
- 模型推导:`docs/12_spc_hgt_v2_design.md`
- 论文背景:`hr_kg_gnn_skill_pack/references/reading_priority.md`(LightGCN → HGT → KGAT 顺序读)
