# Model Card: SPC-HGT v2 (Skill-Path Contrastive Heterogeneous GNN)

版本:v2.0.0(2026-06-12)
代码:`src/jobmatch_gnn/models/spc_hgt_v2.py`、`src/jobmatch_gnn/training/train_spc_hgt_v2.py`
配置:`configs/v2_spc_hgt.yaml`

## 任务

输入候选人画像,输出 Top-K 岗位推荐及结构化解释(matched_skills 含注意力权重、missing_skills、graph_paths、reasons)。

## 架构

- 节点:Candidate(4,500)/ Job(91,479)/ Skill(4,765);城市/岗位类型/学历/年限作为输入层嵌入特征。
- 输入特征:bge-small-zh-v1.5 文本嵌入(512d)+ 结构化嵌入 + 可学习 ID 嵌入,投影到 128d。
- 编码器:2 层关系感知 scatter-mean 消息传递(6 种关系,残差 + LayerNorm)。
- 匹配:内积召回 top-500 → [z_c; z_j; 技能路径注意力表征; 技能缺口表征; 7 维规则特征] MLP 重排。
- 损失:BPR + 0.2·BCE + 0.1·行为分级 InfoNCE(satisfied τ=0.07 / delivered τ=0.10)。
- 负采样:均匀负采样 4:1(语义难负课程经消融为负贡献,已禁用,见 docs/15 §4)。

## 训练数据

智联招聘人岗匹配数据(`data/datasets.zip`):4,500 用户、91,479 有行为岗位、53,408 训练正样本(delivered∪satisfied)。行序 leave-last-out 划分(无时间戳,为已声明局限)。

## 评估(全库排序,4,095 测试用户,3 seeds)

| 指标 | SPC-HGT v2 | 最强 baseline(LightGCN) |
|---|---:|---:|
| NDCG@10 | **0.0266 ± 0.0005** | 0.0233 ± 0.0011 |
| Recall@10 | **0.0513 ± 0.0006** | 0.0446 ± 0.0017 |
| Recall@50 | **0.1231 ± 0.0024** | 0.1027 ± 0.0008 |
| MRR | **0.0245 ± 0.0005** | 0.0203 ± 0.0009 |

配对 t 检验(seed 42,按用户 NDCG@10):p = 0.018。完整对比与消融见 `docs/15_results_v2.md`。

## 预期用途与限制

- 用于人岗匹配召回+粗排研究原型;解释输出可支撑可解释推荐。
- 不应直接用于生产决策:无真实时间切分验证、候选池排除零行为岗位、行为数据存在曝光偏差。
- 公平性:未对性别/年龄等敏感属性做审计;上线前必须补充。

## 复现

```bash
.venv/bin/python -m jobmatch_gnn.data.preprocess_v2 --config configs/v2_data.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m jobmatch_gnn.text.encode_sbert --config configs/v2_data.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m jobmatch_gnn.training.train_v2 --config configs/v2_baselines.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m jobmatch_gnn.training.train_v2 --config configs/v2_spc_hgt.yaml
```
