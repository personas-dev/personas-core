# 18 OSTA 清洗 KG 构建与模型管线评估

日期:2026-06-20

关联: `docs/17_chinese_skill_ontology_cleaning.md`, `docs/15_results_v2.md`

## 1. 运行摘要

本次基于 OSTA 职业分类/国家职业标准参照词表重建 `data/processed`，重新生成 BGE 嵌入，并沿用原 v2 模型管线评估:

```bash
.venv/bin/python scripts/build_osta_reference.py \
  --output data/external/osta_reference_terms.json \
  --max-subordinate-requests 600 \
  --max-details 500 \
  --page-size 200

.venv/bin/python -m jobmatch_gnn.data.preprocess_v2 --config configs/v2_data.yaml
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m jobmatch_gnn.text.encode_sbert --config configs/v2_data.yaml
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m jobmatch_gnn.training.train_v2 --config configs/v2_osta_baselines.yaml
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m jobmatch_gnn.training.train_v2 --config configs/v2_osta_spc_hgt.yaml
```

运行目录:

- `experiments/runs/v2_osta_baselines`
- `experiments/runs/v2_osta_spc_hgt`

## 2. KG 规模变化

| 指标 | v2 原始 KG | OSTA 清洗 KG | 变化 |
|---|---:|---:|---:|
| users | 4,500 | 4,500 | 0 |
| jobs | 91,479 | 91,479 | 0 |
| positives | 61,297 | 61,297 | 0 |
| train/valid/test positives | 53,408 / 3,794 / 4,095 | 53,408 / 3,794 / 4,095 | 0 |
| skill_vocab | 4,765 | 4,634 | -131 |
| avg_user_skills | 33.37 | 32.09 | -3.8% |
| avg_job_skills | 27.31 | 26.60 | -2.6% |
| zero-skill jobs | 2 | 63 | +61 |

本次质量门加载:

| 参照项 | 数量 |
|---|---:|
| OSTA occupation terms | 2,139 |
| OSTA career categories | 523 |
| OSTA standard terms | 933 |
| OSTA work type terms | 0 |
| explicit drop terms | 29 |
| transversal terms | 21 |

`work_type_terms=0` 是本次运行限制: `--max-details 500` 按职业编码前缀采样,前 500 个职业详情没有返回工种列表。下一轮应优先采样 `workNum > 0` 的职业详情,否则“工种名转 JobType/Occupation”的清洗没有充分发挥。

## 3. 高频噪声检查

已从 skill vocab 删除的噪声:

```text
自我评价, 协助, 客户, 材料, 管理, 学历, 经理, 员工, or, ex, cc, em
```

保留但降权的 transversal 词:

```text
责任心, 沟通, 协调, 组织, 计划, 团队领导 等
```

新 KG 高频 job skills 仍有泛词:

```text
工程, 软件, 培训, 责任, 熟练, 销售, 资格, 处理, 业务, 分析,
活动, 沟通, 维护, 理工, 服务, 行政, 施工, 安排, 行业, 制度
```

结论:本轮清洗有效移除了已知高频噪声和短英文 token,但还没有完全解决泛化词问题;`熟练/资格/业务/服务/行业/制度/责任/活动` 应进入下一轮 `generic_non_skill` 或低权重类型审核。

## 4. 模型结果

测试协议同 `docs/15_results_v2.md`:全库排序,4,095 个 test users。

| 模型 | v2 原始 KG NDCG@10 | OSTA 清洗 KG NDCG@10 | OSTA Recall@10 | OSTA Recall@50 | OSTA MRR |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.0016 | 0.0016 | 0.0039 | 0.0164 | 0.0019 |
| SBERT | 0.0030 | 0.0028 | 0.0049 | 0.0149 | 0.0031 |
| BM25 | 0.0047 | 0.0049 | 0.0090 | 0.0278 | 0.0050 |
| Rule | 0.0147 | 0.0113 | 0.0247 | 0.0945 | 0.0124 |
| LightGCN | 0.0233 ± 0.0011 | 0.0224 | 0.0425 | 0.1021 | 0.0199 |
| SPC-HGT | 0.0266 ± 0.0005 | 0.0258 | 0.0498 | 0.1192 | 0.0237 |

SPC-HGT 本次配置:

- `configs/v2_osta_spc_hgt.yaml`
- seed 42 only
- `epochs=60`, `patience=8`, `use_hard_neg=false`
- best valid NDCG@10 = 0.0268
- early stop at epoch 54

## 5. 判断

1. KG 可用性有所改善:明显噪声词和短英文 token 已从词表删除,技能节点从 4,765 降到 4,634。
2. 推荐效果没有超过 v2 final:SPC-HGT 单 seed NDCG@10=0.0258,低于 v2 final 3-seed 均值 0.0266,也低于旧 seed 42 消融基准 0.0273。
3. 主模型仍强于新 KG 上最强 baseline:SPC-HGT 0.0258 vs LightGCN 0.0224,相对提升约 +14.9%。
4. Rule 降幅最大:从 0.0147 到 0.0113。原因是 Rule 强依赖技能覆盖率,本轮删除/降权泛词后覆盖率信号变稀疏,但没有同步调参。

当前结论:OSTA 清洗提升了技能本体质量,但这版清洗策略对排序指标是**小幅负向**或中性,不能直接替换为最终线上 KG。更合理的方向是继续清洗泛词、限制 transversal 边数量,并重新调 Rule/LightGCN/SPC-HGT 超参。

## 6. 下一步

1. 修正 OSTA 详情采样:优先拉取 `workNum > 0` 的职业详情,补齐 `work_type_terms`。
2. 新增第二批泛词审核:`熟练`, `资格`, `业务`, `服务`, `行业`, `制度`, `责任`, `活动`, `理工`。
3. 对 job skill 增加类型限额:每个 job 最多保留 3 个 `transversal`,最多保留 2 个 `generic_low_signal`。
4. 降低 `max_job_skills` 从 30 到 20 或引入动态阈值,避免大量岗位仍卡在 30 个技能上。
5. 完成 3 seeds SPC-HGT 复评后再更新 `docs/15_results_v2.md` 的主结果。

## 7. `max_job_skills=20` 复验

已执行第 6 节第 4 项,选择把 `configs/v2_data.yaml` 中的 `max_job_skills` 从 30 降到 20,并新增独立实验配置:

- `configs/v2_osta_job20_baselines.yaml`
- `configs/v2_osta_job20_spc_hgt.yaml`

复验命令:

```bash
.venv/bin/python -m jobmatch_gnn.data.preprocess_v2 --config configs/v2_data.yaml
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m jobmatch_gnn.text.encode_sbert --config configs/v2_data.yaml
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m jobmatch_gnn.training.train_v2 --config configs/v2_osta_job20_baselines.yaml
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m jobmatch_gnn.training.train_v2 --config configs/v2_osta_job20_spc_hgt.yaml
```

运行目录:

- `experiments/runs/v2_osta_job20_baselines`
- `experiments/runs/v2_osta_job20_spc_hgt`

### 7.1 KG 密度变化

| 指标 | OSTA job30 | OSTA job20 | 变化 |
|---|---:|---:|---:|
| skill_vocab | 4,634 | 4,634 | 0 |
| avg_user_skills | 32.09 | 32.09 | 0 |
| avg_job_skills | 26.60 | 18.94 | -28.8% |
| zero-skill jobs | 63 | 63 | 0 |
| job-skill edges | 约 2.43M | 1.73M | -28.8% |

岗位技能数量分位数显示,job20 版仍有大量岗位到达上限:

| 分位数 | 0% | 25% | 50% | 75% | 90% | 95% | 99% |
|---|---:|---:|---:|---:|---:|---:|---:|
| job skill count | 0 | 20 | 20 | 20 | 20 | 20 | 20 |

判断:降低上限有效解决了“岗位技能边过密”和“大量岗位卡 30 个技能”的问题,但只是把饱和点移动到 20;它不能单独解决技能本体中 `业务/处理/活动/服务/熟练/理工` 等泛词质量问题。

### 7.2 模型结果

测试协议仍为全库排序,4,095 个 test users。job20 结果为 seed 42 单次运行。

| 模型 | OSTA job30 NDCG@10 | OSTA job20 NDCG@10 | job20 Recall@10 | job20 Recall@50 | job20 MRR |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.0016 | 0.0016 | 0.0039 | 0.0164 | 0.0019 |
| SBERT | 0.0028 | 0.0028 | 0.0049 | 0.0149 | 0.0031 |
| BM25 | 0.0049 | 0.0049 | 0.0090 | 0.0278 | 0.0050 |
| Rule | 0.0113 | 0.0116 | 0.0256 | 0.1001 | 0.0126 |
| LightGCN | 0.0224 | 0.0224 | 0.0425 | 0.1021 | 0.0199 |
| SPC-HGT | 0.0258 | 0.0267 | 0.0508 | 0.1206 | 0.0248 |

SPC-HGT job20 配置:

- `configs/v2_osta_job20_spc_hgt.yaml`
- seed 42 only
- `epochs=60`, `patience=8`, `use_hard_neg=false`
- best valid NDCG@10 = 0.0267
- test NDCG@10 = 0.0267
- runtime = 899.9s

### 7.3 结论

1. `max_job_skills=20` 达到了降密度目标:avg_job_skills 从 26.60 降到 18.94,job-skill edges 从约 2.43M 降到 1.73M。
2. 排序效果没有下降:SPC-HGT NDCG@10 从 0.0258 提升到 0.0267,Recall@10 从 0.0498 提升到 0.0508。
3. 主模型仍强于最强 baseline:job20 SPC-HGT 0.0267 vs LightGCN 0.0224,相对提升约 +19.2%。
4. job20 单 seed 已接近 `docs/15_results_v2.md` 的 v2 原始 KG 三 seed 均值 0.0266,但尚不能替代主结果;需要完成 3 seeds 后再更新主结果文档。
5. 泛词问题仍存在:降上限控制的是图密度,不是技能语义质量。下一步仍应执行类型限额和第二批泛词审核,尤其是 `业务`, `处理`, `活动`, `服务`, `熟练`, `理工`, `专员`, `招聘`。

当前建议:把 `max_job_skills=20` 保留为 OSTA 清洗 KG 的下一轮默认设置,并在此基础上继续做技能类型限额与泛词抑制。
