# 09 文档维护规范

## 1. 必须维护的文档

```text
docs/01_algorithm_api.md
docs/02_kg_schema.md
docs/03_model_design.md
docs/04_training_evaluation.md
docs/05_baseline_and_ablation.md
docs/06_innovation_and_advantage.md
docs/07_deployment.md
docs/08_versioning_and_workflow.md
references/references.md
```

## 2. 每次模型更新必须同步修改

| 变更类型 | 必改文档 |
|---|---|
| API 输入输出变化 | `01_algorithm_api.md`, `MODEL_CARD.md` |
| Schema 变化 | `02_kg_schema.md`, `CHANGELOG.md` |
| 模型结构变化 | `03_model_design.md`, `06_innovation_and_advantage.md` |
| 新 baseline | `05_baseline_and_ablation.md`, `references/references.md` |
| 新实验结果 | `04_training_evaluation.md`, `05_baseline_and_ablation.md` |
| 部署方式变化 | `07_deployment.md` |

## 3. References 维护格式

每条资料必须包含：

```text
title
year
type
task
core_idea
what_to_learn
risk
adoption_decision
url
```

不要把 references 混在模型设计正文里。正文只写采用理由，资料统一在 `references/` 下维护。

## 4. 实验文档命名

```text
experiments/runs/{YYYYMMDD}_{model_name}_{dataset}_{seed}/
  config.yaml
  metrics.json
  metrics.csv
  predictions.parquet
  ablation.csv
  error_cases.md
  notes.md
```

## 5. 失败实验也要记录

失败实验必须说明：

```text
为什么失败
失败时的指标
和哪个 baseline 对比失败
可能原因
下一步修复方案
```

不要只保留成功实验。
