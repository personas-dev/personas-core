# Manifest：路径与功能总览

```text
hr_kg_gnn_skill_pack/
  README.md
  MANIFEST.md
  pyproject.toml

  skills/hr_kg_gnn/SKILL.md

  docs/
    00_path_function_map.md
    01_algorithm_api.md
    02_kg_schema.md
    03_model_design.md
    04_training_evaluation.md
    05_baseline_and_ablation.md
    06_innovation_and_advantage.md
    07_deployment.md
    08_versioning_and_workflow.md
    09_document_maintenance.md

  references/
    references.md
    references.csv
    reading_priority.md
    search_queries.md
    uploaded_source_notes.md

  templates/
    configs/train.yaml
    configs/eval.yaml
    configs/infer.yaml
    configs/model/spc_hgt.yaml
    configs/baseline/rule.yaml
    configs/baseline/bm25.yaml
    configs/baseline/sbert.yaml
    MODEL_CARD.md
    CHANGELOG.md
    experiments/experiment_record_template.md

  src_layout/
    tree.md
    src/jobmatch_gnn/...
    tests/README.md
```

## 路径职责

| 路径 | 功能 | 交付标准 |
|---|---|---|
| `skills/hr_kg_gnn/SKILL.md` | Agent 总控指令 | 能指导 Agent 从数据、建图、训练、评估、推理到解释闭环实现 |
| `docs/` | 技术文档 | 接口、图谱 Schema、模型设计、实验、部署、版本流程全部拆分 |
| `references/` | 参考资料库 | 论文、博客、官方文档、代码库单独维护，不混入主 Skill |
| `templates/configs/` | 训练/评估/推理配置模板 | 可直接复制到项目的 `configs/` 下修改 |
| `src_layout/src/jobmatch_gnn/` | 推荐代码结构骨架 | 给 Agent 明确模块边界和函数职责 |
| `src_layout/tests/` | 测试说明 | 明确每类测试必须覆盖什么 |
