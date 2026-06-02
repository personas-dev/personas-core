# HR KG-GNN Skill Pack

版本：v0.1.0
用途：指导 Agent 实现“基于知识图谱的 GNN 人岗匹配核心算法模块”。

本包不是只给一个大段说明，而是按**路径、功能、接口、实验、部署、文档、references**拆开维护。推荐从下面三个文件开始读：

1. `skills/hr_kg_gnn/SKILL.md`：Agent 总控 Skill。
2. `docs/00_path_function_map.md`：路径与功能拆分。
3. `references/references.md`：单独整理的论文、博客、代码、官方文档参考资料。

## 快速使用

把 `skills/hr_kg_gnn/` 放入 Agent 的 skills 目录，把 `docs/` 和 `references/` 放入项目根目录。Agent 执行任务时，先读取 `SKILL.md`，再根据 `docs/00_path_function_map.md` 创建或修改代码。

建议项目根目录最终形态：

```text
project_root/
  skills/hr_kg_gnn/SKILL.md
  docs/
  references/
  configs/
  src/jobmatch_gnn/
  tests/
  experiments/
  pyproject.toml
  README.md
  CHANGELOG.md
  MODEL_CARD.md
```

## 本包的设计重点

- 主任务：输入候选人画像，输出 Top-K 推荐岗位。
- 主模型：`SPC-HGT MatchNet`，即技能路径增强的对比学习异构图 Transformer 人岗匹配模型。
- 最小图谱核心路径：`Candidate -> Skill <- Job`。
- 必跑 baseline：Rule、BM25、SBERT、Node2Vec、LightGCN、KGCN/KGAT、HGT without path features。
- 必报指标：Recall@K、Precision@K、NDCG@K、MRR、HitRate@K、AUC。
- 必做消融：去文本、去技能路径、去对比学习、去 hard negative、去边特征、替换为 LightGCN/KGAT/KGCN。
- 必有解释：匹配技能、缺失技能、图谱路径、结构化推荐理由。
