# Tests

必须覆盖：

```text
test_schema.py：输入输出 Schema 校验
test_build_graph.py：节点/边数量、reverse edge、边特征
test_negative_sampling.py：负样本不包含正样本，hard negative 来源正确
test_metrics.py：Recall@K、NDCG@K、MRR 正确性
test_inference.py：rank_jobs_for_candidate 输出格式正确
test_explanation.py：matched_skills、missing_skills、graph_paths 正确
```

PR 前必须运行：

```bash
ruff check src tests
black --check src tests
mypy src
pytest tests
```
