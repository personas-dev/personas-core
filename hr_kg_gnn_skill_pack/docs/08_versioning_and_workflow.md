# 08 版本维护与开发流程

## 1. 版本号规则

采用 SemVer：

```text
MAJOR.MINOR.PATCH
```

```text
MAJOR：Schema/API 不兼容变更
MINOR：新增模型、新增 baseline、新增指标，兼容旧接口
PATCH：bugfix、性能优化、文档修正
```

示例：

```text
v0.1.0: Rule/BM25/SBERT baseline
v0.2.0: Node2Vec/LightGCN baseline
v0.3.0: SPC-HGT 初版
v0.4.0: hard negative + contrastive learning
v0.5.0: explanation module
v1.0.0: 稳定发布版
```

## 2. 分支流程

```text
main：稳定版本
develop：集成开发版本
feature/*：功能开发
experiment/*：实验模型
fix/*：bug 修复
docs/*：文档更新
```

## 3. PR 合并条件

```text
[ ] pytest 全部通过
[ ] ruff/black/mypy 全部通过
[ ] baseline_comparison.md 已更新
[ ] CHANGELOG.md 已更新
[ ] MODEL_CARD.md 已更新
[ ] experiments/ 下有实验记录
[ ] references/ 下新增资料已记录
```

## 4. Checkpoint 元数据

每个模型 checkpoint 必须包含：

```json
{
  "model_version": "0.3.0",
  "schema_version": "1.0.0",
  "git_commit": "abc123",
  "graph_snapshot_id": "graph_20260602_001",
  "train_config": {},
  "metric_summary": {},
  "created_at": "2026-06-02T00:00:00Z"
}
```

## 5. 代码质量

```text
black: 格式化
ruff: lint
mypy: 类型检查
pytest: 测试
pre-commit: 提交前检查
```

所有 public API 必须有 docstring 和类型注解。
