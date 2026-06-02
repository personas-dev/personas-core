# Changelog

## [0.1.0] - 2026-06-02

### Added

- 初始化 KG-GNN 人岗匹配 Skill Pack。
- 拆分 docs、references、configs、src_layout。
- 添加 SPC-HGT MatchNet 设计、baseline、消融、部署和版本流程。


## [0.2.0] - 2026-06-02

### Added

- 规范化并安装 `hr-kg-gnn` Codex skill，补充 `agents/openai.yaml`。
- 将 P0/P1 关键引用下载到 `references/local/` 并生成 `references/local_manifest.csv`。
- 实现 personas-core 采样训练闭环：Rule、BM25、semantic_hash baseline、LightGCN/SPC-HGT-lite PyTorch 训练入口、指标与实验输出。
- 新增 `docs/baseline_comparison.md` 记录 baseline 对比和 GNN 训练阻塞原因。

### Notes

- 当前 Python 环境未安装 PyTorch，GPU 空闲但 GNN 模型未完成训练。
