# Uploaded Source Notes：本轮上传材料可复用结论

## 1. 调研 5_14晚会.pdf

可复用结论：

```text
1. 项目最终需要回答：数据从哪里来、知识图谱怎么建、简历/JD怎么结构化、推荐算法怎么做、系统怎么展示。
2. 推荐最小知识图谱 Schema：Candidate、Job、Skill、Company、City、Industry、Education、Experience。
3. 推荐核心关系：Candidate-HAS_SKILL-Skill、Job-REQUIRES_SKILL-Skill、Candidate-APPLIED_TO-Job、Candidate-MATCHES-Job 等。
4. 主任务建议定为：输入候选人画像，输出 Top-K 推荐岗位。
5. baseline 建议包含：规则匹配、BM25、SBERT。
6. 图谱推荐最稳路径：Candidate -> Skill <- Job。
7. 评价指标建议：Recall@K、Precision@K、NDCG@K、MRR、HitRate@K；无真实标签时做人评和解释覆盖率。
8. 展示层建议：候选人画像、岗位画像、Top-K 推荐、推荐理由解释、技能差距分析。
```

## 2. A Comprehensive Survey on Graph Neural Networks.pdf

可复用结论：

```text
1. 图卷积可理解为对节点及其邻居特征做聚合。
2. GNN 可以分为 RecGNN、ConvGNN、GAE、STGNN。
3. GNN 的 edge-level 输出适合 link prediction，人岗匹配可视为 Candidate-Job 链接预测/排序。
4. 过深 GNN 可能产生 over-smoothing，因此原型默认 2 层。
5. 异构性、动态性、可扩展性是 GNN 重要挑战，正好对应人岗图谱的多实体、多关系、新岗位/新候选人持续变化。
```
