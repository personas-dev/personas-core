# 06 创新性与优势说明

## 1. 为什么不是只用规则？

规则匹配稳定、可解释，但只能表达显式条件。例如技能、学历、经验、城市。它无法学习：

```text
技能同义关系
候选人潜在能力
岗位之间的隐含相似性
行为反馈中的偏好模式
跨技能路径的高阶关系
```

因此规则必须作为 baseline，而不是最终主模型。

## 2. 为什么不是只用 BM25/SBERT？

BM25 和 SBERT 擅长文本相似，但人岗匹配不是“简历和 JD 文字越像越好”。常见失败：

```text
JD 写得很泛，文本相似度虚高
候选人技能满足但表达方式不同
岗位技能重要性不同，但文本模型不一定区分
候选人城市/经验/学历不满足，但语义仍然很像
```

因此文本模型要作为召回或特征，而不是唯一依据。

## 3. 为什么不是只用 LightGCN？

LightGCN 是推荐系统强 baseline，但它主要依赖 candidate-job 行为二部图。人岗匹配常有冷启动：

```text
新候选人没有行为
新岗位没有行为
技能要求变化快
投递行为有偏差
```

所以需要引入 Skill、City、Industry 等知识图谱节点，增强冷启动和解释能力。

## 4. 为什么不是直接套 KGAT/KGCN？

KGAT/KGCN 是通用知识图谱推荐模型，但人岗匹配有特殊结构：

```text
Candidate 和 Job 是双边匹配，不是普通 user-item 消费
Skill 路径是最关键解释证据
技能有 required/preferred/importance/min_years 等边特征
岗位匹配需要显式处理 missing skills
```

因此本方案不是直接套 KGAT/KGCN，而是在异构图 Transformer 上显式注入技能路径和人岗规则特征。

## 5. SPC-HGT 的创新点

```text
1. Heterogeneous Graph Transformer 处理多实体多关系。
2. Skill-Path Features 显式建模 Candidate -> Skill <- Job。
3. Contrastive Learning 学习正负岗位边界。
4. Hard Negative Mining 处理“看起来相似但实际不匹配”的岗位。
5. Rule/Text/KG Path 三类特征融合，提高工程稳健性。
6. Structured Explanation 输出可审计证据。
```

## 6. 如何证明创新有效？

必须用消融实验证明：

```text
去掉 skill_path_features 后 NDCG@10 下降 -> 证明技能路径有效
去掉 hard_negative 后 MRR 下降 -> 证明困难负样本有效
去掉 contrastive_loss 后 Recall@10 下降 -> 证明对比学习有效
替换成 LightGCN 后冷启动指标下降 -> 证明异构知识图谱有效
替换成 KGAT/KGCN 后解释覆盖率下降 -> 证明人岗特化设计有效
```

## 7. 线上价值

```text
更准：融合文本、规则、行为和图谱结构
更稳：baseline 可降级，避免复杂模型失效
更可解释：返回匹配技能、缺失技能和路径证据
更抗冷启动：新岗位和新候选人可通过技能与文本接入图谱
更可维护：Schema、接口、实验、references 独立维护
```
