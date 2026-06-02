# References：人岗匹配 KG-GNN 参考资料库

更新时间：2026-06-02
维护规则：references 单独维护，不混入 `SKILL.md`。新增资料必须写明采用价值、风险和是否纳入实现。

## 字段说明

| 字段 | 含义 |
|---|---|
| 类型 | paper / blog / code / official-docs / uploaded-note |
| 价值 | 对当前算法、baseline、文档或展示的作用 |
| 风险 | 直接采用时的风险 |
| 采用决策 | P0/P1/P2/P3 或不采用 |

## 参考资料表

| 优先级 | 类型 | 标题 | 年份 | 任务 | 核心思想 | 采用建议 |
|---|---|---|---:|---|---|---|
| P0 | uploaded-note | [调研 5_14晚会.pdf](uploaded:/mnt/data/调研 5_14晚会.pdf) | 2026 | 项目路线与最小 KG Schema | 主任务定为候选人到 Top-K 岗位推荐；采用 Rule/BM25/SBERT/KG 路径打分；图谱核心为 Candidate-Skill-Job。 | 采用为项目约束和最小闭环依据。 |
| P0 | uploaded-paper | [A Comprehensive Survey on Graph Neural Networks](uploaded:/mnt/data/A Comprehensive Survey on Graph Neural Networks.pdf) | 2019 | GNN 基础综述 | 将 GNN 分为 RecGNN、ConvGNN、GAE、STGNN；强调邻域聚合、消息传递、异构性、动态性和可扩展性。 | 采用为理论基础。 |
| P1 | paper | [Person-Job Fit: Adapting the Right Talent for the Right Job with Joint Representation Learning](https://arxiv.org/abs/1810.04040) | 2018 | Person-Job Fit | PJFNN 用 CNN 学习候选人资格与岗位要求的联合表示，并能定位哪些岗位要求被满足。 | 作为人岗匹配深度 baseline/参考。 |
| P2 | paper | [Enhancing Person-Job Fit for Talent Recruitment: An Ability-aware Neural Network Approach](https://arxiv.org/abs/1812.08947) | 2018 | Person-Job Fit | 能力感知神经网络，利用历史岗位申请数据，为匹配结果提供更强解释。 | 作为能力建模参考。 |
| P2 | paper | [An Enhanced Neural Network Approach to Person-Job Fit in Talent Recruitment / TAPJFNN](https://dl.acm.org/doi/abs/10.1145/3376927) | 2020 | Person-Job Fit | Topic-based Ability-aware Person-Job Fit Neural Network，结合主题和能力感知结构。 | 作为解释型人岗匹配参考。 |
| P1 | paper+code | [Person-job fit estimation from candidate profile and related recruitment history with co-attention neural networks](https://arxiv.org/abs/2206.09116) | 2022 | Person-Job Fit | PJFCANN 融合 co-attention 局部语义表示和历史招聘记录的 GNN 全局经验表示。 | 作为非纯 KG 但含图经验的强 baseline。 |
| P0 | paper | [conSultantBERT: Fine-tuned Siamese Sentence-BERT for Matching Jobs and Job Seekers](https://arxiv.org/abs/2109.06501) | 2021 | Text semantic matching | 微调 Siamese SBERT，构建岗位和求职者文本 embedding，支持大规模匹配。 | 作为 SBERT baseline 重要参考。 |
| P0 | paper | [ConFit: Improving Resume-Job Matching using Data Augmentation and Contrastive Learning](https://arxiv.org/html/2401.16349v1) | 2024 | Resume-job matching | 使用数据增强和对比学习提升简历-岗位匹配 dense retrieval。 | 采用对比学习和 hard negative 思路。 |
| P2 | paper | [CONFIT V2: Improving Resume-Job Matching using Hypothetical Reference Resume and Runner-up Hard Negative Mining](https://aclanthology.org/2025.findings-acl.661.pdf) | 2025 | Resume-job matching | 在 ConFit 基础上，用 LLM 生成 hypothetical reference resume，并引入 runner-up hard negative mining。 | 作为后续增强方案，不作为 P0 必需。 |
| P1 | industrial-paper | [LinkSAGE: Optimizing Job Matching Using Graph Neural Networks](https://arxiv.org/html/2402.13430v1) | 2024/2025 | Large-scale job matching | LinkedIn 将 GNN 集成到大规模岗位匹配系统，处理巨型异构 job marketplace graph 和 nearline inference。 | 采用工程原则，不作为原型必需。 |
| P1 | paper+code | [HRGraph: Leveraging LLMs for HR Data Knowledge Graphs with Information Propagation-based Job Recommendation](https://arxiv.org/abs/2408.13521) | 2024 | HR KG + recommendation | 使用 LLM 构建 HR 知识图谱，并用于岗位推荐、技能差距等下游任务。 | 作为 HR KG 构建参考。 |
| P1 | paper | [Graph Neural Networks for Candidate-Job Matching](https://link.springer.com/article/10.1007/s41019-025-00293-y) | 2025 | Candidate-job matching with GNN | 构造候选人-岗位对的二部图/属性图，用 GNN 处理极端类别不平衡的人岗匹配。 | 作为 GNN 人岗匹配近期参考。 |
| P3 | paper | [Resume-Job Compatibility Scoring Using Graph Neural Networks and LLMs](https://dl.acm.org/doi/full/10.1145/3787330.3787359) | 2025 | Resume-job compatibility scoring | 结合 GNN 和 LLM 提取的结构化信息进行简历-岗位兼容性评分。 | 作为增强路线参考。 |
| P0 | paper | [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) | 2020 | Heterogeneous GNN | 使用节点类型和边类型相关参数建模异构图注意力，并支持 Web-scale 异构图训练。 | 作为主模型骨架。 |
| P0 | paper+code | [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126) | 2020 | Graph recommendation baseline | 去掉特征变换和非线性，只保留邻域聚合，在协同过滤上表现强。 | 必须实现 baseline。 |
| P1 | paper+code | [KGCN: Knowledge Graph Convolutional Networks for Recommender Systems](https://arxiv.org/abs/1904.12575) | 2019 | KG recommendation baseline | 通过 KG 邻居采样和用户特定的邻域聚合学习 item/entity 表示。 | KGCN/KGAT 至少二选一。 |
| P1 | paper+code | [KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854) | 2019 | KG recommendation baseline | 端到端建模 user-item 图和知识图谱中的高阶连通性，并用 attention 聚合。 | KGCN/KGAT 至少二选一。 |
| P0 | official-docs | [PyTorch Geometric HeteroConv](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.HeteroConv.html) | 2026 | Heterogeneous GNN implementation | HeteroConv 为异构图每种边类型指定不同 message passing 模块，并聚合同一目标节点的多关系结果。 | PyG 实现首选参考。 |
| P1 | official-docs | [DGL HeteroGraphConv](https://www.dgl.ai/dgl_docs/generated/dgl.nn.pytorch.HeteroGraphConv.html) | 2026 | Heterogeneous GNN implementation | DGL HeteroGraphConv 对每种关系图应用子模块，并将结果写入目标节点后聚合。 | 作为备选实现参考。 |
| P0 | official-docs | [PyTorch Geometric Heterogeneous Graph Learning](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html) | 2026 | Heterogeneous graph data modeling | 异构图中不同类型节点和边拥有独立特征张量和类型定义。 | PyG 数据结构参考。 |
| P3 | blog | [Knowledge Graph with Job Recommendation](https://community.sap.com/t5/technology-blog-posts-by-sap/knowledge-graph-with-job-recommendation/ba-p/13568551) | 2022 | KG + job recommendation product understanding | 从产品角度介绍知识图谱及岗位推荐应用。 | 用于背景和展示，不用于核心算法证明。 |
| P3 | blog/tutorial | [Building a Recommendation Engine with Neo4j](https://neo4j.com/blog/developer/recommendation-engine-hands-on-1/) | 2023 | Graph recommendation engineering | 介绍用 Neo4j 构建推荐引擎的基础流程。 | 用于系统展示和原型教程。 |
| P2 | blog | [Building Knowledge Graphs for Recruitment Technology](https://www.textkernel.com/learn-support/blog/textkernel-acquires-us-based-software-company-sovren-to-become-the-global-leader-in-ai-powered-recruitment-technology/) | 2023 | Recruitment ontology/KG | 招聘本体与外部知识图谱/分类体系映射，如 ESCO、O*NET 等。 | 用于技能标准化和知识治理参考。 |
| P3 | blog | [Revolutionizing HR Recruiting with Knowledge Graphs and Large Language Models](https://blog.metaphacts.com/revolutionizing-hr-recruiting-with-knowledge-graphs-and-large-language-models) | 2024 | HR KG + LLM product design | KG 与 LLM 结合招聘数据，统一候选人与岗位上下文。 | 用于系统展示和答辩背景。 |


## 本地缓存

已将 P0/P1 关键论文和实现文档下载到 `references/local/`，机器可读清单见 `references/local_manifest.csv`。优先使用这些本地文件进行实现依据核对；若本地清单中没有目标资料，再访问原始 URL。

## 资料使用建议

P0 必读：上传调研材料、GNN 综述、conSultantBERT、ConFit、HGT、LightGCN、PyG HeteroConv。
P1 推荐：PJFNN、PJFCANN、LinkSAGE、HRGraph、KGCN/KGAT、GNN candidate-job matching。
P2 增强：TAPJFNN、CONFIT V2、Textkernel 招聘 KG。
P3 展示：SAP/Neo4j/Metaphacts 等工程和产品博客。

## 更新 Checklist

```text
[ ] 新论文是否和人岗匹配、GNN、KG 推荐或解释性直接相关？
[ ] 是否有公开代码或可复现实验？
[ ] 是否明确 baseline 和数据集？
[ ] 是否能转化为本项目的模型模块、特征模块或实验设计？
[ ] 是否已写入 references.csv？
```
