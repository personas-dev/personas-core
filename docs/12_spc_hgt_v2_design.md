# 12 SPC-HGT v2 模型设计

日期:2026-06-12
上游:`docs/10_optimization_plan.md` 阶段 B/D
代码:`src/jobmatch_gnn/models/spc_hgt_v2.py`、`src/jobmatch_gnn/training/train_v2.py`

## 1. 总体结构

```text
                ┌─ SBERT(bge-small-zh-v1.5, 512d) ─ 冻结,只做初始化/输入特征
输入特征        │   user 画像文本 → x_c     jd 标题+描述 → x_j     技能短语 → x_s
                └─ 结构化 one-hot/嵌入:City / Industry / JobType / Degree / Years

异构编码器      RGAT 风格关系感知消息传递 × 2 层(每种边类型独立投影 + 注意力聚合)
                输出:z_c(候选人) z_j(岗位) z_s(技能)

技能路径模块    对 pair (c, j):P(c,j) = {s | s ∈ skills(c) ∩ skills(j)}
                α_s = softmax_s( w^T tanh(W [z_s ; z_c ; z_j]) )     (路径注意力)
                p_cj = Σ α_s · z_s                                   (路径表征)
                gap_cj = mean({z_s | s ∈ skills(j) \ skills(c)})     (缺口表征)

匹配头          score(c,j) = <z_c, z_j>/√d + MLP([z_c ; z_j ; p_cj ; gap_cj ; f_rule])
                f_rule = 7 维规则特征(覆盖率/城市/学历/年限/类型匹配等)

解释输出        matched_skills = top-α 技能(带权重)    missing_skills = 缺口技能
                graph_paths = Candidate→s←Job           reasons = 规则特征命中项
```

设计原则:注意力权重 α_s 既参与评分又直接作为解释 —— **评分即解释**,这是 v2 的第一创新点。

## 2. 异构编码器

节点类型 6 类、边类型约 8 种(含反向)。每层对每种关系 r:

```text
m_{u→v,r} = α_{u,v,r} · W_r h_u
h_v' = LayerNorm( h_v + σ( Σ_r mean_{u∈N_r(v)} m_{u→v,r} ) )
```

- 实现优先用 PyG `HGTConv` / `HeteroConv(GATConv)`;若依赖不可用,自实现上式(scatter-mean 足够)。
- 层数默认 2(skill pack 规定,避免 over-smoothing)。
- Job 节点 26.9 万,SBERT 512d 输入投影到 d=128;全图驻留 96GB GPU 无压力(<2GB)。
- 训练时图上只放 train 交互边。

## 3. 损失函数

```text
L = L_bpr + λ1·L_bce + λ2·L_nce          默认 λ1=0.2, λ2=0.1

L_bpr = -log σ( s(c,j+) - s(c,j-) )                       成对排序
L_bce = BCE( s(c,j±), y )                                  绝对校准
L_nce = -log  exp(<z_c,z_j+>/τ_l) / Σ_b exp(<z_c,z_jb>/τ_l)   批内 InfoNCE
```

**行为分级温度(创新点 2)**:正对的温度 τ_l 按行为等级缩放 —— satisfied τ=0.07、delivered τ=0.1、browsed τ=0.2(等级越高对齐越紧)。browsed 对只进 L_nce,不进 L_bpr/L_bce 正例。

## 4. 难负样本(创新点 3)

```text
N1 随机负:全库均匀(排除该用户全部正例)
N2 语义难负:SBERT 检索与 c 的正例 job 最相似的 top-200 中无任何行为的 job
N3 结构难负:与正例同 JobType 且同 City 但无行为的 job
课程:epoch < E_warm 全用 N1;之后按 (0.5, 0.3, 0.2) 混合
```

语义难负直接攻击"看起来匹配但用户不投"的判别难点;课程式引入避免训练初期崩塌。

## 5. 与 baseline/消融的关系

| 变体 | 说明 |
|---|---|
| SPC-HGT v2(完整) | 上述全部 |
| w/o path-attn | 去掉技能路径模块(p、gap 置零) |
| w/o contrastive | λ2=0 |
| w/o hard-neg | 只用 N1 |
| w/o text | SBERT 输入替换为随机初始化嵌入 |
| w/o hetero | 退化为 LightGCN 交互图 + 规则特征(≈v1 的 SPC-HGT-lite) |

每个创新组件必须由对应消融证明贡献,否则从主模型移除。

## 6. 超参数默认值(configs/v2_spc_hgt.yaml)

| 参数 | 默认 |
|---|---|
| hidden dim | 128 |
| layers | 2 |
| heads | 4 |
| dropout | 0.2 |
| lr / weight_decay | 1e-3 / 1e-5 |
| batch(正对) | 2048 |
| epochs | 30(早停 patience=5,看 valid NDCG@10) |
| 负样本/正对 | 4 |
| τ(satisfied/delivered/browsed) | 0.07 / 0.10 / 0.20 |
| λ1, λ2 | 0.2, 0.1 |
| seed | 42/43/44(报告均值) |
