# 第四章 KG-GCN 方法 — Reviewer Audit v1

## Reviewer A：图学习/机器学习

### Blocking
- 无。

### Major
1. **GCN语义图→参数图投影必须与实现一致。** 当前正文采用标准GCN二值参数邻接图，合理；后续代码不得私自改为带权/R-GCN而正文不更新。
2. **诊断残差窗口长度不能直接继承GRU窗口。** 当前已明确 \(L_r\) 独立选择，PASS。
3. **预训练实现必须严格对应“mask-only reconstruction loss”。** 后续代码和实验表需报告mask ratio、encoder/decoder结构。
4. **冻结encoder是当前师兄路线的关键假设。** 若实验后改为微调encoder，需要补充对比并修改正文。

### Minor
- 第五章补充随机种子、优化器、early stopping等复现实验细节。

## Reviewer B：汽轮机热力/故障诊断

### Blocking
- 无。

### Major
1. **残差物理含义表述合理。** GRU吸收运行边界变化，残差表示健康关系无法解释的偏离，但不能声称残差已完全消除工况影响。
2. **系统KG与VHP任务子图区分清楚。** PASS。后续第五章必须继续坚持“VHP为代表性验证对象”。
3. **故障解释边界合理。** 当前只将高残差节点、设备归属和知识路径用于辅助解释，没有把GCN传播写成严格物理因果，PASS。
4. **参数邻接图投影规则需要最终列表化。** 第五章或附录应给VHP 24节点的邻接边/邻接矩阵来源，防止KG成为不可复现实验部件。

## Reviewer C：论文结构/写作

### Blocking
- 无。

### Major
1. 章节逻辑已形成“为什么融合→GCN基础→输入构建→两阶段训练→解释→小结”，与杨昊东第四章节奏相符但不是原文换词。
2. 图4-1至图4-4必须后续真正绘制，否则当前仍是方法文字完整、视觉证据不足。
3. 目前大量符号定义合理，但第五章不要重复推导公式，应只报告数据、超参数和结果。
4. 章节内已经明确未确定参数留待实验，不应在后续为了“文章完整”倒填虚假网络配置。

## Editor Synthesis

### 当前可接受结论
第四章v1可作为“方法正文初稿”进入下一阶段，Blocking=0。

### 下一步必须完成
1. 根据第五章故障样本确定VHP参数子图的24节点映射和具体边集合。
2. 完成KG-GCN代码并冻结：GCN层数、hidden dim、mask ratio、residual window、optimizer、训练策略。
3. 绘制4张方法图。
4. 设计至少两个关键消融：
   - Raw-KG-GCN vs Residual-KG-GCN，回答为什么需要第二章健康模型；
   - Data/No-KG GCN vs KG-GCN，回答为什么需要知识图谱。
5. 若故障样本为构造/仿真样本，正式稿必须如实说明来源，不得表述为真实现场故障。

## Gate
- Method structure: PASS
- Formula consistency: PASS
- Research-object consistency: PASS
- Experimental evidence: PENDING（第五章）
- Figure completeness: PENDING
