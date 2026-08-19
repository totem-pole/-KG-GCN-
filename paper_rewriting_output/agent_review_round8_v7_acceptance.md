# 多Agent审稿 Round 8：v7终稿语言接受审查

## 审查对象
- sections/03_knowledge_graph_v7.tex
- sections/04_kg_gcn_v7.tex
- sections/05_experiments_v7.tex

## Yang-Style Reviewer
通过。v7保留杨昊东第四章的段落功能和信息顺序，但未采用逐句同义替换：
- Ch3：问题说明→Schema→候选三元组→领域规则→Embedding标准化→关系可靠性→KG结果；
- Ch4：GCN基础→KG参数子图→单时刻状态偏差→标准化→滑动窗口→图输入→Mask预训练→冻结编码器→分类；
- Ch5：主指标表→重复实验稳定性→混淆矩阵→CNN解释→AE解释→KG-GCN解释→少标签预训练补充验证。

## Similarity Reviewer
Blocking=0。与杨文保持的是宏观结构、公式类型和论证节奏；汽轮机对象、机组拓扑、KG证据来源、VHP节点、故障类别、训练口径及实验数据均为本文内容。禁止后续采用杨文逐句同义替换方式继续润色。

## Evidence Reviewer
Blocking=0。
- KG口径：127实体，123候选关系，121进入主KG，2条冲突候选关系不入主图；
- 主实验seed=123：CNN 97.86/97.89/97.86/97.85，AE 95.54/95.57/95.54/95.54，KG-GCN 98.57/98.60/98.57/98.58；
- 3-seed Macro-F1：CNN 98.33±0.45%，AE 96.43±0.82%，KG-GCN 98.46±0.11%；
- 20%标签Mask：94.48±0.93%→96.67±0.27%，+2.19个百分点；
- 半物理场景限制仍明确保留。

## Turbine Reviewer
Blocking=0。系统级方法、VHP代表性算例边界保持清晰。叶片沉积文献已显式限定为跨型式蒸汽轮机机理参考；未把地热汽轮机证据写成本文燃煤机组现场事实。

## Academic-Language Reviewer
Blocking=0。v7较v6进一步拆分长段落，每段主要承担单一功能；减少了项目审计腔和连续连接词堆叠。结果段已经形成“总体结果→误判现象→模型原因”的稳定节奏。

## Minor
1. Ch4第一次定义$e_i(t)$后可增加“下文称为状态残差”，统一“状态偏差/残差”术语；
2. `Xu2011HPIPLeakage` BibTeX作者可补全Yong-feng Shi；
3. 最终排版阶段统一第二章$B+U\to Y$与全文$U+B\to Y$符号顺序。

## Round 8结论
Blocking=0，Major=0，Minor=3。v7可进入主稿。

下一轮：符号/BibTeX/交叉引用Agent + 图表前后文Agent。