# 论文最终定稿检查表

> 项目：基于KG-GCN的燃煤机组汽轮机系统的故障诊断方法
>
> 当前阶段：正文、主实验、KG审计、摘要/总结和核心论文图已冻结。现在只处理引用、符号、完整编译、PDF级排版和最终盲审。

## A. 已完成并冻结

- [x] 中文摘要：`sections/00_abstract_cn.tex`
- [x] English Abstract：`sections/00_abstract_en.tex`
- [x] 第一章绪论主线
- [x] 第二章文字事实源：`sections/02_gru_monitoring_v4.tex`
- [x] 第三章审计后文字事实源：`sections/03_knowledge_graph_v8.tex`
- [x] 第四章文字事实源：`sections/04_kg_gcn_v7.tex`
- [x] 第五章文字事实源：`sections/05_experiments_v7.tex`
- [x] 第六章总结与展望：`sections/06_conclusion_v2.tex`
- [x] Ch2--Ch5 final wrappers已接入 `main.tex`
- [x] 杨昊东式可见版式：`styles/yang_haodong_thesis.tex`
- [x] GB/T 7714顺序编码参考文献置于论文结尾
- [x] SVG→PDF矢量构建桥接及统一编译脚本

## B. 冻结研究结果

### B1. VHP健康状态模型

- 2024-05训练；2024-06独立测试；1 min采样
- 最终评价24点
- GRU宏平均R²：**0.9511**
- ANN / TCN / iTransformer-small / GRU对比：0.9087 / 0.9397 / 0.9356 / 0.9511
- GRU在24点中的17点取得四模型最高R²

### B2. 汽轮机系统故障KG

- 标准实体：127
- 候选关系：123
- 主KG关系：121
- 冲突待核：2
- DCS映射：45，其中44确认
- B类热力推断不进入主KG/GCN
- Ch3 v8已删除未核实的“具体抽汽级次—单台加热器”确定性表述；除VHP一级抽汽→1号高加外，HP/IP/LP仅保留到抽汽系统—回热设备组层级。

### B3. KG-GCN主诊断结果

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| CNN | 98.33% ± 0.45% | 98.33% ± 0.45% |
| AE | 96.43% ± 0.82% | 96.43% ± 0.82% |
| KG-GCN | **98.45% ± 0.10%** | **98.46% ± 0.11%** |

seed=123主表：KG-GCN Accuracy 98.57%，F1 98.58%。

20%标签：Scratch 94.48%±0.93% → Mask预训练 96.67%±0.27%，提升2.19个百分点。

## C. 论文图表

### Ch2
- [x] 汽轮机系统简化图
- [x] 设备级建模框架图
- [x] VHP输入输出/测点配置图
- [x] GRU网络结构图
- [x] 训练/验证Loss图
- [x] 四模型宏平均R²图
- [x] 真实DCS/GRU代表性预测曲线图

### Ch3
- [x] KG构建流程图
- [x] 审计后系统拓扑及代表性故障语义图

### Ch4
- [x] KG-GCN输入构建图
- [x] Mask预训练与分类流程图

### Ch5
- [x] CNN/AE/KG-GCN混淆矩阵图
- [x] 3-seed Macro-F1比较图
- [x] 20%标签Mask预训练消融图

> 所有最终图在编译时由SVG统一转换为PDF，正文使用矢量版本。

## D. 仍需清零的任务

### D1. 引用与证据
- [ ] 提取当前主稿全部 `\\cite{}` key并与四个BibTeX库一一核对
- [ ] 关键文献标题、作者、年份、DOI/报告号二次核验
- [ ] 标准/NRC/运行事件等非期刊条目的GB/T 7714显示核验
- [ ] 跨型式证据继续保持限定性表述

### D2. 符号与术语
- [ ] 清除第二章残留 `$B+U\\rightarrow Y$`，全文统一 `U+B\\rightarrow Y`
- [ ] 第四章首次定义 `e_i(t)` 后统一称“状态残差”
- [ ] VHP/RH1/HP/RH2/IP/LP/CND缩写首次出现核验
- [ ] Accuracy/Precision/Recall/F1/Macro-F1大小写统一
- [ ] `N`、`L_r`、`M_u`、`M_l` 等符号唯一性检查

### D3. 编译与PDF排版
- [ ] 完整执行 `bash scripts/build_thesis.sh`
- [ ] undefined citation/reference = 0
- [ ] 目录页码/章节另起页核验
- [ ] 图表浮动、跨页、越界核验
- [ ] 表格宽度、公式宽度核验
- [ ] 孤行/寡行/连续大空白核验
- [ ] 参考文献独立成章并进入目录

### D4. 最终Reviewer循环
- [ ] 汽轮机Reviewer：拓扑、接口、边界、故障机理
- [ ] ML Reviewer：训练/验证/测试、Mask预训练、信息泄漏
- [ ] KG Reviewer：来源、投影、循环验证风险
- [ ] 统计Reviewer：主表、3-seed与结论强度
- [ ] 写作Reviewer：AI套话、重复、段落长度
- [ ] 杨昊东母版Reviewer：章节功能、公式/图表位置、结果分析节奏、可见版式

## E. 禁止越界结论

- 不把半物理故障写成真实现场故障样本。
- 不声称HP/IP/LP/凝汽器已完成与VHP同等故障分类验证。
- 不把未核实抽汽级次—单台加热器关系写成本文机组事实。
- 不声称Mask预训练在所有标签比例均提升。
- 不声称KG在任意图结构消融条件下都必然最好。
- 不将跨型式汽轮机研究直接写成本文燃煤机组现场证据。

## F. 最终执行顺序

1. BibTeX/cite全量核验；
2. 符号与缩写清零；
3. 完整构建PDF；
4. PDF版式检查；
5. 六类Reviewer循环，Blocking/Major归零；
6. 最终摘要/总结数字一致性复核；
7. 交付最终PDF与GitHub版本断点。
