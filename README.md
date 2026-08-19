# 基于KG-GCN的燃煤机组汽轮机系统的故障诊断方法

本仓库为论文、知识图谱、实验代码、审稿记录与最终排版的单一事实入口。

> **当前状态：正文、主实验、KG审计和论文主图已经冻结并接入主稿。当前仅剩全文引用/BibTeX核验、少量符号统一、完整编译与最终多Reviewer盲审。**
>
> 写作与可见版式以杨昊东学位论文为母版：保持相近的章节功能、信息展开顺序、算法说明节奏、算例组织和图表位置；汽轮机对象、DCS数据、知识图谱、故障场景、实验结果及正文措辞均为本研究独立内容。

---

## 1. 当前正式主稿

`main.tex` 当前引用：

```text
中文摘要：sections/00_abstract_cn.tex
英文摘要：sections/00_abstract_en.tex
第一章：sections/01_introduction.tex
第二章：sections/02_gru_monitoring_final.tex
第三章：sections/03_knowledge_graph_final.tex
第四章：sections/04_kg_gcn_final.tex
第五章：sections/05_experiments_final.tex
第六章：sections/06_conclusion_v2.tex
版式：styles/yang_haodong_thesis.tex
图文件桥接：styles/final_figure_bridge.tex
```

其中final wrapper只负责定稿图表嵌入；底层主要文字事实源为：

```text
Ch2: sections/02_gru_monitoring_v4.tex
Ch3: sections/03_knowledge_graph_v8.tex
Ch4: sections/04_kg_gcn_v7.tex
Ch5: sections/05_experiments_v7.tex
```

第三章已在v8中收紧抽汽拓扑口径：**VHP一级抽汽→1号高压加热器为已核实机组关系；HP/IP/LP仅写到相应高压/中压/低压抽汽系统与回热设备组的系统级连接，不再把未完成机组级核验的具体抽汽级次—单台加热器关系写成确定性事实。**

---

## 2. 论文主线

```text
真实DCS健康运行数据
        ↓
设备级 U+B→Y 健康状态建模
        ↓
GRU健康参考与状态残差
        ↓
运行生产资料 + 外部故障机理
        ↓
汽轮机系统故障知识图谱
        ↓
参数子图 + 多参数残差窗口
        ↓
KG-GCN Mask重构预训练
        ↓
冻结Encoder + FC/Softmax
        ↓
VHP代表性故障诊断算例
```

全文研究对象为**燃煤机组汽轮机热力系统**，VHP仅作为完整展示方法的代表性算例。

---

## 3. 已冻结核心结果

### 3.1 VHP健康状态模型

- 2024-05正常运行DCS数据训练；2024-06独立测试；1 min采样。
- 最终评价：24个状态参数。
- 实际网络：`24×16 → GRU(32) → LayerNorm → Dropout(0.05) → Linear(27)`。
- 参数量：5755。

| 模型 | 24点测试宏平均R² |
|---|---:|
| ANN | 0.9087 |
| TCN | 0.9397 |
| iTransformer-small | 0.9356 |
| **GRU** | **0.9511** |

GRU在24个状态点中的17个点取得四种模型最高R²。

### 3.2 汽轮机系统故障知识图谱

- 标准实体：**127**
- 候选关系：**123**
- 进入主知识图谱：**121**
- 冲突候选关系：**2**，保留待核，不进入确定性KG/GCN
- DCS映射：45条，其中44条已确认
- 点级B类热力推断独立保存，不作为主KG证据

机组结构知识优先来自DCS点表、设备IO与系统接口核对；公开论文、标准和运行事件只用于补充可迁移的故障机理。LLM仅生成候选知识，不作为唯一证据。

### 3.3 VHP代表性KG-GCN诊断

3组随机种子正式汇总：

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| CNN | 98.33% ± 0.45% | 98.33% ± 0.45% |
| AE | 96.43% ± 0.82% | 96.43% ± 0.82% |
| **KG-GCN** | **98.45% ± 0.10%** | **98.46% ± 0.11%** |

固定seed=123主实验：

| 模型 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| CNN | 97.86% | 97.89% | 97.86% | 97.85% |
| AE | 95.54% | 95.57% | 95.54% | 95.54% |
| **KG-GCN** | **98.57%** | **98.60%** | **98.57%** | **98.58%** |

20%带标签故障窗口条件下：

```text
KG-GCN scratch                 94.48% ± 0.93%
Mask预训练 + frozen encoder   96.67% ± 0.27%
提升                           +2.19个百分点
```

---

## 4. 数据与结论边界

第五章主故障样本不是现场真实故障，而是：

```text
真实健康运行残差背景
+
独立故障机理约束的空间响应
+
随机严重度与时间演化
=
半物理 / 机理一致故障场景
```

硬约束：

- 不将半物理样本称为真实电厂故障样本；
- 不把VHP结果外推成HP/IP/LP/凝汽器已经完成同等独立验证；
- 故障空间签名不得由同一个KG邻接矩阵生成，避免循环验证；
- 不声称Mask预训练在所有标签比例下均优于scratch；
- 不声称KG在任意图消融条件下都必然优于统计相关图；
- 跨型式蒸汽轮机证据只作为通用机理参考，不写成本文机组现场证据。

---

## 5. 最终论文图表

最终图已经进入仓库并由final wrapper接入主稿：

```text
figures/ch2/
  fig2_1_turbine_system_final.svg
  fig2_2_device_model_framework_final.svg
  fig2_3_vhp_io.svg
  fig2_4_gru_architecture_final.svg
  fig2_5_selection_loss.svg
  fig2_6_macro_r2.svg
  fig2_7_prediction_panels.svg

figures/ch3/
  fig3_1_kg_construction_flow.svg
  fig3_2_turbine_kg_overview.svg

figures/ch4/
  fig4_1_kg_gcn_input.svg
  fig4_2_mask_training.svg

figures/ch5/
  fig5_1_confusion_matrices.svg
  fig5_2_main_comparison.svg
  fig5_3_mask_ablation.svg
```

编译时所有SVG由 `scripts/build_thesis.sh` 在隔离目录中统一转换为PDF，确保论文中保持矢量清晰度。

---

## 6. 杨昊东式学位论文版式

统一版式位于：

```text
styles/yang_haodong_thesis.tex
```

当前结构：

```text
中文摘要
→ English Abstract
→ 目录
→ 第一章 绪论
→ 第二章 状态监测
→ 第三章 知识图谱
→ 第四章 KG-GCN
→ 第五章 算例分析
→ 第六章 总结与展望
→ 参考文献
```

并统一：章/节/三级标题编号、图2-1/表2-1/式(2-1)、前置罗马页码、正文阿拉伯页码、GB/T 7714顺序编码参考文献等。

封面姓名、导师、专业、学校等元数据尚未硬编码，避免编造。

---

## 7. 编译

```bash
bash scripts/build_thesis.sh
```

流程：

```text
复制至隔离构建目录
→ SVG批量转PDF
→ XeLaTeX
→ BibTeX
→ XeLaTeX
→ XeLaTeX
→ main.pdf
```

GitHub Actions工作流已建立，但当前连接环境未观察到有效run，因此**定稿流程不再依赖Actions触发**；源文件、final wrapper和图表均已直接写入main分支。

---

## 8. 当前剩余定稿任务

- [x] 第一至第六章主线写作
- [x] 中文摘要 / English Abstract
- [x] 总结与展望
- [x] 杨昊东母版式章节重排与多轮Agent审稿
- [x] 主实验及稳定性结果冻结
- [x] KG语义/证据审计
- [x] 抽汽拓扑过度表述修正（Ch3 v8）
- [x] 第二至第五章核心论文图接入
- [x] 统一学位论文版式
- [x] 参考文献置于全文结尾
- [ ] 全量正文 `\\cite{}`—BibTeX key核验
- [ ] BibTeX标题/作者/年份/DOI或报告号最终核验
- [ ] `U+B→Y`、状态残差及缩写等少量术语清理
- [ ] 完整XeLaTeX/BibTeX编译并清除undefined引用
- [ ] PDF级浮动图表、分页、孤行/寡行检查
- [ ] 汽轮机/ML/KG/统计/写作/杨昊东母版最终盲审
- [ ] 最终可提交PDF交付

详细断点见 `paper_rewriting_output/FINALIZATION_CHECKLIST.md`。

---

## 9. 单一事实来源优先级

若旧草稿、聊天记录或历史实验与当前口径冲突：

```text
main.tex + final wrappers
>
Ch2 v4 / Ch3 v8 / Ch4 v7 / Ch5 v7
>
experiments/levelC_v07/正式汇总
>
最新KG审计文件
>
最新Agent Reviewer记录
>
历史草稿与开发期实验
```
