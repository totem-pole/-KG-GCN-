# 基于KG-GCN的燃煤机组汽轮机系统的故障诊断方法

本仓库是论文正文、实验结果、知识图谱、审稿记录、图表与最终构建流程的**单一事实来源**。

> **当前状态（2026-08-20）：论文研究内容、主实验、KG审计、摘要/总结、参考文献、杨昊东式版式和主图均已完成。已有一版49页PDF完成完整XeLaTeX/BibTeX编译与逐页视觉审稿，Blocking=0、Major=0。**
>
> 该逐页审稿PDF生成后，`main` 又进行了两次仅针对第二章图位稳定性的收口修正：`9c960f0...`（7张图改为稳定章节锚点）和当前HEAD `f76a7249...`（修正第二章系统图标题锚点）。因此**最后只需从当前HEAD重新干净构建一次PDF，并回归检查第二章图位/分页，再刷新最终PDF SHA-256，即可正式交付。**

---

## 1. 当前主分支断点

当前 `main` HEAD：

```text
f76a7249e01451a7d383d1960a901554df45a909
Fix Chapter-2 system-figure anchor to match v4 heading exactly
```

最近两项正文收口：

```text
9c960f0...  Place all seven final Chapter-2 figures by stable section anchors
f76a724...  Fix Chapter-2 system-figure anchor to match v4 heading exactly
```

这些提交**没有改变实验数据、模型结果、KG证据、论文结论或章节内容，只改变第二章最终图的插入稳定性**。

详细交接见：

```text
paper_rewriting_output/HANDOFF_20260820.md
```

完整逐页审稿记录见：

```text
paper_rewriting_output/final_pdf_review_v1.md
```

---

## 2. 当前正式主稿

`main.tex` 当前引用：

```text
中文摘要：sections/00_abstract_cn.tex
英文摘要：sections/00_abstract_en.tex
第一章：sections/01_introduction_final.tex
第二章：sections/02_gru_monitoring_final.tex
第三章：sections/03_knowledge_graph_final.tex
第四章：sections/04_kg_gcn_final.tex
第五章：sections/05_experiments_final.tex
第六章：sections/06_conclusion_v2.tex
版式：styles/yang_haodong_thesis.tex
```

底层主要事实/文字源：

```text
Ch1: sections/01_introduction.tex
Ch2: sections/02_gru_monitoring_v4.tex
Ch3: sections/03_knowledge_graph_v8.tex
Ch4: sections/04_kg_gcn_v7.tex
Ch5: sections/05_experiments_v8.tex
Ch6: sections/06_conclusion_v2.tex
```

`final` wrapper只负责最终图位、少量实现口径澄清和定稿排版，不改变冻结实验结果。

---

## 3. 论文技术主线

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

全文研究对象为**燃煤机组汽轮机热力系统**；VHP仅作为完整展示方法链路的代表性算例。

---

## 4. 已冻结核心结果

### 4.1 VHP健康状态模型

- 2024-05正常运行DCS数据训练；2024-06独立测试；1 min采样。
- 训练阶段保留27个候选输出，最终评价/诊断映射为24个状态参数。
- 实际网络：`24×16 → GRU(32) → LayerNorm → Dropout(0.05) → Linear(27)`。
- 参数量：5755。

| 模型 | 24点测试宏平均R² |
|---|---:|
| ANN | 0.9087 |
| TCN | 0.9397 |
| iTransformer-small | 0.9356 |
| **GRU** | **0.9511** |

GRU在24个状态点中的17个点取得四种模型最高R²。

### 4.2 汽轮机系统故障知识图谱

- 标准实体：**127**
- 候选关系：**123**
- 进入主知识图谱：**121**
- 冲突候选：**2**，保留待核，不进入确定性KG/GCN
- DCS映射：45条，其中44条已确认
- B类点级热力推断独立保存，不作为主KG证据

第三章v8已经收紧机组拓扑口径：

- **VHP一级抽汽 → 1号高压加热器**：机组资料已核实；
- HP/IP/LP：只写到相应抽汽系统与回热设备组的系统级连接；
- 未完成机组一致性核验的“具体抽汽级次 → 单台加热器”与除氧器蒸汽来源，不作为确定性主图关系或GCN投影边。

### 4.3 VHP代表性KG-GCN诊断

3组随机种子正式汇总：

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| CNN | 98.33% ± 0.45% | 98.33% ± 0.45% |
| AE | 96.43% ± 0.82% | 96.43% ± 0.82% |
| **KG-GCN** | **98.45% ± 0.10%** | **98.46% ± 0.11%** |

固定seed=123：

| 模型 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| CNN | 97.86% | 97.89% | 97.86% | 97.85% |
| AE | 95.54% | 95.57% | 95.54% | 95.54% |
| **KG-GCN** | **98.57%** | **98.60%** | **98.57%** | **98.58%** |

20%带标签故障窗口：

```text
KG-GCN scratch                 94.48% ± 0.93%
Mask预训练 + frozen encoder   96.67% ± 0.27%
提升                           +2.19个百分点
```

---

## 5. 关键实验与结论边界

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

必须保持：

- 不将半物理样本称为现场真实故障；
- 不把VHP结果外推成HP/IP/LP/凝汽器已完成同等独立验证；
- 故障空间响应 `S_k` 不由分类KG邻接矩阵生成，避免循环验证；
- 不声称Mask预训练在所有标签比例下都提升；
- 不声称KG在所有图结构消融中都必然最好；
- 跨型式蒸汽轮机文献只作通用机理参考，不写成本文机组现场证据。

---

## 6. 杨昊东式论文写法与版式

论文写法和信息顺序以杨昊东学位论文为母版：

```text
问题背景
→ 参数/对象特性
→ 方法原理
→ 模型结构与公式
→ 算例设置
→ 主结果表
→ 混淆矩阵/图
→ 分模型解释
→ 小样本/消融补充
→ 本章小结
```

当前可见版式不是通用模板估计，而是依据杨昊东原始DOCX的OOXML反向核对：

- A4；上/下约2.0 cm，左/右约2.5 cm；
- 正文12 pt、小四、1.25倍行距、首行约2字符；
- 章标题16 pt黑体；
- 二级标题14 pt宋体加粗；
- 三级标题12 pt黑体；
- 图题/表内约10.5 pt；
- 三线表；
- GB/T 7714数字顺序编码参考文献。

版式审计记录：

```text
paper_rewriting_output/yang_format_ooxml_audit_v1.md
```

---

## 7. PDF构建与逐页审稿状态

已经完成过一次完整定稿构建与逐页审稿：

```text
页数：49页
A4：正常
undefined citation：0
undefined reference：0
Overfull hbox：0
Underfull hbox：0
Blocking：0
Major：0
```

该已审PDF记录：

```text
GitHub Actions run: 32295100893
SHA-256: 4676eee674a3d7205cbb942f08bb55b5542fbfc4eff43e127aab59fdb1f6d34b
```

**注意：这份PDF对应的是当前第二章“稳定锚点”最终修正之前的审稿快照。** 当前main已经比它更新，因此它是“已完成逐页审稿的基准PDF”，而不是当前HEAD的最终交付hash。

当前HEAD最终交付只需：

```text
1. 从 main@f76a7249... 干净构建
2. undefined citation/reference 必须保持0
3. 对比上一份49页基准PDF
4. 重点回归第二章7张图、图号、图位和分页
5. 若其余页面像素/排版未变化，可继承上一轮逐页审稿结论
6. 生成 final_pdf_review_v2.md 并记录新run id + SHA-256
7. 交付最终PDF
```

不要再生成整篇contact sheet做“一张大图分析”；应按**PDF页级回归**处理，避免工具卡住。

---

## 8. 当前论文图表

### 第一章
- 总体技术路线图

### 第二章
- 汽轮机系统简化图
- 设备级建模框架图
- VHP输入输出/测点配置图
- GRU网络结构图
- 真实训练/验证Loss图
- 四模型宏平均R²图
- 代表性DCS实测/GRU预测曲线图

### 第三章
- KG构建流程图
- 审计后系统KG拓扑及代表性故障语义图

### 第四章
- KG-GCN输入构建图
- Mask预训练与分类流程图

### 第五章
- CNN/AE/KG-GCN混淆矩阵
- 3-seed Macro-F1比较
- 20%标签Mask预训练消融

---

## 9. 构建方式

本地：

```bash
bash scripts/build_thesis.sh
```

GitHub Actions快速构建：

```text
.github/workflows/pr-build-thesis-fast.yml
```

构建过程只允许：

```text
必要的术语/转义规范化
→ SVG转换为PDF
→ XeLaTeX/BibTeX完整构建
→ undefined citation/reference检查
→ 上传main.pdf + main.log
```

不要再次用旧的 `apply_final_figures.py` 对final wrapper已经控制的章节重复插图。

---

## 10. 最终交付前唯一剩余任务

- [x] 第一至第六章正文
- [x] 中文摘要 / English Abstract
- [x] 总结与展望
- [x] 杨昊东式章节逻辑与可见版式
- [x] 主实验冻结
- [x] KG证据/拓扑审计
- [x] 引用/BibTeX核验
- [x] undefined citation/reference清零
- [x] 主图全部进入PDF
- [x] 一轮完整49页逐页视觉审稿
- [x] Blocking/Major清零
- [x] 第二章图位逻辑改为稳定章节锚点
- [ ] **从当前HEAD `f76a7249...` 重新构建最终PDF**
- [ ] **只回归检查第二章图位/分页以及由其引起的后续页码变化**
- [ ] 写入 `final_pdf_review_v2.md` 的最终run id、页数、SHA-256
- [ ] 最终PDF交付

用户本人仍需在学校正式提交前补：作者、导师、学校、专业、学号、答辩日期、学校封面/声明页等个人元数据；仓库不会擅自编造。

---

## 11. 单一事实来源优先级

如果旧草稿、历史README、聊天记录或开发期实验与当前口径冲突：

```text
main@f76a7249... + final wrappers
>
paper_rewriting_output/HANDOFF_20260820.md
>
paper_rewriting_output/final_pdf_review_v1.md
>
Ch2 v4 / Ch3 v8 / Ch4 v7 / Ch5 v8 / Ch6 v2
>
experiments/levelC_v07/正式汇总
>
最新KG审计文件
>
历史草稿与开发期实验
```
