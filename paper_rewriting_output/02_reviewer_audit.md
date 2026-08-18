# 第二章预审稿审计（Nature-reviewer思路）

## Reviewer A：研究对象与叙事边界
### Major Concern A1 — Blocking: No
正文必须始终保持“燃煤机组汽轮机系统”为研究对象，VHP仅作为代表性案例。当前2.1先给出VHP-RH1-HP-RH2-IP-LP-凝汽器系统链，再说明统一设备级建模范式，符合要求。

### Major Concern A2 — Blocking: Yes if violated later
不能在没有对应结果文件时宣称全部设备均完成训练并达到某一精度。当前正文只写“框架可统一应用，VHP详细展示”，没有伪造其他设备结果，保留。

## Reviewer B：技术一致性
### Major Concern B1 — Blocking: No, but must explain
真实训练模型为16输入、27候选输出；最终对比与KG节点为24状态点。当前正文已区分“27候选健康状态”和“24最终诊断状态”。后续建议在最终表格脚注中明确不进入24点统计的3个候选变量，避免读者疑问。

### Major Concern B2 — Blocking: No
ANN使用当前时刻输入（window=1），GRU/TCN/iTransformer使用24点历史窗口，因此四模型并非获得完全相同的时间信息。当前正文没有称其为“严格相同输入张量”，而表述为“相同16个原始变量、相同候选输出，ANN作为无显式历史的静态基线”，技术上可接受。投稿前若时间允许，可增加24×16展开后的MLP作为额外公平静态基线。

### Minor B3
训练代码中的architecture字符串仍残留历史17输入/21输出描述，但configuration.json和实际模型对象为24×16->GRU(32)->27。论文已以configuration和实际模型为准，不能复制历史字符串。

## Reviewer C：结果与证据链
### Major Concern C1 — Blocking: No
2024-05训练、2024-06测试的24点宏平均R2为：ANN 0.9087，TCN 0.9397，iTransformer 0.9356，GRU 0.9511。GRU在17/24点最优，iTransformer在7/24点最优。当前正文没有声称GRU在所有点均最优，并主动披露CT111A上iTransformer更优，叙事可信。

### Major Concern C2 — Blocking: No
仅一个月训练、一个月测试不足以证明整个汽轮机系统的长期泛化，但第二章在全文中只承担“健康基准可用性验证”，而非全文核心创新。主文可保持2024协议，2025协议或更长跨度结果可视篇幅放入补充材料/稳健性检查。

### Major Concern C3 — Blocking: No
不应对压力与温度的RMSE/MAE直接做宏平均，因为单位不同。当前正文只对无量纲R2做24点宏平均，RMSE/MAE限定为同物理量或逐点比较，保留。

## 当前状态
ready_with_author_checks

## 需要作者重点核对的4项
1. “某1000 MW超超临界、二次再热、单轴、四缸四排汽凝汽式”是否允许在投稿中公开且描述完全准确。
2. 27候选输出缩减为24最终诊断状态时，是否希望正文明确写出3个未进入最终统计的点号。
3. 图2-3最终使用哪一版VHP漂亮设备/测点图，必须去掉历史‘待核/高旁专项/消融’文字。
4. 2025-08->09是否放Supplementary作为稳健性验证，主文仍只展示2024-05->06。
