#!/usr/bin/env python3
"""Apply final figure insertions and minor thesis-consistency fixes.

The script is idempotent: it can be run repeatedly without duplicating figures.
It only edits the current thesis source files referenced by main.tex.
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_figure_block(text: str, label: str, body: str) -> str:
    pattern = re.compile(
        r"\\begin\{figure\}\[htbp\].*?\\label\{" + re.escape(label) + r"\}.*?\\end\{figure\}",
        re.S,
    )
    if not pattern.search(text):
        raise RuntimeError(f"figure block not found: {label}")
    return pattern.sub(body, text, count=1)


def patch_ch2() -> None:
    path = "sections/02_gru_monitoring_v4.tex"
    s = read(path)

    s = replace_figure_block(
        s,
        "fig:vhp_io_v4",
        r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.96\textwidth]{figures/ch2/fig2_3_vhp_io.pdf}
    \caption{VHP结构及状态监测测点配置}
    \label{fig:vhp_io_v4}
\end{figure}""",
    )

    s = replace_figure_block(
        s,
        "fig:gru_loss_v4",
        r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.82\textwidth]{figures/ch2/fig2_5_selection_loss.pdf}
    \caption{GRU模型选择阶段训练与验证损失}
    \label{fig:gru_loss_v4}
\end{figure}""",
    )

    s = replace_figure_block(
        s,
        "fig:prediction_examples_v4",
        r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/ch2/fig2_7_prediction_panels.pdf}
    \caption{VHP代表性状态参数的实测值与GRU预测值}
    \label{fig:prediction_examples_v4}
\end{figure}""",
    )

    if "fig:macro_r2_v4" not in s:
        anchor = (
            "GRU取得最高总体精度，相比ANN提高0.0424，相比TCN和iTransformer-small分别提高0.0114和0.0155。"
            "需要指出，ANN作为静态基线不具有与三种时序模型相同的历史信息，因此GRU相对ANN的提升同时包含“引入时间历史”和“采用门控时序结构”两方面影响；"
            "GRU与TCN、iTransformer-small之间的比较更能反映不同动态特征提取机制的差异。"
        )
        fig = r"""

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.82\textwidth]{figures/ch2/fig2_6_macro_r2.pdf}
    \caption{四种模型在VHP最终24个状态参数上的测试宏平均$R^2$}
    \label{fig:macro_r2_v4}
\end{figure}"""
        if anchor not in s:
            raise RuntimeError("Ch2 macro-R2 insertion anchor not found")
        s = s.replace(anchor, anchor + fig, 1)

    # Global terminology contract: U+B -> Y.
    s = s.replace("$B+U\\rightarrow Y$", "$U+B\\rightarrow Y$")

    # Existing final SVG figures are converted to PDF by build_thesis.sh.
    s = s.replace("figures/ch2/fig2_1_turbine_system_final.svg", "figures/ch2/fig2_1_turbine_system_final.pdf")
    s = s.replace("figures/ch2/fig2_2_device_model_framework_final.svg", "figures/ch2/fig2_2_device_model_framework_final.pdf")
    s = s.replace("figures/ch2/fig2_4_gru_architecture_final.svg", "figures/ch2/fig2_4_gru_architecture_final.pdf")

    write(path, s)


def patch_ch3() -> None:
    path = "sections/03_knowledge_graph_v7.tex"
    s = read(path)

    if "fig:kg_construction_flow_v7" not in s:
        anchor = r"\subsubsection{节点和关系的构建与提取}"
        fig = r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/ch3/fig3_1_kg_construction_flow.pdf}
    \caption{汽轮机系统故障知识图谱构建流程}
    \label{fig:kg_construction_flow_v7}
\end{figure}

"""
        if anchor not in s:
            raise RuntimeError("Ch3 construction-flow anchor not found")
        s = s.replace(anchor, fig + anchor, 1)

    if "fig:turbine_kg_overview_v7" not in s:
        anchor = r"\subsection{本章小结}"
        fig = r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.99\textwidth]{figures/ch3/fig3_2_turbine_kg_overview.pdf}
    \caption{汽轮机系统知识图谱设备拓扑及代表性故障语义关系}
    \label{fig:turbine_kg_overview_v7}
\end{figure}

图~\ref{fig:turbine_kg_overview_v7}仅展示进入主知识图谱的可审计设备拓扑及代表性故障语义链路，用于说明系统结构与故障知识的组织方式；完整关系仍以审查后的知识三元组为准。对于五级抽汽与除氧器等存在资料差异的候选接口，本文保持待核状态，不将其作为确定性图关系或后续GCN投影边。

"""
        if anchor not in s:
            raise RuntimeError("Ch3 KG-overview anchor not found")
        s = s.replace(anchor, fig + anchor, 1)

    write(path, s)


def patch_ch4() -> None:
    path = "sections/04_kg_gcn_v7.tex"
    s = read(path)

    residual_anchor = r"""\begin{equation}
e_i(t)=y_i(t)-\hat y_i(t).
\end{equation}"""
    residual_sentence = "本文将$e_i(t)$称为状态残差，下文故障诊断的动态输入均基于该定义。"
    if residual_sentence not in s:
        if residual_anchor not in s:
            raise RuntimeError("Ch4 residual-definition anchor not found")
        s = s.replace(residual_anchor, residual_anchor + "\n" + residual_sentence, 1)

    if "fig:kg_gcn_input_v7" not in s:
        anchor = r"\subsection{KG-GCN模型训练}"
        fig = r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/ch4/fig4_1_kg_gcn_input.pdf}
    \caption{KG-GCN模型输入构建过程}
    \label{fig:kg_gcn_input_v7}
\end{figure}

"""
        if anchor not in s:
            raise RuntimeError("Ch4 input-figure anchor not found")
        s = s.replace(anchor, fig + anchor, 1)

    if "fig:mask_training_v7" not in s:
        anchor = r"\subsection{本章小结}"
        fig = r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/ch4/fig4_2_mask_training.pdf}
    \caption{KG-GCN掩码预训练与故障分类流程}
    \label{fig:mask_training_v7}
\end{figure}

"""
        if anchor not in s:
            raise RuntimeError("Ch4 training-figure anchor not found")
        s = s.replace(anchor, fig + anchor, 1)

    write(path, s)


def patch_ch5() -> None:
    path = "sections/05_experiments_v7.tex"
    s = read(path)

    # Replace three raw matrix equations by a publication figure while retaining the same numerical results.
    if "fig:confusion_matrices_v7" not in s:
        pattern = re.compile(
            r"为进一步观察各故障类别的误判方向，使用相同随机种子下的测试结果构造混淆矩阵。CNN、AE和KG-GCN的混淆矩阵分别为\n"
            r"\\begin\{equation\}.*?\\end\{equation\}\n"
            r"\\begin\{equation\}.*?\\end\{equation\}\n"
            r"\\begin\{equation\}.*?\\end\{equation\}\n"
            r"矩阵行表示真实类别，列表示预测类别。",
            re.S,
        )
        replacement = r"""为进一步观察各故障类别的误判方向，使用相同随机种子下的测试结果构造混淆矩阵，如图~\ref{fig:confusion_matrices_v7}所示。

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.99\textwidth]{figures/ch5/fig5_1_confusion_matrices.pdf}
    \caption{CNN、AE和KG-GCN故障诊断混淆矩阵}
    \label{fig:confusion_matrices_v7}
\end{figure}

图中横轴表示预测类别，纵轴表示真实类别。"""
        if not pattern.search(s):
            raise RuntimeError("Ch5 confusion-matrix equations not found")
        s = pattern.sub(replacement, s, count=1)

    if "fig:main_comparison_v7" not in s:
        anchor = (
            "为考察结果对随机初始化的敏感程度，采用3组随机种子重复训练。CNN、AE和KG-GCN的平均Macro-F1分别为98.33\\%$\\pm$0.45\\%、96.43\\%$\\pm$0.82\\%和98.46\\%$\\pm$0.11\\%。"
            "KG-GCN平均F1最高，同时标准差最小，说明其在不同随机初始化条件下保持了较稳定的诊断结果。"
        )
        fig = r"""

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.78\textwidth]{figures/ch5/fig5_2_main_comparison.pdf}
    \caption{不同诊断模型三次重复实验Macro-F1比较}
    \label{fig:main_comparison_v7}
\end{figure}"""
        if anchor not in s:
            raise RuntimeError("Ch5 3-seed comparison anchor not found")
        s = s.replace(anchor, anchor + fig, 1)

    if "fig:mask_ablation_v7" not in s:
        anchor = (
            "考虑故障标签数量有限时分类性能可能进一步下降，在主对比实验之后对掩码预训练进行补充验证。"
            "当分类阶段仅使用20\\%带标签故障窗口时，直接从随机初始化训练KG-GCN的Macro-F1为94.48\\%$\\pm$0.93\\%；"
            "采用残差窗口掩码预训练并固定GCN编码器后，Macro-F1提高至96.67\\%$\\pm$0.27\\%，平均提高2.19个百分点。"
            "结果表明，在当前构造场景下，预训练能够利用不带类别标签的残差窗口提前学习参数之间的结构关联，并在故障标签减少时提供更稳定的特征表示。"
        )
        fig = r"""

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.76\textwidth]{figures/ch5/fig5_3_mask_ablation.pdf}
    \caption{20\%带标签样本下掩码预训练效果比较}
    \label{fig:mask_ablation_v7}
\end{figure}"""
        if anchor not in s:
            raise RuntimeError("Ch5 mask-ablation anchor not found")
        s = s.replace(anchor, anchor + fig, 1)

    write(path, s)


def main() -> None:
    patch_ch2()
    patch_ch3()
    patch_ch4()
    patch_ch5()
    print("Final figure insertions and consistency fixes applied.")


if __name__ == "__main__":
    main()
