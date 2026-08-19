#!/usr/bin/env python3
"""Materialize the final thesis figures into the current semantic source files.

This script is intentionally narrow and idempotent. It is used only in the
isolated/CI build copy, so the tracked semantic source remains auditable while
the compiled thesis cannot fall back to historical placeholder boxes.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_figure_block(text: str, label: str, body: str) -> str:
    """Replace exactly the figure environment containing the requested label."""
    label_token = f"\\label{{{label}}}"
    label_pos = text.find(label_token)
    if label_pos < 0:
        raise RuntimeError(f"figure label not found: {label}")
    begin_token = "\\begin{figure}[htbp]"
    end_token = "\\end{figure}"
    start = text.rfind(begin_token, 0, label_pos)
    end_start = text.find(end_token, label_pos)
    if start < 0 or end_start < 0:
        raise RuntimeError(f"figure boundaries not found around label: {label}")
    end = end_start + len(end_token)
    return text[:start] + body + text[end:]


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
    \includegraphics[width=0.84\textwidth]{figures/ch2/fig2_5_selection_loss.pdf}
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
        anchor = r"\subsubsection{典型状态参数预测结果}"
        fig = r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.78\textwidth]{figures/ch2/fig2_6_macro_r2.pdf}
    \caption{不同状态监测模型24点测试宏平均$R^2$比较}
    \label{fig:macro_r2_v4}
\end{figure}

"""
        if anchor not in s:
            raise RuntimeError("Ch2 macro-R2 insertion anchor not found")
        s = s.replace(anchor, fig + anchor, 1)

    s = s.replace("$B+U\\rightarrow Y$", "$U+B\\rightarrow Y$")
    s = s.replace("figures/ch2/fig2_1_turbine_system_final.svg", "figures/ch2/fig2_1_turbine_system_final.pdf")
    s = s.replace("figures/ch2/fig2_2_device_model_framework_final.svg", "figures/ch2/fig2_2_device_model_framework_final.pdf")
    s = s.replace("figures/ch2/fig2_4_gru_architecture_final.svg", "figures/ch2/fig2_4_gru_architecture_final.pdf")
    write(path, s)


def patch_ch3() -> None:
    path = "sections/03_knowledge_graph_v8.tex"
    s = read(path)

    if "fig:kg_construction_flow_v8" not in s:
        anchor = r"\subsubsection{节点和关系的构建与提取}"
        fig = r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/ch3/fig3_1_kg_construction_flow.pdf}
    \caption{汽轮机系统故障知识图谱构建流程}
    \label{fig:kg_construction_flow_v8}
\end{figure}

"""
        if anchor not in s:
            raise RuntimeError("Ch3 v8 construction-flow anchor not found")
        s = s.replace(anchor, fig + anchor, 1)

    if "fig:turbine_kg_overview_v8" not in s:
        anchor = r"\subsection{本章小结}"
        fig = r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/ch3/fig3_2_turbine_kg_overview.pdf}
    \caption{汽轮机系统知识图谱设备拓扑及代表性故障语义关系}
    \label{fig:turbine_kg_overview_v8}
\end{figure}

图~\ref{fig:turbine_kg_overview_v8}按已核实层级展示抽汽拓扑；未完成机组级一致性核验的具体级次--单台设备关系及除氧器蒸汽来源不进入确定性主图或GCN投影。

"""
        if anchor not in s:
            raise RuntimeError("Ch3 v8 KG-overview anchor not found")
        s = s.replace(anchor, fig + anchor, 1)

    write(path, s)


def main() -> None:
    patch_ch2()
    patch_ch3()
    print("Final Ch2/Ch3 figures materialized into the build copy.")


if __name__ == "__main__":
    main()
