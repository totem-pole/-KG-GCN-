#!/usr/bin/env python3
"""Replace the Chapter-1 overall-framework placeholder by the final vector figure.

This runs only in the disposable build copy.  It identifies the figure by its
stable label instead of relying on the ordinal position of an \fbox command.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "sections/01_introduction.tex"
LABEL = r"\label{fig:overall_framework}"
BEGIN = r"\begin{figure}[htbp]"
END = r"\end{figure}"

text = PATH.read_text(encoding="utf-8")
pos = text.find(LABEL)
if pos < 0:
    raise RuntimeError("fig:overall_framework label not found")
start = text.rfind(BEGIN, 0, pos)
end_start = text.find(END, pos)
if start < 0 or end_start < 0:
    raise RuntimeError("overall-framework figure boundaries not found")
end = end_start + len(END)
replacement = r"""\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/ch1/fig1_overall_framework.pdf}
    \caption{基于健康状态建模与KG-GCN的燃煤机组汽轮机系统故障诊断总体框架}
    \label{fig:overall_framework}
\end{figure}"""
PATH.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
print("Chapter-1 overall-framework placeholder replaced by final figure.")
