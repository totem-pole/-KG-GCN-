"""第二章跨设备状态监测补充图（会议论文版）。

使用冻结的2024-06独立测试预测CSV，绘制H1与DEA两个代表状态：
- H1：1号高压给水加热器出口温度；
- DEA：1号机除氧器压力1。

图形目的不是再次做完整四模型排名，而是用一张紧凑图说明统一U+B→Y健康建模方式并非只在VHP有效。
每个设备一行：左侧为完整测试月预测曲线并嵌入固定前48 h局部放大，右侧为实测-预测散点。
颜色与VHP主图统一：实际值黑、ANN红、GRU绿。

注意：
1. 只读取已经冻结的plot_data，不重新训练模型；
2. 局部放大固定为测试月起始48 h，不根据曲线好坏挑选；
3. H1/DEA正文R²采用三随机种子mean±std，图中不以单种子R²替代正文统计量；
4. CND1不进入代表图，保留在表格中作为边界结果。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "chapter2_state_monitoring" / "non_vhp_representatives_20260822" / "deliverables" / "plot_data"
OUT = ROOT / "figures" / "ch2"
OUT.mkdir(parents=True, exist_ok=True)

C_ACTUAL = "#333333"
C_ANN = "#E64B35"
C_GRU = "#00A087"
C_ZOOM = "#D62728"


def paper_rc():
    plt.rcParams.update({
        "font.sans-serif": ["SimSun", "Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.8,
        "savefig.dpi": 600,
    })


def load_case(filename):
    df = pd.read_csv(SRC / filename)
    required = {"timestamp", "actual", "pred_ANN", "pred_GRU"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"{filename} 缺少字段: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def limits_with_margin(values, margin=0.04):
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = max(hi - lo, 1e-9)
    return lo - margin * span, hi + margin * span


def draw_row(fig, gs, df, title, unit, panel_label):
    ax = fig.add_subplot(gs[0, :3])
    axs = fig.add_subplot(gs[0, 3])

    n = len(df)
    x = np.arange(n)
    stride = max(1, n // 7000)
    p = x[::stride]

    actual = df["actual"].to_numpy()
    ann = df["pred_ANN"].to_numpy()
    gru = df["pred_GRU"].to_numpy()

    ax.plot(p, actual[::stride], color=C_ACTUAL, lw=0.82, label="实际值", zorder=4)
    ax.plot(p, ann[::stride], color=C_ANN, lw=0.70, alpha=0.88, label="ANN", zorder=2)
    ax.plot(p, gru[::stride], color=C_GRU, lw=0.76, alpha=0.95, label="GRU", zorder=3)
    ax.set_xlim(0, n - 1)
    ylo, yhi = limits_with_margin(np.r_[actual, ann, gru])
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel(unit)
    ax.set_xlabel("样本")
    ax.legend(frameon=False, ncol=3, loc="upper right", handlelength=2.0, columnspacing=0.9)
    ax.grid(False)

    # 固定前48 h作为局部放大，不按结果筛选。
    t0 = df["timestamp"].iloc[0]
    zoom_n = int((df["timestamp"] < t0 + pd.Timedelta(hours=48)).sum())
    zoom_n = max(2, min(zoom_n, n))
    rect = Rectangle((0, ylo), zoom_n - 1, yhi - ylo,
                     fill=False, edgecolor=C_ZOOM, linewidth=1.05)
    ax.add_patch(rect)

    axins = inset_axes(ax, width="43%", height="45%", loc="upper left", borderpad=1.2)
    z = np.arange(zoom_n)
    axins.plot(z, actual[:zoom_n], color=C_ACTUAL, lw=0.80)
    axins.plot(z, ann[:zoom_n], color=C_ANN, lw=0.68, alpha=0.90)
    axins.plot(z, gru[:zoom_n], color=C_GRU, lw=0.73, alpha=0.95)
    zylo, zyhi = limits_with_margin(np.r_[actual[:zoom_n], ann[:zoom_n], gru[:zoom_n]], margin=0.06)
    axins.set_xlim(0, zoom_n - 1)
    axins.set_ylim(zylo, zyhi)
    axins.tick_params(labelsize=7.2)
    axins.grid(False)
    axins.set_title("固定48 h局部放大", fontsize=8.2, pad=2)

    con = ConnectionPatch(
        xyA=(zoom_n - 1, yhi), coordsA=ax.transData,
        xyB=(0.96, 0.02), coordsB=axins.transAxes,
        arrowstyle="-|>", mutation_scale=8,
        color=C_ZOOM, lw=0.85
    )
    fig.add_artist(con)

    # 散点仅固定等间隔下采样用于可视化。
    m = min(5000, n)
    idx = np.linspace(0, n - 1, m, dtype=int)
    axs.scatter(actual[idx], ann[idx], s=5.0, marker="s", color=C_ANN,
                alpha=0.30, edgecolors="none", label="ANN")
    axs.scatter(actual[idx], gru[idx], s=5.5, marker="^", color=C_GRU,
                alpha=0.38, edgecolors="none", label="GRU")
    lo = min(float(np.nanmin(actual[idx])), float(np.nanmin(ann[idx])), float(np.nanmin(gru[idx])))
    hi = max(float(np.nanmax(actual[idx])), float(np.nanmax(ann[idx])), float(np.nanmax(gru[idx])))
    span = max(hi - lo, 1e-9)
    lo -= 0.035 * span
    hi += 0.035 * span
    axs.plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=0.85, label="$y=x$")
    axs.set_xlim(lo, hi)
    axs.set_ylim(lo, hi)
    axs.set_xlabel(f"实际值/{unit}")
    axs.set_ylabel(f"预测值/{unit}")
    axs.legend(frameon=False, loc="upper left", handletextpad=0.3)
    axs.grid(False)

    ax.text(0.5, -0.23, f"{panel_label} {title}", transform=ax.transAxes,
            ha="center", va="top", fontsize=11.2)


def main():
    paper_rc()
    cases = [
        ("H1_SYG1_10LAB61CT101_predictions.csv", "1号高压给水加热器出口温度预测结果", "℃", "(a)"),
        ("DEA_SYG1_10LAA10CP101_predictions.csv", "1号机除氧器压力预测结果", "MPa", "(b)"),
    ]

    fig = plt.figure(figsize=(10.8, 7.2))
    outer = fig.add_gridspec(2, 1, hspace=0.48)
    for i, case in enumerate(cases):
        filename, title, unit, label = case
        df = load_case(filename)
        gs = outer[i].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.34)
        draw_row(fig, gs, df, title, unit, label)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.98, bottom=0.07)
    for ext in ["svg", "pdf"]:
        fig.savefig(OUT / f"fig2_8_cross_device_predictions.{ext}", bbox_inches="tight")
    fig.savefig(OUT / "fig2_8_cross_device_predictions.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
