"""第二章结果图可复现绘图脚本（会议论文版）。

输入文件：
- selection_loss.csv
- full_train_loss.csv
- month_expanded_2024_ANN.npz
- month_expanded_2024_GRU-RNN.npz
- V25_24输出_四算法_七任务逐点指标.csv

输出：
- 图2-5 模型选择阶段训练/验证损失
- 图2-6 四模型最终24点宏平均R2
- 图2-7 杨昊东式代表测点预测曲线、局部放大与散点分布

图2-7遵循以下固定协议：
1. 主图使用完整2024-06独立测试月；
2. 局部放大固定使用测试月起始48 h，不根据图形好坏重新选段；
3. 仅显示DCS实测、ANN基线与GRU，完整四模型数值比较由表格承担；
4. 散点图采用固定等间隔下采样，仅用于可视化，不改变任何评价指标。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ch2"
OUT = ROOT / "figures" / "ch2"
OUT.mkdir(parents=True, exist_ok=True)

TASK = "2024-05训练_2024-06测试"

# 颜色参照论文常见的高区分度组合：实测黑、基线红、GRU绿。
C_ACTUAL = "#333333"
C_ANN = "#E64B35"
C_GRU = "#00A087"
C_ZOOM = "#D62728"


def _paper_rc():
    plt.rcParams.update({
        "font.sans-serif": ["SimSun", "Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 9.2,
        "axes.linewidth": 0.8,
        "lines.linewidth": 0.85,
        "savefig.dpi": 600,
    })


def plot_selection_loss():
    _paper_rc()
    df = pd.read_csv(DATA / "selection_loss.csv")
    best = df.loc[df["val_loss"].idxmin()]
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    ax.plot(df["epoch"], df["train_loss"], label="训练损失")
    ax.plot(df["epoch"], df["val_loss"], label="验证损失")
    ax.scatter([best["epoch"]], [best["val_loss"]], s=42,
               label=f"最佳验证 epoch={int(best['epoch'])}")
    ax.annotate(f"{best['val_loss']:.4f}",
                (best["epoch"], best["val_loss"]),
                xytext=(best["epoch"]-42, best["val_loss"]+0.12),
                arrowprops={"arrowstyle": "->"})
    ax.set_xlabel("训练轮次 / epoch")
    ax.set_ylabel("标准化 MSE")
    ax.set_xlim(1, df["epoch"].max())
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(False)
    fig.savefig(OUT / "fig2_5_selection_loss.svg", bbox_inches="tight")
    fig.savefig(OUT / "fig2_5_selection_loss.pdf", bbox_inches="tight")
    plt.close(fig)


def load_metrics():
    df = pd.read_csv(DATA / "V25_24输出_四算法_七任务逐点指标.csv")
    return df[(df["task"] == TASK) & (df["split"] == "test")].copy()


def plot_macro_r2():
    _paper_rc()
    df = load_metrics()
    algorithms = ["ANN", "TCN", "iTransformer-small", "GRU-RNN"]
    values = [df[df["algorithm"] == a]["r2"].mean() for a in algorithms]
    fig, ax = plt.subplots(figsize=(8.1, 5.0))
    bars = ax.bar(algorithms, values)
    ax.set_ylabel("24点宏平均 $R^2$")
    ax.set_ylim(0.88, 0.97)
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, v+0.0015, f"{v:.4f}", ha="center")
    ax.grid(False)
    fig.savefig(OUT / "fig2_6_macro_r2.svg", bbox_inches="tight")
    fig.savefig(OUT / "fig2_6_macro_r2.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_prediction(model_name: str):
    z = np.load(DATA / f"month_expanded_2024_{model_name}.npz", allow_pickle=True)
    return {
        "tags": z["target_tags"].astype(str),
        "time": pd.to_datetime(z["test_time"], unit="s"),
        "actual": z["test_actual"],
        "pred": z["test_pred"],
    }


def _limits_with_margin(y):
    lo = float(np.nanmin(y))
    hi = float(np.nanmax(y))
    span = max(hi - lo, 1e-9)
    return lo - 0.035 * span, hi + 0.035 * span


def _draw_yang_panel(fig, outer_spec, sample_idx, actual, ann, gru,
                     name, unit, panel_label, r2_ann, r2_gru, zoom_n):
    """单个测点：上左局部放大 + 上右散点 + 下方全样本曲线。"""
    gs = outer_spec.subgridspec(
        2, 4, height_ratios=[1.25, 0.86], width_ratios=[1, 1, 1, 1.08],
        hspace=0.18, wspace=0.35
    )
    ax_zoom = fig.add_subplot(gs[0, :3])
    ax_scatter = fig.add_subplot(gs[0, 3])
    ax_full = fig.add_subplot(gs[1, :])

    # ---------- 全样本主图 ----------
    n = len(actual)
    stride = max(1, n // 6000)
    p = np.arange(n)[::stride]
    ax_full.plot(p, actual[::stride], color=C_ACTUAL, lw=0.80, label="实际值", zorder=4)
    ax_full.plot(p, ann[::stride], color=C_ANN, lw=0.72, alpha=0.90, label="ANN", zorder=2)
    ax_full.plot(p, gru[::stride], color=C_GRU, lw=0.78, alpha=0.95, label="GRU", zorder=3)
    ax_full.set_xlim(0, n - 1)
    ylo, yhi = _limits_with_margin(np.r_[actual, ann, gru])
    ax_full.set_ylim(ylo, yhi)
    ax_full.set_ylabel(f"{unit}")
    ax_full.set_xlabel("样本")
    ax_full.legend(loc="upper right", ncol=3, frameon=False, handlelength=2.2, columnspacing=1.0)
    ax_full.grid(False)

    # 固定测试月起始48 h作为放大窗口，不按测试结果挑选。
    zoom_n = min(zoom_n, n)
    rect = Rectangle((0, ylo), zoom_n - 1, yhi - ylo,
                     fill=False, edgecolor=C_ZOOM, linewidth=1.15)
    ax_full.add_patch(rect)

    # ---------- 局部放大 ----------
    zidx = np.arange(zoom_n)
    ax_zoom.plot(zidx, actual[:zoom_n], color=C_ACTUAL, lw=0.95, label="实际值")
    ax_zoom.plot(zidx, ann[:zoom_n], color=C_ANN, lw=0.85, label="ANN")
    ax_zoom.plot(zidx, gru[:zoom_n], color=C_GRU, lw=0.90, label="GRU")
    zylo, zyhi = _limits_with_margin(np.r_[actual[:zoom_n], ann[:zoom_n], gru[:zoom_n]])
    ax_zoom.set_ylim(zylo, zyhi)
    ax_zoom.set_xlim(0, zoom_n - 1)
    ax_zoom.set_ylabel(f"{unit}")
    ax_zoom.legend(loc="upper left", ncol=3, frameon=False, handlelength=2.0, columnspacing=0.9)
    ax_zoom.grid(False)

    # 红色箭头连接主图框选区域和局部放大图。
    con = ConnectionPatch(
        xyA=(0.50, 1.00), coordsA=ax_full.transAxes,
        xyB=(0.50, 0.00), coordsB=ax_zoom.transAxes,
        arrowstyle="-|>", mutation_scale=11,
        color=C_ZOOM, linewidth=1.05, shrinkA=5, shrinkB=5
    )
    fig.add_artist(con)

    # ---------- 实测-预测散点 ----------
    # 固定等间隔采样，避免4万点完全遮挡；评价指标仍来自完整测试月。
    m = min(6000, n)
    sidx = np.linspace(0, n - 1, m, dtype=int)
    ax_scatter.scatter(actual[sidx], ann[sidx], s=5.5, marker="s",
                       color=C_ANN, alpha=0.35,
                       label=f"ANN  $R^2$={r2_ann:.4f}", edgecolors="none")
    ax_scatter.scatter(actual[sidx], gru[sidx], s=6.0, marker="^",
                       color=C_GRU, alpha=0.40,
                       label=f"GRU  $R^2$={r2_gru:.4f}", edgecolors="none")
    slo = float(np.nanmin(actual[sidx]))
    shi = float(np.nanmax(actual[sidx]))
    span = max(shi - slo, 1e-9)
    slo -= 0.035 * span
    shi += 0.035 * span
    ax_scatter.plot([slo, shi], [slo, shi], color="#555555", ls="--", lw=0.9, label="$y=x$")
    ax_scatter.set_xlim(slo, shi)
    ax_scatter.set_ylim(slo, shi)
    ax_scatter.set_xlabel(f"实际值/{unit}")
    ax_scatter.set_ylabel(f"预测值/{unit}")
    ax_scatter.legend(loc="upper left", frameon=False, handletextpad=0.35, borderaxespad=0.15)
    ax_scatter.grid(False)

    # 与杨昊东图一致：子图说明放在每组图下方，而不是再加总标题。
    ax_full.text(0.5, -0.36,
                 f"{panel_label} {name}模型预测曲线与散点分布图",
                 ha="center", va="top", transform=ax_full.transAxes, fontsize=11.5)


def plot_prediction_examples():
    _paper_rc()
    metrics = load_metrics()
    ann_metrics = metrics[metrics["algorithm"] == "ANN"]
    gru_metrics = metrics[metrics["algorithm"] == "GRU-RNN"]

    ann_z = _load_prediction("ANN")
    gru_z = _load_prediction("GRU-RNN")

    if not np.array_equal(ann_z["tags"], gru_z["tags"]):
        raise RuntimeError("ANN与GRU的target_tags顺序不一致，禁止直接叠图。")
    if len(ann_z["time"]) != len(gru_z["time"]):
        raise RuntimeError("ANN与GRU测试样本长度不一致。")
    if not np.array_equal(ann_z["time"].values, gru_z["time"].values):
        raise RuntimeError("ANN与GRU测试时间轴不一致。")
    if not np.allclose(ann_z["actual"], gru_z["actual"], equal_nan=True):
        raise RuntimeError("ANN与GRU所使用的测试实测值不一致。")

    tags = gru_z["tags"]
    time = gru_z["time"]
    actual_all = gru_z["actual"]
    ann_all = ann_z["pred"]
    gru_all = gru_z["pred"]
    zoom_n = int((time < time.min() + pd.Timedelta(hours=48)).sum())

    examples = [
        ("SYG1_10MAA50CP101", "超高压缸叶片级压力", "MPa", "(a)"),
        ("SYG1_10MAA50CT123A", "超高压排汽蒸汽温度", "℃", "(b)"),
    ]

    fig = plt.figure(figsize=(10.8, 10.5))
    outer = fig.add_gridspec(2, 1, hspace=0.48)

    for row_i, (tag, name, unit, label) in enumerate(examples):
        idx = np.where(tags == tag)[0]
        if len(idx) != 1:
            raise RuntimeError(f"无法唯一定位代表测点 {tag}")
        idx = idx[0]
        r_ann = ann_metrics[ann_metrics["target_tag"] == tag]
        r_gru = gru_metrics[gru_metrics["target_tag"] == tag]
        if len(r_ann) != 1 or len(r_gru) != 1:
            raise RuntimeError(f"逐点指标表中无法唯一定位 {tag}")

        _draw_yang_panel(
            fig, outer[row_i], np.arange(len(time)),
            actual_all[:, idx], ann_all[:, idx], gru_all[:, idx],
            name, unit, label,
            float(r_ann.iloc[0]["r2"]), float(r_gru.iloc[0]["r2"]), zoom_n
        )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.985, bottom=0.06)
    for ext in ["svg", "pdf"]:
        fig.savefig(OUT / f"fig2_7_prediction_panels.{ext}", bbox_inches="tight")
    fig.savefig(OUT / "fig2_7_prediction_panels.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_selection_loss()
    plot_macro_r2()
    plot_prediction_examples()
