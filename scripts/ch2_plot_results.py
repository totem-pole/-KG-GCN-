"""第二章结果图可复现绘图脚本。

输入文件：
- selection_loss.csv
- full_train_loss.csv
- month_expanded_2024_GRU-RNN.npz
- V25_24输出_四算法_七任务逐点指标.csv

输出：
- 图2-5 模型选择阶段训练/验证损失
- 图2-6 四模型最终24点宏平均R2
- 图2-7 四个代表测点预测曲线

图2-7固定使用2024-06独立测试集前48 h作局部可视化，R2仍引用完整测试月指标；
不得根据图形好坏重新选择时间段。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ch2"
OUT = ROOT / "figures" / "ch2"
OUT.mkdir(parents=True, exist_ok=True)

TASK = "2024-05训练_2024-06测试"


def plot_selection_loss():
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


def plot_prediction_examples():
    metrics = load_metrics()
    gru = metrics[metrics["algorithm"] == "GRU-RNN"]
    z = np.load(DATA / "month_expanded_2024_GRU-RNN.npz", allow_pickle=True)
    tags = z["target_tags"].astype(str)
    time = pd.to_datetime(z["test_time"], unit="s")
    actual = z["test_actual"]
    pred = z["test_pred"]
    mask = (time >= time.min()) & (time < time.min() + pd.Timedelta(hours=48))

    examples = [
        ("SYG1_10MAA50CP101", "超高压缸叶片级压力#1", "MPa"),
        ("SYG1_10MAA50CT123A", "超高压排汽蒸汽温度#3", "℃"),
        ("SYG1_10LBQ10CT104", "一级抽汽逆止门后水平管管顶温度", "℃"),
        ("SYG1_10MAA50CT111A", "超高压内缸壁温90%处#1", "℃"),
    ]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
    for ax, (tag, name, unit), label in zip(axes.flat, examples, labels):
        idx = np.where(tags == tag)[0][0]
        row = gru[gru["target_tag"] == tag].iloc[0]
        ax.plot(time[mask], actual[mask, idx], label="DCS实测值", linewidth=1.1)
        ax.plot(time[mask], pred[mask, idx], label="GRU预测值", linewidth=1.0)
        ax.set_ylabel(f"{name} / {unit}")
        ax.set_title(f"{label} {name} ($R^2$={row['r2']:.4f})")
        ax.legend(frameon=False)
        ax.grid(False)
        ax.tick_params(axis="x", rotation=20)
    for ax in axes[1, :]:
        ax.set_xlabel("时间")
    fig.savefig(OUT / "fig2_7_prediction_panels.svg", bbox_inches="tight")
    fig.savefig(OUT / "fig2_7_prediction_panels.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_selection_loss()
    plot_macro_r2()
    plot_prediction_examples()
