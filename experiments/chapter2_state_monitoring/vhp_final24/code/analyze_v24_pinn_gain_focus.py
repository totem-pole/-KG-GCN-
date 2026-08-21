from __future__ import annotations

"""Re-evaluate saved V24 GRU/PINN predictions without retraining.

The analysis separates near-saturated outputs from weak outputs and from the
three physics-direct groups fixed by mechanism before looking at model wins.
"""

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
V24 = ROOT / "58_v24_fair_comparison"
SOURCE = V24 / "pinn_gru" / "predictions"
CACHE = V24 / "cache"
OUT = V24 / "pinn_gain_focus"
FIGURES = OUT / "figures"
DETAILS = FIGURES / "four_panel_details"
TABLES = OUT / "tables"
PDF_DIR = OUT / "output" / "pdf"
SEED = 20260813

TASK_SPECS = [
    ("2024-05训练_2024-06测试", "2024月度", "month_expanded_2024", None),
    ("2025-08训练_2025-09测试", "2025月度", "month_expanded_2025", None),
    ("2024_10d训练_100d测试", "2024 100d", "10d300d_expanded_2024", 100),
    ("2024_10d训练_200d测试", "2024 200d", "10d300d_expanded_2024", 200),
    ("2024_10d训练_300d测试", "2024 300d", "10d300d_expanded_2024", 300),
    ("2025_10d训练_100d测试", "2025 100d", "10d200d_expanded_2025", 100),
    ("2025_10d训练_200d测试", "2025 200d", "10d200d_expanded_2025", 200),
]
TASK_ORDER = [item[0] for item in TASK_SPECS]
TASK_SHORT = {item[0]: item[1] for item in TASK_SPECS}

FLOW_PRESSURE = {
    "SYG1_10MAA50CP101",
    "SYG1_10MAA50CP102",
    "SYG1_10MAA50CP103",
    "SYG1_10LBC01CP101",
    "SYG1_10LBC02CP101",
    "SYG1_10LBQ10CP101",
    "SYG1_10LBQ10CP102",
}
INTERNAL_STEAM = {"SYG1_10MAA50CT131A", "SYG1_10MAA50CT132A"}
METAL = {
    "SYG1_10MAA50CT111A",
    "SYG1_10MAA50CT112A",
    "SYG1_10MAA50CT151A",
    "SYG1_10MAA50CT152A",
}
DIRECT_PHYSICS = FLOW_PRESSURE | INTERNAL_STEAM | METAL
DETAIL_TARGETS = [
    "SYG1_10MAA50CP101",
    "SYG1_10MAA50CP102",
    "SYG1_10MAA50CP103",
    "SYG1_10MAA50CT111A",
    "SYG1_10MAA50CT112A",
    "SYG1_10MAA50CT131A",
    "SYG1_10MAA50CT132A",
    "SYG1_10MAA50CT151A",
    "SYG1_10MAA50CT152A",
]
DETAIL_TASKS = {
    "2024_10d训练_300d测试": ("10d300d_expanded_2024", 300),
    "2025_10d训练_200d测试": ("10d200d_expanded_2025", 200),
}


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def physics_group(tag: str) -> str:
    if tag in FLOW_PRESSURE:
        return "A_通流压力"
    if tag in INTERNAL_STEAM:
        return "B_缸内蒸汽"
    if tag in METAL:
        return "C_金属温度"
    return "D_辅助输出"


def baseline_band(score: float) -> str:
    if score >= 0.98:
        return "A_近饱和(≥0.98)"
    if score >= 0.95:
        return "B_高精度(0.95-0.98)"
    if score >= 0.80:
        return "C_中等(0.80-0.95)"
    return "D_薄弱(<0.80)"


def short_tag(tag: str) -> str:
    return tag.replace("SYG1_10MAA50", "").replace("SYG1_10", "")


def load_metrics() -> tuple[pd.DataFrame, dict]:
    path = V24 / "pinn_gru" / "same_panel_comparisons" / "V24_GRU对PINNGRU_逐点配对差值.csv"
    frame = pd.read_csv(path)
    metadata = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8-sig"))
    frame["physics_group"] = frame.target_tag.map(physics_group)
    frame["baseline_band"] = frame["r2_GRU-RNN"].map(baseline_band)
    frame["rmse_reduction_pct"] = (
        (frame["rmse_GRU-RNN"] - frame["rmse_PINN-GRU"]) / frame["rmse_GRU-RNN"].replace(0, np.nan) * 100
    )
    headroom = (1.0 - frame["r2_GRU-RNN"]).clip(lower=1e-4)
    frame["r2_headroom_closed_pct"] = frame["delta_r2_PINN_minus_GRU"] / headroom * 100
    frame["PINN胜"] = frame["delta_r2_PINN_minus_GRU"] > 0
    frame["显著改善"] = (frame["delta_r2_PINN_minus_GRU"] >= 0.01) | (frame["rmse_reduction_pct"] >= 5.0)
    frame["任务简称"] = frame.task.map(TASK_SHORT)
    frame["测点简称"] = frame.target_tag.map(short_tag)
    return frame, metadata


def make_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / "V24_PINN逐点增益分层.csv", index=False, encoding="utf-8-sig")

    bands = (
        frame.groupby("baseline_band", as_index=False)
        .agg(
            样本对数=("target_tag", "size"),
            GRU平均R2=("r2_GRU-RNN", "mean"),
            PINN平均R2=("r2_PINN-GRU", "mean"),
            平均DeltaR2=("delta_r2_PINN_minus_GRU", "mean"),
            R2中位增益=("delta_r2_PINN_minus_GRU", "median"),
            PINN胜数=("PINN胜", "sum"),
            平均RMSE降幅pct=("rmse_reduction_pct", "mean"),
            平均剩余方差解释pct=("r2_headroom_closed_pct", "mean"),
        )
        .sort_values("baseline_band")
    )
    bands["胜率pct"] = bands.PINN胜数 / bands.样本对数 * 100
    bands.to_csv(TABLES / "V24_PINN基线精度分层汇总.csv", index=False, encoding="utf-8-sig")

    groups = (
        frame.groupby(["task", "任务简称", "physics_group"], as_index=False)
        .agg(
            测点数=("target_tag", "size"),
            GRU平均R2=("r2_GRU-RNN", "mean"),
            PINN平均R2=("r2_PINN-GRU", "mean"),
            平均DeltaR2=("delta_r2_PINN_minus_GRU", "mean"),
            PINN胜数=("PINN胜", "sum"),
            显著改善数=("显著改善", "sum"),
            平均RMSE降幅pct=("rmse_reduction_pct", "mean"),
        )
    )
    groups["胜率pct"] = groups.PINN胜数 / groups.测点数 * 100
    groups["direct_physics"] = groups.physics_group != "D_辅助输出"
    groups.to_csv(TABLES / "V24_PINN物理组逐任务汇总.csv", index=False, encoding="utf-8-sig")

    weak = frame[frame["r2_GRU-RNN"] < 0.80].copy()
    weak["跨过0.80门槛"] = (weak["r2_PINN-GRU"] >= 0.80) & (weak["r2_GRU-RNN"] < 0.80)
    weak = weak.sort_values(["r2_GRU-RNN", "delta_r2_PINN_minus_GRU"], ascending=[True, False])
    weak.to_csv(TABLES / "V24_PINN薄弱测点救助清单.csv", index=False, encoding="utf-8-sig")
    return bands, groups, weak


def plot_heatmap(frame: pd.DataFrame, metadata: dict) -> plt.Figure:
    order = metadata["targets"]
    matrix = frame.pivot(index="target_tag", columns="task", values="delta_r2_PINN_minus_GRU").reindex(
        index=order, columns=TASK_ORDER
    )
    labels = [f"{short_tag(tag)}  {metadata['target_metadata'][tag]['name_cn']}" for tag in order]
    values = matrix.to_numpy(dtype=float)
    limit = max(0.005, float(np.nanpercentile(np.abs(values), 97)))
    fig, ax = plt.subplots(figsize=(13.8, 11.5))
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    ax.set_xticks(np.arange(len(TASK_ORDER)), [TASK_SHORT[t] for t in TASK_ORDER], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(order)), labels)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=7.4,
                        color="white" if abs(value) > limit * 0.58 else "#202020")
    ax.set_title("PINN-GRU 相对 GRU 的逐点 ΔR²（正值代表 PINN 改善）", pad=14)
    ax.set_xlabel("评测任务")
    ax.set_ylabel("27个输出测点")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.018)
    colorbar.set_label("ΔR² = R²(PINN-GRU) - R²(GRU)")
    fig.tight_layout()
    return fig


def plot_scatter(frame: pd.DataFrame) -> plt.Figure:
    colors = {
        "A_通流压力": "#31688E",
        "B_缸内蒸汽": "#35B779",
        "C_金属温度": "#F8961E",
        "D_辅助输出": "#9E9E9E",
    }
    fig, ax = plt.subplots(figsize=(12.8, 7.3))
    for group, subset in frame.groupby("physics_group", sort=False):
        ax.scatter(
            subset["r2_GRU-RNN"], subset["delta_r2_PINN_minus_GRU"],
            s=38 if group != "D_辅助输出" else 22, alpha=0.78,
            color=colors[group], edgecolor="white", linewidth=0.35, label=group.split("_", 1)[1],
        )
    ax.axhline(0, color="#333333", lw=1)
    for x, label in [(0.80, "薄弱/中等"), (0.95, "高精度"), (0.98, "近饱和")]:
        ax.axvline(x, color="#777777", lw=0.8, ls="--")
        ax.text(x, ax.get_ylim()[1] if ax.get_ylim()[1] else 0.02, label, rotation=90, va="top", ha="right", fontsize=8)
    candidates = pd.concat(
        [frame.nsmallest(5, "r2_GRU-RNN"), frame.nlargest(5, "delta_r2_PINN_minus_GRU")]
    ).drop_duplicates(["task", "target_tag"])
    for _, row in candidates.iterrows():
        ax.annotate(
            f"{row['任务简称']} {row['测点简称']}",
            (row["r2_GRU-RNN"], row["delta_r2_PINN_minus_GRU"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7.4, alpha=0.85,
        )
    ax.set_xlabel("GRU基线 R²")
    ax.set_ylabel("PINN增益 ΔR²")
    ax.set_title("基线精度与PINN增益：99%测点看剩余误差，薄弱点看是否被救起")
    ax.grid(alpha=0.2)
    ax.legend(ncol=4, loc="lower left")
    fig.tight_layout()
    return fig


def plot_group_bars(groups: pd.DataFrame) -> plt.Figure:
    group_order = ["A_通流压力", "B_缸内蒸汽", "C_金属温度", "D_辅助输出"]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), sharey=False)
    x = np.arange(len(TASK_ORDER))
    width = 0.36
    for ax, group in zip(axes.ravel(), group_order):
        subset = groups[groups.physics_group == group].set_index("task").reindex(TASK_ORDER)
        ax.bar(x - width / 2, subset.GRU平均R2, width, label="GRU", color="#4C72B0")
        ax.bar(x + width / 2, subset.PINN平均R2, width, label="PINN-GRU", color="#C44E52")
        for i, delta in enumerate(subset.平均DeltaR2):
            y = max(subset.GRU平均R2.iloc[i], subset.PINN平均R2.iloc[i])
            ax.text(i, y + 0.012, f"Δ{delta:+.3f}", ha="center", va="bottom", fontsize=7.8)
        low = min(0.45, float(np.nanmin([subset.GRU平均R2.min(), subset.PINN平均R2.min()])) - 0.04)
        ax.set_ylim(max(-0.1, low), 1.035)
        ax.set_xticks(x, [TASK_SHORT[t] for t in TASK_ORDER], rotation=30, ha="right")
        ax.set_title(group.split("_", 1)[1])
        ax.set_ylabel("组内平均 R²")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("预先按控制方程分组：GRU 与 PINN-GRU 的成组比较", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_weak_points(weak: pd.DataFrame) -> plt.Figure:
    shown = weak.nsmallest(min(28, len(weak)), "r2_GRU-RNN").sort_values("r2_GRU-RNN")
    labels = [f"{row.任务简称} | {row.测点简称} | GRU={row._asdict().get('_8', '')}" for row in shown.itertuples()]
    labels = [f"{row['任务简称']} | {row['测点简称']} | GRU={row['r2_GRU-RNN']:.3f}" for _, row in shown.iterrows()]
    delta = shown["delta_r2_PINN_minus_GRU"].to_numpy()
    colors = np.where(delta >= 0, "#2A9D8F", "#D1495B")
    fig, ax = plt.subplots(figsize=(12.5, max(7.0, len(shown) * 0.31)))
    y = np.arange(len(shown))
    ax.barh(y, delta, color=colors, alpha=0.88)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(y, labels)
    for yi, value in zip(y, delta):
        ax.text(value + (0.0007 if value >= 0 else -0.0007), yi, f"{value:+.3f}",
                ha="left" if value >= 0 else "right", va="center", fontsize=8)
    ax.set_xlabel("ΔR²（正值=PINN救助，负值=PINN退化）")
    ax.set_ylabel("GRU基线低于0.80的测点-任务对")
    ax.set_title("最薄弱测点是否获得PINN助力（按GRU基线从低到高）")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def load_predictions(protocol: str, horizon_days: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    gru_path = SOURCE / f"{protocol}_GRU_seed{SEED}.npz"
    pinn_path = SOURCE / f"{protocol}_PINN_FULL_ADAPTIVE_seed{SEED}.npz"
    with np.load(gru_path) as gru, np.load(pinn_path) as pinn:
        times = np.intersect1d(gru["test_time"].astype(np.int64), pinn["test_time"].astype(np.int64))
        if horizon_days is not None:
            times = times[times < times[0] + horizon_days * 86400]
        gi = np.searchsorted(gru["test_time"], times)
        pi = np.searchsorted(pinn["test_time"], times)
        actual = gru["test_actual"][gi]
        if not np.allclose(actual, pinn["test_actual"][pi], rtol=1e-5, atol=1e-5):
            raise ValueError(f"Actual mismatch for {protocol}")
        tags = [str(item) for item in gru["target_tags"]]
        return times, actual, gru["test_pred"][gi], pinn["test_pred"][pi], tags


def align_mw(protocol: str, times: np.ndarray, metadata: dict) -> np.ndarray:
    with np.load(CACHE / f"{protocol}.npz") as cache:
        cache_time = cache["test_time"].astype(np.int64)
        indices = np.searchsorted(cache_time, times)
        if np.any(indices >= len(cache_time)) or not np.array_equal(cache_time[indices], times):
            raise ValueError(f"MW alignment failed for {protocol}")
        mw_index = metadata["baseline_inputs"].index("SYG1_MW")
        return cache["test_x"][indices, mw_index]


def dynamic_window(times: np.ndarray, mw: np.ndarray) -> tuple[pd.Timestamp, pd.Timestamp]:
    indexed = pd.Series(mw, index=pd.to_datetime(times, unit="s")).resample("1h").mean().interpolate(limit=3)
    movement = indexed.diff().abs().rolling(72, min_periods=24).sum()
    end = movement.idxmax()
    start = end - pd.Timedelta(hours=72)
    return start, end


def resample_frame(times: np.ndarray, values: dict[str, np.ndarray], rule: str) -> pd.DataFrame:
    frame = pd.DataFrame(values, index=pd.to_datetime(times, unit="s"))
    return frame.resample(rule).mean()


def plot_four_panel(
    task: str,
    tag: str,
    name: str,
    unit: str,
    times: np.ndarray,
    actual: np.ndarray,
    gru: np.ndarray,
    pinn: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
    metrics: pd.Series,
) -> plt.Figure:
    full = resample_frame(times, {"DCS": actual, "GRU": gru, "PINN": pinn}, "6h")
    raw = pd.DataFrame({"DCS": actual, "GRU": gru, "PINN": pinn}, index=pd.to_datetime(times, unit="s"))
    zoom = raw.loc[start:end].resample("5min").mean()
    hourly_sq = pd.DataFrame(
        {"GRU": (gru - actual) ** 2, "PINN": (pinn - actual) ** 2}, index=pd.to_datetime(times, unit="s")
    ).resample("1h").mean()
    rolling_rmse = np.sqrt(hourly_sq.rolling(6, min_periods=1).mean())
    delta_error = pd.Series(np.abs(gru - actual) - np.abs(pinn - actual), index=pd.to_datetime(times, unit="s")).resample("6h").mean()

    fig, axes = plt.subplots(2, 2, figsize=(15.3, 9.1))
    ax = axes[0, 0]
    ax.plot(full.index, full.DCS, color="#8A8A8A", lw=0.75, alpha=0.6, label="DCS 6h均值")
    ax.plot(full.index, full.GRU, color="#4C72B0", lw=1.15, label="GRU")
    ax.plot(full.index, full.PINN, color="#C44E52", lw=1.15, label="PINN-GRU")
    ax.set_title("(a) 完整测试周期：6小时趋势")
    ax.set_ylabel(f"{name} ({unit})" if unit else name)
    ax.legend(ncol=3, fontsize=8)

    ax = axes[0, 1]
    ax.plot(zoom.index, zoom.DCS, color="#777777", lw=0.85, alpha=0.65, label="DCS")
    ax.plot(zoom.index, zoom.GRU, color="#4C72B0", lw=1.1, label="GRU")
    ax.plot(zoom.index, zoom.PINN, color="#C44E52", lw=1.1, label="PINN-GRU")
    ax.set_title(f"(b) 最大自然变负荷72h：{start:%Y-%m-%d} 至 {end:%m-%d}")
    ax.legend(ncol=3, fontsize=8)

    ax = axes[1, 0]
    ax.plot(rolling_rmse.index, rolling_rmse.GRU, color="#4C72B0", lw=0.85, label="GRU 6h滚动RMSE")
    ax.plot(rolling_rmse.index, rolling_rmse.PINN, color="#C44E52", lw=0.85, label="PINN-GRU 6h滚动RMSE")
    ax.set_title("(c) 误差曲线：避免预测线重合掩盖差异")
    ax.set_ylabel(f"RMSE ({unit})" if unit else "RMSE")
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 1]
    ax.plot(delta_error.index, delta_error, color="#2A9D8F", lw=0.85)
    ax.axhline(0, color="#333333", lw=0.9)
    positive = delta_error >= 0
    ax.fill_between(delta_error.index, 0, delta_error, where=positive, color="#2A9D8F", alpha=0.28, label="PINN误差更小")
    ax.fill_between(delta_error.index, 0, delta_error, where=~positive, color="#D1495B", alpha=0.22, label="GRU误差更小")
    ax.set_title("(d) 相对改善 Δe=|e_GRU|-|e_PINN|（6小时均值）")
    ax.set_ylabel(f"Δ绝对误差 ({unit})" if unit else "Δ绝对误差")
    ax.legend(ncol=2, fontsize=8)

    for ax in axes.ravel():
        ax.grid(alpha=0.18)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    title = (
        f"{TASK_SHORT[task]} | {tag} | {name} | "
        f"R² {metrics['r2_GRU-RNN']:.3f}→{metrics['r2_PINN-GRU']:.3f} "
        f"(Δ{metrics['delta_r2_PINN_minus_GRU']:+.3f}, RMSE降幅{metrics['rmse_reduction_pct']:+.1f}%)"
    )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def save_figure(fig: plt.Figure, name: str, pdf: PdfPages) -> None:
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    configure_plot()
    for directory in (FIGURES, DETAILS, TABLES, PDF_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    frame, metadata = load_metrics()
    bands, groups, weak = make_tables(frame)
    pdf_path = PDF_DIR / "V24_PINN增益与薄弱点分析_仅重绘未重训.pdf"

    with PdfPages(pdf_path) as pdf:
        save_figure(plot_heatmap(frame, metadata), "01_27点7任务_DeltaR2热力图", pdf)
        save_figure(plot_scatter(frame), "02_GRU基线R2与PINN增益散点图", pdf)
        save_figure(plot_group_bars(groups), "03_物理组_GRU对PINN成组比较", pdf)
        save_figure(plot_weak_points(weak), "04_薄弱测点_PINN救助情况", pdf)

        indexed_metrics = frame.set_index(["task", "target_tag"])
        for task, (protocol, horizon) in DETAIL_TASKS.items():
            times, actual, gru, pinn, tags = load_predictions(protocol, horizon)
            mw = align_mw(protocol, times, metadata)
            start, end = dynamic_window(times, mw)
            task_dir = DETAILS / TASK_SHORT[task].replace(" ", "_")
            task_dir.mkdir(parents=True, exist_ok=True)
            for tag in DETAIL_TARGETS:
                j = tags.index(tag)
                meta = metadata["target_metadata"][tag]
                fig = plot_four_panel(
                    task, tag, meta["name_cn"], meta["unit"], times,
                    actual[:, j], gru[:, j], pinn[:, j], start, end,
                    indexed_metrics.loc[(task, tag)],
                )
                output = task_dir / f"{tag}_{meta['name_cn']}.png"
                fig.savefig(output, bbox_inches="tight")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    manifest_sources = sorted(SOURCE.glob("*.npz"))
    manifest = {
        "created_at": pd.Timestamp.now().isoformat(),
        "training_executed": False,
        "statement": "All metrics and figures were recomputed only from saved V24 NPZ predictions and cache arrays.",
        "source_prediction_files": [
            {"path": str(path), "size": path.stat().st_size, "sha256": file_sha256(path)} for path in manifest_sources
        ],
        "definitions": {
            "near_saturated": "GRU R2 >= 0.98",
            "weak": "GRU R2 < 0.80",
            "meaningful_improvement": "delta R2 >= 0.01 OR RMSE reduction >= 5%",
            "physics_groups_fixed_before_evaluation": {
                "flow_pressure": sorted(FLOW_PRESSURE),
                "internal_steam": sorted(INTERNAL_STEAM),
                "metal": sorted(METAL),
            },
        },
        "final_pdf": str(pdf_path),
    }
    (OUT / "V24_PINN增益分析_未重训清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(bands.to_string(index=False))
    print(groups[groups.direct_physics].to_string(index=False))
    print(f"weak_pairs={len(weak)}")
    print(pdf_path)


if __name__ == "__main__":
    main()
