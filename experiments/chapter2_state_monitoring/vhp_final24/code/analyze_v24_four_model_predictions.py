from __future__ import annotations

"""Create per-point, same-panel V24 comparisons for all four algorithms."""

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
V24 = ROOT / "58_v24_fair_comparison"
PREDICTIONS = V24 / "four_models" / "predictions"
OUT = V24 / "four_models" / "same_panel_comparisons"
FIGURES = OUT / "per_point_png"
PDFS = OUT / "per_task_pdf"
ALGORITHMS = ["ANN", "GRU-RNN", "TCN", "iTransformer-small"]
TITLE_SUFFIX = "四算法同图对比"
COLORS = {
    "ANN": "#4C72B0",
    "GRU-RNN": "#DD8452",
    "TCN": "#55A868",
    "iTransformer-small": "#8172B2",
}


def configure_plot() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", text).strip("_")


def load_prediction(protocol: str, algorithm: str) -> dict[str, np.ndarray]:
    path = PREDICTIONS / f"{protocol}_{algorithm}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def align_split(
    predictions: dict[str, dict[str, np.ndarray]], split: str, cutoff_s: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    time_key = f"{split}_time"
    actual_key = f"{split}_actual"
    pred_key = f"{split}_pred"
    common: np.ndarray | None = None
    for algorithm in ALGORITHMS:
        times = predictions[algorithm][time_key].astype(np.int64)
        common = times if common is None else np.intersect1d(common, times, assume_unique=False)
    assert common is not None
    if cutoff_s is not None:
        common = common[common < cutoff_s]
    if len(common) == 0:
        raise ValueError(f"No common timestamps for {split}")

    aligned: dict[str, np.ndarray] = {}
    actual: np.ndarray | None = None
    for algorithm in ALGORITHMS:
        times = predictions[algorithm][time_key].astype(np.int64)
        indices = np.searchsorted(times, common)
        if np.any(indices >= len(times)) or not np.array_equal(times[indices], common):
            raise ValueError(f"Timestamp alignment failed for {algorithm} {split}")
        candidate_actual = predictions[algorithm][actual_key][indices]
        if actual is None:
            actual = candidate_actual
        elif not np.allclose(actual, candidate_actual, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Actual values differ across algorithms for {split}")
        aligned[algorithm] = predictions[algorithm][pred_key][indices]
    assert actual is not None
    return common, actual, aligned


def peak_preserving_indices(y: np.ndarray, max_points: int = 5000) -> np.ndarray:
    n = len(y)
    if n <= max_points:
        return np.arange(n)
    bins = max(1, max_points // 2)
    edges = np.linspace(0, n, bins + 1, dtype=np.int64)
    chosen: list[int] = [0, n - 1]
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        segment = y[start:end]
        chosen.extend((start + int(np.nanargmin(segment)), start + int(np.nanargmax(segment))))
    return np.unique(np.asarray(chosen, dtype=np.int64))


def break_at_real_gaps(time_s: np.ndarray, selected: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Insert NaNs when a plotted segment would bridge a real DCS data gap."""
    plotted = values[selected].astype(np.float64, copy=True)
    if len(selected) < 2:
        return plotted
    positive = np.diff(time_s)
    positive = positive[positive > 0]
    nominal = int(np.median(positive)) if len(positive) else 60
    gap_edges = np.diff(time_s) > nominal * 1.5
    prefix = np.r_[0, np.cumsum(gap_edges, dtype=np.int64)]
    crosses_gap = (prefix[selected[1:]] - prefix[selected[:-1]]) > 0
    plotted[1:][crosses_gap] = np.nan
    return plotted


def metric_rows(
    task: str,
    split: str,
    times: np.ndarray,
    actual: np.ndarray,
    predicted: dict[str, np.ndarray],
    targets: list[str],
    target_meta: dict[str, dict[str, str]],
    train_reference: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for j, tag in enumerate(targets):
        train_min = float(np.min(train_reference[:, j]))
        train_max = float(np.max(train_reference[:, j]))
        train_range = train_max - train_min
        truth = actual[:, j]
        upper_threshold = float(np.quantile(truth, 0.99))
        upper = truth >= upper_threshold
        for algorithm, matrix in predicted.items():
            estimate = matrix[:, j]
            rmse = math.sqrt(mean_squared_error(truth, estimate))
            mae = mean_absolute_error(truth, estimate)
            rows.append(
                {
                    "task": task,
                    "split": split,
                    "algorithm": algorithm,
                    "target_tag": tag,
                    "name_cn": target_meta[tag]["name_cn"],
                    "unit": target_meta[tag]["unit"],
                    "group": target_meta[tag]["group"],
                    "r2": r2_score(truth, estimate),
                    "rmse": rmse,
                    "mae": mae,
                    "nrmse_train_range": rmse / train_range if train_range > 0 else np.nan,
                    "nmae_train_range": mae / train_range if train_range > 0 else np.nan,
                    "train_actual_min": train_min,
                    "train_actual_max": train_max,
                    "actual_min": float(np.min(truth)),
                    "actual_max": float(np.max(truth)),
                    "pred_min": float(np.min(estimate)),
                    "pred_max": float(np.max(estimate)),
                    "peak_error": float(np.max(estimate) - np.max(truth)),
                    "upper_1pct_bias": float(np.mean(estimate[upper] - truth[upper])),
                    "samples": len(times),
                    "first_time": pd.to_datetime(times[0], unit="s"),
                    "last_time": pd.to_datetime(times[-1], unit="s"),
                }
            )
    return rows


def plot_task(
    task: str,
    protocol: str,
    predictions: dict[str, dict[str, np.ndarray]],
    targets: list[str],
    target_meta: dict[str, dict[str, str]],
    cutoff_s: int | None,
    pdf: PdfPages,
) -> list[dict[str, object]]:
    train_time, train_actual, train_pred = align_split(predictions, "train")
    test_time, test_actual, test_pred = align_split(predictions, "test", cutoff_s)
    rows = metric_rows(task, "train", train_time, train_actual, train_pred, targets, target_meta, train_actual)
    rows += metric_rows(task, "test", test_time, test_actual, test_pred, targets, target_meta, train_actual)

    destination = FIGURES / safe_name(task)
    destination.mkdir(parents=True, exist_ok=True)
    test_metrics = pd.DataFrame(rows)
    test_metrics = test_metrics[test_metrics.split == "test"].set_index(["target_tag", "algorithm"])
    for j, tag in enumerate(targets):
        name = target_meta[tag]["name_cn"]
        unit = target_meta[tag]["unit"]
        train_idx = peak_preserving_indices(train_actual[:, j], 3000)
        test_idx = peak_preserving_indices(test_actual[:, j], 6000)
        train_dt = pd.to_datetime(train_time[train_idx], unit="s")
        test_dt = pd.to_datetime(test_time[test_idx], unit="s")
        fig, axes = plt.subplots(2, 1, figsize=(15.0, 7.7), sharey=False)
        for ax, split_name, dt, idx, actual, estimates in (
            (axes[0], "训练集", train_dt, train_idx, train_actual, train_pred),
            (axes[1], "测试集", test_dt, test_idx, test_actual, test_pred),
        ):
            raw_time = train_time if split_name == "训练集" else test_time
            ax.plot(
                dt,
                break_at_real_gaps(raw_time, idx, actual[:, j]),
                color="black",
                lw=1.15,
                label="DCS实测",
                zorder=6,
            )
            for algorithm in ALGORITHMS:
                label = algorithm
                if split_name == "测试集":
                    score = test_metrics.loc[(tag, algorithm), "r2"]
                    label += f" (R²={score:.3f})"
                ax.plot(
                    dt,
                    break_at_real_gaps(raw_time, idx, estimates[algorithm][:, j]),
                    color=COLORS[algorithm],
                    lw=0.8,
                    alpha=0.88,
                    label=label,
                )
            ax.set_title(f"{split_name} | {pd.Timestamp(dt[0]).date()} 至 {pd.Timestamp(dt[-1]).date()}")
            ax.set_ylabel(f"{name} ({unit})" if unit else name)
            ax.grid(alpha=0.22)
            ax.legend(ncol=5, fontsize=8, loc="best")
            locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axes[1].set_xlabel("时间")
        fig.suptitle(f"{task} | {tag} | {name} | {TITLE_SUFFIX}", fontsize=13)
        fig.tight_layout()
        fig_path = destination / f"{j + 1:02d}_{tag}_{safe_name(name)}.png"
        fig.savefig(fig_path, dpi=170, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return rows


def main() -> None:
    configure_plot()
    FIGURES.mkdir(parents=True, exist_ok=True)
    PDFS.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((V24 / "cache" / "metadata.json").read_text(encoding="utf-8-sig"))
    targets = list(metadata["targets"])
    target_meta = metadata["target_metadata"]
    scenarios = [
        ("2024-05训练_2024-06测试", "month_expanded_2024", None),
        ("2025-08训练_2025-09测试", "month_expanded_2025", None),
        ("2024_10d训练_100d测试", "10d300d_expanded_2024", 100),
        ("2024_10d训练_200d测试", "10d300d_expanded_2024", 200),
        ("2024_10d训练_300d测试", "10d300d_expanded_2024", 300),
        ("2025_10d训练_100d测试", "10d200d_expanded_2025", 100),
        ("2025_10d训练_200d测试", "10d200d_expanded_2025", 200),
    ]
    all_rows: list[dict[str, object]] = []
    loaded: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for task, protocol, horizon_days in scenarios:
        if protocol not in loaded:
            loaded[protocol] = {algorithm: load_prediction(protocol, algorithm) for algorithm in ALGORITHMS}
        predictions = loaded[protocol]
        cutoff_s: int | None = None
        if horizon_days is not None:
            first_test = max(int(predictions[a]["test_time"][0]) for a in ALGORITHMS)
            cutoff_s = first_test + horizon_days * 86400
        pdf_path = PDFS / f"{safe_name(task)}_27测点四算法预测曲线.pdf"
        with PdfPages(pdf_path) as pdf:
            all_rows.extend(plot_task(task, protocol, predictions, targets, target_meta, cutoff_s, pdf))
        print(f"finished {task}", flush=True)

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(OUT / "V24_27测点_四算法_逐任务指标.csv", index=False, encoding="utf-8-sig")
    test = metrics[metrics.split == "test"]
    summary = (
        test.groupby(["task", "algorithm"], as_index=False)
        .agg(
            macro_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            macro_nrmse=("nrmse_train_range", "mean"),
            macro_nmae=("nmae_train_range", "mean"),
            qualified_r2_ge_0_8=("r2", lambda s: int((s >= 0.8).sum())),
            outputs=("target_tag", "count"),
        )
        .sort_values(["task", "macro_r2"], ascending=[True, False])
    )
    summary.to_csv(OUT / "V24_四算法_任务级汇总.csv", index=False, encoding="utf-8-sig")

    architecture_rows = []
    for protocol in ("month_expanded_2024", "month_expanded_2025", "10d300d_expanded_2024", "10d200d_expanded_2025"):
        for algorithm in ALGORITHMS:
            config = json.loads(
                (V24 / "four_models" / "models" / protocol / algorithm / "configuration.json").read_text(
                    encoding="utf-8-sig"
                )
            )
            architecture_rows.append({"protocol": protocol, **config})
    pd.DataFrame(architecture_rows).to_csv(OUT / "V24_四算法_网络结构与参数.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
