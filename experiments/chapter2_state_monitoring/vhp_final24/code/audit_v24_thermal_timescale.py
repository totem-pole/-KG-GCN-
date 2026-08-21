from __future__ import annotations

"""Training-only identification of VHP local steam/metal thermal time scales."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
V24 = ROOT / "58_v24_fair_comparison"
CACHE = V24 / "cache"
OUT = V24 / "thermal_timescale_audit"
HORIZONS_MIN = (1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360)
LAGS_MIN = tuple(range(0, 181, 2))


def configure_plot() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def aligned_future(time_s: np.ndarray, horizon_min: int) -> tuple[np.ndarray, np.ndarray]:
    future_time = time_s + horizon_min * 60
    future_idx = np.searchsorted(time_s, future_time)
    valid = (future_idx < len(time_s))
    current_idx = np.flatnonzero(valid)
    future_idx = future_idx[valid]
    exact = time_s[future_idx] == future_time[current_idx]
    return current_idx[exact], future_idx[exact]


def trimmed_lstsq(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    finite = np.isfinite(design).all(axis=1) & np.isfinite(target)
    design, target = design[finite], target[finite]
    if len(target) < 100:
        return np.full(design.shape[1], np.nan)
    low, high = np.quantile(target, [0.0025, 0.9975])
    keep = (target >= low) & (target <= high)
    design, target = design[keep], target[keep]
    ridge = 1e-8 * np.eye(design.shape[1])
    ridge[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + ridge, design.T @ target)


def identify_horizon(
    time_s: np.ndarray,
    driver: np.ndarray,
    response: np.ndarray,
    load: np.ndarray,
    horizon_min: int,
) -> dict[str, float]:
    current, future = aligned_future(time_s, horizon_min)
    split_time = np.quantile(time_s, 0.70)
    train = time_s[current] <= split_time
    valid = ~train
    gap = driver[current] - response[current]
    delta_load = load[future] - load[current]
    design = np.column_stack([gap, delta_load, load[current], np.ones(len(current))])
    delta_response = response[future] - response[current]
    beta = trimmed_lstsq(design[train], delta_response[train])
    predicted_delta = design[valid] @ beta
    predicted = response[current[valid]] + predicted_delta
    truth = response[future[valid]]
    persistence = response[current[valid]]
    rmse = math.sqrt(mean_squared_error(truth, predicted))
    persistence_rmse = math.sqrt(mean_squared_error(truth, persistence))
    a = float(beta[0])
    tau = -horizon_min / math.log1p(-a) if 0 < a < 0.999 else np.nan
    return {
        "horizon_min": horizon_min,
        "samples_train": int(train.sum()),
        "samples_validation": int(valid.sum()),
        "r2_future": r2_score(truth, predicted),
        "rmse_future": rmse,
        "rmse_persistence": persistence_rmse,
        "skill_vs_persistence": 1.0 - rmse / persistence_rmse if persistence_rmse > 0 else np.nan,
        "gap_coefficient_a": a,
        "estimated_tau_min": tau,
        "delta_load_coefficient": float(beta[1]),
        "load_coefficient": float(beta[2]),
        "intercept": float(beta[3]),
    }


def lag_correlation(
    time_s: np.ndarray, driver: np.ndarray, response: np.ndarray, lag_min: int, diff_min: int = 10
) -> tuple[float, int]:
    # Compare response change over the latest diff_min with an earlier driver
    # change.  Exact timestamp matching prevents gaps from becoming fake lags.
    target_times = time_s
    y0_idx = np.searchsorted(time_s, target_times - diff_min * 60)
    u1_idx = np.searchsorted(time_s, target_times - lag_min * 60)
    u0_idx = np.searchsorted(time_s, target_times - (lag_min + diff_min) * 60)
    valid = (y0_idx < len(time_s)) & (u1_idx < len(time_s)) & (u0_idx < len(time_s))
    idx = np.flatnonzero(valid)
    valid_exact = (
        (time_s[y0_idx[idx]] == target_times[idx] - diff_min * 60)
        & (time_s[u1_idx[idx]] == target_times[idx] - lag_min * 60)
        & (time_s[u0_idx[idx]] == target_times[idx] - (lag_min + diff_min) * 60)
    )
    idx = idx[valid_exact]
    if len(idx) < 100:
        return np.nan, len(idx)
    dy = response[idx] - response[y0_idx[idx]]
    du = driver[u1_idx[idx]] - driver[u0_idx[idx]]
    threshold_y = np.quantile(np.abs(dy), 0.30)
    threshold_u = np.quantile(np.abs(du), 0.30)
    active = (np.abs(dy) >= threshold_y) & (np.abs(du) >= threshold_u)
    dy, du = dy[active], du[active]
    return float(np.corrcoef(du, dy)[0, 1]) if len(dy) > 3 else np.nan, len(dy)


def main() -> None:
    configure_plot()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    metadata = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8-sig"))
    inputs = list(metadata["baseline_inputs"])
    targets = list(metadata["targets"])
    input_temp_indices = [inputs.index(tag) for tag in inputs if "10LBA21CT" in tag or "10LBA22CT" in tag]
    load_index = inputs.index("SYG1_MW")
    pairs = [
        ("main_to_internal100", "主蒸汽平均温度→超高压缸蒸汽温度100%", None, "SYG1_10MAA50CT131A"),
        ("main_to_internal50", "主蒸汽平均温度→超高压缸蒸汽温度50%", None, "SYG1_10MAA50CT132A"),
        ("internal100_to_wall90_1", "缸内蒸汽100%→内缸壁温90%#1", "SYG1_10MAA50CT131A", "SYG1_10MAA50CT111A"),
        ("internal100_to_wall90_2", "缸内蒸汽100%→内缸壁温90%#2", "SYG1_10MAA50CT131A", "SYG1_10MAA50CT112A"),
        ("internal50_to_casing_top", "缸内蒸汽50%→中部上半汽缸温度", "SYG1_10MAA50CT132A", "SYG1_10MAA50CT151A"),
        ("internal50_to_casing_bottom", "缸内蒸汽50%→中部下半汽缸温度", "SYG1_10MAA50CT132A", "SYG1_10MAA50CT152A"),
    ]

    horizon_rows: list[dict[str, object]] = []
    lag_rows: list[dict[str, object]] = []
    for year, protocol in ((2024, "month_expanded_2024"), (2025, "month_expanded_2025")):
        with np.load(CACHE / f"{protocol}.npz") as data:
            x = data["train_x"].astype(np.float64)
            y = data["train_y"].astype(np.float64)
            time_s = data["train_time"].astype(np.int64)
        load = x[:, load_index]
        main_temperature = np.median(x[:, input_temp_indices], axis=1)
        for pair_key, pair_name, driver_tag, response_tag in pairs:
            driver = main_temperature if driver_tag is None else y[:, targets.index(driver_tag)]
            response = y[:, targets.index(response_tag)]
            for horizon in HORIZONS_MIN:
                row = identify_horizon(time_s, driver, response, load, horizon)
                horizon_rows.append(
                    {
                        "year": year,
                        "protocol": protocol,
                        "pair_key": pair_key,
                        "pair_name": pair_name,
                        "driver_tag": driver_tag or "median_6_main_steam_temperature_inputs",
                        "response_tag": response_tag,
                        **row,
                    }
                )
            for lag in LAGS_MIN:
                correlation, samples = lag_correlation(time_s, driver, response, lag)
                lag_rows.append(
                    {
                        "year": year,
                        "protocol": protocol,
                        "pair_key": pair_key,
                        "pair_name": pair_name,
                        "lag_min": lag,
                        "change_correlation": correlation,
                        "samples": samples,
                    }
                )

    horizons = pd.DataFrame(horizon_rows)
    lags = pd.DataFrame(lag_rows)
    horizons.to_csv(OUT / "thermal_first_order_horizon_scan.csv", index=False, encoding="utf-8-sig")
    lags.to_csv(OUT / "thermal_change_lag_scan.csv", index=False, encoding="utf-8-sig")

    best_skill = horizons.loc[horizons.groupby(["year", "pair_key"])["skill_vs_persistence"].idxmax()].copy()
    best_lag = lags.loc[lags.groupby(["year", "pair_key"])["change_correlation"].idxmax()].copy()
    summary = best_skill.merge(
        best_lag[["year", "pair_key", "lag_min", "change_correlation"]], on=["year", "pair_key"], how="left"
    )
    summary.to_csv(OUT / "thermal_timescale_summary.csv", index=False, encoding="utf-8-sig")

    for pair_key, pair_name, _, _ in pairs:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        for year, color in ((2024, "#4C72B0"), (2025, "#C44E52")):
            h = horizons[(horizons.year == year) & (horizons.pair_key == pair_key)]
            l = lags[(lags.year == year) & (lags.pair_key == pair_key)]
            axes[0].plot(h.horizon_min, h.skill_vs_persistence, marker="o", ms=3, color=color, label=str(year))
            axes[1].plot(l.lag_min, l.change_correlation, color=color, label=str(year))
        axes[0].axhline(0, color="black", lw=0.7)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("一阶热惯性预测步长 (分钟)")
        axes[0].set_ylabel("相对持久性基线技能")
        axes[0].grid(alpha=0.25)
        axes[0].legend()
        axes[1].set_xlabel("驱动温度变化领先时间 (分钟)")
        axes[1].set_ylabel("10分钟温变相关系数")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        fig.suptitle(pair_name)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"{pair_key}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    report_lines = [
        "# V24 局部热响应时间尺度审计",
        "",
        "## 方法",
        "",
        "- 仅使用各协议的训练月（2024-05、2025-08），不触碰测试月。",
        "- 扫描 1–360 min，拟合 `ΔT_m = a(T_s-T_m)+bΔMW+cMW+d`，前70%拟合、后30%验证。",
        "- 同时扫描 0–180 min 的自然温变滞后相关，只配对时间戳连续的样本。",
        "- 这是时间尺度反辨识，不是把测量输出历史喂给故障诊断网络。",
        "",
        "## 训练段结果摘要",
        "",
        "| 年份 | 物理对 | 最佳辨识步长(min) | skill vs persistence | 变化最佳领先(min) | corr | τ估计(min) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["pair_key", "year"]).iterrows():
        tau = "N/A" if not np.isfinite(row.estimated_tau_min) else f"{row.estimated_tau_min:.1f}"
        report_lines.append(
            f"| {int(row.year)} | {row.pair_name} | {int(row.horizon_min)} | {row.skill_vs_persistence:.3f} | "
            f"{int(row.lag_min)} | {row.change_correlation:.3f} | {tau} |"
        )
    report_lines += [
        "",
        "## 文献与建模解读",
        "",
        "- 蒸汽容积/通流动态和金属导热不是同一时间尺度；文献中再热蒸汽容积时间常数可为秒级，不能用它代替汽缸金属热惯性。",
        "- 汽轮机缸体瞬态研究使用三维导热/CHT，说明不同位置响应不能用一个统一固定窗口代替。",
        "- 因此 V24 的候选窗口应由上表和跨年一致性决定；若分钟级更稳定，就不应强行使用 1/3/6 h。",
        "",
        "## 一手来源",
        "",
        "1. [Finite element analysis of the three-dimensional transient temperature field in steam turbine casings](https://doi.org/10.1016/0020-7403(93)90003-D)",
        "2. [Thermo-Structural Analysis of Steam Turbine Start-Up with and Without Integrated Pre-Warming](https://doi.org/10.1115/1.4049502)",
        "3. [Full conjugate heat transfer modelling for steam turbines in transient operations](https://doi.org/10.1016/j.ijthermalsci.2017.10.025)",
        "4. [Automatic generation control under varying steam-turbine dynamic parameters](https://doi.org/10.1049/joe.2016.0178)",
        "",
        "## 限制",
        "",
        "当前只能辨识自然运行中的等效时间尺度，它混合了测点套管、局部对流、金属导热和控制动作。未有缸壁厚度、材料热物性和热电偶安装深度时，不应把辨识的 τ 声称为纯材料导热常数。",
        "",
        "AI-assisted search and analysis were used; all cited claims should be checked against the linked primary papers before manuscript submission.",
    ]
    (OUT / "V24_热响应时间尺度审计.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(summary[["year", "pair_name", "horizon_min", "skill_vs_persistence", "lag_min", "change_correlation", "estimated_tau_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
