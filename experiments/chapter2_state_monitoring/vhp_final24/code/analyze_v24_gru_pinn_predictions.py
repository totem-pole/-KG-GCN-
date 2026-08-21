from __future__ import annotations

"""Create exact 27-output GRU versus PINN-GRU per-point comparisons."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import analyze_v24_four_model_predictions as common


ROOT = Path(__file__).resolve().parent
V24 = ROOT / "58_v24_fair_comparison"
SOURCE = V24 / "pinn_gru" / "predictions"
OUT = V24 / "pinn_gru" / "same_panel_comparisons"
SEED = 20260813
DISPLAY_TO_FILE = {
    "GRU-RNN": "GRU",
    "PINN-GRU": "PINN_FULL_ADAPTIVE",
}


def load_prediction(protocol: str, display_name: str) -> dict[str, np.ndarray]:
    profile = DISPLAY_TO_FILE[display_name]
    path = SOURCE / f"{protocol}_{profile}_seed{SEED}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    common.configure_plot()
    common.ALGORITHMS = list(DISPLAY_TO_FILE)
    common.TITLE_SUFFIX = "GRU-RNN 与 PINN-GRU 同图对比"
    common.COLORS = {"GRU-RNN": "#4C72B0", "PINN-GRU": "#C44E52"}
    common.FIGURES = OUT / "per_point_png"
    pdf_dir = OUT / "per_task_pdf"
    common.FIGURES.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

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
    loaded: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    rows: list[dict[str, object]] = []
    for task, protocol, horizon_days in scenarios:
        if protocol not in loaded:
            loaded[protocol] = {name: load_prediction(protocol, name) for name in DISPLAY_TO_FILE}
        predictions = loaded[protocol]
        cutoff_s = None
        if horizon_days is not None:
            first_test = max(int(predictions[name]["test_time"][0]) for name in DISPLAY_TO_FILE)
            cutoff_s = first_test + horizon_days * 86400
        pdf_path = pdf_dir / f"{common.safe_name(task)}_27测点_GRU对PINNGRU.pdf"
        with PdfPages(pdf_path) as pdf:
            rows.extend(common.plot_task(task, protocol, predictions, targets, target_meta, cutoff_s, pdf))
        print(f"finished {task}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "V24_27测点_GRU对PINNGRU_逐任务指标.csv", index=False, encoding="utf-8-sig")
    test = metrics[metrics.split == "test"]
    summary = (
        test.groupby(["task", "algorithm"], as_index=False)
        .agg(
            macro_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            macro_nrmse=("nrmse_train_range", "mean"),
            macro_nmae=("nmae_train_range", "mean"),
            upper_1pct_bias=("upper_1pct_bias", "mean"),
            qualified_r2_ge_0_8=("r2", lambda s: int((s >= 0.8).sum())),
        )
        .sort_values(["task", "macro_r2"], ascending=[True, False])
    )
    summary.to_csv(OUT / "V24_GRU对PINNGRU_任务级汇总.csv", index=False, encoding="utf-8-sig")
    paired = test.pivot_table(
        index=["task", "target_tag", "name_cn", "unit", "group"], columns="algorithm", values=["r2", "rmse", "mae"]
    ).reset_index()
    paired.columns = ["_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else col for col in paired.columns]
    if "r2_PINN-GRU" in paired and "r2_GRU-RNN" in paired:
        paired["delta_r2_PINN_minus_GRU"] = paired["r2_PINN-GRU"] - paired["r2_GRU-RNN"]
        paired["delta_rmse_PINN_minus_GRU"] = paired["rmse_PINN-GRU"] - paired["rmse_GRU-RNN"]
        paired["delta_mae_PINN_minus_GRU"] = paired["mae_PINN-GRU"] - paired["mae_GRU-RNN"]
    paired.to_csv(OUT / "V24_GRU对PINNGRU_逐点配对差值.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
