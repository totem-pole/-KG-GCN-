from __future__ import annotations

"""Exact 27-output GRU/PINN-GRU comparison for V24.

The prediction network is identical between GRU and PINN-GRU.  Physics enters
only through training loss; no analytical output overwrites network outputs.
"""

import pickle
from pathlib import Path
from typing import Any

import train_v23_expanded_pinn_gru as base


ROOT = Path(__file__).resolve().parent
V24 = ROOT / "58_v24_fair_comparison"
CACHE = V24 / "cache"
OUT = V24 / "pinn_gru"
MODELS = OUT / "models"
PREDICTIONS = OUT / "predictions"
FIGURES = OUT / "figures"
REFERENCE_CALIBRATIONS = ROOT / "43_v21_soft_pinn_gru" / "calibration_cache"

for folder in (OUT, MODELS, PREDICTIONS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

v21 = base.v21
v21.CACHE = CACHE
v21.OUT = OUT
v21.MODELS = MODELS
v21.PREDICTIONS = PREDICTIONS
v21.FIGURES = FIGURES
v21.CALIBRATIONS = REFERENCE_CALIBRATIONS


def load_audited_reference(
    year: int,
    inputs: list[str],
    targets: list[str],
    rebuild: bool = False,
) -> dict[str, Any]:
    del inputs, targets
    if rebuild:
        raise ValueError("V24 reuses only the audited 21-point physical calibration")
    path = REFERENCE_CALIBRATIONS / f"physics_calibration_{year}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_architecture(metadata: dict[str, Any]) -> None:
    ni, no = len(metadata["baseline_inputs"]), len(metadata["targets"])
    text = f"""# V24 GRU / PINN-GRU 公平对比

`24×{ni} → 单层 GRU(hidden=32) → LayerNorm(32) → Dropout(0.05) → Linear(32,{no})`

- {no} 个输出全部由网络直接预测，不输入任何真实输出历史。
- GRU 与 PINN-GRU 的输入、结构、参数量、划分、种子和优化器一致。
- PINN-GRU 只在训练 Loss 增加已审计的无量纲物理残差；推理时仍为 `X → GRU → Ŷ`。
- 物理解不覆盖、不替换、不门控任何预测值。
- 本次先作为 V24 的精确 27 输出基准；新的缸体热惯性 Loss 必须等时间尺度反辨识后再加入。
"""
    (OUT / "V24_GRU_PINNGRU_公平结构.md").write_text(text, encoding="utf-8")


v21.load_or_build_calibration = load_audited_reference
v21.write_architecture = write_architecture


if __name__ == "__main__":
    v21.main()
