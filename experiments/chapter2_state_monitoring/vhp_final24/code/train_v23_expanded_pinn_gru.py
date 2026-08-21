from __future__ import annotations

"""V23 entry point: identical GRU and loss-only PINN-GRU on 29 outputs."""

import pickle
from pathlib import Path
from typing import Any

import train_v21_soft_pinn_gru as v21


ROOT = Path(__file__).resolve().parent
V23 = ROOT / "55_v23_expanded_outputs"
V23_CACHE = V23 / "cache"
V23_MODELS = V23 / "models"
V23_PREDICTIONS = V23 / "predictions"
V23_FIGURES = V23 / "figures"
REFERENCE_CALIBRATIONS = ROOT / "43_v21_soft_pinn_gru" / "calibration_cache"

for folder in (V23, V23_CACHE, V23_MODELS, V23_PREDICTIONS, V23_FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

v21.CACHE = V23_CACHE
v21.OUT = V23
v21.MODELS = V23_MODELS
v21.PREDICTIONS = V23_PREDICTIONS
v21.FIGURES = V23_FIGURES
v21.CALIBRATIONS = REFERENCE_CALIBRATIONS


def load_audited_reference(
    year: int,
    inputs: list[str],
    targets: list[str],
    rebuild: bool = False,
) -> dict[str, Any]:
    """Reuse the measured-state audit built from the original physical chain.

    The eight new outputs do not yet enter Physics Loss.  Rebuilding a prior
    while pretending equations exist for them would be scientifically wrong.
    The frozen 21-point calibration is therefore reused, while all 29 outputs
    remain direct neural predictions under the same data loss.
    """

    del inputs, targets
    if rebuild:
        raise ValueError("V23 must not rebuild an unaudited 29-output physics calibration")
    path = REFERENCE_CALIBRATIONS / f"physics_calibration_{year}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing audited V21 calibration: {path}")
    print(f"[physics] loading audited 21-point {year} prior for the shared physical chain", flush=True)
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_architecture(metadata: dict[str, Any]) -> None:
    n_inputs = len(metadata["baseline_inputs"])
    n_outputs = len(metadata["targets"])
    text = f"""# V23 同架构 GRU / PINN-GRU

## 唯一预测结构

`24×{n_inputs} → 单层 GRU(hidden=32) → LayerNorm(32) → Dropout(0.05) → Linear(32,{n_outputs})`

- 两种算法的输入、窗口、网络参数、优化器、验证划分和随机种子相同。
- {n_outputs} 个输出全部由神经网络直接预测；不使用真实输出历史作为输入。
- 新增缸体/蒸汽/疏水温度仅作为输出，不回灌输入，避免故障状态被直接打印。
- PINN-GRU 只在训练 Loss 中加入已由实测数据审计的 Stodola、叶片压力、IF97 和管道热惯性残差。
- 推理路径始终是 `X → GRU → Ŷ`，物理解不覆盖、不门控、不替换任何预测值。
- 新增 8 点暂不强加未经审计的物理方程；其改善只能来自共享特征表示和物理正则化的间接作用。
"""
    (V23 / "V23_网络结构与公平对比说明.md").write_text(text, encoding="utf-8")


v21.load_or_build_calibration = load_audited_reference
v21.write_architecture = write_architecture


if __name__ == "__main__":
    v21.main()
