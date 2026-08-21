from __future__ import annotations

"""Run the V24 27-output fair comparison with the existing compact models."""

import json
from pathlib import Path

import train_compare_four_models as base


ROOT = Path(__file__).resolve().parent
V24 = ROOT / "58_v24_fair_comparison"
CACHE = V24 / "cache"
OUT = V24 / "four_models"
OUT.mkdir(parents=True, exist_ok=True)

base.CACHE = CACHE
base.OUT = OUT


def v24_specs() -> dict[str, base.ModelSpec]:
    metadata = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8-sig"))
    ni = len(metadata["baseline_inputs"])
    no = len(metadata["targets"])
    return {
        "ANN": base.ModelSpec(
            "ANN",
            1,
            8192,
            400,
            lambda n_inputs, n_outputs: base.SmallANN(n_inputs, n_outputs),
            f"{ni}→32→16→{no}；LayerNorm+SiLU+Dropout(0.05)；当前输入同步软测量",
        ),
        "GRU-RNN": base.ModelSpec(
            "GRU-RNN",
            base.WINDOW,
            4096,
            320,
            lambda n_inputs, n_outputs: base.SmallGRU(n_inputs, n_outputs),
            f"24×{ni}→单层GRU(32)→LayerNorm+Dropout(0.05)→{no}",
        ),
        "TCN": base.ModelSpec(
            "TCN",
            base.WINDOW,
            4096,
            320,
            lambda n_inputs, n_outputs: base.SmallTCN(n_inputs, n_outputs),
            f"24×{ni}→1×1投影(16)→3个因果深度可分离残差块(k=5,d=1/2/4)→{no}",
        ),
        "iTransformer-small": base.ModelSpec(
            "iTransformer-small",
            base.WINDOW,
            2048,
            320,
            lambda n_inputs, n_outputs: base.SmallITransformer(n_inputs, n_outputs, base.WINDOW),
            f"24点历史→{ni}个变量token；1层4头Encoder(d=32,ff=64)+{no}输出查询token",
        ),
    }


base.specs = v24_specs


if __name__ == "__main__":
    base.main()
