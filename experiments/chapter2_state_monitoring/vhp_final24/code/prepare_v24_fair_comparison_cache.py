from __future__ import annotations

"""Build the V24 fair-comparison cache.

The cache keeps the V23 causal 16-input boundary and removes only the two
structurally unobservable drain-temperature outputs.  All four data-driven
models therefore see exactly the same 27 targets, timestamps and samples.
"""

import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "56_v23_no_condenser_ablation" / "cache"
OUT = ROOT / "58_v24_fair_comparison" / "cache"
DROP_TARGETS = {
    "SYG1_10MAL41CT101A": "#1抽汽逆止门前疏水阀后温度：阀未动作时无稳定可观测通流边界",
    "SYG1_10MAL67CT101A": "超高压冷再热逆止阀前疏水阀后温度：阀未动作时无稳定可观测通流边界",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8-sig"))
    old_targets = list(metadata["targets"])
    keep_indices = [i for i, tag in enumerate(old_targets) if tag not in DROP_TARGETS]
    targets = [old_targets[i] for i in keep_indices]
    if len(old_targets) != 29 or len(targets) != 27:
        raise ValueError(f"Expected 29 -> 27 targets, got {len(old_targets)} -> {len(targets)}")

    for protocol in metadata["protocols"]:
        source_path = SOURCE / f"{protocol}.npz"
        with np.load(source_path) as data:
            payload = {key: data[key] for key in data.files}
        payload["train_y"] = payload["train_y"][:, keep_indices]
        payload["test_y"] = payload["test_y"][:, keep_indices]
        np.savez_compressed(OUT / source_path.name, **payload)

    for audit_name in ("new_point_raw_audit.csv", "protocol_audit.csv", "target_split_quality.csv"):
        source = SOURCE / audit_name
        if source.exists():
            shutil.copy2(source, OUT / audit_name)

    metadata["task"] = "V24 fair 4-algorithm comparison: causal 16 inputs -> 27 observable VHP outputs"
    metadata["targets"] = targets
    metadata["target_metadata"] = {tag: metadata["target_metadata"][tag] for tag in targets}
    metadata["v24_removed_unobservable_outputs"] = DROP_TARGETS
    metadata["fair_comparison"] = {
        "algorithms": ["ANN", "GRU-RNN", "TCN", "iTransformer-small"],
        "same_inputs": True,
        "same_targets": True,
        "same_train_test_rows": True,
        "target_history_input": False,
        "condenser_backpressure_input": False,
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8-sig"
    )
    print(f"Built {OUT}")
    print(f"inputs={len(metadata['baseline_inputs'])}; targets={len(targets)}")
    print("removed=" + ",".join(DROP_TARGETS))


if __name__ == "__main__":
    main()
