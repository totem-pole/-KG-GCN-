"""Generate Level-C VHP semi-physical fault windows from real healthy GRU residuals.

Input NPZ must contain: test_time, test_actual, test_pred, target_tags.
The split is fixed by calendar day before window construction, preventing overlap leakage.
Fault signatures are B-level semi-physical hypotheses and are kept separate from the A0 KG graph.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EXCLUDE = {"SYG1_10MAA50CT132A", "SYG1_10MAA50CT151A", "SYG1_10MAA50CT152A"}
FAULTS = ["F1_SPE通流侵蚀", "F2_一级抽汽逆止门后管路泄漏", "F3_VHP汽封泄漏"]


def split_of_day(day: int) -> str:
    return "train" if day % 5 in (1, 2, 3) else ("val" if day % 5 == 4 else "test")


def temporal_profile(kind: str, onset: int, L: int) -> np.ndarray:
    x = np.arange(L)
    if kind == "step":
        return (x >= onset).astype(np.float32)
    if kind == "ramp":
        g = np.zeros(L, np.float32)
        m = x >= onset
        g[m] = (x[m] - onset) / max(1, L - 1 - onset)
        return g
    center = onset + (L - onset) * 0.45
    beta = 8.0 / max(4, L - onset)
    g = 1.0 / (1.0 + np.exp(-beta * (x - center)))
    g = (g - g[0]) / (g[-1] - g[0] + 1e-9)
    g[x < onset] *= 0.15
    return g.astype(np.float32)


def load_signatures(path: str, tags: list[str]) -> dict[str, np.ndarray]:
    sig = pd.read_csv(path)
    out = {}
    for fault in FAULTS:
        v = np.zeros(len(tags), np.float32)
        part = sig[sig["fault"] == fault]
        for _, r in part.iterrows():
            if r["target_tag"] not in tags:
                raise KeyError(f"Signature tag is not in frozen VHP nodes: {r['target_tag']}")
            v[tags.index(r["target_tag"])] = float(r["weight_sigma"])
        out[fault] = v
    return out


def make_split(z, time, split, which, signatures, L, seed, sev_range, noise):
    rng = np.random.default_rng(seed)
    X, y, rows = [], [], []
    label_map = {"F0_normal": 0, **{f: i + 1 for i, f in enumerate(FAULTS)}}

    for date in pd.Series(time.normalize()).drop_duplicates():
        inds = np.where(time.normalize() == date)[0]
        if len(inds) < L or split[inds[0]] != which:
            continue
        for s in range(0, len(inds) - L + 1, L):
            ii = inds[s:s + L]
            base = z[ii].T.copy()
            X.append(base)
            y.append(0)
            rows.append([str(pd.Timestamp(date).date()), int(ii[0]), "F0_normal", 0.0, "none", -1])

            for fault in FAULTS:
                severity = float(rng.uniform(*sev_range))
                kind = str(rng.choice(["ramp", "sigmoid", "step"], p=[0.45, 0.40, 0.15]))
                onset = int(rng.integers(8, 24))
                spatial = signatures[fault].copy()
                active = np.abs(spatial) > 0
                spatial[active] *= rng.normal(1.0, 0.12, active.sum()).astype(np.float32)
                inject = severity * spatial[:, None] * temporal_profile(kind, onset, L)[None, :]
                sample = base + inject + rng.normal(0.0, noise, base.shape).astype(np.float32)
                X.append(sample)
                y.append(label_map[fault])
                rows.append([str(pd.Timestamp(date).date()), int(ii[0]), fault, severity, kind, onset])

    manifest = pd.DataFrame(rows, columns=["date", "start_index", "fault", "severity", "profile", "onset"])
    return np.stack(X).astype(np.float32), np.asarray(y, np.int64), manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gru-npz", required=True)
    ap.add_argument("--signature-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=int, default=60)
    args = ap.parse_args()

    src = np.load(args.gru_npz, allow_pickle=True)
    all_tags = list(src["target_tags"].astype(str))
    keep = [i for i, t in enumerate(all_tags) if t not in EXCLUDE]
    tags = [all_tags[i] for i in keep]
    if len(tags) != 24:
        raise ValueError(f"Expected 24 final VHP targets, got {len(tags)}")

    time = pd.to_datetime(src["test_time"], unit="s")
    residual = (src["test_actual"][:, keep] - src["test_pred"][:, keep]).astype(np.float32)
    split = np.array([split_of_day(d) for d in time.day.to_numpy()])

    train_mask = split == "train"
    mu = residual[train_mask].mean(0)
    sd = residual[train_mask].std(0, ddof=1)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = ((residual - mu) / sd).astype(np.float32)
    signatures = load_signatures(args.signature_csv, tags)

    configs = {
        "train": (701, (0.55, 1.25), 0.035),
        "val": (702, (0.50, 1.30), 0.040),
        "test": (703, (0.50, 1.30), 0.045),
    }
    made = {}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, (seed, sev, noise) in configs.items():
        X, y, manifest = make_split(z, time, split, name, signatures, args.window, seed, sev, noise)
        made[name] = (X, y)
        manifest.to_csv(out / f"{name}_manifest.csv", index=False, encoding="utf-8-sig")
        print(name, X.shape, np.bincount(y))

    np.savez_compressed(
        out / "levelC_dataset_v03_regenerated.npz",
        X_train=made["train"][0], y_train=made["train"][1],
        X_val=made["val"][0], y_val=made["val"][1],
        X_test=made["test"][0], y_test=made["test"][1],
        tags=np.asarray(tags),
        healthy_residual_mean=mu.astype(np.float32),
        healthy_residual_std=sd.astype(np.float32),
    )


if __name__ == "__main__":
    main()
