from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def split_of_day(day: int) -> str:
    return "train" if day % 5 in (1, 2, 3) else ("val" if day % 5 == 4 else "test")


def temporal_profile(kind: str, onset: int, length: int) -> np.ndarray:
    x = np.arange(length)
    if kind == "step":
        return (x >= onset).astype(np.float32)
    if kind == "ramp":
        y = np.zeros(length, np.float32)
        active = x >= onset
        y[active] = (x[active] - onset) / max(1, length - 1 - onset)
        return y
    center = onset + (length - onset) * 0.5
    beta = 8.0 / max(4, length - onset)
    y = 1.0 / (1.0 + np.exp(-beta * (x - center)))
    y = (y - y[0]) / (y[-1] - y[0] + 1e-9)
    y[x < onset] *= 0.1
    return y.astype(np.float32)


def contiguous_windows(times: pd.DatetimeIndex, length: int):
    sec = times.astype("datetime64[s]").astype(np.int64)
    day = times.normalize()
    for date in pd.Series(day).drop_duplicates():
        ids = np.where(day == date)[0]
        if not len(ids):
            continue
        gap = np.r_[True, np.diff(sec[ids]) > 90]
        segment = np.cumsum(gap)
        for sid in np.unique(segment):
            sids = ids[segment == sid]
            for start in range(0, len(sids) - length + 1, length):
                yield sids[start:start + length]


def build_signature_matrix(device_cfg: dict, tags: list[str], out: Path):
    names = list(device_cfg["faults"])
    matrix = np.zeros((len(names), len(tags)), np.float32)
    rows = []
    for i, name in enumerate(names):
        fault = device_cfg["faults"][name]
        for tag, weight in fault["weights"].items():
            if tag not in tags:
                raise KeyError(f"{name}: missing tag {tag}")
            matrix[i, tags.index(tag)] = float(weight)
            rows.append([name, fault["region"], tag, float(weight)])
    pd.DataFrame(rows, columns=["fault", "region", "target_tag", "weight_sigma"]).to_csv(
        out / "frozen_fault_signature_matrix.csv", index=False, encoding="utf-8-sig")
    return names, matrix


def build_adjacency(device_cfg: dict, tags: list[str], out: Path):
    a = np.zeros((len(tags), len(tags)), np.float32)
    for left, right in device_cfg["edges"]:
        i, j = tags.index(left), tags.index(right)
        a[i, j] = a[j, i] = 1
    pd.DataFrame(a, index=tags, columns=tags).to_csv(out / "kg_adjacency.csv", encoding="utf-8-sig")
    return a


def make_samples(z, times, which, names, signatures, length, seed, severity, profiles, jitter, dropout, noise):
    rng = np.random.default_rng(seed)
    xs, ys, rows = [], [], []
    for ii in contiguous_windows(times, length):
        if split_of_day(times[ii[0]].day) != which:
            continue
        base = z[ii].T.copy()
        xs.append(base); ys.append(0)
        rows.append([str(times[ii[0]].date()), int(ii[0]), "F0_normal", 0.0, "none", -1, 0])
        for fidx, name in enumerate(names, 1):
            sev = float(rng.uniform(*severity))
            kind = str(rng.choice(profiles))
            onset = int(rng.integers(8, 24))
            spatial = signatures[fidx - 1].copy()
            active = np.where(spatial != 0)[0]
            spatial[active] *= rng.normal(1.0, jitter, len(active)).astype(np.float32)
            dropped = active[rng.random(len(active)) < dropout]
            spatial[dropped] = 0.0
            inject = sev * spatial[:, None] * temporal_profile(kind, onset, length)[None, :]
            sample = base + inject + rng.normal(0, noise, base.shape).astype(np.float32)
            xs.append(sample); ys.append(fidx)
            rows.append([str(times[ii[0]].date()), int(ii[0]), name, sev, kind, onset, len(dropped)])
    manifest = pd.DataFrame(rows, columns=["date", "start_index", "fault", "severity", "profile", "onset", "dropped_active_nodes"])
    return np.stack(xs).astype(np.float32), np.asarray(ys, np.int64), manifest


def build_prediction_device(device: str, pred_path: Path, device_cfg: dict, root: Path):
    out = root / device
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(pred_path, allow_pickle=True)
    tags = list(d["target_tags"].astype(str))
    names_cn = list(d["target_names"].astype(str))
    units = list(d["target_units"].astype(str))
    times = pd.to_datetime(d["test_time"].astype(np.int64), unit="s")
    residual = (d["test_actual"] - d["test_pred"]).astype(np.float32)
    train_mask = np.asarray([split_of_day(day) == "train" for day in times.day])
    mu = residual[train_mask].mean(0)
    sd = residual[train_mask].std(0, ddof=1)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = ((residual - mu) / sd).astype(np.float32)
    fault_names, signatures = build_signature_matrix(device_cfg, tags, out)
    build_adjacency(device_cfg, tags, out)
    specs = {
        "train": (2101, (2.0, 3.0), ["ramp", "sigmoid"], 0.15, 0.05, 0.035),
        "val": (2102, (2.0, 3.0), ["ramp", "sigmoid"], 0.15, 0.05, 0.040),
        "test": (2103, (2.0, 3.0), ["ramp", "sigmoid"], 0.15, 0.05, 0.045),
    }
    save = {}
    for split, spec in specs.items():
        x, y, manifest = make_samples(z, times, split, fault_names, signatures, 60, *spec)
        save[f"X_{split}"] = x; save[f"y_{split}"] = y
        manifest.to_csv(out / f"{split}_manifest.csv", index=False, encoding="utf-8-sig")
    x, y, m = make_samples(z, times, "test", fault_names, signatures, 60, 2104, (2.0, 3.0), ["step"], 0.25, 0.10, 0.045)
    save["X_test_ood"] = x; save["y_test_ood"] = y
    m.to_csv(out / "test_ood_profile_manifest.csv", index=False, encoding="utf-8-sig")
    x, y, m = make_samples(z, times, "test", fault_names, signatures, 60, 2105, (0.5, 1.0), ["ramp", "sigmoid"], 0.25, 0.10, 0.045)
    save["X_test_weak"] = x; save["y_test_weak"] = y
    m.to_csv(out / "test_weak_manifest.csv", index=False, encoding="utf-8-sig")
    save.update(tags=np.asarray(tags), class_names=np.asarray(["F0_normal"] + fault_names), healthy_mean=mu, healthy_std=sd)
    np.savez_compressed(out / f"{device}_fault_dataset.npz", **save)
    pd.DataFrame({"tag": tags, "name_cn": names_cn, "unit": units}).to_csv(out / "node_manifest.csv", index=False, encoding="utf-8-sig")
    return {"device": device, "nodes": len(tags), "classes": len(fault_names) + 1,
            "train": len(save["y_train"]), "val": len(save["y_val"]), "test": len(save["y_test"])}


def build_rh1(system_path: Path, system_adj_path: Path, root: Path):
    device = "RH1"; out = root / device; out.mkdir(parents=True, exist_ok=True)
    d = np.load(system_path, allow_pickle=True)
    tags = list(d["tags"].astype(str)); rh_idx = [i for i, t in enumerate(tags) if t.startswith("SYG1_10LBB21") or t.startswith("SYG1_10LBB22")]
    keep_names = ["F0_normal", "F5_RH1换热能力下降", "F6_RH1喷水过量", "F7_高旁异常开启"]
    old_names = list(d["class_names"].astype(str)); old_ids = [old_names.index(n) for n in keep_names]; remap = {old: new for new, old in enumerate(old_ids)}
    save = {}
    for suffix in ["train", "val", "test", "test_ood", "test_weak"]:
        x = d[f"X_{suffix}"]; y = d[f"y_{suffix}"]
        mask = np.isin(y, old_ids)
        save[f"X_{suffix}"] = x[mask][:, rh_idx, :].astype(np.float32)
        save[f"y_{suffix}"] = np.asarray([remap[int(v)] for v in y[mask]], np.int64)
    rh_tags = [tags[i] for i in rh_idx]
    save.update(tags=np.asarray(rh_tags), class_names=np.asarray(keep_names))
    np.savez_compressed(out / "RH1_fault_dataset.npz", **save)
    a = pd.read_csv(system_adj_path, index_col=0).loc[rh_tags, rh_tags]
    a.to_csv(out / "kg_adjacency.csv", encoding="utf-8-sig")
    pd.DataFrame({"tag": rh_tags, "name_cn": rh_tags, "unit": "-"}).to_csv(out / "node_manifest.csv", index=False, encoding="utf-8-sig")
    return {"device": device, "nodes": len(rh_tags), "classes": len(keep_names),
            "train": len(save["y_train"]), "val": len(save["y_val"]), "test": len(save["y_test"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1-pred", required=True); ap.add_argument("--dea-pred", required=True)
    ap.add_argument("--system-data", required=True); ap.add_argument("--system-adj", required=True)
    ap.add_argument("--config", required=True); ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(); root = Path(args.out_dir); root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = [
        build_prediction_device("H1", Path(args.h1_pred), cfg["H1"], root),
        build_prediction_device("DEA", Path(args.dea_pred), cfg["DEA"], root),
        build_rh1(Path(args.system_data), Path(args.system_adj), root),
    ]
    pd.DataFrame(summary).to_csv(root / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
