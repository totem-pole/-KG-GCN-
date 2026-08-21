from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from generate_other_device_datasets import build_prediction_device


def is_pressure(name: str) -> bool:
    return "压力" in name


def is_level(name: str) -> bool:
    return "水位" in name


def is_drain_temp(name: str) -> bool:
    return "疏水温度" in name


def is_outlet_temp(name: str) -> bool:
    return "出口温度" in name or "出水温度" in name or "进口凝结水温度" in name


def build_config(pred_path: Path, device: str):
    d = np.load(pred_path, allow_pickle=True)
    tags = list(d["target_tags"].astype(str)); names = list(d["target_names"].astype(str))
    weights_heat, weights_leak, weights_level = {}, {}, {}
    pressures, levels, temps = [], [], []
    for tag, name in zip(tags, names):
        if is_pressure(name):
            pressures.append(tag); weights_heat[tag] = 0.10; weights_leak[tag] = -1.00; weights_level[tag] = 0.15
        elif is_level(name):
            levels.append(tag); weights_heat[tag] = 0.05; weights_leak[tag] = -0.25; weights_level[tag] = 1.00
        elif is_drain_temp(name):
            temps.append(tag); weights_heat[tag] = 0.30; weights_leak[tag] = -0.55; weights_level[tag] = -0.25
        elif is_outlet_temp(name):
            temps.append(tag); weights_heat[tag] = -1.00; weights_leak[tag] = -0.70; weights_level[tag] = -0.40
        else:
            temps.append(tag); weights_heat[tag] = -0.70; weights_leak[tag] = -0.55; weights_level[tag] = -0.30

    edges = []
    for group in [pressures, levels]:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                edges.append([group[i], group[j]])
    for i in range(len(temps) - 1): edges.append([temps[i], temps[i + 1]])
    for p in pressures:
        for t in temps: edges.append([p, t])
        if levels: edges.append([p, levels[0]])
    if temps and levels:
        edges.append([temps[-1], levels[0]])

    return {
        "faults": {
            f"F1_{device}换热能力下降": {"region": f"{device}汽侧-疏水-出水接口", "weights": weights_heat},
            f"F2_{device}汽侧供汽不足": {"region": f"{device}抽汽入口-汽侧", "weights": weights_leak},
            f"F3_{device}水位过高": {"region": f"{device}壳侧水位-换热区", "weights": weights_level}
        },
        "edges": edges
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--runs-dir",required=True); ap.add_argument("--out-dir",required=True);a=ap.parse_args()
    runs=Path(a.runs_dir); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    configs={}; summary=[]
    for device in ["H2","H3","H4","L6","L9"]:
        pred=runs/device/"GRU"/"seed20260801"/"predictions.npz"
        cfg=build_config(pred,device);configs[device]=cfg
        summary.append(build_prediction_device(device,pred,cfg,out))
    (out/"expanded_device_configs.json").write_text(json.dumps(configs,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
