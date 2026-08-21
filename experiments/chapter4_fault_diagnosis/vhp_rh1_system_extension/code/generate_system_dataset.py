from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

VHP_EXCLUDE = {"SYG1_10MAA50CT132A", "SYG1_10MAA50CT151A", "SYG1_10MAA50CT152A"}


def split_of_day(day: int) -> str:
    return "train" if day % 5 in (1, 2, 3) else ("val" if day % 5 == 4 else "test")


def profile(kind: str, onset: int, length: int) -> np.ndarray:
    x = np.arange(length)
    if kind == "step":
        return (x >= onset).astype(np.float32)
    if kind == "ramp":
        y = np.zeros(length, np.float32); active = x >= onset
        y[active] = (x[active] - onset) / max(1, length - 1 - onset); return y
    center = onset + (length - onset) * 0.5
    beta = 8.0 / max(4, length - onset)
    y = 1.0 / (1.0 + np.exp(-beta * (x - center)))
    y = (y - y[0]) / (y[-1] - y[0] + 1e-9); y[x < onset] *= 0.1
    return y.astype(np.float32)


def build_signatures(config: dict, tags: list[str], out: Path) -> tuple[list[str], np.ndarray]:
    names = list(config)
    matrix = np.zeros((len(names), len(tags)), np.float32)
    rows = []
    for i, name in enumerate(names):
        for tag, value in config[name]["weights"].items():
            if tag not in tags:
                raise KeyError(f"signature tag not available: {name} / {tag}")
            matrix[i, tags.index(tag)] = float(value)
            rows.append([name, config[name]["region"], tag, float(value)])
    pd.DataFrame(rows, columns=["fault", "region", "target_tag", "weight_sigma"]).to_csv(
        out / "frozen_fault_signature_matrix.csv", index=False, encoding="utf-8-sig")
    active = matrix != 0
    jac = np.zeros((len(names), len(names)), float)
    for i in range(len(names)):
        for j in range(len(names)):
            union = np.logical_or(active[i], active[j]).sum()
            jac[i, j] = np.logical_and(active[i], active[j]).sum() / max(union, 1)
    pd.DataFrame(jac, index=names, columns=names).to_csv(out / "fault_active_node_jaccard.csv", encoding="utf-8-sig")
    return names, matrix


def make_windows(z: np.ndarray, times: pd.DatetimeIndex, split: np.ndarray, which: str,
                 fault_names: list[str], signatures: np.ndarray, length: int, seed: int,
                 severity: tuple[float, float], kinds: list[str], jitter: float,
                 node_dropout: float, noise: float):
    rng = np.random.default_rng(seed); xs=[]; ys=[]; rows=[]
    for date in pd.Series(times.normalize()).drop_duplicates():
        ids = np.where(times.normalize() == date)[0]
        if len(ids) < length or split[ids[0]] != which: continue
        for start in range(0, len(ids)-length+1, length):
            ii=ids[start:start+length]; base=z[ii].T.copy()
            xs.append(base); ys.append(0); rows.append([str(date.date()),int(ii[0]),"F0_normal",0,"none",-1,0])
            for fidx,name in enumerate(fault_names,1):
                sev=float(rng.uniform(*severity)); kind=str(rng.choice(kinds)); onset=int(rng.integers(8,24))
                spatial=signatures[fidx-1].copy(); active=np.where(spatial!=0)[0]
                spatial[active] *= rng.normal(1.0,jitter,len(active)).astype(np.float32)
                drop=active[rng.random(len(active)) < node_dropout]
                spatial[drop]=0.0
                inject=sev*spatial[:,None]*profile(kind,onset,length)[None,:]
                sample=base+inject+rng.normal(0,noise,base.shape).astype(np.float32)
                xs.append(sample); ys.append(fidx); rows.append([str(date.date()),int(ii[0]),name,sev,kind,onset,len(drop)])
    manifest=pd.DataFrame(rows,columns=["date","start_index","fault","severity","profile","onset","dropped_active_nodes"])
    return np.stack(xs).astype(np.float32),np.asarray(ys,np.int64),manifest


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--vhp-npz",required=True); ap.add_argument("--rh1-npz",required=True)
    ap.add_argument("--rh1-metrics",required=True); ap.add_argument("--signatures",required=True); ap.add_argument("--out-dir",required=True)
    ap.add_argument("--main-severity-low",type=float,default=0.70); ap.add_argument("--main-severity-high",type=float,default=1.50)
    ap.add_argument("--main-jitter",type=float,default=0.20); ap.add_argument("--main-node-dropout",type=float,default=0.10)
    ap.add_argument("--dataset-name",default="system_vhp_rh1_dataset")
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    v=np.load(args.vhp_npz,allow_pickle=True); r=np.load(args.rh1_npz,allow_pickle=True)
    all_vtags=list(v["target_tags"].astype(str)); keep=[i for i,t in enumerate(all_vtags) if t not in VHP_EXCLUDE]
    vtags=[all_vtags[i] for i in keep]
    rm=pd.read_csv(args.rh1_metrics); rtags=rm["tag"].astype(str).tolist()
    tv=v["test_time"].astype(np.int64); tr=r["time"].astype("datetime64[s]").astype(np.int64); common=np.intersect1d(tv,tr)
    iv=np.searchsorted(tv,common); ir=np.searchsorted(tr,common)
    vres=v["test_actual"][iv][:,keep]-v["test_pred"][iv][:,keep]
    rres=r["y_true"][ir]-r["y_pred"][ir]
    raw=np.concatenate([vres,rres],axis=1).astype(np.float32); tags=vtags+rtags
    times=pd.to_datetime(common,unit="s"); split=np.asarray([split_of_day(x) for x in times.day])
    train=split=="train"; mu=raw[train].mean(0); sd=raw[train].std(0,ddof=1); sd=np.where(sd<1e-8,1,sd)
    z=((raw-mu)/sd).astype(np.float32)
    config=json.loads(Path(args.signatures).read_text(encoding="utf-8")); names,signatures=build_signatures(config,tags,out)
    specs={
      "train":(801,(args.main_severity_low,args.main_severity_high),["ramp","sigmoid"],args.main_jitter,args.main_node_dropout,0.035),
      "val":(802,(args.main_severity_low,args.main_severity_high),["ramp","sigmoid"],args.main_jitter,args.main_node_dropout,0.040),
      "test":(803,(args.main_severity_low,args.main_severity_high),["ramp","sigmoid"],args.main_jitter,args.main_node_dropout,0.045),
    }
    saved={}
    for name,spec in specs.items():
        x,y,m=make_windows(z,times,split,name,names,signatures,60,*spec); saved[name]=(x,y); m.to_csv(out/f"{name}_manifest.csv",index=False,encoding="utf-8-sig")
        print(name,x.shape,np.bincount(y))
    xood,yood,mood=make_windows(z,times,split,"test",names,signatures,60,804,(args.main_severity_low,args.main_severity_high),["step"],0.35,0.20,0.045)
    xweak,yweak,mweak=make_windows(z,times,split,"test",names,signatures,60,805,(0.25,0.70),["ramp","sigmoid"],0.35,0.20,0.045)
    mood.to_csv(out/"test_ood_profile_manifest.csv",index=False,encoding="utf-8-sig"); mweak.to_csv(out/"test_weak_manifest.csv",index=False,encoding="utf-8-sig")
    np.savez_compressed(out/f"{args.dataset_name}.npz",X_train=saved["train"][0],y_train=saved["train"][1],X_val=saved["val"][0],y_val=saved["val"][1],X_test=saved["test"][0],y_test=saved["test"][1],X_test_ood=xood,y_test_ood=yood,X_test_weak=xweak,y_test_weak=yweak,tags=np.asarray(tags),class_names=np.asarray(["F0_normal"]+names),healthy_mean=mu,healthy_std=sd,main_severity=np.asarray([args.main_severity_low,args.main_severity_high],np.float32))
    pd.DataFrame({"tag":tags,"source":["VHP"]*len(vtags)+["RH1"]*len(rtags)}).to_csv(out/"system_node_manifest.csv",index=False,encoding="utf-8-sig")

if __name__=="__main__": main()
