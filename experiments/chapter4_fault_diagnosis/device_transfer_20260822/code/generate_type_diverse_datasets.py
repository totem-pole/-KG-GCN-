from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from generate_other_device_datasets import build_prediction_device


def subset_prediction(src:Path,out:Path,keep):
 d=np.load(src,allow_pickle=True);tags=list(d["target_tags"].astype(str));idx=[tags.index(t) for t in keep];save={}
 for split in ["train","val","test"]:
  for kind in ["actual","pred"]:save[f"{split}_{kind}"]=d[f"{split}_{kind}"][:,idx]
  save[f"{split}_time"]=d[f"{split}_time"]
 save["target_tags"]=d["target_tags"][idx];save["target_names"]=d["target_names"][idx];save["target_units"]=d["target_units"][idx];save["input_tags"]=d["input_tags"]
 np.savez_compressed(out,**save)


def bfpt_config(tags):
 pressure=[t for t in tags if "CP" in t];temps=[t for t in tags if "CT" in t];speeds=[t for t in tags if "CS" in t or "SEL-SP" in t]
 f1={t:(-0.85 if t in speeds else -0.55 if t in temps else -0.35) for t in tags}
 f2={t:(1.00 if t in pressure else 0.70 if t in temps else -0.30) for t in tags}
 f3={t:(1.00 if t in speeds else 0.15) for t in tags}
 edges=[]
 for g in [temps,speeds]:
  for i in range(len(g)):
   for j in range(i+1,len(g)):edges.append([g[i],g[j]])
 for p in pressure:
  for t in temps+speeds:edges.append([p,t])
 return {"faults":{"F1_BFPT进汽不足":{"region":"小机进汽-通流-转速","weights":f1},"F2_BFPT排汽真空恶化":{"region":"小机排汽-凝汽器背压","weights":f2},"F3_BFPT调速异常":{"region":"小机调节系统-转速","weights":f3}},"edges":edges}


def cnd_config(tags,names):
 pressures=[t for t,n in zip(tags,names) if "压力" in n];temps=[t for t,n in zip(tags,names) if "温度" in n];levels=[t for t,n in zip(tags,names) if "液位" in n]
 f1={t:(1.0 if t in pressures else .55 if t in temps else .05) for t in tags}
 f2={t:(.70 if t in pressures else 1.0 if t in temps else .05) for t in tags}
 f3={t:(1.0 if t in levels else .15) for t in tags}
 edges=[]
 for g in [pressures,temps,levels]:
  for i in range(len(g)):
   for j in range(i+1,len(g)):edges.append([g[i],g[j]])
 for p in pressures:
  for t in temps:edges.append([p,t])
 if pressures and levels:edges.append([pressures[0],levels[0]])
 return {"faults":{"F1_CND1真空恶化":{"region":"凝汽器汽侧-真空","weights":f1},"F2_CND1循环水不足":{"region":"循环水侧-换热管束","weights":f2},"F3_CND1热井液位过高":{"region":"凝汽器热井-液位","weights":f3}},"edges":edges}


def main():
 ap=argparse.ArgumentParser();ap.add_argument("--bfpt-pred",required=True);ap.add_argument("--cnd-pred",required=True);ap.add_argument("--out-dir",required=True);a=ap.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 bd=np.load(a.bfpt_pred,allow_pickle=True);btags=list(bd["target_tags"].astype(str));keep=[t for t in btags if t not in {"SYG1_10XAC10CT113","SYG1_10XAC10CT114"}];bsub=out/"BFPT_selected_predictions.npz";subset_prediction(Path(a.bfpt_pred),bsub,keep);bcfg=bfpt_config(keep)
 cd=np.load(a.cnd_pred,allow_pickle=True);ctags=list(cd["target_tags"].astype(str));cnames=list(cd["target_names"].astype(str));ccfg=cnd_config(ctags,cnames)
 summary=[build_prediction_device("BFPT",bsub,bcfg,out),build_prediction_device("CND1",Path(a.cnd_pred),ccfg,out)]
 (out/"type_diverse_configs.json").write_text(json.dumps({"BFPT":bcfg,"CND1":ccfg},ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
