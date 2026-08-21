from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--vhp-adj",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    d=np.load(a.dataset,allow_pickle=True); tags=list(d["tags"].astype(str)); n=len(tags); A=np.zeros((n,n),np.float32); idx={t:i for i,t in enumerate(tags)}
    v=pd.read_csv(a.vhp_adj,index_col=0); vtags=v.index.astype(str).tolist()
    for i,x in enumerate(vtags):
      for j,y in enumerate(vtags):
        if v.iloc[i,j]>0: A[idx[x],idx[y]]=1
    def add(x,y):
      if x not in idx or y not in idx: raise KeyError((x,y))
      A[idx[x],idx[y]]=A[idx[y],idx[x]]=1
    for side in ("21","22"):
      ps=[f"SYG1_10LBB{side}CP10{i}" for i in (1,2,3)]; ts=[f"SYG1_10LBB{side}CT10{i}" for i in (1,2,3)]
      for seq in (ps,ts):
        add(seq[0],seq[1]); add(seq[1],seq[2])
      for p,t in zip(ps,ts): add(p,t)
    for typ in ("CP","CT"):
      for i in (1,2,3): add(f"SYG1_10LBB21{typ}10{i}",f"SYG1_10LBB22{typ}10{i}")
    for cold in ("SYG1_10LBC20CT101","SYG1_10LBC20CT103","SYG1_10LBC20CT104"):
      add(cold,"SYG1_10LBB21CT101"); add(cold,"SYG1_10LBB22CT101")
    add("SYG1_10LBC01CP101","SYG1_10LBB21CP101"); add("SYG1_10LBC02CP101","SYG1_10LBB22CP101")
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(A,index=tags,columns=tags).to_csv(out,encoding="utf-8-sig")
    print({"nodes":n,"edges":int(A.sum()/2)})
if __name__=="__main__": main()
