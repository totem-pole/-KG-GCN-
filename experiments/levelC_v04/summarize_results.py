from pathlib import Path
import argparse
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('inputs', nargs='+')
p.add_argument('--model', required=True)
p.add_argument('--out', required=True)
a=p.parse_args()
frames=[]
for f in a.inputs:
    d=pd.read_csv(f)
    d.insert(0,'model',a.model)
    frames.append(d)
x=pd.concat(frames,ignore_index=True)
s=pd.DataFrame([{
    'model':a.model,
    'n_seeds':len(x),
    'accuracy_mean':x['accuracy'].mean(),
    'accuracy_std':x['accuracy'].std(ddof=1),
    'macro_precision_mean':x['macro_precision'].mean(),
    'macro_recall_mean':x['macro_recall'].mean(),
    'macro_f1_mean':x['macro_f1'].mean(),
    'macro_f1_std':x['macro_f1'].std(ddof=1),
}])
out=Path(a.out)
out.parent.mkdir(parents=True,exist_ok=True)
s.to_csv(out,index=False,encoding='utf-8-sig')
print(s.to_string(index=False))
