from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parent; D=np.load(ROOT/'data/system_vhp_rh1_dataset.npz',allow_pickle=True)
def hashes(x): return {hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in x}
sets={k:hashes(D[f'X_{k}']) for k in ('train','val','test')}
man={k:pd.read_csv(ROOT/f'data/{k}_manifest.csv') for k in ('train','val','test')}
report={
 'nodes':int(D['X_train'].shape[1]),'classes':D['class_names'].astype(str).tolist(),
 'samples':{k:int(len(D[f'X_{k}'])) for k in ('train','val','test')},
 'exact_duplicate_train_val':len(sets['train']&sets['val']),
 'exact_duplicate_train_test':len(sets['train']&sets['test']),
 'exact_duplicate_val_test':len(sets['val']&sets['test']),
 'date_overlap_train_val':sorted(set(man['train'].date)&set(man['val'].date)),
 'date_overlap_train_test':sorted(set(man['train'].date)&set(man['test'].date)),
 'date_overlap_val_test':sorted(set(man['val'].date)&set(man['test'].date)),
 'chance_accuracy':1/len(D['class_names'])
}
(ROOT/'results/dataset_leakage_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
