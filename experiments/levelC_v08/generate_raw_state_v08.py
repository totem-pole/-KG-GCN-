"""Generate Raw-DCS-state counterpart of Level-C v07.

The fault deviation is identical to v07 in physical DCS units. Only the healthy
baseline is changed from GRU residual to raw measured DCS state.

Required local files:
  month_expanded_2024_GRU-RNN.npz
  ../levelC_v07/levelC_dataset_v07.npz
  ../levelC_v07/{train,val,test}_manifest_v07.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

HEALTH=Path('month_expanded_2024_GRU-RNN.npz')
V07=Path('../levelC_v07')
OUT=Path('.')

health=np.load(HEALTH,allow_pickle=True)
resd=np.load(V07/'levelC_dataset_v07.npz',allow_pickle=True)
tags27=list(health['target_tags'].astype(str))
tags=list(resd['tags'].astype(str))
idx=[tags27.index(t) for t in tags]

actual=health['test_actual'][:,idx].astype(np.float32)
pred=health['test_pred'][:,idx].astype(np.float32)
time=pd.to_datetime(health['test_time'],unit='s')
residual=actual-pred
day=time.day.to_numpy()
split=np.array(['train' if d%5 in (1,2,3) else 'val' if d%5==4 else 'test' for d in day])

raw_mu=actual[split=='train'].mean(0)
raw_sd=actual[split=='train'].std(0,ddof=1); raw_sd=np.where(raw_sd<1e-8,1,raw_sd)
res_sd=residual[split=='train'].std(0,ddof=1); res_sd=np.where(res_sd<1e-8,1,res_sd)

save={'tags':np.array(tags)}
rows=[]
for sp in ['train','val','test']:
    Xres=resd[f'X_{sp}']; y=resd[f'y_{sp}']
    man=pd.read_csv(V07/f'{sp}_manifest_v07.csv')
    Xraw=np.empty_like(Xres,dtype=np.float32)
    for i in range(0,len(Xres),4):
        start=int(man.loc[i,'start_index'])
        base_phys=actual[start:start+60].T
        Xraw[i]=((base_phys-raw_mu[:,None])/raw_sd[:,None]).astype(np.float32)
        for j in (1,2,3):
            k=i+j
            # v07 fault difference is in residual z-score units. Convert it back
            # to physical DCS units, then add exactly the same physical fault
            # perturbation to the raw healthy DCS window.
            delta_phys=(Xres[k]-Xres[i])*res_sd[:,None]
            fault_phys=base_phys+delta_phys
            Xraw[k]=((fault_phys-raw_mu[:,None])/raw_sd[:,None]).astype(np.float32)
    save[f'X_{sp}']=Xraw; save[f'y_{sp}']=y
    rows.append({'split':sp,'samples':len(Xraw),'mean_abs_raw_z':float(np.mean(np.abs(Xraw)))})

np.savez_compressed(OUT/'levelC_rawstate_v08.npz',**save)
pd.DataFrame(rows).to_csv(OUT/'rawstate_v08_summary.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(rows).to_string(index=False))
