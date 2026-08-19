"""Current v04 masked-pretraining audit implementation.

This file is intentionally kept because the present result is NEGATIVE:
masked pretraining + frozen encoder underperforms scratch training. Do not cite it as a
positive result; it is a reproducible development checkpoint for the next fine-tuning revision.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def row_norm(A):
    deg=A.sum(1)
    deg=np.where(deg>0,deg,1.0)
    return (A/deg[:,None]).astype(np.float32)


class Encoder(nn.Module):
    def __init__(self,A,window=60,hidden=8):
        super().__init__()
        self.register_buffer('A',torch.tensor(row_norm(A)))
        self.self_fc=nn.Linear(window,hidden)
        self.neigh_fc=nn.Linear(window,hidden,bias=False)
        self.gate=nn.Parameter(torch.tensor(-1.5))
    def forward(self,x):
        return torch.relu(self.self_fc(x)+torch.sigmoid(self.gate)*self.neigh_fc(torch.matmul(self.A,x)))


class MaskAutoencoder(nn.Module):
    def __init__(self,A,window=60,hidden=8):
        super().__init__()
        self.encoder=Encoder(A,window,hidden)
        self.decoder=nn.Linear(hidden,window)
    def forward(self,x):
        return self.decoder(self.encoder(x))


class FrozenClassifier(nn.Module):
    def __init__(self,encoder,n_nodes=24,hidden=8,n_classes=4):
        super().__init__()
        self.encoder=encoder
        for p in self.encoder.parameters(): p.requires_grad=False
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(n_nodes*hidden,48),nn.ReLU(),nn.Dropout(.05),nn.Linear(48,n_classes))
    def forward(self,x):
        with torch.no_grad(): h=self.encoder(x)
        return self.head(h)


def dense_healthy_windows(gru_npz,tags,window=60,stride=10):
    src=np.load(gru_npz,allow_pickle=True)
    all_tags=list(src['target_tags'].astype(str))
    idx=[all_tags.index(t) for t in tags]
    time=pd.to_datetime(src['test_time'],unit='s')
    residual=(src['test_actual'][:,idx]-src['test_pred'][:,idx]).astype(np.float32)
    split=np.array(['train' if d%5 in (1,2,3) else ('val' if d%5==4 else 'test') for d in time.day.to_numpy()])
    mu=residual[split=='train'].mean(0); sd=residual[split=='train'].std(0,ddof=1); sd=np.where(sd<1e-8,1,sd)
    z=((residual-mu)/sd).astype(np.float32)
    def make(which):
        out=[]
        for date in pd.Series(time.normalize()).drop_duplicates():
            inds=np.where(time.normalize()==date)[0]
            if len(inds)<window or split[inds[0]]!=which: continue
            for s in range(0,len(inds)-window+1,stride): out.append(z[inds[s:s+window]].T)
        return np.stack(out).astype(np.float32)
    return make('train'),make('val')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--gru-npz',required=True)
    ap.add_argument('--adjacency',required=True)
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--mask-ratio',type=float,default=.30)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed)

    d=np.load(a.dataset,allow_pickle=True)
    Xtr,ytr=d['X_train'].astype(np.float32),d['y_train'].astype(np.int64)
    Xv,yv=d['X_val'].astype(np.float32),d['y_val'].astype(np.int64)
    Xte,yte=d['X_test'].astype(np.float32),d['y_test'].astype(np.int64)
    tags=list(d['tags'].astype(str))
    A=pd.read_csv(a.adjacency,index_col=0).loc[tags,tags].to_numpy(np.float32)
    Utr,Uv=dense_healthy_windows(a.gru_npz,tags)

    ae=MaskAutoencoder(A,window=Xtr.shape[-1]); opt=torch.optim.AdamW(ae.parameters(),lr=1e-3,weight_decay=1e-4)
    dl=DataLoader(TensorDataset(torch.tensor(Utr)),batch_size=128,shuffle=True)
    uv=torch.tensor(Uv); best=None; bestvl=float('inf'); bad=0
    for ep in range(1,41):
        ae.train()
        for (xb,) in dl:
            mask=torch.rand_like(xb)<a.mask_ratio; xm=xb.masked_fill(mask,0.0)
            opt.zero_grad(); rec=ae(xm); loss=((rec-xb)[mask]**2).mean(); loss.backward(); opt.step()
        ae.eval()
        with torch.no_grad():
            torch.manual_seed(1000+ep); vm=torch.rand_like(uv)<a.mask_ratio
            vr=ae(uv.masked_fill(vm,0.0)); vl=((vr-uv)[vm]**2).mean().item()
        if vl<bestvl-1e-4:
            bestvl=vl; bestep=ep; bad=0; best={k:v.detach().clone() for k,v in ae.state_dict().items()}
        else:
            bad+=1
            if bad>=7: break
    ae.load_state_dict(best)

    model=FrozenClassifier(ae.encoder,n_nodes=len(tags)); opt=torch.optim.AdamW(model.head.parameters(),lr=2e-3,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
    dl=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(ytr)),batch_size=128,shuffle=True)
    xv,yvv=torch.tensor(Xv),torch.tensor(yv); best=None; bestvl_cls=float('inf'); bad=0
    for ep in range(1,81):
        model.train()
        for xb,yb in dl:
            opt.zero_grad(); loss=lf(model(xb),yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl=lf(model(xv),yvv).item()
        if vl<bestvl_cls-1e-4:
            bestvl_cls=vl; bestep_cls=ep; bad=0; best={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=10: break
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): pred=model(torch.tensor(Xte)).argmax(1).numpy()
    result=pd.DataFrame([{
        'seed':a.seed,'accuracy':accuracy_score(yte,pred),
        'macro_precision':precision_score(yte,pred,average='macro',zero_division=0),
        'macro_recall':recall_score(yte,pred,average='macro',zero_division=0),
        'macro_f1':f1_score(yte,pred,average='macro',zero_division=0),
        'pretrain_best_epoch':bestep,'pretrain_val_masked_mse':bestvl,
        'classifier_best_epoch':bestep_cls,'classifier_best_val_loss':bestvl_cls,
        'mask_ratio':a.mask_ratio,
    }])
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out,index=False,encoding='utf-8-sig')
    print(result.to_string(index=False))


if __name__=='__main__': main()
