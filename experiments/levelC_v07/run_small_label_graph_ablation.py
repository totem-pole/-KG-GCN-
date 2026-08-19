"""Low-label KG/Corr/Identity graph ablation for Level-C v07.

Run one graph and one seed at a time to avoid execution timeouts.
Example:
  python run_small_label_graph_ablation.py --graph KG --seed 123 --label-fraction 0.05

Required local files:
  levelC_dataset_v07.npz
  ../levelC_v04/kg_ordered_sensor_adjacency.csv
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


def stratified_subset(y, frac, seed):
    rng=np.random.default_rng(seed); out=[]
    for c in np.unique(y):
        ids=np.where(y==c)[0]
        n=max(1,int(round(len(ids)*frac)))
        out.extend(rng.choice(ids,n,replace=False).tolist())
    return np.asarray(sorted(out))


def normalize_graph(A):
    A=A.copy().astype(np.float32)
    np.fill_diagonal(A,1.0)
    d=A.sum(1); inv=1/np.sqrt(np.maximum(d,1e-8))
    return (inv[:,None]*A*inv[None,:]).astype(np.float32)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--graph',choices=['KG','Corr','Identity'],required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--label-fraction',type=float,default=0.05)
    ap.add_argument('--data',default='levelC_dataset_v07.npz')
    ap.add_argument('--kg-adj',default='../levelC_v04/kg_ordered_sensor_adjacency.csv')
    args=ap.parse_args()

    torch.set_num_threads(1)
    d=np.load(args.data,allow_pickle=True)
    Xtr,ytr=d['X_train'],d['y_train']; Xv,yv=d['X_val'],d['y_val']; Xte,yte=d['X_test'],d['y_test']
    N,L=Xtr.shape[1],Xtr.shape[2]
    Akg=pd.read_csv(args.kg_adj,index_col=0).to_numpy(np.float32)
    if args.graph=='KG':
        A=Akg
    elif args.graph=='Identity':
        A=np.zeros_like(Akg)
    else:
        nedges=int(Akg.sum()/2)
        series=Xtr[ytr==0].transpose(1,0,2).reshape(N,-1)
        C=np.corrcoef(series)
        pairs=sorted([(abs(C[i,j]),i,j) for i in range(N) for j in range(i+1,N)],reverse=True)
        A=np.zeros_like(Akg)
        for _,i,j in pairs[:nedges]: A[i,j]=A[j,i]=1
    Ahat=normalize_graph(A)

    class Encoder(nn.Module):
        def __init__(self,hid=32):
            super().__init__(); self.register_buffer('Ahat',torch.tensor(Ahat))
            self.inp=nn.Linear(L,hid); self.g1=nn.Linear(hid,hid,bias=False); self.g2=nn.Linear(hid,hid,bias=False)
            self.n1=nn.LayerNorm(hid); self.n2=nn.LayerNorm(hid); self.out=nn.LayerNorm(hid*2)
        def forward(self,x):
            h0=torch.relu(self.inp(x))
            h1=torch.relu(self.n1(h0+self.g1(torch.matmul(self.Ahat,h0))))
            h2=torch.relu(self.n2(h1+self.g2(torch.matmul(self.Ahat,h1))))
            return self.out(torch.cat([h0,h2],-1))
    class MaskAE(nn.Module):
        def __init__(self):
            super().__init__(); self.enc=Encoder(); self.dec=nn.Linear(64,L)
        def forward(self,x): return self.dec(self.enc(x))

    seed=args.seed; torch.manual_seed(seed); np.random.seed(seed)
    ae=MaskAE(); opt=torch.optim.AdamW(ae.parameters(),lr=1e-3,weight_decay=1e-4)
    dl=DataLoader(TensorDataset(torch.tensor(Xtr)),batch_size=128,shuffle=True)
    xv=torch.tensor(Xv); best=None; best_mse=1e9; bad=0
    for ep in range(1,36):
        ae.train()
        for (xb,) in dl:
            mask=torch.rand_like(xb)<0.15
            opt.zero_grad(); rec=ae(xb.masked_fill(mask,0.0)); loss=((rec-xb)[mask]**2).mean()
            loss.backward(); opt.step()
        ae.eval()
        with torch.no_grad():
            torch.manual_seed(seed+7000+ep); vm=torch.rand_like(xv)<0.15
            vl=((ae(xv.masked_fill(vm,0.0))-xv)[vm]**2).mean().item()
        if vl<best_mse-1e-4:
            best_mse=vl; best={k:v.detach().clone() for k,v in ae.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=5: break
    ae.load_state_dict(best); enc=ae.enc
    for p in enc.parameters(): p.requires_grad=False

    sel=stratified_subset(ytr,args.label_fraction,seed)
    class Clf(nn.Module):
        def __init__(self):
            super().__init__(); self.fc=nn.Sequential(nn.Flatten(),nn.Linear(N*64,96),nn.ReLU(),nn.Dropout(.05),nn.Linear(96,4))
        def forward(self,x):
            with torch.no_grad(): h=enc(x)
            return self.fc(h)
    m=Clf(); opt=torch.optim.AdamW(m.fc.parameters(),lr=1.5e-3,weight_decay=1e-4); ce=nn.CrossEntropyLoss()
    dl=DataLoader(TensorDataset(torch.tensor(Xtr[sel]),torch.tensor(ytr[sel])),batch_size=64,shuffle=True)
    yvv=torch.tensor(yv); best=None; best_f1=-1; best_loss=1e9; bad=0
    for ep in range(1,71):
        m.train()
        for xb,yb in dl:
            opt.zero_grad(); loss=ce(m(xb),yb); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            logits=m(xv); vl=ce(logits,yvv).item(); vf=f1_score(yv,logits.argmax(1).numpy(),average='macro')
        if vf>best_f1+1e-4 or (abs(vf-best_f1)<=1e-4 and vl<best_loss):
            best_f1,best_loss=vf,vl; best={k:v.detach().clone() for k,v in m.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=8: break
    m.load_state_dict(best); m.eval()
    with torch.no_grad(): pred=m(torch.tensor(Xte)).argmax(1).numpy()
    result=pd.DataFrame([{'graph':args.graph,'seed':seed,'label_fraction':args.label_fraction,'labeled_train':len(sel),
        'accuracy':accuracy_score(yte,pred),'macro_f1':f1_score(yte,pred,average='macro'),
        'best_val_macro_f1':best_f1,'pretrain_val_mse':best_mse}])
    out=f"small{int(args.label_fraction*100):02d}_graph_{args.graph}_seed{seed}.csv"
    result.to_csv(out,index=False,encoding='utf-8-sig'); print(result.to_string(index=False))

if __name__=='__main__': main()
