from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix

def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def subset(y,frac,seed):
 if frac>=.999:return np.arange(len(y))
 rng=np.random.default_rng(seed); out=[]
 for c in np.unique(y):
  ids=np.where(y==c)[0]; out+=rng.choice(ids,max(1,int(round(len(ids)*frac))),replace=False).tolist()
 return np.asarray(sorted(out))

def graph_matrix(kind,kg,x,y):
 if kind=='Identity': return np.zeros_like(kg)
 if kind=='KG': return kg.copy()
 edges=int(kg.sum()/2); series=x[y==0].transpose(1,0,2).reshape(x.shape[1],-1); C=np.corrcoef(series)
 pairs=sorted([(abs(C[i,j]),i,j) for i in range(len(C)) for j in range(i+1,len(C))],reverse=True)
 A=np.zeros_like(kg)
 for _,i,j in pairs[:edges]:A[i,j]=A[j,i]=1
 return A

def norm_graph(A):
 A=A.copy();np.fill_diagonal(A,1);d=A.sum(1);inv=1/np.sqrt(np.maximum(d,1e-8));return (inv[:,None]*A*inv[None,:]).astype(np.float32)

def metrics(y,p):return dict(accuracy=accuracy_score(y,p),macro_precision=precision_score(y,p,average='macro',zero_division=0),macro_recall=recall_score(y,p,average='macro',zero_division=0),macro_f1=f1_score(y,p,average='macro'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['CNN','AE','GCN-NodeAware','GCN-GraphOnly','GCN-GAP'],required=True);ap.add_argument('--graph',choices=['KG','Corr','Identity'],default='KG');ap.add_argument('--seed',type=int,required=True);ap.add_argument('--label-fraction',type=float,default=1);ap.add_argument('--shuffle-labels',action='store_true');ap.add_argument('--hidden',type=int,default=32);ap.add_argument('--head-hidden',type=int,default=128);ap.add_argument('--dropout',type=float,default=.1);ap.add_argument('--learning-rate',type=float,default=1.5e-3);ap.add_argument('--finetune-encoder',action='store_true');ap.add_argument('--stage',choices=['select','final'],default='final');ap.add_argument('--data',required=True);ap.add_argument('--kg-adj',required=True);ap.add_argument('--out-dir',required=True);a=ap.parse_args()
 torch.set_num_threads(1);seed_all(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');D=np.load(a.data,allow_pickle=True)
 Xtr,ytr=D['X_train'].astype(np.float32),D['y_train'].astype(np.int64);Xv,yv=D['X_val'].astype(np.float32),D['y_val'].astype(np.int64);tests={'iid':(D['X_test'].astype(np.float32),D['y_test'].astype(np.int64)),'ood_profile':(D['X_test_ood'].astype(np.float32),D['y_test_ood'].astype(np.int64)),'weak':(D['X_test_weak'].astype(np.float32),D['y_test_weak'].astype(np.int64))};classes=len(np.unique(ytr));N,L=Xtr.shape[1:];ids=subset(ytr,a.label_fraction,a.seed+10000);yfit=ytr.copy()
 if a.shuffle_labels: np.random.default_rng(a.seed+30000).shuffle(yfit)
 def loader(x,y,shuffle=True):return DataLoader(TensorDataset(torch.tensor(x),torch.tensor(y)),batch_size=128,shuffle=shuffle)
 def supervised_fit(model,params):
  params=list(params);opt=torch.optim.AdamW(params,lr=a.learning_rate,weight_decay=1e-4);ce=nn.CrossEntropyLoss();best=None;bf=-1;bl=1e9;bad=0
  for ep in range(1,81):
   model.train()
   for xb,yb in loader(Xtr[ids],yfit[ids]):
    xb,yb=xb.to(dev),yb.to(dev);opt.zero_grad(set_to_none=True);loss=ce(model(xb),yb);loss.backward();torch.nn.utils.clip_grad_norm_(params,5);opt.step()
   model.eval()
   with torch.no_grad():lg=model(torch.tensor(Xv,device=dev));vl=ce(lg,torch.tensor(yv,device=dev)).item();vf=f1_score(yv,lg.argmax(1).cpu().numpy(),average='macro')
   if vf>bf+1e-4 or(abs(vf-bf)<=1e-4 and vl<bl):bf,bl,bep,bad=vf,vl,ep,0;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   else:
    bad+=1
    if bad>=10:break
  model.load_state_dict(best);model.to(dev);return bep,bf,bl
 if a.model=='CNN':
  class M(nn.Module):
   def __init__(self):
    super().__init__();self.net=nn.Sequential(nn.Conv2d(1,16,(3,5),padding=(1,2)),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.AdaptiveAvgPool2d((3,4)),nn.Flatten(),nn.Linear(32*3*4,64),nn.ReLU(),nn.Dropout(.05),nn.Linear(64,classes))
   def forward(self,x):return self.net(x[:,None])
  model=M().to(dev);bep,bf,bl=supervised_fit(model,model.parameters());pre_mse=np.nan
 elif a.model=='AE':
  dim=N*L
  class Auto(nn.Module):
   def __init__(self):super().__init__();self.enc=nn.Sequential(nn.Linear(dim,512),nn.ReLU(),nn.Linear(512,128),nn.ReLU());self.dec=nn.Sequential(nn.Linear(128,512),nn.ReLU(),nn.Linear(512,dim))
   def forward(self,x):z=self.enc(x.flatten(1));return self.dec(z)
  ae=Auto().to(dev);opt=torch.optim.AdamW(ae.parameters(),lr=1e-3,weight_decay=1e-4);best=None;bm=1e9;bad=0
  for ep in range(1,51):
   ae.train()
   for xb,_ in loader(Xtr,ytr):xb=xb.to(dev);opt.zero_grad(set_to_none=True);loss=((ae(xb)-xb.flatten(1))**2).mean();loss.backward();opt.step()
   ae.eval()
   with torch.no_grad():vm=((ae(torch.tensor(Xv,device=dev))-torch.tensor(Xv,device=dev).flatten(1))**2).mean().item()
   if vm<bm-1e-4:bm,best,bad=vm,{k:v.detach().cpu().clone() for k,v in ae.state_dict().items()},0
   else:
    bad+=1
    if bad>=7:break
  ae.load_state_dict(best);enc=ae.enc
  for p in enc.parameters():p.requires_grad=a.finetune_encoder
  class M(nn.Module):
   def __init__(self):super().__init__();self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.05),nn.Linear(64,classes))
   def forward(self,x):
    with torch.no_grad():z=enc(x.flatten(1))
    return self.head(z)
  model=M().to(dev);bep,bf,bl=supervised_fit(model,model.parameters() if a.finetune_encoder else model.head.parameters());pre_mse=bm
 else:
  kg=pd.read_csv(a.kg_adj,index_col=0).to_numpy(np.float32);Ah=norm_graph(graph_matrix(a.graph,kg,Xtr,ytr));hidden=a.hidden
  class Enc(nn.Module):
   def __init__(self):super().__init__();self.register_buffer('A',torch.tensor(Ah));self.inp=nn.Linear(L,hidden);self.g1=nn.Linear(hidden,hidden,bias=False);self.g2=nn.Linear(hidden,hidden,bias=False);self.n1=nn.LayerNorm(hidden);self.n2=nn.LayerNorm(hidden);self.out=nn.LayerNorm(hidden*2)
   def forward(self,x):h0=torch.relu(self.inp(x));h1=torch.relu(self.n1(h0+self.g1(torch.matmul(self.A,h0))));h2=torch.relu(self.n2(h1+self.g2(torch.matmul(self.A,h1))));return self.out(torch.cat([h0,h2],-1))
  class MAE(nn.Module):
   def __init__(self):super().__init__();self.enc=Enc();self.dec=nn.Linear(hidden*2,L)
   def forward(self,x):return self.dec(self.enc(x))
  ae=MAE().to(dev);opt=torch.optim.AdamW(ae.parameters(),lr=1e-3,weight_decay=1e-4);best=None;bm=1e9;bad=0
  xv=torch.tensor(Xv,device=dev)
  for ep in range(1,41):
   ae.train()
   for xb,_ in loader(Xtr,ytr):xb=xb.to(dev);mask=torch.rand_like(xb)<.15;opt.zero_grad(set_to_none=True);rec=ae(xb.masked_fill(mask,0));loss=((rec-xb)[mask]**2).mean();loss.backward();opt.step()
   ae.eval();seed_all(a.seed+7000+ep)
   with torch.no_grad():mask=torch.rand_like(xv)<.15;vm=((ae(xv.masked_fill(mask,0))-xv)[mask]**2).mean().item()
   if vm<bm-1e-4:bm,best,bad=vm,{k:v.detach().cpu().clone() for k,v in ae.state_dict().items()},0
   else:
    bad+=1
    if bad>=6:break
  ae.load_state_dict(best);enc=ae.enc
  for p in enc.parameters():p.requires_grad=False
  class M(nn.Module):
   def __init__(self):
    super().__init__();indim=N*(hidden*2) if a.model=='GCN-NodeAware' else (N*hidden if a.model=='GCN-GraphOnly' else hidden*2);self.head=nn.Sequential(nn.Linear(indim,a.head_hidden),nn.ReLU(),nn.Dropout(a.dropout),nn.Linear(a.head_hidden,classes))
   def forward(self,x):
    if a.finetune_encoder: h=enc(x)
    else:
     with torch.no_grad():h=enc(x)
    if a.model=='GCN-NodeAware': h=h.flatten(1)
    elif a.model=='GCN-GraphOnly': h=h[:,:,hidden:].flatten(1)
    else: h=h.mean(1)
    return self.head(h)
  model=M().to(dev);bep,bf,bl=supervised_fit(model,model.head.parameters());pre_mse=bm
 out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
 if a.stage=='select':
  row=dict(stage='select',model=a.model,graph=a.graph,seed=a.seed,label_fraction=a.label_fraction,shuffle_labels=a.shuffle_labels,hidden=a.hidden,head_hidden=a.head_hidden,dropout=a.dropout,learning_rate=a.learning_rate,finetune_encoder=a.finetune_encoder,best_val_macro_f1=bf,best_epoch=bep,pretrain_val_mse=pre_mse)
  pd.DataFrame([row]).to_csv(out/f"select_{a.model}_{a.graph}_h{a.hidden}_c{a.head_hidden}_d{a.dropout:g}_seed{a.seed}.csv",index=False,encoding='utf-8-sig');print(json.dumps(row,ensure_ascii=False));return
 for name,(x,y) in tests.items():
  model.eval()
  with torch.no_grad():p=model(torch.tensor(x,device=dev)).argmax(1).cpu().numpy()
  row=dict(stage='final',model=a.model,graph=a.graph,seed=a.seed,label_fraction=a.label_fraction,shuffle_labels=a.shuffle_labels,hidden=a.hidden,head_hidden=a.head_hidden,dropout=a.dropout,learning_rate=a.learning_rate,finetune_encoder=a.finetune_encoder,test=name,best_val_macro_f1=bf,best_epoch=bep,pretrain_val_mse=pre_mse,**metrics(y,p));rows.append(row)
  pd.DataFrame(confusion_matrix(y,p),index=D['class_names'],columns=D['class_names']).to_csv(out/f"cm_{a.model}_{a.graph}_lf{a.label_fraction:g}_{name}_seed{a.seed}.csv",encoding='utf-8-sig')
 pd.DataFrame(rows).to_csv(out/f"metrics_{a.model}_{a.graph}_lf{a.label_fraction:g}_seed{a.seed}.csv",index=False,encoding='utf-8-sig');print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
