from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler

def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

def sequences(x,y,t,L):
 xs=[];ys=[];ts=[]
 sec=t.astype('datetime64[s]').astype('int64');gaps=np.r_[True,np.diff(sec)>90];seg=np.cumsum(gaps)
 for sid in np.unique(seg):
  ids=np.where(seg==sid)[0]
  if len(ids)<L:continue
  for k in range(L-1,len(ids)):ii=ids[k-L+1:k+1];xs.append(x[ii]);ys.append(y[ids[k]]);ts.append(sec[ids[k]])
 return np.asarray(xs,np.float32),np.asarray(ys,np.float32),np.asarray(ts,np.int64)

class ANN(nn.Module):
 def __init__(self,di,do):super().__init__();self.net=nn.Sequential(nn.Linear(di,32),nn.LayerNorm(32),nn.SiLU(),nn.Dropout(.05),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,do))
 def forward(self,x):return self.net(x[:,-1])
class GRU(nn.Module):
 def __init__(self,di,do):super().__init__();self.gru=nn.GRU(di,32,batch_first=True);self.norm=nn.LayerNorm(32);self.drop=nn.Dropout(.05);self.out=nn.Linear(32,do)
 def forward(self,x):h,_=self.gru(x);return self.out(self.drop(self.norm(h[:,-1])))
class Block(nn.Module):
 def __init__(self,c,d):super().__init__();self.dw=nn.Conv1d(c,c,5,padding=2*d,dilation=d,groups=c);self.pw=nn.Conv1d(c,c,1);self.n=nn.BatchNorm1d(c)
 def forward(self,x):return torch.relu(self.n(self.pw(self.dw(x)))+x)
class TCN(nn.Module):
 def __init__(self,di,do):super().__init__();self.inp=nn.Conv1d(di,16,1);self.blocks=nn.Sequential(Block(16,1),Block(16,2),Block(16,4));self.out=nn.Linear(16,do)
 def forward(self,x):h=self.blocks(self.inp(x.transpose(1,2)));return self.out(h[:,:,-1])
class ITransformer(nn.Module):
 def __init__(self,di,do,L):super().__init__();self.proj=nn.Linear(L,32);layer=nn.TransformerEncoderLayer(32,4,64,.05,batch_first=True,norm_first=True);self.enc=nn.TransformerEncoder(layer,1);self.out=nn.Linear(32,do)
 def forward(self,x):tokens=self.proj(x.transpose(1,2));return self.out(self.enc(tokens).mean(1))

def tol(unit):
 u=str(unit).lower()
 if 'mpa' in u:return .1
 if 'kpa' in u:return .2
 if '℃' in u or '°c' in u:return 1.0
 if 'mm' in u:return 10.0
 if 'μg' in u:return 5.0
 return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--devices',required=True);ap.add_argument('--audit',required=True);ap.add_argument('--device',required=True);ap.add_argument('--model',choices=['ANN','GRU','TCN','iTransformer-small'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--window',type=int,default=24);a=ap.parse_args();torch.set_num_threads(1);seed_all(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 df=pd.read_pickle(a.cache);defs=json.loads(Path(a.devices).read_text(encoding='utf-8'));audit=pd.read_csv(a.audit);rows=defs[a.device];meta={r['tag']:r for r in rows};train=df[(df.time>='2024-05-01')&(df.time<'2024-06-01')].copy();test=df[(df.time>='2024-06-01')&(df.time<'2024-07-01')].copy()
 def num(d,tag):return pd.to_numeric(d[tag],errors='coerce')
 inputs=[]
 for r in rows:
  if r['role']=='Y' or r['tag'] not in df:continue
  stat=audit[(audit.device==a.device)&(audit.tag==r['tag'])]
  if stat.empty or stat.numeric_rate.iloc[0]<.95 or stat.nonzero_rate.iloc[0]<.02:continue
  z=num(train,r['tag']);
  if z.std()<1e-7:continue
  inputs.append(r['tag'])
 outputs=[r['tag'] for r in rows if r['role']=='Y' and r['tag'] in df and audit[(audit.device==a.device)&(audit.tag==r['tag'])].numeric_rate.iloc[0]>.99]
 for d in (train,test):
  for c in inputs+outputs:d[c]=num(d,c)
 mw='SYG1_MW';sp='SYG1_DEH-SPEED';train=train[(train[mw]>200)&train[sp].between(2950,3050)];test=test[(test[mw]>200)&test[sp].between(2950,3050)]
 med=train[inputs].median();train[inputs]=train[inputs].fillna(med);test[inputs]=test[inputs].fillna(med);train=train.dropna(subset=outputs);test=test.dropna(subset=outputs)
 cut=int(len(train)*.8);fit=train.iloc[:cut];val=train.iloc[cut:];xs=StandardScaler().fit(fit[inputs]);ys=StandardScaler().fit(fit[outputs])
 def prep(d):return sequences(xs.transform(d[inputs]),ys.transform(d[outputs]),d.time.to_numpy(),a.window)
 xtr,ytr,ttr=prep(fit);xv,yv,tv=prep(val);xte,yte,tte=prep(test)
 cls={'ANN':ANN,'GRU':GRU,'TCN':TCN};model=(ITransformer(len(inputs),len(outputs),a.window) if a.model=='iTransformer-small' else cls[a.model](len(inputs),len(outputs))).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);lossfn=nn.MSELoss();dl=DataLoader(TensorDataset(torch.tensor(xtr),torch.tensor(ytr)),batch_size=1024,shuffle=True);best=None;bv=1e9;bad=0;hist=[]
 xv_t,yv_t=torch.tensor(xv,device=dev),torch.tensor(yv,device=dev)
 for ep in range(1,201):
  model.train();losses=[]
  for xb,yb in dl:xb,yb=xb.to(dev),yb.to(dev);opt.zero_grad(set_to_none=True);loss=lossfn(model(xb),yb);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
  model.eval();
  with torch.no_grad():vl=lossfn(model(xv_t),yv_t).item()
  hist.append([ep,float(np.mean(losses)),vl])
  if vl<bv-1e-5:bv,bep,bad=vl,ep,0;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  else:
   bad+=1
   if bad>=15:break
 model.load_state_dict(best);model.to(dev).eval()
 def pred(x):
  out=[]
  with torch.no_grad():
   for i in range(0,len(x),4096):out.append(model(torch.tensor(x[i:i+4096],device=dev)).cpu().numpy())
  return np.concatenate(out)
 ptr,pv,pte=pred(xtr),pred(xv),pred(xte);atr,av,ate=ys.inverse_transform(ytr),ys.inverse_transform(yv),ys.inverse_transform(yte);ptr,pv,pte=ys.inverse_transform(ptr),ys.inverse_transform(pv),ys.inverse_transform(pte)
 metrics=[]
 for split,act,est in [('train',atr,ptr),('val',av,pv),('test',ate,pte)]:
  for j,tag in enumerate(outputs):
   e=act[:,j]-est[:,j];threshold=tol(meta[tag]['unit']);metrics.append(dict(device=a.device,model=a.model,seed=a.seed,split=split,tag=tag,name_cn=meta[tag]['name_cn'],unit=meta[tag]['unit'],r2=r2_score(act[:,j],est[:,j]),rmse=mean_squared_error(act[:,j],est[:,j])**.5,mae=mean_absolute_error(act[:,j],est[:,j]),bias=float(e.mean()),within_tolerance=float((np.abs(e)<=threshold).mean()) if threshold else np.nan,train_min=float(atr[:,j].min()),train_max=float(atr[:,j].max()),samples=len(act)))
 out=Path(a.out_dir)/a.device/a.model/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(metrics).to_csv(out/'metrics_by_point.csv',index=False,encoding='utf-8-sig');pd.DataFrame(hist,columns=['epoch','train_loss','val_loss']).to_csv(out/'training_history.csv',index=False);np.savez_compressed(out/'predictions.npz',train_time=ttr,train_actual=atr,train_pred=ptr,val_time=tv,val_actual=av,val_pred=pv,test_time=tte,test_actual=ate,test_pred=pte,input_tags=np.asarray(inputs),target_tags=np.asarray(outputs),target_names=np.asarray([meta[t]['name_cn'] for t in outputs]),target_units=np.asarray([meta[t]['unit'] for t in outputs]));(out/'run_manifest.json').write_text(json.dumps(dict(device=a.device,model=a.model,seed=a.seed,window=a.window,inputs=inputs,outputs=outputs,best_epoch=bep,best_val_loss=bv,parameters=sum(p.numel() for p in model.parameters()),device_runtime=str(dev),train_samples=len(xtr),val_samples=len(xv),test_samples=len(xte)),ensure_ascii=False,indent=2),encoding='utf-8');print(dict(device=a.device,model=a.model,seed=a.seed,inputs=len(inputs),outputs=len(outputs),best_epoch=bep,test_macro_r2=pd.DataFrame(metrics).query("split=='test'").r2.mean()),flush=True)
if __name__=='__main__':main()
