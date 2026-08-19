from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--data', required=True)
parser.add_argument('--adjacency', required=True)
parser.add_argument('--out-dir', required=True)
args = parser.parse_args()

torch.set_num_threads(1)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
D = np.load(args.data, allow_pickle=True)
Xtr,ytr=D['X_train'],D['y_train']; Xv,yv=D['X_val'],D['y_val']; Xte,yte=D['X_test'],D['y_test']
A = pd.read_csv(args.adjacency,index_col=0).to_numpy(np.float32)
N,L=Xtr.shape[1],Xtr.shape[2]
At=A.copy(); np.fill_diagonal(At,1.0)
deg=At.sum(1); inv=1/np.sqrt(np.maximum(deg,1e-8)); Ahat=(inv[:,None]*At*inv[None,:]).astype(np.float32)

class Encoder(nn.Module):
    def __init__(self,hid=32):
        super().__init__(); self.register_buffer('Ahat',torch.tensor(Ahat))
        self.inp=nn.Linear(L,hid); self.g1=nn.Linear(hid,hid,bias=False); self.g2=nn.Linear(hid,hid,bias=False)
        self.n1=nn.LayerNorm(hid); self.n2=nn.LayerNorm(hid); self.outnorm=nn.LayerNorm(hid*2)
    def forward(self,x):
        h0=torch.relu(self.inp(x))
        h1=torch.relu(self.n1(h0+self.g1(torch.matmul(self.Ahat,h0))))
        h2=torch.relu(self.n2(h1+self.g2(torch.matmul(self.Ahat,h1))))
        return self.outnorm(torch.cat([h0,h2],dim=-1))

class MaskAE(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=Encoder(32); self.decoder=nn.Linear(64,L)
    def forward(self,x): return self.decoder(self.encoder(x))

# Stage 1: Yang-style masked reconstruction on all TRAIN residual windows, labels ignored.
ae=MaskAE(); opt=torch.optim.AdamW(ae.parameters(),lr=1e-3,weight_decay=1e-4)
dl=DataLoader(TensorDataset(torch.tensor(Xtr)),batch_size=128,shuffle=True)
xv=torch.tensor(Xv); best=None; best_mse=float('inf'); bad=0
for ep in range(1,41):
    ae.train()
    for (xb,) in dl:
        mask=torch.rand_like(xb)<0.15
        opt.zero_grad(); rec=ae(xb.masked_fill(mask,0.0)); loss=((rec-xb)[mask]**2).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(ae.parameters(),5.0); opt.step()
    ae.eval()
    with torch.no_grad():
        torch.manual_seed(args.seed+7000+ep); vm=torch.rand_like(xv)<0.15
        val_mse=((ae(xv.masked_fill(vm,0.0))-xv)[vm]**2).mean().item()
    if val_mse<best_mse-1e-4:
        best_mse=val_mse; best_pre_ep=ep; bad=0; best={k:v.detach().clone() for k,v in ae.state_dict().items()}
    else:
        bad+=1
        if bad>=6: break
ae.load_state_dict(best)

# Stage 2: freeze GCN encoder, train FC+Softmax classifier.
encoder=ae.encoder
for p in encoder.parameters(): p.requires_grad=False
class Classifier(nn.Module):
    def __init__(self):
        super().__init__(); self.fc=nn.Sequential(nn.Flatten(),nn.Linear(N*64,96),nn.ReLU(),nn.Dropout(0.05),nn.Linear(96,4))
    def forward(self,x):
        with torch.no_grad(): h=encoder(x)
        return self.fc(h)

model=Classifier(); opt=torch.optim.AdamW(model.fc.parameters(),lr=1.5e-3,weight_decay=1e-4); ce=nn.CrossEntropyLoss()
dl=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(ytr)),batch_size=128,shuffle=True)
xvv,yvv=torch.tensor(Xv),torch.tensor(yv); best=None; best_f1=-1; best_loss=float('inf'); bad=0
for ep in range(1,61):
    model.train()
    for xb,yb in dl:
        opt.zero_grad(); loss=ce(model(xb),yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits=model(xvv); val_loss=ce(logits,yvv).item(); val_f1=f1_score(yv,logits.argmax(1).numpy(),average='macro')
    if val_f1>best_f1+1e-4 or (abs(val_f1-best_f1)<=1e-4 and val_loss<best_loss):
        best_f1,best_loss,best_cls_ep=val_f1,val_loss,ep; bad=0; best={k:v.detach().clone() for k,v in model.state_dict().items()}
    else:
        bad+=1
        if bad>=8: break
model.load_state_dict(best); model.eval()
with torch.no_grad(): pred=model(torch.tensor(Xte)).argmax(1).numpy()

result=pd.DataFrame([{'model':'KG-GCN','seed':args.seed,'accuracy':accuracy_score(yte,pred),
 'macro_precision':precision_score(yte,pred,average='macro',zero_division=0),
 'macro_recall':recall_score(yte,pred,average='macro',zero_division=0),
 'macro_f1':f1_score(yte,pred,average='macro'),'best_val_macro_f1':best_f1,
 'pretrain_val_mse':best_mse,'pretrain_epoch':best_pre_ep,'classifier_epoch':best_cls_ep}])
out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
result.to_csv(out/f'kg2layer_pretrain_seed{args.seed}.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(confusion_matrix(yte,pred),index=['F0','F1','F2','F3'],columns=['F0','F1','F2','F3']).to_csv(out/f'cm_kg2layer_seed{args.seed}.csv',encoding='utf-8-sig')
print(result.to_string(index=False))
