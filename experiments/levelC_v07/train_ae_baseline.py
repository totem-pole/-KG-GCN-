from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int,default=123); parser.add_argument('--data',required=True); parser.add_argument('--out-dir',required=True); args=parser.parse_args()
torch.set_num_threads(1); torch.manual_seed(args.seed); np.random.seed(args.seed)
D=np.load(args.data,allow_pickle=True); Xtr,ytr=D['X_train'],D['y_train']; Xv,yv=D['X_val'],D['y_val']; Xte,yte=D['X_test'],D['y_test']
F=Xtr.shape[1]*Xtr.shape[2]; Xtr=Xtr.reshape(len(Xtr),F); Xv=Xv.reshape(len(Xv),F); Xte=Xte.reshape(len(Xte),F)

class AE(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=nn.Sequential(nn.Linear(F,256),nn.ReLU(),nn.Linear(256,64),nn.ReLU()); self.decoder=nn.Sequential(nn.Linear(64,256),nn.ReLU(),nn.Linear(256,F))
    def forward(self,x): return self.decoder(self.encoder(x))

ae=AE(); opt=torch.optim.AdamW(ae.parameters(),lr=1e-3,weight_decay=1e-4); mse=nn.MSELoss()
dl=DataLoader(TensorDataset(torch.tensor(Xtr)),batch_size=256,shuffle=True); xv=torch.tensor(Xv); best=None; best_mse=float('inf'); bad=0
for ep in range(1,31):
    ae.train()
    for (xb,) in dl:
        opt.zero_grad(); loss=mse(ae(xb),xb); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad(): val_mse=mse(ae(xv),xv).item()
    if val_mse<best_mse-1e-4:
        best_mse=val_mse; bad=0; best={k:v.detach().clone() for k,v in ae.state_dict().items()}
    else:
        bad+=1
        if bad>=5: break
ae.load_state_dict(best)
for p in ae.encoder.parameters(): p.requires_grad=False
with torch.no_grad(): Ztr=ae.encoder(torch.tensor(Xtr)); Zv=ae.encoder(torch.tensor(Xv)); Zte=ae.encoder(torch.tensor(Xte))
clf=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.05),nn.Linear(32,4)); opt=torch.optim.AdamW(clf.parameters(),lr=1e-3,weight_decay=1e-4); ce=nn.CrossEntropyLoss()
dl=DataLoader(TensorDataset(Ztr,torch.tensor(ytr)),batch_size=256,shuffle=True); yvv=torch.tensor(yv); best=None; best_f1=-1; best_loss=float('inf'); bad=0
for ep in range(1,51):
    clf.train()
    for zb,yb in dl:
        opt.zero_grad(); loss=ce(clf(zb),yb); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        logits=clf(Zv); val_loss=ce(logits,yvv).item(); val_f1=f1_score(yv,logits.argmax(1).numpy(),average='macro')
    if val_f1>best_f1+1e-4 or (abs(val_f1-best_f1)<=1e-4 and val_loss<best_loss):
        best_f1,best_loss=val_f1,val_loss; bad=0; best={k:v.detach().clone() for k,v in clf.state_dict().items()}
    else:
        bad+=1
        if bad>=7: break
clf.load_state_dict(best); clf.eval()
with torch.no_grad(): pred=clf(Zte).argmax(1).numpy()
r=pd.DataFrame([{'model':'AE','seed':args.seed,'accuracy':accuracy_score(yte,pred),'macro_precision':precision_score(yte,pred,average='macro',zero_division=0),'macro_recall':recall_score(yte,pred,average='macro',zero_division=0),'macro_f1':f1_score(yte,pred,average='macro'),'best_val_macro_f1':best_f1,'pretrain_val_mse':best_mse}])
out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); r.to_csv(out/f'ae_seed{args.seed}.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(confusion_matrix(yte,pred),index=['F0','F1','F2','F3'],columns=['F0','F1','F2','F3']).to_csv(out/f'cm_ae_seed{args.seed}.csv',encoding='utf-8-sig')
print(r.to_string(index=False))
