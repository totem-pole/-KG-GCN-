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

class CNN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat=nn.Sequential(
            nn.Conv2d(1,16,(3,5),padding=(1,2)),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d((2,2)),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d((2,2)),
            nn.AdaptiveAvgPool2d((3,4)))
        self.cls=nn.Sequential(nn.Flatten(),nn.Linear(32*3*4,64),nn.ReLU(),nn.Dropout(0.05),nn.Linear(64,4))
    def forward(self,x): return self.cls(self.feat(x.unsqueeze(1)))

m=CNN2D(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4); ce=nn.CrossEntropyLoss()
dl=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(ytr)),batch_size=128,shuffle=True)
xv,yvv=torch.tensor(Xv),torch.tensor(yv); best=None; best_f1=-1; best_loss=float('inf'); bad=0
for ep in range(1,61):
    m.train()
    for xb,yb in dl:
        opt.zero_grad(); loss=ce(m(xb),yb); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5.0); opt.step()
    m.eval()
    with torch.no_grad():
        logits=m(xv); val_loss=ce(logits,yvv).item(); val_f1=f1_score(yv,logits.argmax(1).numpy(),average='macro')
    if val_f1>best_f1+1e-4 or (abs(val_f1-best_f1)<=1e-4 and val_loss<best_loss):
        best_f1,best_loss,best_ep=val_f1,val_loss,ep; bad=0; best={k:v.detach().clone() for k,v in m.state_dict().items()}
    else:
        bad+=1
        if bad>=8: break
m.load_state_dict(best); m.eval()
with torch.no_grad(): pred=m(torch.tensor(Xte)).argmax(1).numpy()
r=pd.DataFrame([{'model':'CNN','seed':args.seed,'accuracy':accuracy_score(yte,pred),'macro_precision':precision_score(yte,pred,average='macro',zero_division=0),'macro_recall':recall_score(yte,pred,average='macro',zero_division=0),'macro_f1':f1_score(yte,pred,average='macro'),'best_val_macro_f1':best_f1,'best_epoch':best_ep}])
out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); r.to_csv(out/f'cnn2d_seed{args.seed}.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(confusion_matrix(yte,pred),index=['F0','F1','F2','F3'],columns=['F0','F1','F2','F3']).to_csv(out/f'cm_cnn2d_seed{args.seed}.csv',encoding='utf-8-sig')
print(r.to_string(index=False))
