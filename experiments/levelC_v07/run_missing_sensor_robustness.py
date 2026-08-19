"""Evaluate trained CNN/KG-GCN models under DCS sensor loss.

This script records the evaluation protocol used in v07. It assumes the caller
has already trained/loaded a model with the same v07 split. Missing standardized
DCS nodes are filled with 0.0. Use identical masks for all models.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

RANDOM_MISSING=[0,1,2,4,6]
TARGETED={
    'clean':[],
    'post_NRV_2T':[14,15],
    'post_NRV_plus_heater_PT':[14,15,16,17],
    'NRV_prepost_plus_heater':[12,13,14,15,16,17],
}

NODE_INDEX={
    12:'一级抽汽逆止门前水平管管顶温度',
    13:'一级抽汽逆止门前水平管管底温度',
    14:'一级抽汽逆止门后水平管管顶温度',
    15:'一级抽汽逆止门后水平管管底温度',
    16:'1号高加入口抽汽压力',
    17:'1号高加入口抽汽温度',
}


def evaluate_random(model_predict, Xtest, ytest, model_name, seed=20260819):
    rng=np.random.default_rng(seed); rows=[]; n=Xtest.shape[1]
    for miss in RANDOM_MISSING:
        X=Xtest.copy()
        if miss:
            for i in range(len(X)):
                ids=rng.choice(n,miss,replace=False); X[i,ids,:]=0.0
        pred=model_predict(X)
        rows.append({'scenario':'random','model':model_name,'missing_nodes':miss,
                     'missing_ratio':miss/n,'accuracy':accuracy_score(ytest,pred),
                     'macro_f1':f1_score(ytest,pred,average='macro')})
    return pd.DataFrame(rows)


def evaluate_targeted(model_predict, Xtest, ytest, model_name):
    rows=[]
    for name,ids in TARGETED.items():
        X=Xtest.copy()
        if ids: X[:,ids,:]=0.0
        pred=model_predict(X)
        rows.append({'scenario':name,'model':model_name,'missing_nodes':len(ids),
                     'accuracy':accuracy_score(ytest,pred),
                     'macro_f1':f1_score(ytest,pred,average='macro')})
    return pd.DataFrame(rows)

# v07 representative seed=123 reference results:
# random missing Macro-F1
# CNN:    0:0.97853, 1:0.95899, 2:0.94333, 4:0.90286, 6:0.83838
# KG-GCN: 0:0.98753, 1:0.97689, 2:0.95745, 4:0.90967, 6:0.84983
# targeted branch loss Macro-F1
# post-NRV 2T: CNN 0.96971, KG 0.94826
# post-NRV + heater P/T (4 nodes): CNN 0.62326, KG 0.65180
# NRV pre/post + heater (6 nodes): CNN 0.58721, KG 0.62250
