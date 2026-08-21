from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"expanded_deliverables";(OUT/"tables").mkdir(parents=True,exist_ok=True);(OUT/"figures").mkdir(parents=True,exist_ok=True)

rows=[]
for device_dir in (ROOT/"expanded_results").iterdir():
    if not device_dir.is_dir():continue
    for p in device_dir.rglob("metrics_*.csv"):
        d=pd.read_csv(p);d.insert(0,"device",device_dir.name);rows.append(d)
all_metrics=pd.concat(rows,ignore_index=True)
# Bring H1 into the same final comparison.
for p in (ROOT/"results"/"H1").rglob("metrics_*.csv"):
    d=pd.read_csv(p);d.insert(0,"device","H1");all_metrics=pd.concat([all_metrics,d],ignore_index=True)
all_metrics.to_csv(OUT/"tables"/"selected_devices_all_seed_metrics.csv",index=False,encoding="utf-8-sig")
summary=(all_metrics.groupby(["device","model","graph","test"],as_index=False)
         .agg(seeds=("seed","nunique"),accuracy_mean=("accuracy","mean"),accuracy_std=("accuracy","std"),
              macro_precision_mean=("macro_precision","mean"),macro_precision_std=("macro_precision","std"),
              macro_recall_mean=("macro_recall","mean"),macro_recall_std=("macro_recall","std"),
              macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std")))
summary.to_csv(OUT/"tables"/"selected_devices_3seed_summary.csv",index=False,encoding="utf-8-sig")

selection=[]
for device in ["H2","H3","H4","L6","L9"]:
    p=next((ROOT/"expanded_selection"/device).glob("select_*.csv"));r=pd.read_csv(p).iloc[0]
    selection.append({"device":device,"validation_macro_f1":r.best_val_macro_f1,"selected_for_test":device!="L9","selected_for_paper":device in ["H2","H3","H4","L6"]})
pd.DataFrame(selection).to_csv(OUT/"tables"/"expanded_device_selection_audit.csv",index=False,encoding="utf-8-sig")

selected=["H1","H2","H3","H4","L6"]
perclass=[]
for device in selected:
    base=(ROOT/"results"/"H1" if device=="H1" else ROOT/"expanded_results"/device)/"GCN-NodeAware_KG"
    p=base/"cm_GCN-NodeAware_KG_lf1_iid_seed123.csv";cm_df=pd.read_csv(p,index_col=0);cm=cm_df.to_numpy(float);tp=np.diag(cm)
    precision=tp/np.maximum(cm.sum(0),1);recall=tp/np.maximum(cm.sum(1),1);f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    for name,pr,re,ff,sup in zip(cm_df.index,precision,recall,f1,cm.sum(1)):
        perclass.append({"device":device,"class":name,"precision":pr,"recall":re,"f1":ff,"support":int(sup)})
pd.DataFrame(perclass).to_csv(OUT/"tables"/"selected_devices_seed123_per_class.csv",index=False,encoding="utf-8-sig")

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False,"font.size":9,"axes.linewidth":.8,"xtick.direction":"in","ytick.direction":"in"})

# Multi-device three-model comparison.
fig,ax=plt.subplots(figsize=(8.8,4.3));x=np.arange(len(selected));width=.24
defs=[("GCN-NodeAware","KG","KG-GCN","#2F5597"),("CNN","KG","CNN","#70AD47"),("AE","KG","AE","#ED7D31")]
for k,(model,graph,label,color) in enumerate(defs):
    vals=[];errs=[]
    for device in selected:
        r=summary[(summary.device==device)&(summary.model==model)&(summary.graph==graph)&(summary.test=="iid")].iloc[0]
        vals.append(r.macro_f1_mean*100);errs.append(r.macro_f1_std*100)
    ax.bar(x+(k-1)*width,vals,width,yerr=errs,capsize=3,label=label,color=color,edgecolor="black",linewidth=.6)
ax.axhline(95,color="#666666",linestyle="--",linewidth=.9,label="95%")
ax.set_xticks(x,selected);ax.set_ylabel("Macro-F1 / %");ax.set_ylim(88,101);ax.grid(axis="y",linestyle="--",linewidth=.5,alpha=.4);ax.legend(frameon=False,ncol=4,loc="lower center")
fig.tight_layout()
for ext in ["png","pdf"]:fig.savefig(OUT/"figures"/f"五设备KG-GCN迁移三模型对比.{ext}",dpi=400 if ext=="png" else None,bbox_inches="tight")
plt.close(fig)

# Four newly-added device confusion matrices.
fig,axes=plt.subplots(2,2,figsize=(8.8,7.8));labels=["正常","换热下降","供汽不足","水位过高"]
for ax,device in zip(axes.ravel(),["H2","H3","H4","L6"]):
    p=ROOT/"expanded_results"/device/"GCN-NodeAware_KG"/"cm_GCN-NodeAware_KG_lf1_iid_seed123.csv";cm=pd.read_csv(p,index_col=0).to_numpy(float);norm=cm/cm.sum(1,keepdims=True)
    im=ax.imshow(norm,cmap="Blues",vmin=0,vmax=1);ax.set_xticks(range(4),labels,rotation=22,ha="right");ax.set_yticks(range(4),labels);ax.set_title(device)
    for i in range(4):
        for j in range(4):ax.text(j,i,f"{int(cm[i,j])}\n{norm[i,j]*100:.1f}%",ha="center",va="center",fontsize=7,color="white" if norm[i,j]>.55 else "black")
fig.supxlabel("预测类别");fig.supylabel("真实类别");fig.subplots_adjust(left=.12,right=.90,top=.95,bottom=.10,wspace=.28,hspace=.34);cb=fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.028,pad=.03);cb.set_label("行归一化比例")
for ext in ["png","pdf"]:fig.savefig(OUT/"figures"/f"H2_H3_H4_L6混淆矩阵.{ext}",dpi=400 if ext=="png" else None,bbox_inches="tight")
plt.close(fig)

main=summary[(summary.model=="GCN-NodeAware")&(summary.graph=="KG")&(summary.test=="iid")&summary.device.isin(selected)].sort_values("macro_f1_mean",ascending=False)
main.to_csv(OUT/"tables"/"paper_selected_KG_GCN_results.csv",index=False,encoding="utf-8-sig")
print(main[["device","accuracy_mean","accuracy_std","macro_f1_mean","macro_f1_std"]].to_string(index=False))
