from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/"type_diverse_deliverables";(OUT/"tables").mkdir(parents=True,exist_ok=True);(OUT/"figures").mkdir(parents=True,exist_ok=True)
sources={"H1":ROOT/"results"/"H1","L6":ROOT/"expanded_results"/"L6","BFPT":ROOT/"type_diverse_results"/"BFPT","CND1":ROOT/"type_diverse_results"/"CND1"};rows=[]
for device,base in sources.items():
 for p in base.rglob("metrics_*.csv"):
  d=pd.read_csv(p);d.insert(0,"device",device);rows.append(d)
allm=pd.concat(rows,ignore_index=True);allm.to_csv(OUT/"tables"/"type_diverse_all_seed_metrics.csv",index=False,encoding="utf-8-sig")
summary=(allm.groupby(["device","model","graph","test"],as_index=False).agg(seeds=("seed","nunique"),accuracy_mean=("accuracy","mean"),accuracy_std=("accuracy","std"),macro_precision_mean=("macro_precision","mean"),macro_precision_std=("macro_precision","std"),macro_recall_mean=("macro_recall","mean"),macro_recall_std=("macro_recall","std"),macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std")))
summary.to_csv(OUT/"tables"/"type_diverse_3seed_summary.csv",index=False,encoding="utf-8-sig")
main=summary[(summary.model=="GCN-NodeAware")&(summary.graph=="KG")&(summary.test=="iid")].sort_values("macro_f1_mean",ascending=False);main.to_csv(OUT/"tables"/"paper_type_diverse_KG_GCN_results.csv",index=False,encoding="utf-8-sig")
per=[]
for device,base in sources.items():
 p=base/"GCN-NodeAware_KG"/"cm_GCN-NodeAware_KG_lf1_iid_seed123.csv";cmf=pd.read_csv(p,index_col=0);cm=cmf.to_numpy(float);tp=np.diag(cm);pr=tp/np.maximum(cm.sum(0),1);re=tp/np.maximum(cm.sum(1),1);f1=2*pr*re/np.maximum(pr+re,1e-12)
 for n,a,b,c,s in zip(cmf.index,pr,re,f1,cm.sum(1)):per.append({"device":device,"class":n,"precision":a,"recall":b,"f1":c,"support":int(s)})
pd.DataFrame(per).to_csv(OUT/"tables"/"type_diverse_seed123_per_class.csv",index=False,encoding="utf-8-sig")
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False,"font.size":9,"axes.linewidth":.8,"xtick.direction":"in","ytick.direction":"in"})
devices=["H1","L6","BFPT","CND1"];fig,ax=plt.subplots(figsize=(7.8,4.2));x=np.arange(4);w=.24
for k,(m,g,l,c) in enumerate([("GCN-NodeAware","KG","KG-GCN","#2F5597"),("CNN","KG","CNN","#70AD47"),("AE","KG","AE","#ED7D31")]):
 vals=[];err=[]
 for d in devices:
  r=summary[(summary.device==d)&(summary.model==m)&(summary.graph==g)&(summary.test=="iid")].iloc[0];vals.append(r.macro_f1_mean*100);err.append(r.macro_f1_std*100)
 ax.bar(x+(k-1)*w,vals,w,yerr=err,capsize=3,label=l,color=c,edgecolor="black",linewidth=.6)
ax.axhline(95,color="#666",ls="--",lw=.9,label="95%");ax.set_xticks(x,["H1高加","L6低加","BFPT小汽轮机","CND1凝汽器"]);ax.set_ylabel("Macro-F1 / %");ax.set_ylim(88,100);ax.grid(axis="y",ls="--",lw=.5,alpha=.4);ax.legend(frameon=False,ncol=4,loc="lower center");fig.tight_layout()
for ext in ["png","pdf"]:fig.savefig(OUT/"figures"/f"四类设备KG-GCN迁移对比.{ext}",dpi=400 if ext=="png" else None,bbox_inches="tight")
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(8.8,4.1));labels={"BFPT":["正常","进汽不足","真空恶化","调速异常"],"CND1":["正常","真空恶化","循环水不足","液位过高"]}
for ax,device in zip(axes,["BFPT","CND1"]):
 base=sources[device];cm=pd.read_csv(base/"GCN-NodeAware_KG"/"cm_GCN-NodeAware_KG_lf1_iid_seed123.csv",index_col=0).to_numpy(float);norm=cm/cm.sum(1,keepdims=True);im=ax.imshow(norm,cmap="Blues",vmin=0,vmax=1);ax.set_xticks(range(4),labels[device],rotation=22,ha="right");ax.set_yticks(range(4),labels[device]);ax.set_title(device)
 for i in range(4):
  for j in range(4):ax.text(j,i,f"{int(cm[i,j])}\n{norm[i,j]*100:.1f}%",ha="center",va="center",fontsize=7,color="white" if norm[i,j]>.55 else "black")
fig.supxlabel("预测类别");fig.supylabel("真实类别");fig.subplots_adjust(left=.11,right=.90,top=.94,bottom=.18,wspace=.30);cb=fig.colorbar(im,ax=axes.tolist(),fraction=.035,pad=.03);cb.set_label("行归一化比例")
for ext in ["png","pdf"]:fig.savefig(OUT/"figures"/f"BFPT_CND1混淆矩阵.{ext}",dpi=400 if ext=="png" else None,bbox_inches="tight")
plt.close(fig);print(main[["device","accuracy_mean","accuracy_std","macro_f1_mean","macro_f1_std"]].to_string(index=False))
