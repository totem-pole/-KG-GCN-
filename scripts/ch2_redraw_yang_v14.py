"""Chapter 2 publication figures in Yang-Haodong-style layout.

Reads frozen experiment archive only; no retraining.  Outputs full-test prediction
curves + fixed 48 h zoom + actual-vs-predicted scatter for VHP, H1 and DEA.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

ROOT = Path(__file__).resolve().parents[1]
VHP = ROOT / "experiments/chapter2_state_monitoring/vhp_final24"
NON = ROOT / "experiments/chapter2_state_monitoring/non_vhp_representatives_20260822/deliverables/plot_data"
OUT = ROOT / "artifacts/ch2_yang_v14"
OUT.mkdir(parents=True, exist_ok=True)

C_ACTUAL, C_ANN, C_GRU, C_ZOOM = "#222222", "#E64B35", "#00A087", "#D62728"


def rc():
    plt.rcParams.update({
        "font.sans-serif": ["Noto Sans CJK SC", "SimSun", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10.5, "axes.labelsize": 10.5,
        "xtick.labelsize": 9.0, "ytick.labelsize": 9.0, "legend.fontsize": 9.0,
        "axes.linewidth": 0.8, "savefig.dpi": 600,
    })


def limits(y, m=.04):
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y)); s=max(hi-lo,1e-9)
    return lo-m*s, hi+m*s


def load_vhp(model):
    z=np.load(VHP / "predictions" / f"month_expanded_2024_{model}.npz", allow_pickle=True)
    tags=z["target_tags"].astype(str)
    time=pd.to_datetime(z["test_time"], unit="s")
    return tags, time, z["test_actual"], z["test_pred"]


def vhp_join(tag):
    ta, t_ann, a_ann, p_ann = load_vhp("ANN")
    tg, t_gru, a_gru, p_gru = load_vhp("GRU-RNN")
    ia=np.where(ta==tag)[0]; ig=np.where(tg==tag)[0]
    if len(ia)!=1 or len(ig)!=1: raise RuntimeError(f"cannot locate {tag}")
    da=pd.DataFrame({"timestamp":t_ann,"actual_ann":a_ann[:,ia[0]],"ANN":p_ann[:,ia[0]]})
    dg=pd.DataFrame({"timestamp":t_gru,"actual_gru":a_gru[:,ig[0]],"GRU":p_gru[:,ig[0]]})
    d=da.merge(dg,on="timestamp",how="inner",validate="one_to_one").sort_values("timestamp").reset_index(drop=True)
    diff=np.nanmax(np.abs(d["actual_ann"]-d["actual_gru"]))
    if diff>1e-4: raise RuntimeError(f"actual mismatch for {tag}: {diff}")
    return pd.DataFrame({"timestamp":d.timestamp,"actual":d.actual_gru,"ANN":d.ANN,"GRU":d.GRU})


def draw_panel(fig, spec, df, title, unit, label, r2_ann=None, r2_gru=None):
    gs=spec.subgridspec(2,4,height_ratios=[1.25,.86],width_ratios=[1,1,1,1.08],hspace=.18,wspace=.35)
    az=fig.add_subplot(gs[0,:3]); asc=fig.add_subplot(gs[0,3]); af=fig.add_subplot(gs[1,:])
    x=np.arange(len(df)); y=df.actual.to_numpy(); ann=df.ANN.to_numpy(); gru=df.GRU.to_numpy(); n=len(df)
    stride=max(1,n//6500); p=x[::stride]
    for axx in (af,az,asc): axx.grid(False)
    af.plot(p,y[::stride],color=C_ACTUAL,lw=.80,label="实际值",zorder=4)
    af.plot(p,ann[::stride],color=C_ANN,lw=.70,alpha=.90,label="ANN",zorder=2)
    af.plot(p,gru[::stride],color=C_GRU,lw=.76,alpha=.96,label="GRU",zorder=3)
    lo,hi=limits(np.r_[y,ann,gru]); af.set_xlim(0,n-1); af.set_ylim(lo,hi); af.set_ylabel(unit); af.set_xlabel("样本")
    af.legend(frameon=False,ncol=3,loc="upper right",handlelength=2.0,columnspacing=.9)
    t0=df.timestamp.iloc[0]; zn=int((df.timestamp < t0+pd.Timedelta(hours=48)).sum()); zn=max(2,min(zn,n))
    af.add_patch(Rectangle((0,lo),zn-1,hi-lo,fill=False,edgecolor=C_ZOOM,lw=1.1))
    z=np.arange(zn)
    az.plot(z,y[:zn],color=C_ACTUAL,lw=.92,label="实际值"); az.plot(z,ann[:zn],color=C_ANN,lw=.80,label="ANN"); az.plot(z,gru[:zn],color=C_GRU,lw=.85,label="GRU")
    zlo,zhi=limits(np.r_[y[:zn],ann[:zn],gru[:zn]],.06); az.set_xlim(0,zn-1); az.set_ylim(zlo,zhi); az.set_ylabel(unit)
    az.legend(frameon=False,ncol=3,loc="upper left",handlelength=2,columnspacing=.8)
    fig.add_artist(ConnectionPatch(xyA=(.50,1.00),coordsA=af.transAxes,xyB=(.50,0.00),coordsB=az.transAxes,
                                   arrowstyle="-|>",mutation_scale=10,color=C_ZOOM,lw=1.0,shrinkA=5,shrinkB=5))
    m=min(6000,n); idx=np.linspace(0,n-1,m,dtype=int)
    la="ANN" if r2_ann is None else f"ANN  $R^2$={r2_ann:.4f}"
    lg="GRU" if r2_gru is None else f"GRU  $R^2$={r2_gru:.4f}"
    asc.scatter(y[idx],ann[idx],s=5.5,marker="s",c=C_ANN,alpha=.30,edgecolors="none",label=la)
    asc.scatter(y[idx],gru[idx],s=6.0,marker="^",c=C_GRU,alpha=.38,edgecolors="none",label=lg)
    slo=min(np.nanmin(y[idx]),np.nanmin(ann[idx]),np.nanmin(gru[idx])); shi=max(np.nanmax(y[idx]),np.nanmax(ann[idx]),np.nanmax(gru[idx])); span=max(shi-slo,1e-9); slo-=.035*span; shi+=.035*span
    asc.plot([slo,shi],[slo,shi],"--",color="#555555",lw=.85,label="$y=x$"); asc.set_xlim(slo,shi); asc.set_ylim(slo,shi)
    asc.set_xlabel(f"实际值/{unit}"); asc.set_ylabel(f"预测值/{unit}"); asc.legend(frameon=False,loc="upper left",handletextpad=.3,borderaxespad=.15)
    af.text(.5,-.35,f"{label} {title}模型预测曲线与散点分布图",ha="center",va="top",transform=af.transAxes,fontsize=11.3)


def main():
    rc()
    metrics=pd.read_csv(VHP / "metrics/V25_24输出_四算法_七任务逐点指标.csv")
    metrics=metrics[(metrics.task=="2024-05训练_2024-06测试")&(metrics.split=="test")]
    examples=[("SYG1_10MAA50CP101","超高压缸叶片级压力","MPa","(a)"),("SYG1_10MAA50CT123A","超高压排汽蒸汽温度","℃","(b)")]
    fig=plt.figure(figsize=(10.8,10.5)); outer=fig.add_gridspec(2,1,hspace=.50)
    for i,(tag,name,unit,label) in enumerate(examples):
        d=vhp_join(tag)
        ra=float(metrics[(metrics.algorithm=="ANN")&(metrics.target_tag==tag)].r2.iloc[0]); rg=float(metrics[(metrics.algorithm=="GRU-RNN")&(metrics.target_tag==tag)].r2.iloc[0])
        draw_panel(fig,outer[i],d,name,unit,label,ra,rg)
    fig.subplots_adjust(left=.075,right=.985,top=.985,bottom=.06)
    fig.savefig(OUT/"fig_vhp_prediction_yang_v14.png",dpi=600,bbox_inches="tight"); fig.savefig(OUT/"fig_vhp_prediction_yang_v14.pdf",bbox_inches="tight"); plt.close(fig)

    cases=[("H1_SYG1_10LAB61CT101_predictions.csv","1号高压给水加热器出口温度","℃","(a)"),("DEA_SYG1_10LAA10CP101_predictions.csv","1号机除氧器压力","MPa","(b)")]
    fig=plt.figure(figsize=(10.8,10.5)); outer=fig.add_gridspec(2,1,hspace=.50)
    for i,(fn,name,unit,label) in enumerate(cases):
        d=pd.read_csv(NON/fn); d["timestamp"]=pd.to_datetime(d["timestamp"]); d=d.sort_values("timestamp").rename(columns={"pred_ANN":"ANN","pred_GRU":"GRU"}).reset_index(drop=True)
        draw_panel(fig,outer[i],d[["timestamp","actual","ANN","GRU"]],name,unit,label)
    fig.subplots_adjust(left=.075,right=.985,top=.985,bottom=.06)
    fig.savefig(OUT/"fig_cross_device_prediction_yang_v14.png",dpi=600,bbox_inches="tight"); fig.savefig(OUT/"fig_cross_device_prediction_yang_v14.pdf",bbox_inches="tight"); plt.close(fig)

    print(OUT)

if __name__=="__main__": main()
