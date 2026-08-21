from __future__ import annotations
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,ConnectionPatch

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'deliverables';(OUT/'figures').mkdir(parents=True,exist_ok=True);(OUT/'tables').mkdir(parents=True,exist_ok=True);(OUT/'plot_data').mkdir(parents=True,exist_ok=True)
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
files=list((ROOT/'runs').glob('*/*/seed*/metrics_by_point.csv'));allm=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);allm.to_csv(OUT/'tables/all_device_point_metrics.csv',index=False,encoding='utf-8-sig')
test=allm[allm.split.eq('test')];agg=(test.groupby(['device','model','tag','name_cn','unit'],as_index=False).agg(seeds=('seed','nunique'),r2_mean=('r2','mean'),r2_std=('r2','std'),rmse_mean=('rmse','mean'),rmse_std=('rmse','std'),mae_mean=('mae','mean'),mae_std=('mae','std'),tolerance_mean=('within_tolerance','mean')));agg.to_csv(OUT/'tables/all_device_point_metrics_3seed_summary.csv',index=False,encoding='utf-8-sig')
devsum=(test.groupby(['device','model'],as_index=False).agg(seeds=('seed','nunique'),outputs=('tag','nunique'),macro_r2_mean=('r2','mean'),macro_r2_std=('r2','std'),macro_rmse_mean=('rmse','mean'),macro_mae_mean=('mae','mean')));devsum.to_csv(OUT/'tables/device_model_macro_summary.csv',index=False,encoding='utf-8-sig')
piv=agg.pivot_table(index=['device','tag','name_cn','unit'],columns='model',values='r2_mean').reset_index();piv['other_best']=piv[['ANN','TCN','iTransformer-small']].max(axis=1);piv['selected']=(piv.GRU>=.95)&(piv.GRU>=piv.other_best);piv.to_csv(OUT/'tables/representative_point_selection_audit.csv',index=False,encoding='utf-8-sig')
selected={'DEA':'SYG1_10LAA10CP101','H1':'SYG1_10LAB61CT101'};cache=pd.read_pickle(ROOT/'cache/may_june_2024.pkl');aud=[];load_rows=[]
colors={'ANN':'#d95f5f','GRU':'#2f6db3','TCN':'#3a9d5d','iTransformer-small':'#8b6bb1'}
for device,tag in selected.items():
 preds={};actual=None;time=None;name=unit=''
 for model in colors:
  p=ROOT/'runs'/device/model/'seed20260801/predictions.npz';d=np.load(p,allow_pickle=True);tags=list(d['target_tags'].astype(str));j=tags.index(tag);preds[model]=d['test_pred'][:,j];
  if actual is None:actual=d['test_actual'][:,j];time=d['test_time'];name=str(d['target_names'][j]);unit=str(d['target_units'][j])
 frame=pd.DataFrame({'timestamp':pd.to_datetime(time,unit='s'),'actual':actual,**{f'pred_{k}':v for k,v in preds.items()}});frame.to_csv(OUT/'plot_data'/f'{device}_{tag}_predictions.csv',index=False,encoding='utf-8-sig')
 load_map=cache[['time','SYG1_MW']].copy();load_map['time']=pd.to_datetime(load_map['time']);frame_load=frame.merge(load_map,left_on='timestamp',right_on='time',how='left')
 for band,mask in [('low_<400',frame_load['SYG1_MW']<400),('mid_400_700',frame_load['SYG1_MW'].between(400,700,inclusive='left')),('high_>=700',frame_load['SYG1_MW']>=700)]:
  q=frame_load[mask]
  if len(q):
   e=q.actual-q.pred_GRU;load_rows.append(dict(device=device,tag=tag,name_cn=name,load_band=band,samples=len(q),r2=1-float((e**2).sum()/((q.actual-q.actual.mean())**2).sum()) if ((q.actual-q.actual.mean())**2).sum()>0 else np.nan,rmse=float(np.sqrt((e**2).mean())),mae=float(abs(e).mean()),bias=float(e.mean())))
 err=actual-preds['GRU'];row={'device':device,'tag':tag,'name_cn':name,'unit':unit,'mean':err.mean(),'std':err.std(),'p95_abs':np.quantile(abs(err),.95),'max_abs':abs(err).max()}
 for lag in (1,10,30,60):row[f'acf_lag{lag}']=np.corrcoef(err[:-lag],err[lag:])[0,1]
 aud.append(row)
 n=len(actual);w=min(2500,max(500,n//15));ranges=pd.Series(actual).rolling(w).max()-pd.Series(actual).rolling(w).min();end=int(ranges.idxmax()) if ranges.notna().any() else min(n,w);start=max(0,end-w);zoom=np.arange(start,min(n,start+w));full=np.arange(0,n,max(1,n//12000));scatter=np.linspace(0,n-1,min(6000,n),dtype=int)
 fig=plt.figure(figsize=(11.2,7.5));gs=fig.add_gridspec(2,2,width_ratios=[3.2,1.25],height_ratios=[1,1.05],hspace=.42,wspace=.25);axz=fig.add_subplot(gs[0,0]);axs=fig.add_subplot(gs[0,1]);axf=fig.add_subplot(gs[1,:])
 axz.plot(zoom,actual[zoom],color='black',lw=1,label='实际值');axz.plot(zoom,preds['GRU'][zoom],color=colors['GRU'],lw=.9,label='GRU');axz.set_ylabel(f'{name}/{unit}');axz.legend(frameon=False,ncol=2,loc='upper left');axz.grid(alpha=.15)
 axs.scatter(actual[scatter],preds['GRU'][scatter],s=5,c=colors['GRU'],alpha=.45,label='GRU');lo=min(actual[scatter].min(),preds['GRU'][scatter].min());hi=max(actual[scatter].max(),preds['GRU'][scatter].max());axs.plot([lo,hi],[lo,hi],'k--',lw=1,label='y=x');axs.set_xlabel(f'实际值/{unit}');axs.set_ylabel(f'预测值/{unit}');axs.legend(frameon=False,fontsize=8)
 axf.plot(full,actual[full],color='black',lw=.75,label='实际值');axf.plot(full,preds['GRU'][full],color=colors['GRU'],lw=.65,label='GRU');ylo,yhi=axf.get_ylim();rect=Rectangle((start,ylo),len(zoom),yhi-ylo,fill=False,edgecolor='#e74c3c',lw=1.4);axf.add_patch(rect);axf.set_xlabel('样本');axf.set_ylabel(f'{name}/{unit}');axf.legend(frameon=False,ncol=2,loc='upper right');axf.grid(alpha=.12)
 con=ConnectionPatch(xyA=(start+len(zoom)/2,yhi),coordsA=axf.transData,xyB=(.5,0),coordsB=axz.transAxes,arrowstyle='-|>',color='#e74c3c',lw=1.2);fig.add_artist(con);fig.suptitle(f'{device}｜{tag}｜{name}：GRU预测曲线与散点分布',fontsize=13);fig.savefig(OUT/'figures'/f'{device}_{tag}_yang_style.png',dpi=220,bbox_inches='tight');fig.savefig(OUT/'figures'/f'{device}_{tag}_yang_style.pdf',bbox_inches='tight');plt.close(fig)
 if device=='H1':
  fig=plt.figure(figsize=(11.2,7.5));gs=fig.add_gridspec(2,2,width_ratios=[3.2,1.25],height_ratios=[1,1.05],hspace=.42,wspace=.25);axz=fig.add_subplot(gs[0,0]);axs=fig.add_subplot(gs[0,1]);axf=fig.add_subplot(gs[1,:])
  axz.plot(zoom,actual[zoom],color='black',lw=1,label='实际值')
  for m,c in colors.items():axz.plot(zoom,preds[m][zoom],color=c,lw=.75,label=m)
  axz.set_ylabel(f'{name}/{unit}');axz.legend(frameon=False,ncol=3,fontsize=8);axz.grid(alpha=.15)
  for m,c in colors.items():axs.scatter(actual[scatter],preds[m][scatter],s=4,c=c,alpha=.32,label=m)
  lo=min(actual[scatter].min(),*(preds[m][scatter].min() for m in colors));hi=max(actual[scatter].max(),*(preds[m][scatter].max() for m in colors));axs.plot([lo,hi],[lo,hi],'k--',lw=1,label='y=x');axs.set_xlabel(f'实际值/{unit}');axs.set_ylabel(f'预测值/{unit}');axs.legend(frameon=False,fontsize=7)
  axf.plot(full,actual[full],color='black',lw=.75,label='实际值')
  for m,c in colors.items():axf.plot(full,preds[m][full],color=c,lw=.55,label=m)
  ylo,yhi=axf.get_ylim();axf.add_patch(Rectangle((start,ylo),len(zoom),yhi-ylo,fill=False,edgecolor='#e74c3c',lw=1.4));axf.set_xlabel('样本');axf.set_ylabel(f'{name}/{unit}');axf.legend(frameon=False,ncol=5,fontsize=8);axf.grid(alpha=.12);fig.add_artist(ConnectionPatch(xyA=(start+len(zoom)/2,yhi),coordsA=axf.transData,xyB=(.5,0),coordsB=axz.transAxes,arrowstyle='-|>',color='#e74c3c',lw=1.2));fig.suptitle(f'{device}｜{tag}｜{name}：四模型预测曲线与散点分布',fontsize=13);fig.savefig(OUT/'figures'/f'{device}_{tag}_yang_style_all_models.png',dpi=220,bbox_inches='tight');fig.savefig(OUT/'figures'/f'{device}_{tag}_yang_style_all_models.pdf',bbox_inches='tight');plt.close(fig)
pd.DataFrame(aud).to_csv(OUT/'tables/representative_residual_audit.csv',index=False,encoding='utf-8-sig');pd.DataFrame(load_rows).to_csv(OUT/'tables/representative_load_segment_error.csv',index=False,encoding='utf-8-sig')
print(piv[piv.selected].to_string(index=False));print(pd.DataFrame(aud).to_string(index=False))
