from __future__ import annotations
import argparse,re,json
from pathlib import Path
import numpy as np,pandas as pd

def parse_list(path):
 rows=[]
 for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
  m=re.match(r'^(SYG\S+)\s*\|\s*(U|B1|B2|B3|Y)\s*\|\s*([^|]+)\|\s*(.+?)\s*$',line.strip())
  if m:rows.append(dict(tag=m.group(1),role=m.group(2),name_cn=m.group(3).strip(),unit=m.group(4).strip()))
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--csv',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--device',action='append',nargs=2,metavar=('NAME','LIST'),required=True);a=ap.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 devices={name:parse_list(p) for name,p in a.device};header=pd.read_csv(a.csv,nrows=0,encoding='utf-8-sig').columns.tolist();available=set(header);tags=sorted({r['tag'] for rows in devices.values() for r in rows if r['tag'] in available});missing=[dict(device=n,**r) for n,rows in devices.items() for r in rows if r['tag'] not in available]
 use=['time']+tags;print({'requested':sum(map(len,devices.values())),'available_unique':len(tags),'missing':len(missing)},flush=True)
 frames=[]
 for chunk in pd.read_csv(a.csv,usecols=use,chunksize=120000,encoding='utf-8-sig',low_memory=False):
  t=pd.to_datetime(chunk['time'],errors='coerce');m=((t>='2024-05-01')&(t<'2024-07-01'))
  if m.any():chunk=chunk.loc[m].copy();chunk['time']=t[m].values;frames.append(chunk)
 data=pd.concat(frames,ignore_index=True).sort_values('time').drop_duplicates('time');data.to_pickle(out/'may_june_2024.pkl')
 audit=[]
 for name,rows in devices.items():
  for r in rows:
   if r['tag'] not in data: audit.append(dict(device=name,**r,available=False,numeric_rate=0,nonzero_rate=0));continue
   x=pd.to_numeric(data[r['tag']],errors='coerce');audit.append(dict(device=name,**r,available=True,numeric_rate=float(x.notna().mean()),nonzero_rate=float((x.fillna(0).abs()>1e-10).mean()),min=float(x.min()) if x.notna().any() else np.nan,max=float(x.max()) if x.notna().any() else np.nan))
 pd.DataFrame(audit).to_csv(out/'point_availability_audit.csv',index=False,encoding='utf-8-sig');pd.DataFrame(missing).to_csv(out/'missing_points.csv',index=False,encoding='utf-8-sig');(out/'devices.json').write_text(json.dumps(devices,ensure_ascii=False,indent=2),encoding='utf-8');print({'rows':len(data),'first':str(data.time.min()),'last':str(data.time.max()),'cache_mb':round((out/'may_june_2024.pkl').stat().st_size/2**20,1)},flush=True)
if __name__=='__main__':main()
