"""Create minute pickup series using ONLY January–June 2025 JFK records."""
import csv, hashlib, json
from datetime import datetime
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'generalization_2025/monthly_jfk'
OUT=HERE/'data'

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 OUT.mkdir(exist_ok=True)
 arrays=[]; sources=[]
 for m in range(1,7):
  p=SOURCE/f'jfk_yellow_2025-{m:02d}.parquet'
  arrays.append(pq.read_table(p,columns=['tpep_pickup_datetime'])['tpep_pickup_datetime'].to_numpy(zero_copy_only=False).astype('datetime64[us]'))
  sources.append({'file':p.name,'sha256':digest(p)})
 events=np.sort(np.concatenate(arrays))
 if np.isnat(events).any():raise ValueError('Null pickup timestamp')
 points=np.arange(np.datetime64('2025-01-01T00:00','m'),np.datetime64('2025-07-01T00:01','m'),np.timedelta64(1,'m')).astype('datetime64[us]')
 ends=np.searchsorted(events,points,side='right');starts=np.searchsorted(events,points-np.timedelta64(1,'m'),side='right')
 midnight=points.astype('datetime64[m]').astype('int64')%1440==0
 ends[midnight]=np.searchsorted(events,points[midnight],side='left')
 counts=ends-starts
 if (counts<0).any():raise ValueError('Invalid interval count')
 missing_dst_start=np.datetime64('2025-03-09T02:00','us');missing_dst_end=np.datetime64('2025-03-09T03:00','us')
 valid=np.logical_not((points>=missing_dst_start)&(points<=missing_dst_end));valid[0]=False
 files={s:(OUT/f'{s}_minutes.csv').open('w',encoding='utf8',newline='') for s in ('train','validation')}
 writers={s:csv.writer(f) for s,f in files.items()};n={'train':0,'validation':0}
 for w in writers.values():w.writerow(['bin_end_local','pickups','valid'])
 try:
  for p,c,v in zip(points,counts,valid):
   t=datetime.fromisoformat(np.datetime_as_string(p,unit='m'))
   group='train' if t<datetime(2025,6,1) else 'validation'
   writers[group].writerow([t.isoformat(sep=' '),int(c),int(v)]);n[group]+=1
 finally:
  for f in files.values():f.close()
 assert n=={'train':151*1440,'validation':30*1440+1},n
 meta={'source_files':sources,'bin_definition':'(bin_end-1min, bin_end], midnight end excludes the midnight instant','scope':'January through June only, plus June 30 final bin at 2025-07-01 00:00, excluding the midnight instant','invalid':'first bin lacks prior December data; 2025-03-09 02:00–03:00 local excluded for DST transition','minute_rows':n,'input_records':len(events)}
 (OUT/'series_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
