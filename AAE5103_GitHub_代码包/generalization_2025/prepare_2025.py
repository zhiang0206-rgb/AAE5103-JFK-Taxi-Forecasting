"""Build JFK 2025 chronological samples and rolling 3-week baseline."""
import csv,json,math,hashlib
from datetime import date,datetime,timedelta
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
FEATURES=['last_10_min','last_30_min','last_60_min','hour_sin','hour_cos']+[f'weekday_{i}' for i in range(7)]

def main():
 summary=json.loads((ROOT/'collection_summary.json').read_text(encoding='utf8'))
 if len(summary)!=12:raise ValueError('12 verified months required')
 chunks=[]
 for m in range(1,13):
  p=ROOT/'monthly_jfk'/f'jfk_yellow_2025-{m:02d}.parquet'
  chunks.append(pq.read_table(p,columns=['tpep_pickup_datetime'])['tpep_pickup_datetime'].to_numpy(zero_copy_only=False).astype('datetime64[us]'))
 stamps=np.sort(np.concatenate(chunks))
 if np.isnat(stamps).any():raise ValueError('Unexpected null timestamps')
 out=ROOT/'shared_samples';out.mkdir(exist_ok=True)
 files={k:(out/(k+'.csv')).open('w',encoding='utf-8',newline='') for k in ['train','validation','test']}
 cols=['as_of_local','window_start_exclusive','window_end_inclusive']+FEATURES+['target_pickups','history_three_week']
 writers={k:csv.writer(f) for k,f in files.items()}
 for w in writers.values():w.writerow(cols)
 count={k:0 for k in files};baseline_missing={k:0 for k in files}
 def count_interval(lo,hi,day):
  # Same semantics as current model: (lo, hi], with next midnight excluded.
  after=np.searchsorted(stamps,lo,side='right');end=np.searchsorted(stamps,hi,side='right')
  cap=np.searchsorted(stamps,day+np.timedelta64(1,'D'),side='left')
  return np.minimum(end,cap)-after
 try:
  for i in range(365):
   d=date(2025,1,1)+timedelta(days=i);group='train' if d.month<=5 else 'validation' if d.month==6 else 'test'
   day=np.datetime64(d,'us');minutes=np.arange(360,1401,dtype=np.int64)
   t=day+minutes.astype('timedelta64[m]'); features=[]
   for m in (10,30,60):
    features.append(np.searchsorted(stamps,t,side='right')-np.searchsorted(stamps,t-np.timedelta64(m,'m'),side='right'))
   target=count_interval(t+np.timedelta64(10,'m'),t+np.timedelta64(40,'m'),day)
   has_history=d>=date(2025,1,22)
   if has_history:
    seq=[]
    for weeks in (1,2,3):
     shift=np.timedelta64(7*weeks,'D');seq.append(count_interval(t-shift+np.timedelta64(10,'m'),t-shift+np.timedelta64(40,'m'),day-shift))
    history=np.rint(np.mean(seq,axis=0)).astype(int)
   else:baseline_missing[group]+=len(t);history=['']*len(t)
   for j,minute in enumerate(minutes):
    hh,mm=divmod(int(minute),60);now=datetime(d.year,d.month,d.day,hh,mm);angle=2*math.pi*int(minute)/1440
    x=[int(features[k][j]) for k in range(3)]+[math.sin(angle),math.cos(angle)]+[int(d.weekday()==k) for k in range(7)]
    writers[group].writerow([now.isoformat(sep=' '),(now+timedelta(minutes=10)).isoformat(sep=' '),(now+timedelta(minutes=40)).isoformat(sep=' '),*x,int(target[j]),history[j]])
   count[group]+=len(t)
 finally:
  for f in files.values():f.close()
 expected={'train':151*1041,'validation':30*1041,'test':184*1041}
 assert count==expected,(count,expected)
 result={'scope':'JFK yellow taxi pickup records 2025-01 through 2025-12','source':'NYC TLC monthly yellow taxi trip records; PULocationID=132','source_record_count':len(stamps),'split':{'train':'2025-01-01 to 2025-05-31','validation':'2025-06-01 to 2025-06-30','test':'2025-07-01 to 2025-12-31'},'rows':count,'baseline_unavailable_rows':baseline_missing,'features':FEATURES,'history_three_week':'Reference baseline only, NOT a model feature. Prior 1,2,3 same weekdays, same target window, rounded to integer; blank before Jan 22. Earlier observed test dates can serve as history.','target':'Counts JFK pickups in (t+10min,t+40min], excluding next-day midnight, same as prior demo','prediction_schedule':'Daily at one-minute intervals from 06:00 to 23:20 inclusive; New York local time','important':['Test set must not be used for fitting, tuning, early stopping, or choosing model features.','Adjacent windows overlap heavily; rows are not independent.','Historical averages depend on actual pickups being observed, not only a fixed static forecast.','January and February etc extracted separately from raw monthly files with out-of-month rows excluded.','No weather or flight features; not observed passenger arrival or waiting; no guarantee of realtime availability.']}
 (out/'metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
