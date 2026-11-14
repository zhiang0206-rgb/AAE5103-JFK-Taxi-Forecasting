"""Evaluate frozen seasonal AR model on June only; never use Jul–Dec samples."""
import csv,json,os
from datetime import datetime,timedelta
import numpy as np
from common import DATA,VALIDATION,LINEAR,RESULTS,forecast_raw,final_count,metrics,read_minutes,sha256
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/aae5103-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 model=json.loads((RESULTS/'time_series_model.json').read_text(encoding='utf8'))
 times,counts,valid=read_minutes(DATA/'validation_minutes.csv',datetime(2025,6,1),datetime(2025,7,1))
 prefix=np.r_[0,np.cumsum(counts)]
 with open(LINEAR,newline='',encoding='utf8') as f:
  linear={r['as_of_local']:r for r in csv.DictReader(f)}
 records=[];start=times[0]
 with open(VALIDATION,newline='',encoding='utf8') as f:
  for row in csv.DictReader(f):
   t=datetime.fromisoformat(row['as_of_local'])
   if not datetime(2025,6,1)<=t<datetime(2025,7,1):raise ValueError('Outside June')
   idx=int((t-start).total_seconds()//60)
   if idx<9 or idx+40>=len(times) or not valid[idx-9:idx+41].all():raise ValueError('Incomplete observed or target minutes: '+str(t))
   if datetime.fromisoformat(row['window_start_exclusive'])!=t+timedelta(minutes=10) or datetime.fromisoformat(row['window_end_inclusive'])!=t+timedelta(minutes=40):raise ValueError('Incorrect target window')
   target=int(row['target_pickups'])
   # Independently cross-check sum of (t+10,t+40] observed bins against shared labels.
   truth=int(prefix[idx+41]-prefix[idx+11])
   if target!=truth:raise ValueError(f'Target mismatch at {t}: {target}, {truth}')
   observed=int(prefix[idx+1]-prefix[idx-9])
   if observed!=int(row['last_10_min']):raise ValueError('Recent history mismatch: '+str(t))
   raw=forecast_raw(model,t,observed);p=final_count(raw)
   old=linear.get(row['as_of_local'])
   if old is None or int(old['actual_pickups'])!=target or int(old['historical_average_three_week'])!=int(row['history_three_week']):raise ValueError('Comparators not aligned')
   records.append((t,target,raw,p,int(row['history_three_week']),int(old['linear_forecast']),observed))
 if len(records)!=31230 or len(linear)!=len(records):raise ValueError('Validation sample count mismatch')
 y=np.asarray([r[1] for r in records]);pred=np.asarray([r[3] for r in records]);base=np.asarray([r[4] for r in records]);lin=np.asarray([r[5] for r in records]);week=np.asarray([r[0].weekday()<5 for r in records])
 result={'period':'2025-06-01 to 2025-06-30','rows':len(records),'series_sha256':sha256(DATA/'validation_minutes.csv'),'windows_sha256':sha256(VALIDATION),'time_series':metrics(y,pred),'historical_average_three_week':metrics(y,base),'linear_regression':metrics(y,lin),'by_day_type':{}}
 for name,mask in [('workday',week),('weekend',~week)]:
  result['by_day_type'][name]={'samples':int(mask.sum()),'time_series':metrics(y[mask],pred[mask]),'historical_average_three_week':metrics(y[mask],base[mask]),'linear_regression':metrics(y[mask],lin[mask])}
 result['notes']=['Fixed model trained Jan–May. Rolling forecast observes the last 10 minutes up to t; future counts used only for scoring.','Overlapping minute windows are dependent.','Linear regression, baseline and AR model use different historical information.']
 (RESULTS/'validation_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 with (RESULTS/'validation_predictions.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.writer(f);w.writerow(['as_of_local','observed_last_10_minutes','actual_pickups','time_series_raw','time_series_forecast','history_three_week','linear_forecast'])
  for t,actual,raw,p,b,l,obs in records:w.writerow([t.isoformat(sep=' '),obs,actual,f'{raw:.8f}',p,b,l])
 ix=[i for i,r in enumerate(records) if r[0].date().isoformat()=='2025-06-10' and r[0].minute%10==0]
 fig,ax=plt.subplots(figsize=(12,4));xx=[records[i][0] for i in ix]
 ax.plot(xx,y[ix],label='Recorded pickups',color='#243e60');ax.plot(xx,pred[ix],label='Seasonal 10-min AR(1)',color='#178cac');ax.plot(xx,base[ix],label='Three-week history',color='#e39835',alpha=.85)
 ax.set(title='JFK time series: 10 June 2025 validation',xlabel='Prediction time (New York local)',ylabel='Pickups in (t+10, t+40]');ax.legend();ax.grid(alpha=.25);fig.autofmt_xdate();fig.tight_layout();fig.savefig(RESULTS/'validation_comparison.png',dpi=170);plt.close(fig)
 print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
