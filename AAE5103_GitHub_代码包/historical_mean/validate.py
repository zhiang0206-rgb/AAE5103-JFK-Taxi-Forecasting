"""Validate the three-same-weekday historical mean on June without opening test data."""
import csv,json,math
from datetime import datetime,timedelta
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=HERE/'data' if (HERE/'data').exists() else HERE.parent/'generalization_2025/shared_samples'
OUT=HERE/'results'

def load(path,lo,hi):
 rows=[]; last=None
 with path.open(newline='',encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):
   t=datetime.fromisoformat(r['as_of_local']);end=datetime.fromisoformat(r['window_end_inclusive'])
   if not lo<=t<hi or (last is not None and t<=last):raise ValueError('Invalid date or order: '+str(t))
   if end!=t+timedelta(minutes=40):raise ValueError('Unexpected target window end')
   rows.append((t,int(r['target_pickups']),r.get('history_three_week',''),end));last=t
 return rows

def score(items):
 err=[p-a for _,a,p,*_ in items]
 return {'mae_trips':sum(abs(e) for e in err)/len(err),'rmse_trips':math.sqrt(sum(e*e for e in err)/len(err))}

def main():
 train=load(DATA/'train.csv',datetime(2025,1,1),datetime(2025,6,1))
 val=load(DATA/'validation.csv',datetime(2025,6,1),datetime(2025,7,1))
 if len(train)!=151*1041 or len(val)!=30*1041:raise ValueError('Unexpected sample count')
 history={t:(target,end) for t,target,_,end in train+val}
 predictions=[]
 for t,actual,precomputed,_ in val:
  past=[]
  for weeks in (1,2,3):
   key=t-timedelta(days=7*weeks)
   if key not in history:raise ValueError('Missing historical observation: '+str(key))
   value,end=history[key]
   if end>t:raise ValueError('Historical target not yet observed: '+str(key))
   past.append(value)
  forecast=round(sum(past)/3)
  if not precomputed or int(precomputed)!=forecast:raise ValueError('Precomputed baseline differs: '+str(t))
  predictions.append((t,actual,forecast,*past))
 OUT.mkdir(exist_ok=True)
 with (OUT/'validation_predictions.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.writer(f);w.writerow(['as_of_local','actual_pickups','history_three_week','one_week_before','two_weeks_before','three_weeks_before'])
  for t,actual,forecast,*past in predictions:w.writerow([t.isoformat(sep=' '),actual,forecast,*past])
 result={'method':'same weekday, same window, mean of previous 3 weeks','validation_period':'2025-06-01 to 2025-06-30','validation_rows':len(predictions),'validation':score(predictions),'workday':score([r for r in predictions if r[0].weekday()<5]),'weekend':score([r for r in predictions if r[0].weekday()>=5]),'note':'No parameters fitted; each forecast uses earlier observed windows. Test set not opened.'}
 (OUT/'validation_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
