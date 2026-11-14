"""Document the two AR lag designs explored on the June validation set."""
import csv,json
from datetime import datetime,timedelta
import numpy as np
from common import DATA,RESULTS,VALIDATION,read_minutes,slot,final_count,metrics

def main():
 model=json.loads((RESULTS/'time_series_model.json').read_text())
 p=np.asarray(model['time_of_week_profile']);t,y,valid=read_minutes(DATA/'train_minutes.csv',datetime(2025,1,1),datetime(2025,5,31,23,59))
 ids=np.fromiter((slot(x) for x in t),dtype=int,count=len(t));r=y-p[ids];pairs=valid[1:]&valid[:-1];phi=float(np.dot(r[:-1][pairs],r[1:][pairs])/np.dot(r[:-1][pairs],r[:-1][pairs]))
 vt,vc,vvalid=read_minutes(DATA/'validation_minutes.csv',datetime(2025,6,1),datetime(2025,7,1));start=vt[0]
 truth=[];pred=[]
 with open(VALIDATION,newline='',encoding='utf8') as f:
  for row in csv.DictReader(f):
   now=datetime.fromisoformat(row['as_of_local']);idx=int((now-start).total_seconds()//60)
   seasonal=sum(p[slot(now+timedelta(minutes=h))] for h in range(11,41))
   forecast=seasonal+sum(phi**h for h in range(11,41))*(vc[idx]-p[slot(now)])
   truth.append(int(row['target_pickups']));pred.append(final_count(forecast))
 selected=json.loads((RESULTS/'validation_metrics.json').read_text())['time_series']
 report={'note':'Both development variants use training-estimated weekly profile and residual AR coefficient. June validation was inspected to compare them, so June is model-selection evidence, not a final unbiased test.','candidates':[{'lag_minutes':1,'ar1_coefficient':phi,'validation':metrics(truth,pred)},{'lag_minutes':10,'ar1_coefficient':model['ar1_coefficient'],'validation':selected}],'selected_lag_minutes':10,'selection':'10-minute residual captures recent changes over the forecast horizon and had lower June MAE; reserve July–December for final assessment.'}
 (RESULTS/'development_candidates.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
