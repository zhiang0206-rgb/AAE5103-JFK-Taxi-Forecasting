"""Time-of-week seasonal profile with AR(1) residuals."""
import csv,hashlib
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
DATA=HERE/'data'; RESULTS=HERE/'results'
BUNDLED=DATA/'validation_windows.csv'
VALIDATION=BUNDLED if BUNDLED.exists() else HERE.parent/'generalization_2025/shared_samples/validation.csv'
LINEAR_BUNDLED=DATA/'linear_validation.csv'
LINEAR=LINEAR_BUNDLED if LINEAR_BUNDLED.exists() else HERE.parent/'linear_regression/results/validation_predictions.csv'

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def read_minutes(path,start,end):
 times=[]; counts=[]; valid=[];previous=None
 with open(path,newline='',encoding='utf8') as f:
  for r in csv.DictReader(f):
   t=datetime.fromisoformat(r['bin_end_local'])
   if not start<=t<=end or (previous is not None and t-previous!=timedelta(minutes=1)):raise ValueError('Minute sequence must be complete and ordered: '+str(t))
   c=int(r['pickups']);v=int(r['valid'])
   if c<0 or v not in (0,1):raise ValueError('Invalid minute count or validity flag')
   times.append(t);counts.append(c);valid.append(bool(v));previous=t
 if not times or times[0]!=start or times[-1]!=end:raise ValueError('Minute coverage is incomplete')
 return times,np.asarray(counts,dtype=np.float64),np.asarray(valid,dtype=bool)

def slot(t): return t.weekday()*144+(t.hour*60+t.minute)//10

def forecast_raw(model,as_of,observed_last_10_minutes):
 """Expected trips in (t+10,t+40], using the last completed 10-minute count."""
 if as_of.second or as_of.microsecond:raise ValueError('Prediction time must be minute aligned')
 if not 360<=as_of.hour*60+as_of.minute<=1400:raise ValueError('Supported hours 06:00–23:20')
 if observed_last_10_minutes<0 or not np.isfinite(observed_last_10_minutes):raise ValueError('Invalid observed count')
 p=model['time_of_week_profile']; phi=model['ar1_coefficient']
 expected_recent=sum(p[slot(as_of+timedelta(minutes=h))] for h in range(-9,1))
 residual=observed_last_10_minutes-expected_recent
 expected_future=sum(p[slot(as_of+timedelta(minutes=h))] for h in range(11,41))
 return expected_future+sum(phi**k for k in (2,3,4))*residual

def final_count(x):return max(0,int(np.rint(x)))
def metrics(y,p):
 e=np.asarray(p,dtype=float)-np.asarray(y,dtype=float)
 return {'mae_trips':float(np.mean(np.abs(e))),'rmse_trips':float(np.sqrt(np.mean(e**2)))}
