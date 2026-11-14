"""Fit weekly minute profile and 10-minute-lag AR(1) residual."""
import json
from datetime import datetime
import numpy as np
from common import DATA,RESULTS,read_minutes,sha256,slot

def main():
 p=DATA/'train_minutes.csv'
 t,y,valid=read_minutes(p,datetime(2025,1,1),datetime(2025,5,31,23,59))
 ids=np.fromiter((slot(x) for x in t),dtype=int,count=len(t))
 n=np.bincount(ids[valid],minlength=1008); sums=np.bincount(ids[valid],weights=y[valid],minlength=1008)
 if np.any(n==0):raise ValueError('Every weekday/10-minute slot needs training data')
 profile=sums/n
 # Rolling 10-minute totals; the previous and current 10-minute windows
 # are disjoint, although successive training examples overlap.
 kernel=np.ones(10)
 residual=np.convolve(y,kernel,mode='valid')-np.convolve(profile[ids],kernel,mode='valid')
 good=np.convolve(valid.astype(int),kernel,mode='valid')==10
 pairs=good[10:]&good[:-10]
 denom=np.sum(residual[:-10][pairs]**2)
 if denom==0:raise ValueError('No residual variation')
 phi=float(np.sum(residual[:-10][pairs]*residual[10:][pairs])/denom)
 if not -1<phi<1:raise ValueError('Nonstationary AR(1); inspect training data')
 model={'model_type':'time-of-week profile + 10-minute AR(1) residual','profile_description':'Mean observed 1-minute counts per weekday and 10-minute slot from training only','time_of_week_profile':profile.tolist(),'profile_slot_observations':n.tolist(),'ar1_coefficient':phi,'ar_lag_minutes':10,'train_start':'2025-01-01','train_end':'2025-05-31','train_minute_rows':len(t),'train_valid_rows':int(valid.sum()),'train_pairs':int(pairs.sum()),'train_sha256':sha256(p),'forecast':'sum expected minute bins h=11,...,40 plus AR corrections for ten-minute segments k=2,3,4; negative total clipped, nearest integer ties-to-even','caveat':'One-month validation not final test; rolling recent ten-minute observations assumed available in real time.'}
 RESULTS.mkdir(exist_ok=True);(RESULTS/'time_series_model.json').write_text(json.dumps(model,indent=2,ensure_ascii=False),encoding='utf8')
 print(json.dumps({k:v for k,v in model.items() if k not in ('time_of_week_profile','profile_slot_observations')},indent=2))
if __name__=='__main__':main()
