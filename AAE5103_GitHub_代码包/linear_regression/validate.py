"""Evaluate the frozen JFK regression and history baseline on June validation only."""
import csv,json,os
from datetime import date
import joblib
import numpy as np
from common import DATA, FEATURES, RESULTS, final_counts, load, metrics, sha256
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/aae5103-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    path=DATA/'validation.csv'; model_path=RESULTS/'linear_model.joblib'
    artifact=joblib.load(model_path)  # Only load trusted local model files.
    if artifact['features']!=FEATURES:raise ValueError('Model feature definitions do not match')
    X,y,times,baseline=load(path,date(2025,6,1),date(2025,6,30),baseline=True)
    if len(X)!=30*1041:raise ValueError(f'Unexpected validation sample count: {len(X)}')
    raw=artifact['estimator'].predict(X); predicted=final_counts(raw); baseline=baseline.astype(np.int64)
    result={'period':'2025-06-01 to 2025-06-30','rows':len(y),'train_sha256':artifact['train_sha256'],'validation_sha256':sha256(path),'linear_regression':metrics(y,predicted),'historical_average_three_week':metrics(y,baseline),'prediction_treatment':'model output clipped at zero and rounded to nearest integer, ties to even','caution':'Overlapping one-minute windows are dependent. Outcomes are recorded JFK taxi pickups, not unmet passenger demand.'}
    weekdays=np.array([t.weekday()<5 for t in times]); result['by_day_type']={}
    for key,mask in [('workday',weekdays),('weekend',~weekdays)]:
        result['by_day_type'][key]={'samples':int(mask.sum()),'linear_regression':metrics(y[mask],predicted[mask]),'historical_average_three_week':metrics(y[mask],baseline[mask])}
    (RESULTS/'validation_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (RESULTS/'validation_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['as_of_local','actual_pickups','linear_raw','linear_forecast','historical_average_three_week'])
        for t,actual,r,p,b in zip(times,y,raw,predicted,baseline):w.writerow([t.isoformat(sep=' '),int(actual),f'{r:.8f}',int(p),int(b)])
    # One validation day displayed in 10-minute increments; scores use every minute.
    indices=[i for i,t in enumerate(times) if t.date()==date(2025,6,10) and t.minute%10==0]
    fig,ax=plt.subplots(figsize=(12,4)); x=[times[i] for i in indices]
    ax.plot(x,y[indices],label='Recorded pickups',color='#243e60')
    ax.plot(x,predicted[indices],label='Linear regression',color='#168aaf')
    ax.plot(x,baseline[indices],label='Three-week historical mean',color='#e39835',alpha=.85)
    ax.set(title='JFK pickup forecasts: 10 June 2025 validation',xlabel='Prediction time (New York local)',ylabel='Pickups in (t+10, t+40]')
    ax.legend();ax.grid(alpha=.25);fig.autofmt_xdate();fig.tight_layout();fig.savefig(RESULTS/'validation_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
