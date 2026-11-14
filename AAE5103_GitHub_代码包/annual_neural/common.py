"""Shared data loading and feature definitions for the JFK neural model."""
import csv
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLED = HERE / 'data'
DATA = BUNDLED if BUNDLED.exists() else HERE.parent / 'generalization_2025' / 'shared_samples'
RESULTS = HERE / 'results'
FEATURES = ['last_10_min','last_30_min','last_60_min','hour_sin','hour_cos'] + [f'weekday_{i}' for i in range(7)]

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load(path, start, end, baseline=False):
    x=[]; y=[]; times=[]; refs=[]; last=None
    with open(path,newline='',encoding='utf-8-sig') as f:
        r=csv.DictReader(f)
        required=['as_of_local','window_start_exclusive','window_end_inclusive','target_pickups']+FEATURES
        if baseline:required+=['history_three_week']
        missing=[k for k in required if k not in (r.fieldnames or [])]
        if missing:raise ValueError(f'Columns missing: {missing}')
        for i,row in enumerate(r,2):
            t=datetime.fromisoformat(row['as_of_local'])
            if not start<=t.date()<=end:raise ValueError(f'Unexpected date at row {i}: {t}')
            if last is not None and t<=last:raise ValueError(f'Non-increasing timestamp at row {i}')
            if datetime.fromisoformat(row['window_start_exclusive'])!=t+timedelta(minutes=10) or datetime.fromisoformat(row['window_end_inclusive'])!=t+timedelta(minutes=40):raise ValueError(f'Incorrect forecast window at row {i}')
            xx=[float(row[k]) for k in FEATURES]; yy=float(row['target_pickups'])
            if not np.isfinite(xx).all() or not np.isfinite(yy) or yy<0:raise ValueError(f'Invalid value at row {i}')
            if any(v not in (0.,1.) for v in xx[5:]) or sum(xx[5:])!=1:raise ValueError(f'Invalid weekday flags at row {i}')
            x.append(xx); y.append(yy); times.append(t); last=t
            if baseline:
                if not row['history_three_week']:raise ValueError(f'Missing baseline at row {i}')
                refs.append(float(row['history_three_week']))
    if not x: raise ValueError('No samples')
    return np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64),times,np.asarray(refs,dtype=np.float64)

def final_counts(x):
    return np.maximum(0,np.rint(x)).astype(np.int64)

def metrics(y,p):
    e=np.asarray(p,dtype=np.float64)-y
    return {'mae_trips':float(np.mean(np.abs(e))),'rmse_trips':float(np.sqrt(np.mean(e**2)))}
