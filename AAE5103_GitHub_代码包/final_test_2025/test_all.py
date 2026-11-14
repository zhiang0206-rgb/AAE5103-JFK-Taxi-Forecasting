"""Reproduce June forecasts and evaluate frozen models on Jul–Dec 2025.

This script NEVER fits a model. Rolling inputs and the historical baseline use
only observations that precede each prediction time. Output is written to the
five method folders and this directory; existing validation files stay intact.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'final_test_2025'
SAMPLES = ROOT / 'generalization_2025' / 'shared_samples'
MONTHLY = ROOT / 'generalization_2025' / 'monthly_jfk'
sys.path.insert(0, str(ROOT / 'annual_neural'))
from common import FEATURES, load, final_counts, metrics, sha256
from train import PickupNet

spec = importlib.util.spec_from_file_location('seasonal_ar_common', ROOT / 'time_series_forecast' / 'common.py')
ar_common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar_common)

METHODS = {
    'neural_network': ('annual_neural', 'neural_forecast', 'Neural network', '#a13370', '-'),
    'decision_tree': ('annual_tree', 'tree_forecast', 'Decision tree', '#66853a', '--'),
    'linear_regression': ('linear_regression', 'linear_forecast', 'Linear regression', '#a84e20', '-.'),
    'time_series': ('time_series_forecast', 'time_series_forecast', 'Seasonal 10-min AR(1)', '#128caf', '-'),
    'historical_average_three_week': ('historical_mean', 'history_three_week', 'Three-week historical mean', '#eda039', '--'),
}


def metric(y, p):
    result = metrics(y, p)
    result['bias_trips'] = float(np.mean(np.asarray(p, dtype=float) - y))
    return result


def forecasts(X, times, history, models):
    result = {}
    for name in ('linear_regression', 'decision_tree'):
        result[name] = final_counts(models[name]['estimator'].predict(X))
    ck = models['neural_network']
    x32 = X.astype(np.float32)
    xmean = np.asarray(ck['x_mean'], dtype=np.float32)
    xstd = np.asarray(ck['x_std'], dtype=np.float32)
    network = PickupNet()
    network.load_state_dict(ck['model_state'])
    network.eval()
    with torch.no_grad():
        result['neural_network'] = final_counts(network(torch.from_numpy((x32-xmean)/xstd)).squeeze(1).numpy() * ck['y_std'] + ck['y_mean'])
    result['time_series'] = np.fromiter((ar_common.final_count(ar_common.forecast_raw(models['time_series'], t, x)) for t, x in zip(times, X[:, 0])), dtype=np.int64, count=len(times))
    result['historical_average_three_week'] = history.astype(np.int64)
    return result


def load_models():
    paths = {
        'linear_regression': ROOT/'linear_regression/results/linear_model.joblib',
        'decision_tree': ROOT/'annual_tree/results/tree_model.joblib',
        'neural_network': ROOT/'annual_neural/results/annual_model.pt',
        'time_series': ROOT/'time_series_forecast/results/time_series_model.json',
    }
    torch.set_num_threads(1)
    models = {k: joblib.load(paths[k]) for k in ('linear_regression', 'decision_tree')}
    models['neural_network'] = torch.load(paths['neural_network'], map_location='cpu')
    models['time_series'] = json.loads(paths['time_series'].read_text(encoding='utf8'))
    train_sha = sha256(SAMPLES/'train.csv')
    validation_sha = sha256(SAMPLES/'validation.csv')
    for name in ('linear_regression', 'decision_tree', 'neural_network'):
        obj = models[name]
        if obj['features'] != FEATURES or obj['train_sha256'] != train_sha:
            raise ValueError(f'{name} model feature/training mismatch')
        if 'validation_sha256' in obj and obj['validation_sha256'] != validation_sha:
            raise ValueError(f'{name} selection data mismatch')
    if models['time_series']['train_sha256'] != sha256(ROOT/'time_series_forecast/data/train_minutes.csv'):
        raise ValueError('Time series training-data hash mismatch')
    return models, {name: sha256(path) for name, path in paths.items()}


def check_june(models):
    x, y, times, history = load(SAMPLES/'validation.csv', date(2025,6,1), date(2025,6,30), baseline=True)
    expected_path = ROOT/'annual_neural/results/validation_predictions.csv'
    with expected_path.open(newline='', encoding='utf8') as f:
        expected = list(csv.DictReader(f))
    if len(y) != 31230 or len(expected) != len(y):
        raise ValueError('Unexpected June validation length')
    calculated = forecasts(x, times, history, models)
    for name, (_, column, _, _, _) in METHODS.items():
        saved = np.fromiter((int(r[column]) for r in expected), dtype=np.int64, count=len(expected))
        diff = np.flatnonzero(calculated[name] != saved)
        if len(diff):
            i = int(diff[0])
            raise ValueError(f'June reproduction mismatch {name} {times[i]} {calculated[name][i]} vs {saved[i]}, {len(diff)} discrepancies')
    for t, actual, row in zip(times, y, expected):
        if t.isoformat(sep=' ') != row['as_of_local'] or int(actual) != int(row['actual_pickups']):
            raise ValueError('June timestamp/actual mismatch')
    print('June: reproduced all five saved forecasts on all 31,230 windows', flush=True)
    return y, times


def validate_raw_and_history(x, y, times, history):
    """Independently cross-check target, last 10 min and baseline for every row."""
    sources = json.loads((ROOT/'generalization_2025/collection_summary.json').read_text(encoding='utf8'))
    arrays = []
    for month in range(6,13):
        p = MONTHLY/f'jfk_yellow_2025-{month:02d}.parquet'
        if sha256(p) != sources[month-1]['filtered_sha256']:
            raise ValueError('Filtered source hash mismatch: '+p.name)
        arrays.append(pq.read_table(p, columns=['tpep_pickup_datetime'])['tpep_pickup_datetime'].to_numpy(zero_copy_only=False).astype('datetime64[us]'))
    events = np.sort(np.concatenate(arrays))
    ts = np.array(times, dtype='datetime64[us]')
    dayend = ts.astype('datetime64[D]').astype('datetime64[us]')+np.timedelta64(1,'D')
    left = np.searchsorted(events,ts+np.timedelta64(10,'m'),side='right')
    right = np.minimum(np.searchsorted(events,ts+np.timedelta64(40,'m'),side='right'),np.searchsorted(events,dayend,side='left'))
    actual = right-left
    if not np.array_equal(actual, y.astype(np.int64)):
        raise ValueError('Raw source does not match targets')
    for i, minutes in enumerate((10,30,60)):
        last = np.searchsorted(events,ts,side='right')-np.searchsorted(events,ts-np.timedelta64(minutes,'m'),side='right')
        if not np.array_equal(last,x[:,i].astype(np.int64)):
            raise ValueError(f'Raw source does not match past {minutes}-minute inputs')
    minutes = np.fromiter((t.hour*60+t.minute for t in times), dtype=np.int64, count=len(times))
    if not np.allclose(x[:,3],np.sin(2*np.pi*minutes/1440),atol=1e-12) or not np.allclose(x[:,4],np.cos(2*np.pi*minutes/1440),atol=1e-12):
        raise ValueError('Clock features mismatch')
    if not np.array_equal(np.argmax(x[:,5:12],axis=1),np.fromiter((t.weekday() for t in times),dtype=np.int64,count=len(times))):
        raise ValueError('Weekday features mismatch')
    previous=[]
    for weeks in (1,2,3):
        past=ts-np.timedelta64(7*weeks,'D')
        past_day_end=past.astype('datetime64[D]').astype('datetime64[us]')+np.timedelta64(1,'D')
        lo=np.searchsorted(events,past+np.timedelta64(10,'m'),side='right')
        hi=np.minimum(np.searchsorted(events,past+np.timedelta64(40,'m'),side='right'),np.searchsorted(events,past_day_end,side='left'))
        if not np.all(past+np.timedelta64(40,'m') < ts):raise ValueError('History not available at prediction time')
        previous.append(hi-lo)
    old=np.stack(previous,axis=1)
    if not np.array_equal(np.rint(np.mean(old,axis=1)).astype(np.int64),history.astype(np.int64)):
        raise ValueError('Historical baseline differs from raw source')
    print('Raw source: all targets, recent 10/30/60-min inputs and rolling three-week baselines checked',flush=True)
    return old


def write_per_method(name, pred, y, times, old, hashes, test_sha):
    folder=ROOT/METHODS[name][0]/'results'
    folder.mkdir(exist_ok=True)
    index=np.array([t.weekday()<5 for t in times])
    result={'method':name,'test_period':'2025-07-01 to 2025-12-31','test_rows':len(y),'test_data_sha256':test_sha,'frozen_model_sha256':hashes.get(name),'overall':metric(y,pred),'by_month':{},'by_day_type':{},'by_prediction_time':{},'forecast_protocol':'rolling origin; historic inputs must have occurred by as_of; no model refitting','target':'JFK recorded pickups in (t+10min,t+40min], excluding next-day midnight','rounding':'nonnegative integer, nearest with ties to even','note':'Overlapping windows are dependent; observed trips not unmet demand; past test observations may be used as inputs.'}
    for month in range(7,13):
        mask=np.fromiter((t.month==month for t in times),dtype=bool,count=len(times))
        result['by_month'][f'2025-{month:02d}']={'rows':int(mask.sum()),**metric(y[mask],pred[mask])}
    for title,mask in [('workday',index),('weekend',~index)]:
        result['by_day_type'][title]={'rows':int(mask.sum()),**metric(y[mask],pred[mask])}
    hours=np.fromiter((t.hour for t in times),dtype=np.int64,count=len(times))
    for title,mask in [('06:00-11:59',hours<12),('12:00-17:59',(hours>=12)&(hours<18)),('18:00-23:20',hours>=18)]:
        result['by_prediction_time'][title]={'rows':int(mask.sum()),**metric(y[mask],pred[mask])}
    (folder/'test_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    with (folder/'test_predictions.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.writer(f);header=['as_of_local','actual_pickups',METHODS[name][1]]
        if name=='historical_average_three_week':header+=['one_week_before','two_weeks_before','three_weeks_before']
        w.writerow(header)
        for i,t in enumerate(times):
            row=[t.isoformat(sep=' '),int(y[i]),int(pred[i])]
            if name=='historical_average_three_week':row+=old[i].tolist()
            w.writerow(row)
    return result


def create_figures(y,times,preds,reports):
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/aae5103-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    months=np.arange(7,13)
    fig,ax=plt.subplots(figsize=(12,5))
    for name,(_,_,label,color,style) in METHODS.items():
        ax.plot(months,[reports[name]['by_month'][f'2025-{m:02d}']['mae_trips'] for m in months],label=label,color=color,linestyle=style,marker='o',linewidth=1.8,markersize=4)
    ax.set_xticks(months,['Jul','Aug','Sep','Oct','Nov','Dec']);ax.set_ylim(bottom=0);ax.set(title='JFK forecasts: monthly test MAE (2025)',xlabel='Test month',ylabel='MAE per 30-minute window (trips)');ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=9);fig.tight_layout();fig.savefig(OUT/'monthly_test_mae.png',dpi=190);plt.close(fig)
    # Readable full-period view: daily means of each 30-minute-window forecast,
    # not daily trip totals. All minute-level windows are included.
    days=sorted({t.date() for t in times}); day_index=np.array([t.date().toordinal() for t in times]);ordinals=np.array([d.toordinal() for d in days]);counts=np.bincount(day_index-ordinals[0]);
    def daily(vals):return np.bincount(day_index-ordinals[0],weights=vals)/counts
    fig,ax=plt.subplots(figsize=(16,6));ax.plot(days,daily(y),label='Recorded pickups',color='#243e60',linewidth=2.3)
    for name,(_,_,label,color,style) in METHODS.items():ax.plot(days,daily(preds[name]),label=label,color=color,linestyle=style,linewidth=1.4)
    ax.set(title='JFK forecasts: daily mean per window, July–December 2025',xlabel='Prediction date (New York local)',ylabel='Mean pickups per 30-minute window (trips)');ax.set_xticks([date(2025,m,1) for m in range(7,13)]);ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'));ax.set_xlim(days[0],days[-1]);ax.grid(alpha=.25);ax.legend(ncol=3,fontsize=9);fig.tight_layout();fig.savefig(OUT/'daily_means_all_methods.png',dpi=190);plt.close(fig)
    for name,(folder,_,label,color,style) in METHODS.items():
        # One fixed test date, chosen before viewing the test results, matching the old demo style.
        ix=[i for i,t in enumerate(times) if t.date()==date(2025,7,10) and t.minute%10==0]
        fig,ax=plt.subplots(figsize=(12,4));xx=[times[i] for i in ix];ax.plot(xx,y[ix],label='Recorded pickups',color='#243e60');ax.plot(xx,preds[name][ix],label=label,color=color,linestyle=style)
        ax.set(title='JFK forecast: 10 July 2025 test',xlabel='Prediction time (New York local)',ylabel='Pickups in (t+10, t+40] (trips)');ax.legend();ax.grid(alpha=.25);fig.autofmt_xdate();fig.tight_layout();fig.savefig(ROOT/folder/'results/test_comparison.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['all']+list(METHODS),default='all')
    args=parser.parse_args()
    models,hashes=load_models()
    check_june(models)
    path=SAMPLES/'test.csv';x,y,times,history=load(path,date(2025,7,1),date(2025,12,31),baseline=True)
    if len(y)!=184*1041:raise ValueError('Unexpected test length')
    old=validate_raw_and_history(x,y,times,history)
    preds=forecasts(x,times,history,models)
    if any(np.any(p<0) for p in preds.values()):raise ValueError('Negative forecast')
    selected=list(METHODS) if args.method=='all' else [args.method]
    reports={name:write_per_method(name,preds[name],y,times,old,hashes,sha256(path)) for name in selected}
    if args.method=='all':
        with (OUT/'all_test_predictions.csv').open('w',encoding='utf8',newline='') as f:
            w=csv.writer(f);w.writerow(['as_of_local','actual_pickups']+[METHODS[n][1] for n in METHODS])
            for i,t in enumerate(times):w.writerow([t.isoformat(sep=' '),int(y[i])]+[int(preds[n][i]) for n in METHODS])
        counts=json.loads((ROOT/'generalization_2025/collection_summary.json').read_text(encoding='utf8'))
        monthly_raw={m['month']:m['jfk_in_month_rows'] for m in counts if m['month']>='2025-07'}
        result={'period':'2025-07-01 to 2025-12-31','window_count':len(y),'test_data_sha256':sha256(path),'raw_pickups_by_month':monthly_raw,'raw_pickups_total':sum(monthly_raw.values()),'mean_actual_per_window':float(np.mean(y)),'methods':reports,'model_hashes':hashes,'notes':['Models fixed before testing; no retraining.','Rolling historical baseline and recent inputs can use earlier observed test records.','Predictions estimate observed trips, not unmet demand; one-minute windows overlap.','Time-series forecast uses same past 10-minute counts as validated implementation.']}
        (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        create_figures(y,times,preds,reports)
    print(json.dumps({name:reports[name]['overall'] for name in selected},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
