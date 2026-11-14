"""Evaluate chosen tree on the same June validation used for model selection."""
import csv,json,os
from datetime import date
import joblib
import numpy as np
from common import DATA, FEATURES, RESULTS, final_counts, load, metrics, sha256
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/aae5103-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def load_comparator(path,column,times,y,baseline):
 with path.open(newline='',encoding='utf-8') as f:
  rows=list(csv.DictReader(f))
 if len(rows)!=len(times):raise ValueError('Comparator row count mismatch: '+str(path))
 for i,(r,t,a,b) in enumerate(zip(rows,times,y,baseline)):
  basekey='history_three_week' if column=='time_series_forecast' else 'historical_average_three_week'
  if r['as_of_local']!=t.isoformat(sep=' ') or int(r['actual_pickups'])!=int(a) or int(r[basekey])!=int(b):raise ValueError('Comparator mismatch at '+str(t))
 return np.array([int(r[column]) for r in rows],dtype=np.int64)

def main():
 artifact=joblib.load(RESULTS/'tree_model.joblib')  # Only load trusted files.
 if artifact['features']!=FEATURES:raise ValueError('Unexpected feature schema')
 path=DATA/'validation.csv'; X,y,times,history=load(path,date(2025,6,1),date(2025,6,30),baseline=True)
 if len(y)!=30*1041 or sha256(path)!=artifact['validation_sha256']:raise ValueError('Validation data differs from model selection')
 # Bundles keep comparator predictions in data/; project uses the original paths.
 local=DATA/'linear_validation.csv'; serieslocal=DATA/'time_series_validation.csv'
 linear=local if local.exists() else DATA.parent.parent/'linear_regression/results/validation_predictions.csv'
 series=serieslocal if serieslocal.exists() else DATA.parent.parent/'time_series_forecast/results/validation_predictions.csv'
 lp=load_comparator(linear,'linear_forecast',times,y,history)
 sp=load_comparator(series,'time_series_forecast',times,y,history)
 estimator=artifact['estimator']; raw=estimator.predict(X);pred=final_counts(raw);hist=history.astype(np.int64)
 methods={'decision_tree':pred,'linear_regression':lp,'time_series':sp,'historical_average_three_week':hist}
 daytypes=np.array([t.weekday()<5 for t in times]); result={'period':'2025-06-01 to 2025-06-30','rows':len(y),'selected_max_depth':artifact['selected_max_depth'],'train_sha256':artifact['train_sha256'],'validation_sha256':artifact['validation_sha256'],'validation':{k:metrics(y,p) for k,p in methods.items()},'by_day_type':{},'note':'June was used for depth selection and is not an independent final test. Candidate methods use different historical information; overlapping windows are dependent.'}
 for name,mask in [('workday',daytypes),('weekend',~daytypes)]:result['by_day_type'][name]={'samples':int(mask.sum()),**{k:metrics(y[mask],p[mask]) for k,p in methods.items()}}
 (RESULTS/'validation_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 with (RESULTS/'validation_predictions.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['as_of_local','actual_pickups','tree_raw','tree_forecast','linear_forecast','time_series_forecast','history_three_week','tree_leaf'])
  for t,a,r,p,l,s,h,leaf in zip(times,y,raw,pred,lp,sp,hist,estimator.apply(X)):
   w.writerow([t.isoformat(sep=' '),int(a),f'{r:.8f}',int(p),int(l),int(s),int(h),int(leaf)])
 # Visualize only the first three levels to keep the trained tree readable.
 fig,ax=plt.subplots(figsize=(20,8));plot_tree(estimator,feature_names=FEATURES,max_depth=2,filled=True,rounded=True,fontsize=8,ax=ax)
 ax.set_title('Decision tree: first three levels (deeper branches omitted)',fontsize=16);fig.tight_layout();fig.savefig(RESULTS/'tree_top_levels.png',dpi=150);plt.close(fig)
 ix=[i for i,t in enumerate(times) if t.date()==date(2025,6,10) and t.minute%10==0]
 fig,ax=plt.subplots(figsize=(12,4));tt=[times[i] for i in ix]
 ax.plot(tt,y[ix],label='Recorded pickups',color='#243e60');ax.plot(tt,pred[ix],label='Decision tree',color='#168aaf');ax.plot(tt,hist[ix],label='Three-week historical mean',color='#e39835',alpha=.85)
 ax.set(title='JFK decision tree: 10 June 2025 validation',xlabel='Prediction time (New York local)',ylabel='Pickups in (t+10, t+40]');ax.legend();ax.grid(alpha=.25);fig.autofmt_xdate();fig.tight_layout();fig.savefig(RESULTS/'validation_comparison.png',dpi=170);plt.close(fig)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
