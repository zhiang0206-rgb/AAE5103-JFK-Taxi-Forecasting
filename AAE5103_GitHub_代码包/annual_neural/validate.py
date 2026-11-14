"""Score the saved MLP on June validation, without opening the test set."""
import csv,json,os
from datetime import date
import numpy as np
import torch
from train import PickupNet
from common import DATA,FEATURES,RESULTS,final_counts,load,metrics,sha256
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/aae5103-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 torch.set_num_threads(1)
 ck=torch.load(RESULTS/'annual_model.pt',map_location='cpu')  # Only load trusted local checkpoints.
 if ck['features']!=FEATURES:raise ValueError('Feature schema mismatch')
 path=DATA/'validation.csv';X,y,times,history=load(path,date(2025,6,1),date(2025,6,30),baseline=True)
 if len(y)!=30*1041 or sha256(path)!=ck['validation_sha256']:raise ValueError('Validation data does not match training selection')
 local=DATA/'other_validation.csv';other=local if local.exists() else DATA.parent.parent/'annual_tree/results/validation_predictions.csv'
 with other.open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
 if len(rows)!=len(times):raise ValueError('Comparator length mismatch')
 for r,t,a,b in zip(rows,times,y,history):
  if r['as_of_local']!=t.isoformat(sep=' ') or int(r['actual_pickups'])!=int(a) or int(r['history_three_week'])!=int(b):raise ValueError('Comparator mismatch at '+str(t))
 X=X.astype(np.float32);xm=np.asarray(ck['x_mean'],dtype=np.float32);xs=np.asarray(ck['x_std'],dtype=np.float32)
 model=PickupNet();model.load_state_dict(ck['model_state']);model.eval()
 with torch.no_grad():raw=model(torch.from_numpy((X-xm)/xs)).squeeze(1).numpy()*ck['y_std']+ck['y_mean']
 pred=final_counts(raw);hist=history.astype(np.int64)
 methods={'neural_network':pred,'decision_tree':np.array([int(r['tree_forecast']) for r in rows]),'linear_regression':np.array([int(r['linear_forecast']) for r in rows]),'time_series':np.array([int(r['time_series_forecast']) for r in rows]),'historical_average_three_week':hist}
 weekdays=np.array([t.weekday()<5 for t in times]); result={'period':'2025-06-01 to 2025-06-30','rows':len(y),'best_epoch':ck['best_epoch'],'train_sha256':ck['train_sha256'],'validation_sha256':ck['validation_sha256'],'validation':{k:metrics(y,p) for k,p in methods.items()},'by_day_type':{},'note':'June was used for early stopping and other models selection; it is development evidence, not an independent final test. Models use different historical inputs; overlapping windows are dependent.'}
 for name,mask in [('workday',weekdays),('weekend',~weekdays)]:result['by_day_type'][name]={'samples':int(mask.sum()),**{k:metrics(y[mask],p[mask]) for k,p in methods.items()}}
 (RESULTS/'validation_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 with (RESULTS/'validation_predictions.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.writer(f);w.writerow(['as_of_local','actual_pickups','neural_raw','neural_forecast','tree_forecast','linear_forecast','time_series_forecast','history_three_week'])
  for i,t in enumerate(times):w.writerow([t.isoformat(sep=' '),int(y[i]),f'{raw[i]:.8f}',int(pred[i]),int(methods['decision_tree'][i]),int(methods['linear_regression'][i]),int(methods['time_series'][i]),int(hist[i])])
 with (RESULTS/'training_history.csv').open(newline='',encoding='utf8') as f:history_rows=list(csv.DictReader(f))
 epochs=[int(r['epoch']) for r in history_rows];tl=[float(r['train_mse_standardized']) for r in history_rows];vl=[float(r['validation_mse_standardized']) for r in history_rows]
 fig,ax=plt.subplots(figsize=(8,4));ax.plot(epochs,tl,label='Training');ax.plot(epochs,vl,label='Validation');ax.axvline(ck['best_epoch'],color='gray',linestyle='--',label=f"Best epoch {ck['best_epoch']}");ax.set(title='JFK neural network: learning curves',xlabel='Epoch',ylabel='MSE (standardized target)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(RESULTS/'training_loss.png',dpi=170);plt.close(fig)
 ix=[i for i,t in enumerate(times) if t.date()==date(2025,6,10) and t.minute%10==0];x=[times[i] for i in ix]
 fig,ax=plt.subplots(figsize=(12,4));ax.plot(x,y[ix],label='Recorded pickups',color='#243e60');ax.plot(x,pred[ix],label='Neural network',color='#168aaf');ax.plot(x,hist[ix],label='Three-week historical mean',color='#e39835',alpha=.85);ax.set(title='JFK neural network: 10 June 2025 validation',xlabel='Prediction time (New York local)',ylabel='Pickups in (t+10, t+40]');ax.grid(alpha=.25);ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(RESULTS/'validation_comparison.png',dpi=170);plt.close(fig)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
