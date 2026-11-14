"""Compare diagnostic runs on June only and create an English learning plot."""
import csv,json,sys,os
from datetime import date
from pathlib import Path
import numpy as np, torch
OUT=Path(__file__).resolve().parent; ROOT=OUT.parent;MODEL_DIR=ROOT/'annual_neural';sys.path.insert(0,str(MODEL_DIR))
from common import DATA,load,final_counts,metrics
from train import PickupNet
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/aae5103-matplotlib')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 torch.set_num_threads(1);X,y,t,_=load(DATA/'validation.csv',date(2025,6,1),date(2025,6,30),baseline=True);X=X.astype(np.float32)
 original=torch.load(MODEL_DIR/'results/annual_model.pt',map_location='cpu');rows=list(csv.DictReader((OUT/'training_history.csv').open()));result=json.loads((OUT/'experiment_summary.json').read_text())
 fig,axes=plt.subplots(1,3,figsize=(16,4.5),sharey=True)
 for ax,seed,run in zip(axes,(42,7,123),result['runs']):
  s=[r for r in rows if int(r['seed'])==seed];assert len(s)==80
  epochs=[int(r['epoch']) for r in s];train=[float(r['train_end_epoch_mse']) for r in s];val=[float(r['validation_mse']) for r in s]
  ck=torch.load(OUT/f'seed_{seed}_best.pt',map_location='cpu');net=PickupNet();net.load_state_dict(ck['model_state']);net.eval()
  with torch.no_grad():raw=net(torch.from_numpy((X-np.array(ck['x_mean'],np.float32))/np.array(ck['x_std'],np.float32))).squeeze(1).numpy()*ck['y_std']+ck['y_mean']
  prediction=final_counts(raw);run['validation_scores']=metrics(y,prediction);run['validation_mae_by_week']={}
  for label,lo,hi in [('Jun 1-7',1,7),('Jun 8-14',8,14),('Jun 15-21',15,21),('Jun 22-30',22,30)]:
   mask=np.array([lo<=dt.day<=hi for dt in t]);run['validation_mae_by_week'][label]=metrics(y[mask],prediction[mask])['mae_trips']
  ax.plot(epochs,train,label='Training MSE',color='#2b608a');ax.plot(epochs,val,label='Validation MSE',color='#e88b34');ax.axvline(run['best_epoch'],color='#777',ls='--',label='Lowest validation');ax.set(title=f"Seed {seed}: best epoch {run['best_epoch']}",xlabel='Epoch');ax.grid(alpha=.25)
  if seed==42:
   result['seed_42_matches_original_state']=all(torch.equal(original['model_state'][k],ck['model_state'][k]) for k in original['model_state'])
   if not result['seed_42_matches_original_state']:raise ValueError('Original seed 42 weights did not reproduce')
 axes[0].set_ylabel('MSE (standardised target)');axes[0].legend(fontsize=9);fig.suptitle('JFK neural network: 80-epoch training and June validation',fontsize=15);fig.tight_layout();fig.savefig(OUT/'training_validation_80_epochs.png',dpi=180);plt.close(fig)
 (OUT/'experiment_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(result['runs'],indent=2),flush=True);print('Seed 42 matches original:',result['seed_42_matches_original_state'])
if __name__=='__main__':main()
