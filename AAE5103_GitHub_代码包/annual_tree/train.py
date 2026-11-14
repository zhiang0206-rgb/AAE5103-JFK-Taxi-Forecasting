"""Fit decision trees on Jan–May and choose depth on June validation, no test."""
import json
from datetime import date
import joblib
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor, export_text
from common import DATA, FEATURES, RESULTS, final_counts, load, metrics, sha256

def main():
 train=DATA/'train.csv';validation=DATA/'validation.csv'
 X,y,t,_=load(train,date(2025,1,1),date(2025,5,31))
 vx,vy,vt,_=load(validation,date(2025,6,1),date(2025,6,30))
 if len(X)!=151*1041 or len(vx)!=30*1041:raise ValueError('Unexpected sample count')
 candidates=[]
 for depth in (3,4,6,8):
  model=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=100,random_state=42)
  model.fit(X,y)
  p=final_counts(model.predict(vx))
  candidates.append({'depth':depth,'validation':metrics(vy,p),'leaves':int(model.get_n_leaves()),'estimator':model})
  print(f'depth={depth} validation_MAE={candidates[-1]["validation"]["mae_trips"]:.4f}',flush=True)
 candidates.sort(key=lambda x:(x['validation']['mae_trips'],x['depth']))
 winner=candidates[0];model=winner['estimator'];RESULTS.mkdir(parents=True,exist_ok=True)
 artifact={'estimator':model,'features':FEATURES,'training_start':str(t[0]),'training_end':str(t[-1]),'training_rows':len(X),'train_sha256':sha256(train),'selection_period':'2025-06-01 to 2025-06-30','selection_rows':len(vx),'validation_sha256':sha256(validation),'selected_max_depth':winner['depth'],'min_samples_leaf':100,'sklearn_version':sklearn.__version__,'target':'JFK pickup count in (t+10, t+40] minutes, excluding next-day midnight'}
 joblib.dump(artifact,RESULTS/'tree_model.joblib')
 report={'selection_rule':'Lowest June MAE after non-negative clipping and nearest integer rounding; ties choose shallower tree','candidates':[{k:v for k,v in c.items() if k!='estimator'} for c in sorted(candidates,key=lambda x:x['depth'])],'selected_max_depth':winner['depth'],'train_rows':len(X),'validation_rows':len(vx),'train_sha256':artifact['train_sha256'],'validation_sha256':artifact['validation_sha256'],'notes':['All candidates fitted on Jan–May only.','June used for model selection; July–December was not used.']}
 (RESULTS/'model_selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 (RESULTS/'tree_rules.txt').write_text(export_text(model,feature_names=FEATURES,decimals=6,max_depth=winner['depth']),encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
