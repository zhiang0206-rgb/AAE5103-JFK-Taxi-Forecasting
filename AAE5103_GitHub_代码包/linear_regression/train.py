"""Fit JFK linear regression using January–May only. Does not open test data."""
import json
from datetime import date
import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from common import DATA, FEATURES, SELECTED, RESULTS, load, sha256

def main():
    path=DATA/'train.csv'
    X,y,times,_=load(path,date(2025,1,1),date(2025,5,31))
    if len(X)!=151*1041:raise ValueError(f'Unexpected training sample count: {len(X)}')
    model=Pipeline([('independent_features',ColumnTransformer([('selected','passthrough',SELECTED)],remainder='drop')),('linear_regression',LinearRegression())])
    model.fit(X,y)
    estimator=model.named_steps['linear_regression']
    RESULTS.mkdir(parents=True,exist_ok=True)
    artifact={'estimator':model,'features':FEATURES,'selected_features':[FEATURES[i] for i in SELECTED],'reference_weekday':'Sunday','training_start':str(times[0]),'training_end':str(times[-1]),'training_rows':len(X),'train_sha256':sha256(path),'sklearn_version':sklearn.__version__,'target':'JFK pickup count in (t+10, t+40] minutes, excluding next-day midnight'}
    joblib.dump(artifact,RESULTS/'linear_model.joblib')
    weights={'intercept':float(estimator.intercept_),'coefficients':{FEATURES[i]:float(c) for i,c in zip(SELECTED,estimator.coef_)},'weekday_sunday_reference':0.0,'effective_rank':int(estimator.rank_),'explanation':'Sunday is the reference category; one weekday flag is omitted because weekday flags sum to 1.'}
    (RESULTS/'coefficients.json').write_text(json.dumps(weights,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Trained {len(X)} samples; saved {RESULTS / "linear_model.joblib"}')
    print(json.dumps(weights,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
