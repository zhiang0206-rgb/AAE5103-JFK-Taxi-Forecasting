"""Repeat the original JFK MLP with fixed settings; train/validate only."""
import csv, hashlib, json, random, sys
from datetime import date
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
MODEL_DIR=ROOT/'annual_neural'
sys.path.insert(0,str(MODEL_DIR))
from common import DATA, FEATURES, load, sha256
from train import PickupNet

def main():
    torch.set_num_threads(1)
    path_train=DATA/'train.csv';path_val=DATA/'validation.csv'
    X,y,_,_=load(path_train,date(2025,1,1),date(2025,5,31))
    V,v,_,_=load(path_val,date(2025,6,1),date(2025,6,30))
    if len(y)!=157191 or len(v)!=31230:raise ValueError('Unexpected train or validation rows')
    X,y,V,v=[a.astype(np.float32) for a in (X,y,V,v)]
    mean=X.mean(0);std=X.std(0);std[std==0]=1
    ym=float(y.mean());ys=float(y.std()) or 1.
    xn=(X-mean)/std;yn=(y-ym)/ys;vx=torch.from_numpy((V-mean)/std);vy=torch.from_numpy((v-ym)/ys).unsqueeze(1)
    ds=TensorDataset(torch.from_numpy(xn),torch.from_numpy(yn).unsqueeze(1));tx=torch.from_numpy(xn);ty=torch.from_numpy(yn).unsqueeze(1)
    records=[];summaries=[];criterion=nn.MSELoss()
    for seed in (42,7,123):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=PickupNet();opt=torch.optim.Adam(model.parameters(),lr=.003)
        loader=DataLoader(ds,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
        best=float('inf');best_epoch=0;best_state=None
        for epoch in range(1,81):
            model.train();total=0.
            for xb,yb in loader:
                opt.zero_grad();loss=criterion(model(xb),yb);loss.backward();opt.step();total+=loss.item()*len(xb)
            model.eval()
            with torch.no_grad():
                validation=float(criterion(model(vx),vy).item())
                training_final=float(criterion(model(tx),ty).item())
            records.append({'seed':seed,'epoch':epoch,'train_online_mse':total/len(ds),'train_end_epoch_mse':training_final,'validation_mse':validation})
            if validation<best-1e-6:
                best=validation;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            if epoch in (1,6) or epoch%10==0:print(f'seed={seed} epoch={epoch} validation={validation:.6f} best_epoch={best_epoch}',flush=True)
        ck={'model_state':best_state,'features':FEATURES,'x_mean':mean.tolist(),'x_std':std.tolist(),'y_mean':ym,'y_std':ys,'best_epoch':best_epoch,'epochs_run':80,'seed':seed,'learning_rate':.003,'batch_size':128,'train_sha256':sha256(path_train),'validation_sha256':sha256(path_val),'target':'JFK pickup count in (t+10,t+40] excluding next-day midnight','torch_version':str(torch.__version__)}
        torch.save(ck,OUT/f'seed_{seed}_best.pt')
        summaries.append({'seed':seed,'best_epoch':best_epoch,'best_validation_mse':best})
        with (OUT/'training_history.csv').open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
        print('Finished seed',seed,'best epoch',best_epoch,'MSE',best,flush=True)
    metadata={'seeds':[42,7,123],'epochs_per_seed':80,'same_original_architecture':'12 → 16 → 1 ReLU','optimiser':'Adam','learning_rate':.003,'batch_size':128,'early_stopping':'disabled for diagnostics','train_sha256':sha256(path_train),'validation_sha256':sha256(path_val),'train_rows':len(y),'validation_rows':len(v),'y_std':ys,'runs':summaries,'notes':['Only January–May train and June validation read.','End-of-epoch training and validation losses both computed after model updates.','Original model and held-out test set untouched.','Selecting seeds or epochs using June may overfit the validation month.']}
    (OUT/'experiment_summary.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print(json.dumps(metadata,indent=2),flush=True)
if __name__=='__main__':main()
