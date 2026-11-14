"""Train the 12→16→1 JFK PyTorch model on Jan–May, early-stop on June."""
import csv,json,random
from datetime import date
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from common import DATA,FEATURES,RESULTS,load,sha256

class PickupNet(nn.Module):
 def __init__(self):
  super().__init__();self.network=nn.Sequential(nn.Linear(12,16),nn.ReLU(),nn.Linear(16,1))
 def forward(self,x):return self.network(x)

def main():
 torch.set_num_threads(1);seed=42;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 train=DATA/'train.csv';val=DATA/'validation.csv'
 X,y,t,_=load(train,date(2025,1,1),date(2025,5,31));vx,vy,vt,_=load(val,date(2025,6,1),date(2025,6,30))
 if len(X)!=151*1041 or len(vx)!=30*1041:raise ValueError('Unexpected sample count')
 X=X.astype(np.float32);y=y.astype(np.float32);vx=vx.astype(np.float32);vy=vy.astype(np.float32)
 xm=X.mean(axis=0);xs=X.std(axis=0);xs[xs==0]=1.;ym=float(y.mean());ys=float(y.std()) or 1.
 xn=(X-xm)/xs;yn=(y-ym)/ys;vxn=(vx-xm)/xs;vyn=(vy-ym)/ys
 model=PickupNet();opt=torch.optim.Adam(model.parameters(),lr=.003);criterion=nn.MSELoss()
 ds=TensorDataset(torch.from_numpy(xn),torch.from_numpy(yn).unsqueeze(1))
 dl=DataLoader(ds,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
 tx=torch.from_numpy(vxn);ty=torch.from_numpy(vyn).unsqueeze(1)
 best=float('inf');best_state=None;best_epoch=0;patience=20;records=[]
 for epoch in range(1,251):
  model.train();total=0.
  for xb,yb in dl:
   opt.zero_grad();loss=criterion(model(xb),yb);loss.backward();opt.step();total+=loss.item()*len(xb)
  model.eval()
  with torch.no_grad():vloss=criterion(model(tx),ty).item()
  records.append((epoch,total/len(ds),vloss))
  if vloss<best-1e-6:best=vloss;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
  elif epoch-best_epoch>=patience:break
  if epoch==1 or epoch%10==0:print(f'epoch {epoch}: validation MSE={vloss:.5f}, best={best_epoch}',flush=True)
 if best_state is None:raise RuntimeError('No valid model checkpoint')
 RESULTS.mkdir(exist_ok=True)
 ck={'model_state':best_state,'features':FEATURES,'x_mean':xm.tolist(),'x_std':xs.tolist(),'y_mean':ym,'y_std':ys,'best_epoch':best_epoch,'epochs_run':len(records),'seed':seed,'optimizer':'Adam','learning_rate':.003,'batch_size':128,'patience':patience,'training_start':str(t[0]),'training_end':str(t[-1]),'train_rows':len(X),'selection_period':'2025-06-01 to 2025-06-30','validation_rows':len(vx),'train_sha256':sha256(train),'validation_sha256':sha256(val),'target':'JFK pickup count in (t+10,t+40] excluding next-day midnight','torch_version':str(torch.__version__)}
 torch.save(ck,RESULTS/'annual_model.pt')
 with (RESULTS/'training_history.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.writer(f);w.writerow(['epoch','train_mse_standardized','validation_mse_standardized']);w.writerows(records)
 info={k:v for k,v in ck.items() if k not in ('model_state','x_mean','x_std')};info['best_validation_mse_standardized']=best
 (RESULTS/'training_summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(info,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
