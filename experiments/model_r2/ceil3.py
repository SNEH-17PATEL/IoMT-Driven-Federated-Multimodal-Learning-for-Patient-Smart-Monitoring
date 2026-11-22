import sys; sys.path.insert(0,'.')
import numpy as np, tune_training as t, xgboost as xgb, torch
from sklearn.metrics import r2_score
H=r'<WORKDIR>/handoff/handoff_full'
c,(Xt,yt)=t.load_clients(False,H)
X=np.vstack([a for a,_ in c]);y=np.concatenate([b for _,b in c])
r=lambda p:round(r2_score(yt,np.clip(p,0,24)),4)
m=xgb.XGBRegressor(n_estimators=1500,max_depth=5,learning_rate=.02,subsample=.7,colsample_bytree=.5,min_child_weight=5,reg_lambda=5,n_jobs=8).fit(X,y)
print('xgb tuned',r(m.predict(Xt)),'train',round(r2_score(y,m.predict(X)),3))
m2=xgb.XGBRegressor(n_estimators=1500,max_depth=5,learning_rate=.02,subsample=.7,colsample_bytree=.5,min_child_weight=5,reg_lambda=5,n_jobs=8).fit(X,np.sqrt(y))
print('xgb sqrt-target',r(m2.predict(Xt)**2))
# centralized MLP
torch.manual_seed(0)
from model_utils import ICUModel,train_model
mm=ICUModel(108)
for ep in range(6):
    train_model(mm,X,y,epochs=5,lr=1e-3*0.9**ep,batch_size=128)
    mm.eval()
    with torch.no_grad(): p=mm(torch.tensor(Xt)).numpy().ravel()
    print('mlp central ep',(ep+1)*5,r(p))
