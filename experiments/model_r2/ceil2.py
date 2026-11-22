import sys; sys.path.insert(0,'.')
import numpy as np, tune_training as t, xgboost as xgb
from sklearn.metrics import r2_score
H=r'<WORKDIR>/handoff/handoff_full'
c,te=t.load_clients(False,H)
X=np.vstack([a for a,_ in c]);y=np.concatenate([b for _,b in c])
print(X.shape,y.mean(),y.std(),(y>=10).mean(),te[1].mean())
for d,n in [(4,400),(6,600)]:
    m=xgb.XGBRegressor(n_estimators=n,max_depth=d,learning_rate=.03,subsample=.8,colsample_bytree=.7,n_jobs=8).fit(X,y)
    print('xgb',d,n,r2_score(te[1],np.clip(m.predict(te[0]),0,24)))
