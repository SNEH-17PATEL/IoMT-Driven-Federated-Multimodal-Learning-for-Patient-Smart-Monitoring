import sys; sys.path.insert(0,'.')
import numpy as np,pandas as pd,scipy.sparse as sp,torch,time
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from model_utils import ICUModel,train_model
S=r'<WORKDIR>/'
d=pd.read_parquet(S+'handoff/handoff_full/bigquery_tables/ml_dataset_final_last_row_per_stay.parquet').sort_values('icustay_id').reset_index(drop=True)
y=d.sofa_score.values.astype(np.float32)
num=[c for c in d.columns if c not in('subject_id','hadm_id','icustay_id','window_time','sofa_score','rn')]
Xn=d[num].fillna(0).values
tr,te=next(GroupShuffleSplit(1,test_size=.2,random_state=42).split(d,groups=d.subject_id))
Bm=sp.load_npz(S+'B.npz').toarray().astype(np.float32)
def run(X,name,hidden=(256,128,64),dropout=0.3,epochs=40,lr=1e-3):
    torch.manual_seed(0); m=ICUModel(X.shape[1],hidden=hidden,dropout=dropout); best=-9
    for ep in range(epochs//5):
        train_model(m,X[tr],y[tr],epochs=5,lr=lr*0.85**ep,batch_size=128)
        with torch.no_grad(): p=np.clip(m(torch.tensor(X[te])).numpy().ravel(),0,24)
        best=max(best,r2_score(y[te],p)); print(name,(ep+1)*5,round(r2_score(y[te],p),4),flush=True)
sc=StandardScaler().fit(Xn[tr]); Xs=np.clip(sc.transform(Xn),-10,10).astype(np.float32)
run(np.hstack([Xs,Bm*10]).astype(np.float32),'raw tfidf x10',epochs=30)
sc2=StandardScaler().fit(Bm[tr]); Bs=np.clip(sc2.transform(Bm),-10,10).astype(np.float32)
run(np.hstack([Xs,Bs]),'standardized tfidf',epochs=30)
