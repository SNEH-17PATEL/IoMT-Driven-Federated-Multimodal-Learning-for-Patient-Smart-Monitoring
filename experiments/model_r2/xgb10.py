import numpy as np,pandas as pd,scipy.sparse as sp,xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
S=r'<WORKDIR>/'
d=pd.read_parquet(S+'handoff/handoff_full/bigquery_tables/ml_dataset_final_last_row_per_stay.parquet').sort_values('icustay_id').reset_index(drop=True)
y=d.sofa_score.values.astype(float)
num=[c for c in d.columns if c not in('subject_id','hadm_id','icustay_id','window_time','sofa_score','rn')]
Xn=d[num].fillna(0).values
tr,te=next(GroupShuffleSplit(1,test_size=.2,random_state=42).split(d,groups=d.subject_id))
X=sp.hstack([sp.csr_matrix(Xn),sp.load_npz(S+'B10.npz')]).tocsr()
m=xgb.XGBRegressor(n_estimators=1500,max_depth=6,learning_rate=.02,subsample=.7,colsample_bytree=.3,min_child_weight=5,reg_lambda=5,n_jobs=8).fit(X[tr],y[tr])
print('xgb num+10k tfidf R2',round(r2_score(y[te],np.clip(m.predict(X[te]),0,24)),4))
