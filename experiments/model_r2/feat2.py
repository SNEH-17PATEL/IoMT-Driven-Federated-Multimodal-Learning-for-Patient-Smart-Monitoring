import re,numpy as np,pandas as pd,scipy.sparse as sp,xgboost as xgb,time,pickle
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
S=r'<WORKDIR>/'
B=S+'handoff/handoff_full/'
d=pd.read_parquet(B+'bigquery_tables/ml_dataset_final_last_row_per_stay.parquet').sort_values('icustay_id').reset_index(drop=True)
n=pd.read_parquet(B+'bigquery_tables/clinical_notes.parquet')
d=d.merge(n[['hadm_id','text']],on='hadm_id',how='left'); d['text']=d.text.fillna('')
y=d.sofa_score.values.astype(float)
num=[c for c in d.columns if c not in('subject_id','hadm_id','icustay_id','window_time','sofa_score','rn','text')]
Xn=d[num].fillna(0).values
tr,te=next(GroupShuffleSplit(1,test_size=.2,random_state=42).split(d,groups=d.subject_id))
LABS={'creat':(r'creat(?:inine)?\w*',0.2,15),'bili':(r'(?:t(?:ot)?\s*bili\w*|bilirubin|bili)',0.1,50),
 'plt':(r'(?:plt|platelets?)',5,1500),'inr':(r'inr',0.5,10),'wbc':(r'wbc',0.5,100),'lactate':(r'lactate|lact',0.3,30),
 'hct':(r'hct|hematocrit',10,60),'fio2':(r'fio2',21,100),'pao2':(r'pao2|po2',20,600),'ph':(r'(?:art\s*)?ph',6.8,7.8),
 'bun':(r'bun|urean',2,200),'na':(r'sodium|na',110,170),'k':(r'potassium|k',2,8),'co2':(r'hco3?|co2',5,50)}
pats={k:(re.compile(r'\b(?:'+p+r')\b[^0-9\n]{0,12}?(\d+(?:\.\d+)?)',re.I),lo,hi) for k,(p,lo,hi) in LABS.items()}
t0=time.time(); rows=[]
for t in d.text.values:
    t=t[:60000]; r=[]
    for k,(rx,lo,hi) in pats.items():
        v=[float(x) for x in rx.findall(t)]; v=[x for x in v if lo<=x<=hi]
        r+= [np.mean(v),np.max(v),np.min(v),len(v)] if v else [np.nan,np.nan,np.nan,0]
    rows.append(r)
L=np.array(rows,dtype=np.float32); print('labs',L.shape,time.time()-t0,'coverage creat',(L[:,3]>0).mean().round(2),flush=True)
np.save(S+'labs.npy',L)
Bm=sp.load_npz(S+'B.npz')
def fit(X,name):
    m=xgb.XGBRegressor(n_estimators=800,max_depth=5,learning_rate=.03,subsample=.7,colsample_bytree=.5,min_child_weight=5,reg_lambda=5,n_jobs=8)
    m.fit(X[tr],y[tr]); p=np.clip(m.predict(X[te]),0,24); print(name,'R2',round(r2_score(y[te],p),4),flush=True)
fit(sp.hstack([sp.csr_matrix(np.hstack([Xn,L])),Bm]).tocsr(),'num+labs+3000 tfidf')
fit(sp.csr_matrix(np.hstack([Xn,L])),'num+labs only (no tfidf)')
