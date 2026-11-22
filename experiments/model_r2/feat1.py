import re,numpy as np,pandas as pd,joblib,scipy.sparse as sp,xgboost as xgb,time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
B=r'<WORKDIR>/handoff/handoff_full/'
d=pd.read_parquet(B+'bigquery_tables/ml_dataset_final_last_row_per_stay.parquet').sort_values('icustay_id').reset_index(drop=True)
n=pd.read_parquet(B+'bigquery_tables/clinical_notes.parquet')
d=d.merge(n[['hadm_id','text']],on='hadm_id',how='left'); d['text']=d.text.fillna('')
y=d.sofa_score.values.astype(float)
num=[c for c in d.columns if c not in('subject_id','hadm_id','icustay_id','window_time','sofa_score','rn','text')]
Xn=d[num].fillna(0).values
tr,te=next(GroupShuffleSplit(1,test_size=.2,random_state=42).split(d,groups=d.subject_id))
def clean(t):
    t=re.sub(r'\[\*\*[^\]]*\*\*\]',' ',t); t=re.sub(r'\b\d+(?:[./]\d+)?\b',' ',t); t=t.lower()
    return ' '.join(re.sub(r'[^a-z\s]',' ',t).split())
t0=time.time(); ct=d.text.apply(clean); print('clean',time.time()-t0)
def fit(X,name,**kw):
    m=xgb.XGBRegressor(n_estimators=800,max_depth=5,learning_rate=.03,subsample=.7,colsample_bytree=.5,min_child_weight=5,reg_lambda=5,n_jobs=8,**kw)
    m.fit(X[tr],y[tr]); p=np.clip(m.predict(X[te]),0,24); print(name,'R2',round(r2_score(y[te],p),4),flush=True)
fit(Xn,'numeric only')
voc=joblib.load(B+'models/tfidf_vectorizer.pkl').get_feature_names_out()
v1=TfidfVectorizer(vocabulary=list(voc),ngram_range=(1,2),token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b',sublinear_tf=True)
A=v1.fit_transform(ct.str[:10000]).toarray(); fit(np.hstack([Xn,A]),'baseline: num+90 whitelist')
v2=TfidfVectorizer(max_features=3000,min_df=20,max_df=.9,ngram_range=(1,2),token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b',sublinear_tf=True,dtype=np.float32)
Bm=v2.fit_transform(ct.str[:30000]); print('tfidf3000',Bm.shape,time.time()-t0)
fit(sp.hstack([sp.csr_matrix(Xn),Bm]).tocsr(),'num+3000 tfidf')
ln=np.log1p(d.text.str.len().values)[:,None]
fit(sp.hstack([sp.csr_matrix(np.hstack([Xn,ln])),Bm]).tocsr(),'num+3000 tfidf+len')
sp.save_npz(r'<WORKDIR>/B.npz',Bm)
