import sys,re,time; sys.path.insert(0,'.')
import numpy as np,pandas as pd,scipy.sparse as sp,torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score
from model_utils import ICUModel,train_model
S=r'<WORKDIR>/'
B=S+'handoff/handoff_full/bigquery_tables/'
d=pd.read_parquet(B+'ml_dataset_final_last_row_per_stay.parquet').sort_values('icustay_id').reset_index(drop=True)
n=pd.read_parquet(B+'clinical_notes.parquet'); d=d.merge(n[['hadm_id','text']],on='hadm_id',how='left'); d['text']=d.text.fillna('')
y=d.sofa_score.values.astype(np.float32)
num=[c for c in d.columns if c not in('subject_id','hadm_id','icustay_id','window_time','sofa_score','rn','text')]
Xn=d[num].fillna(0).values
tr,te=next(GroupShuffleSplit(1,test_size=.2,random_state=42).split(d,groups=d.subject_id))
sc=StandardScaler().fit(Xn[tr]); Xs=np.clip(sc.transform(Xn),-10,10).astype(np.float32)
def run(X,name,dropout=0.3,epochs=20,lr=1e-3,hidden=(256,128,64)):
    torch.manual_seed(0); m=ICUModel(X.shape[1],hidden=hidden,dropout=dropout); r=[]
    for ep in range(epochs//5):
        train_model(m,X[tr],y[tr],epochs=5,lr=lr*0.85**ep,batch_size=128)
        with torch.no_grad(): p=np.clip(m(torch.tensor(X[te])).numpy().ravel(),0,24)
        r.append(round(r2_score(y[te],p),4))
    print(name,r,flush=True)
B3=sp.load_npz(S+'B.npz')
def zs(Bm,k):
    svd=TruncatedSVD(k,random_state=0).fit(Bm[tr]); Z=svd.transform(Bm); return (Z/Z[tr].std(0)).astype(np.float32)
for k in (64,256):
    Z=zs(B3,k)
    for dr in (0.3,0.5): run(np.hstack([Xs,Z]),f'3k svd{k} dr{dr}',dropout=dr)
def clean(t):
    t=re.sub(r'\[\*\*[^\]]*\*\*\]',' ',t[:30000]); t=re.sub(r'\b\d+(?:[./]\d+)?\b',' ',t); t=t.lower()
    return ' '.join(re.sub(r'[^a-z\s]',' ',t).split())
t0=time.time(); ct=d.text.apply(clean)
v=TfidfVectorizer(max_features=10000,min_df=10,max_df=.9,ngram_range=(1,2),token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b',sublinear_tf=True,dtype=np.float32)
B10=v.fit_transform(ct); sp.save_npz(S+'B10.npz',B10); print('tfidf10k',B10.shape,time.time()-t0,flush=True)
for k in (128,256):
    Z=zs(B10,k)
    for dr in (0.3,0.5): run(np.hstack([Xs,Z]),f'10k svd{k} dr{dr}',dropout=dr)
