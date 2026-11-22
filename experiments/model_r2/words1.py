import sys,re,time,pickle; sys.path.insert(0,'.')
import numpy as np,pandas as pd,scipy.sparse as sp,joblib,xgboost as xgb,torch,torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score
S=r'<WORKDIR>/'
B=S+'handoff/handoff_full/'
d=pd.read_parquet(B+'bigquery_tables/ml_dataset_final_last_row_per_stay.parquet').sort_values('icustay_id').reset_index(drop=True)
n=pd.read_parquet(B+'bigquery_tables/clinical_notes.parquet'); d=d.merge(n[['hadm_id','text']],on='hadm_id',how='left'); d['text']=d.text.fillna('')
y=d.sofa_score.values.astype(np.float32)
num=[c for c in d.columns if c not in('subject_id','hadm_id','icustay_id','window_time','sofa_score','rn','text')]
Xn=d[num].fillna(0).values
tr,te=next(GroupShuffleSplit(1,test_size=.2,random_state=42).split(d,groups=d.subject_id))
def clean(t):
    t=re.sub(r'\[\*\*[^\]]*\*\*\]',' ',t[:30000]); t=re.sub(r'\b\d+(?:[./]\d+)?\b',' ',t); t=t.lower()
    return ' '.join(re.sub(r'[^a-z\s]',' ',t).split())
t0=time.time(); ct=d.text.apply(clean); print('clean',round(time.time()-t0),flush=True)
tp=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b'
wl=list(joblib.load(B+'models/tfidf_vectorizer.pkl').get_feature_names_out())
W=TfidfVectorizer(vocabulary=wl,ngram_range=(1,2),token_pattern=tp,sublinear_tf=True,dtype=np.float32).fit_transform(ct.str[:10000]).toarray()
v=TfidfVectorizer(max_features=3000,min_df=20,max_df=.9,ngram_range=(1,2),token_pattern=tp,sublinear_tf=True,dtype=np.float32)
v.fit(ct.iloc[tr]); T=v.transform(ct); names=np.array(v.get_feature_names_out()); print('tfidf',T.shape,round(time.time()-t0),flush=True)
sp.save_npz(S+'T3k.npz',T); pickle.dump((names,wl,tr,te),open(S+'T3k_meta.pkl','wb'))
# importance from XGB on train rows only
X=sp.hstack([sp.csr_matrix(Xn),T]).tocsr()
m=xgb.XGBRegressor(n_estimators=400,max_depth=5,learning_rate=.05,subsample=.7,colsample_bytree=.5,min_child_weight=5,reg_lambda=5,n_jobs=8,importance_type='gain').fit(X[tr],y[tr])
imp=m.feature_importances_[len(num):]; order=np.argsort(-imp); np.save(S+'imp_order.npy',order); print('xgb importance done',round(time.time()-t0),flush=True)
sc=StandardScaler().fit(Xn[tr]); Xs=np.clip(sc.transform(Xn),-10,10).astype(np.float32)
class Net(nn.Module):
    def __init__(s,i,hidden=(256,128,64),dr=.3,indrop=0.):
        super().__init__(); L=[nn.Dropout(indrop)] if indrop>0 else []; p=i
        for h in hidden: L+=[nn.Linear(p,h),nn.ReLU(),nn.Dropout(dr)]; p=h
        s.net=nn.Sequential(*L,nn.Linear(p,1))
    def forward(s,x): return s.net(x)
def run(F,name,wd=1e-4,dr=.3,indrop=0.,epochs=20,lr=1e-3):
    torch.manual_seed(0); m=Net(F.shape[1],dr=dr,indrop=indrop); o=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    Xt=torch.tensor(F[tr]);yt=torch.tensor(y[tr]).view(-1,1);Xe=torch.tensor(F[te]);res=[]
    for ep in range(epochs):
        m.train(); perm=torch.randperm(len(Xt))
        for g in o.param_groups: g['lr']=lr*(0.9**(ep//5))
        for i in range(0,len(perm),128):
            b=perm[i:i+128]; o.zero_grad(); l=((m(Xt[b])-yt[b])**2).mean(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); o.step()
        if (ep+1)%5==0:
            m.eval()
            with torch.no_grad(): p=np.clip(m(Xe).numpy().ravel(),0,24)
            res.append(round(r2_score(y[te],p),4))
    print(name,res,flush=True)
F0=np.hstack([Xs,W*10]).astype(np.float32); run(F0,'whitelist90 (current)')
run(F0,'whitelist90 wd1e-2 indrop.2',wd=1e-2,indrop=.2)
for N in (200,500):
    idx=order[:N]; Fs=np.hstack([Xs,T[:,idx].toarray()*10]).astype(np.float32)
    run(Fs,f'top{N} xgb-chosen terms'); run(Fs,f'top{N} wd1e-2 indrop.2',wd=1e-2,indrop=.2)
    Fu=np.hstack([Xs,W*10,T[:,idx].toarray()*10]).astype(np.float32); run(Fu,f'whitelist+top{N} wd1e-2 indrop.2',wd=1e-2,indrop=.2)
