"""
Protein Stability — Exp3: Smaller model (1M) + MSE loss + lower dropout.
Hypothesis: 2.5M model overfits on short 55-AA sequences. 1M model with
lower dropout may generalize better. Also try MSE loss (Huber might be
too forgiving on outliers).
Usage: modal run stability_run3.py
"""

import modal, os, time

app = modal.App("stability-v3")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scipy", "requests", "matplotlib", "lmdb")
vol = modal.Volume.from_name("autoresearch-bio-stability-data", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=600,
              volumes={"/root/.cache/autoresearch_bio": vol})
def train_stability():
    import gc, math, pickle, tarfile
    import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, requests
    from scipy.stats import spearmanr
    from torch.utils.data import TensorDataset, DataLoader

    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    AA_TO_IDX = {aa: i+2 for i, aa in enumerate(AA_LIST)}
    PAD=0; START=1; AA_TO_IDX["<PAD>"]=PAD; AA_TO_IDX["<START>"]=START
    VOCAB = len(AA_LIST)+2; TIME_BUDGET = 300
    CACHE = os.path.expanduser("~/.cache/autoresearch_bio/stability_data")
    URL = "https://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz"

    # ---- Exp3 Config: Smaller model (1M) ----
    N_EMBD=96; N_CNN=192; N_LAYERS=4; N_HEADS=8; N_HIDDEN=384
    DROPOUT=0.10; BATCH=256; EMB_NOISE=0.01  # less noise, larger batch
    LR=5e-4; WD=0.02  # higher LR for smaller model

    # Data
    def download():
        os.makedirs(CACHE, exist_ok=True)
        dd = os.path.join(CACHE, "stability")
        if os.path.exists(os.path.join(dd, "stability_train.lmdb")): return
        tp = os.path.join(CACHE, "stability.tar.gz")
        if not os.path.exists(tp):
            r = requests.get(URL, stream=True); r.raise_for_status()
            with open(tp, "wb") as f:
                for c in r.iter_content(8192): f.write(c)
        with tarfile.open(tp, "r:gz") as t: t.extractall(CACHE, filter="data")

    def load_lmdb(path):
        import lmdb
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        recs = []
        with env.begin(write=False) as txn:
            raw = txn.get(b"num_examples")
            if raw:
                try: n = int(raw.decode())
                except: n = pickle.loads(raw)
                for i in range(n):
                    r = txn.get(str(i).encode())
                    if r: recs.append(pickle.loads(r))
            else:
                for key, value in txn.cursor():
                    if key != b"num_examples":
                        try: recs.append(pickle.loads(value))
                        except: pass
        env.close(); return recs

    def build():
        download()
        dd = os.path.join(CACHE, "stability")
        tr = load_lmdb(os.path.join(dd, "stability_train.lmdb"))
        va = load_lmdb(os.path.join(dd, "stability_valid.lmdb"))
        tk = None
        for k in ["stability_score","stability","score","target"]:
            if k in tr[0]: tk=k; break
        if not tk:
            for k,v in tr[0].items():
                if k!="primary" and isinstance(v,(float,int)): tk=k; break
        print(f"Target: {tk}")
        def proc(recs):
            ss,tt,ml=[],[],0
            for r in recs:
                s=r.get("primary","")
                if isinstance(s,bytes):s=s.decode()
                t=r.get(tk)
                if t is None:continue
                if isinstance(t,(list,np.ndarray)):t=float(t[0])
                else:t=float(t)
                s="".join(c for c in s if c in AA_LIST)
                if len(s)<5:continue
                ml=max(ml,len(s));ss.append(s);tt.append(t)
            return ss,tt,ml
        str_,ttr_,ml1=proc(tr);sva_,tva_,ml2=proc(va)
        ML=min(max(ml1,ml2)+5,200)
        def enc(s):
            t=[START]
            for a in s[:ML]:t.append(AA_TO_IDX.get(a,PAD))
            while len(t)<ML+1:t.append(PAD)
            return torch.tensor(t,dtype=torch.long)
        Xt=torch.stack([enc(s) for s in str_]);Yt=torch.tensor(ttr_,dtype=torch.float32)
        Xv=torch.stack([enc(s) for s in sva_]);Yv=torch.tensor(tva_,dtype=torch.float32)
        print(f"Train:{len(Xt)},Val:{len(Xv)},ML:{ML},Range:[{Yt.min():.2f},{Yt.max():.2f}]")
        return TensorDataset(Xt,Yt),TensorDataset(Xv,Yv),ML

    # Model
    class CB(nn.Module):
        def __init__(self,ch,dp):
            super().__init__()
            self.n=nn.BatchNorm1d(ch);self.c1=nn.Conv1d(ch,ch,3,padding=1)
            self.c2=nn.Conv1d(ch,ch,3,padding=1);self.d=nn.Dropout(dp)
        def forward(self,x):return F.gelu(self.d(self.c2(F.gelu(self.c1(self.n(x)))))+x)

    class M(nn.Module):
        def __init__(self,ml):
            super().__init__()
            self.emb=nn.Embedding(VOCAB,N_EMBD);self.pos=nn.Embedding(ml+1,N_EMBD)
            self.proj=nn.Conv1d(N_EMBD,N_CNN,1)
            self.blks=nn.ModuleList([CB(N_CNN,DROPOUT) for _ in range(N_LAYERS)])
            self.cn=nn.BatchNorm1d(N_CNN)
            self.mha=nn.MultiheadAttention(N_EMBD,N_HEADS,dropout=DROPOUT,batch_first=True)
            self.q=nn.Parameter(torch.randn(1,1,N_EMBD)*0.02)
            cd=2*N_CNN+2*N_EMBD
            self.head=nn.Sequential(
                nn.LayerNorm(cd),nn.Linear(cd,N_HIDDEN),nn.GELU(),nn.Dropout(DROPOUT),
                nn.Linear(N_HIDDEN,N_HIDDEN//4),nn.GELU(),nn.Dropout(DROPOUT),
                nn.Linear(N_HIDDEN//4,1))
            for m in self.modules():
                if isinstance(m,(nn.Linear,nn.Conv1d)):
                    nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                    if m.bias is not None:nn.init.zeros_(m.bias)
                elif isinstance(m,nn.Embedding):nn.init.normal_(m.weight,std=0.02)
        def forward(self,x):
            B=x.size(0);s=x[:,1:]
            e=self.emb(s)+self.pos(torch.arange(s.size(1),device=x.device).unsqueeze(0))
            if self.training and EMB_NOISE>0:e=e+torch.randn_like(e)*EMB_NOISE
            c=self.proj(e.permute(0,2,1))
            for b in self.blks:c=b(c)
            c=self.cn(c);cm,cx=c.mean(2),c.max(2).values
            a,_=self.mha(self.q.expand(B,-1,-1),e,e)
            return self.head(torch.cat([cm,cx,a.squeeze(1),e.mean(1)],-1))
        def np(self):return sum(p.numel() for p in self.parameters() if p.requires_grad)

    dev=torch.device("cuda")
    tds,vds,ml=build();vol.commit()
    model=M(ml).to(dev)
    print(f"Params:{model.np():,}({model.np()/1e6:.2f}M)")

    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    crit=nn.MSELoss()  # MSE instead of Huber
    loader=DataLoader(tds,batch_size=BATCH,shuffle=True,drop_last=True)
    vloader=DataLoader(vds,batch_size=64,shuffle=False)
    scaler=torch.amp.GradScaler("cuda")

    best_sp=-1.0;best_st=None;top_ckpts=[];TOP_K=5

    def ev():
        model.eval();pa,la=[],[]
        with torch.no_grad():
            for xv,yv in vloader:
                with torch.amp.autocast("cuda"):
                    pa.extend(model(xv.to(dev)).squeeze(-1).cpu().numpy().tolist())
                la.extend(yv.numpy().tolist())
        return spearmanr(pa,la)[0],float(np.mean((np.array(pa)-np.array(la))**2)),pa,la

    tt=0.0;step=0;ep=0;sl=0.0;model.train()
    while True:
        ep+=1
        for xb,yb in loader:
            t0=time.time();xb,yb=xb.to(dev),yb.to(dev)
            with torch.amp.autocast("cuda"):
                p=model(xb).squeeze(-1);loss=crit(p,yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt);scaler.update();torch.cuda.synchronize()
            dt=time.time()-t0
            if step>5:tt+=dt
            prog=min(tt/TIME_BUDGET,1.0)
            if prog<0.05:lr_m=prog/0.05
            else:
                cp=(prog-0.05)/0.95;lr_m=max(0.01,0.5*(1+math.cos(math.pi*cp)))
            for g in opt.param_groups:g["lr"]=LR*lr_m
            lf=loss.item();sl=0.95*sl+0.05*lf;db=sl/(1-0.95**(step+1))
            if step%200==0 or step<5:
                print(f"\rstep {step:05d} ({100*prog:.1f}%) | loss: {db:.6f} | lr: {LR*lr_m:.2e} | ep: {ep} | rem: {max(0,TIME_BUDGET-tt):.0f}s    ",end="",flush=True)
            if step>10 and step%400==0 and tt<TIME_BUDGET*0.95:
                sp,mse,_,_=ev()
                ck={k:v.clone() for k,v in model.state_dict().items()}
                top_ckpts.append((sp,ck));top_ckpts.sort(key=lambda x:-x[0])
                if len(top_ckpts)>TOP_K:top_ckpts.pop()
                if sp>best_sp:best_sp=sp;best_st=ck;print(f"\n  [ckpt] sp={sp:.4f} mse={mse:.4f} (NEW BEST)")
                else:print(f"\n  [ckpt] sp={sp:.4f} (best={best_sp:.4f})")
                model.train()
            if lf>100 or math.isnan(lf):return {"error":"diverged"}
            if step==0:gc.collect()
            step+=1
            if step>5 and tt>=TIME_BUDGET:break
        if step>5 and tt>=TIME_BUDGET:break

    print(f"\nDone. {step} steps, {ep} epochs.")
    if best_st:model.load_state_dict(best_st)
    sp_s,mse_s,pa,la=ev();pr_s=float(np.corrcoef(pa,la)[0,1])
    ens_sp=-1
    if len(top_ckpts)>=2:
        ap=[]
        for _,st in top_ckpts:
            model.load_state_dict(st);model.eval();pp=[]
            with torch.no_grad():
                for xv,yv in vloader:pp.extend(model(xv.to(dev)).squeeze(-1).cpu().numpy().tolist())
            ap.append(np.array(pp))
        avg=np.mean(ap,axis=0);ens_sp=spearmanr(avg,la)[0]
        ens_mse=float(np.mean((avg-np.array(la))**2));ens_pr=float(np.corrcoef(avg,la)[0,1])
        print(f"Single:sp={sp_s:.4f}\nEnsemble:sp={ens_sp:.4f}({len(top_ckpts)} ckpts)")
    pk=torch.cuda.max_memory_allocated()/1024/1024
    if ens_sp>sp_s:sp_f,mse_f,pr_f=ens_sp,ens_mse,ens_pr;mt="ensemble"
    else:sp_f,mse_f,pr_f=sp_s,mse_s,pr_s;mt="single"
    print(f"=>{mt}\n---")
    for k,v in [("val_spearman",sp_f),("val_mse",mse_f),("val_pearson",pr_f),
                ("val_samples",len(la)),("training_seconds",tt),("peak_memory_mb",pk),
                ("num_steps",step),("num_params_M",model.np()/1e6)]:
        print(f"{k}:{v}")
    return {"val_spearman":sp_f,"val_mse":mse_f,"val_pearson":pr_f,
            "val_samples":len(la),"training_seconds":tt,"peak_memory_mb":pk,
            "num_steps":step,"num_params_M":model.np()/1e6,"method":mt}


@app.local_entrypoint()
def main():
    print("Stability Exp3 (smaller model + MSE)...")
    t0=time.time()
    r=train_stability.remote()
    print(f"\n{'='*60}\nEXP3 DONE ({time.time()-t0:.0f}s)\n{'='*60}")
    for k,v in r.items():print(f"  {k}:{v}")
