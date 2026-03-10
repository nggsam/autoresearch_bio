"""
GFP Exp12: Wider model (5.5M) + 600s budget + cosine restarts for diverse checkpoints.
Usage: modal run gfp_run3.py
"""

import modal, os, time

app = modal.App("gfp-direct-v3")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scipy", "requests", "matplotlib", "lmdb")
vol = modal.Volume.from_name("autoresearch-bio-gfp-data", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=900,
              volumes={"/root/.cache/autoresearch_bio": vol})
def train_gfp():
    import gc, math, pickle, tarfile
    import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, requests
    from scipy.stats import spearmanr
    from torch.utils.data import TensorDataset, DataLoader

    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    AA_TO_IDX = {aa: i+2 for i, aa in enumerate(AA_LIST)}
    PAD=0; START=1; AA_TO_IDX["<PAD>"]=PAD; AA_TO_IDX["<START>"]=START
    VOCAB = len(AA_LIST)+2; MAXLEN = 240; TIME_BUDGET = 600
    CACHE = os.path.expanduser("~/.cache/autoresearch_bio/gfp_data")
    URL = "https://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz"

    # ---- Config: wider model ----
    N_EMBD=192; N_CNN=384; N_LAYERS=6; N_HEADS=8; N_HIDDEN=768
    DROPOUT=0.15; BATCH=96; EMB_NOISE=0.015
    LR=2e-4; WD=0.03; N_RESTARTS=3  # cosine restarts

    # ---- Data ----
    def download():
        os.makedirs(CACHE, exist_ok=True)
        dd = os.path.join(CACHE, "fluorescence")
        if os.path.exists(os.path.join(dd, "fluorescence_train.lmdb")): return
        tp = os.path.join(CACHE, "fluorescence.tar.gz")
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
        env.close(); return recs

    def enc(seq):
        t = [START]
        for aa in seq[:MAXLEN]: t.append(AA_TO_IDX.get(aa, PAD))
        while len(t) < MAXLEN+1: t.append(PAD)
        return torch.tensor(t, dtype=torch.long)

    def build():
        download()
        dd = os.path.join(CACHE, "fluorescence")
        def to_t(recs):
            ss, tt = [], []
            for r in recs:
                s = r.get("primary", "")
                if isinstance(s, bytes): s = s.decode()
                t = r.get("log_fluorescence")
                if isinstance(t, (list, np.ndarray)): t = float(t[0])
                elif t is not None: t = float(t)
                if t is None: continue
                s = "".join(c for c in s if c in AA_LIST)
                if len(s) < 5: continue
                ss.append(s); tt.append(t)
            return torch.stack([enc(s) for s in ss]), torch.tensor(tt, dtype=torch.float32)
        Xt, Yt = to_t(load_lmdb(os.path.join(dd, "fluorescence_train.lmdb")))
        Xv, Yv = to_t(load_lmdb(os.path.join(dd, "fluorescence_valid.lmdb")))
        print(f"Train: {len(Xt)}, Val: {len(Xv)}")
        return TensorDataset(Xt, Yt), TensorDataset(Xv, Yv)

    # ---- Model ----
    class CB(nn.Module):
        def __init__(self, ch, dp):
            super().__init__()
            self.n = nn.BatchNorm1d(ch)
            self.c1 = nn.Conv1d(ch, ch, 3, padding=1)
            self.c2 = nn.Conv1d(ch, ch, 3, padding=1)
            self.d = nn.Dropout(dp)
        def forward(self, x):
            return F.gelu(self.d(self.c2(F.gelu(self.c1(self.n(x))))) + x)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, N_EMBD)
            self.pos = nn.Embedding(MAXLEN+1, N_EMBD)
            self.proj = nn.Conv1d(N_EMBD, N_CNN, 1)
            self.blks = nn.ModuleList([CB(N_CNN, DROPOUT) for _ in range(N_LAYERS)])
            self.cn = nn.BatchNorm1d(N_CNN)
            self.mha = nn.MultiheadAttention(N_EMBD, N_HEADS, dropout=DROPOUT, batch_first=True)
            self.q = nn.Parameter(torch.randn(1,1,N_EMBD)*0.02)
            cd = 2*N_CNN + 2*N_EMBD
            self.head = nn.Sequential(
                nn.LayerNorm(cd), nn.Linear(cd, N_HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(N_HIDDEN, N_HIDDEN//4), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(N_HIDDEN//4, 1))
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None: nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
        def forward(self, x):
            B = x.size(0); s = x[:,1:]
            e = self.emb(s) + self.pos(torch.arange(s.size(1), device=x.device).unsqueeze(0))
            if self.training and EMB_NOISE > 0: e = e + torch.randn_like(e)*EMB_NOISE
            c = self.proj(e.permute(0,2,1))
            for b in self.blks: c = b(c)
            c = self.cn(c)
            cm, cx = c.mean(2), c.max(2).values
            a, _ = self.mha(self.q.expand(B,-1,-1), e, e)
            return self.head(torch.cat([cm, cx, a.squeeze(1), e.mean(1)], -1))
        def np(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- Train ----
    dev = torch.device("cuda")
    train_ds, val_ds = build()
    vol.commit()

    model = M().to(dev)
    print(f"Params: {model.np():,} ({model.np()/1e6:.2f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.HuberLoss(delta=1.0)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    vloader = DataLoader(val_ds, batch_size=64, shuffle=False)
    scaler = torch.amp.GradScaler("cuda")

    best_sp = -1.0; best_st = None
    top_ckpts = []; TOP_K = 7  # more checkpoints for restarts

    def eval_model():
        model.eval()
        pa, la = [], []
        with torch.no_grad():
            for xv,yv in vloader:
                pa.extend(model(xv.to(dev)).squeeze(-1).cpu().numpy().tolist())
                la.extend(yv.numpy().tolist())
        return spearmanr(pa,la)[0], float(np.mean((np.array(pa)-np.array(la))**2)), pa, la

    tt = 0.0; step = 0; ep = 0; sl = 0.0
    model.train()

    while True:
        ep += 1
        for xb, yb in loader:
            t0 = time.time()
            xb, yb = xb.to(dev), yb.to(dev)
            with torch.amp.autocast("cuda"):
                p = model(xb).squeeze(-1)
                loss = crit(p, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            dt = time.time() - t0
            if step > 5: tt += dt

            prog = min(tt / TIME_BUDGET, 1.0)
            # Cosine with restarts
            cycle_len = 1.0 / N_RESTARTS
            cycle_prog = (prog % cycle_len) / cycle_len
            if prog < 0.03:
                lr_m = prog / 0.03
            else:
                lr_m = max(0.01, 0.5*(1+math.cos(math.pi*cycle_prog)))
            for g in opt.param_groups: g["lr"] = LR * lr_m

            lf = loss.item()
            sl = 0.95*sl + 0.05*lf
            db = sl / (1-0.95**(step+1))

            if step%100==0 or step<5:
                print(f"\rstep {step:05d} ({100*prog:.1f}%) | loss: {db:.6f} | lr: {LR*lr_m:.2e} | ep: {ep} | rem: {max(0,TIME_BUDGET-tt):.0f}s    ", end="", flush=True)

            if step>10 and step%400==0 and tt<TIME_BUDGET*0.95:
                sp, mse, _, _ = eval_model()
                ck = {k:v.clone() for k,v in model.state_dict().items()}
                top_ckpts.append((sp, ck))
                top_ckpts.sort(key=lambda x: -x[0])
                if len(top_ckpts) > TOP_K: top_ckpts.pop()
                if sp > best_sp:
                    best_sp = sp; best_st = ck
                    print(f"\n  [ckpt] sp={sp:.4f} mse={mse:.4f} (NEW BEST)")
                else:
                    print(f"\n  [ckpt] sp={sp:.4f} (best={best_sp:.4f})")
                model.train()

            if lf>100 or math.isnan(lf): return {"error":"diverged"}
            if step==0: gc.collect()
            step += 1
            if step>5 and tt>=TIME_BUDGET: break
        if step>5 and tt>=TIME_BUDGET: break

    print(f"\nDone. {step} steps, {ep} epochs.")

    # Single
    if best_st: model.load_state_dict(best_st)
    sp_s, mse_s, p_all, l_all = eval_model()
    pr_s = float(np.corrcoef(p_all, l_all)[0,1])

    # Ensemble
    ens_sp = -1
    if len(top_ckpts)>=2:
        all_p = []
        for _,st in top_ckpts:
            model.load_state_dict(st); model.eval()
            pp = []
            with torch.no_grad():
                for xv,yv in vloader:
                    pp.extend(model(xv.to(dev)).squeeze(-1).cpu().numpy().tolist())
            all_p.append(np.array(pp))
        avg = np.mean(all_p, axis=0)
        ens_sp = spearmanr(avg, l_all)[0]
        ens_mse = float(np.mean((avg-np.array(l_all))**2))
        ens_pr = float(np.corrcoef(avg, l_all)[0,1])
        print(f"Single:   sp={sp_s:.4f}")
        print(f"Ensemble: sp={ens_sp:.4f} ({len(top_ckpts)} ckpts)")

    pk = torch.cuda.max_memory_allocated()/1024/1024
    if ens_sp > sp_s:
        sp_f,mse_f,pr_f = ens_sp,ens_mse,ens_pr; mt="ensemble"
    else:
        sp_f,mse_f,pr_f = sp_s,mse_s,pr_s; mt="single"

    print(f"=> Using {mt}\n---")
    print(f"val_spearman:     {sp_f:.6f}")
    print(f"val_mse:          {mse_f:.6f}")
    print(f"val_pearson:      {pr_f:.6f}")
    print(f"val_samples:      {len(l_all)}")
    print(f"training_seconds: {tt:.1f}")
    print(f"peak_memory_mb:   {pk:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {model.np()/1e6:.1f}")

    return {"val_spearman":sp_f, "val_mse":mse_f, "val_pearson":pr_f,
            "val_samples":len(l_all), "training_seconds":tt, "peak_memory_mb":pk,
            "num_steps":step, "num_params_M":model.np()/1e6, "method":mt}


@app.local_entrypoint()
def main():
    print("Launching GFP Exp12...")
    t0 = time.time()
    r = train_gfp.remote()
    print(f"\n{'='*60}\nEXP12 COMPLETE ({time.time()-t0:.0f}s)\n{'='*60}")
    for k,v in r.items(): print(f"  {k}: {v}")
