"""
Protein Stability Prediction — TAPE Benchmark (Rocklin et al.)
All-in-one Modal runner.

Exp1: Baseline CNN+attention (adapted from GFP Exp2/Exp9 champion architecture).
TAPE baselines: Transformer=0.73, LSTM=0.69, ResNet=0.48

Usage: modal run stability_run.py
"""

import modal, os, time

app = modal.App("stability-direct")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scipy", "requests", "matplotlib", "lmdb")
vol = modal.Volume.from_name("autoresearch-bio-stability-data", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=900,
              volumes={"/root/.cache/autoresearch_bio": vol})
def train_stability():
    import gc, math, pickle, tarfile
    import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, requests
    from scipy.stats import spearmanr
    from torch.utils.data import TensorDataset, DataLoader

    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    AA_TO_IDX = {aa: i+2 for i, aa in enumerate(AA_LIST)}
    PAD=0; START=1; AA_TO_IDX["<PAD>"]=PAD; AA_TO_IDX["<START>"]=START
    VOCAB = len(AA_LIST)+2
    MAX_SEQ_LEN = 100  # stability proteins are short (~50-70 AA)
    TIME_BUDGET = 300
    CACHE = os.path.expanduser("~/.cache/autoresearch_bio/stability_data")
    URL = "https://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz"

    # ---- Config ----
    N_EMBD=128; N_CNN=256; N_LAYERS=5; N_HEADS=8; N_HIDDEN=512
    DROPOUT=0.15; BATCH=128; EMB_NOISE=0.015
    LR=3e-4; WD=0.03

    # ---- Data ----
    def download():
        os.makedirs(CACHE, exist_ok=True)
        dd = os.path.join(CACHE, "stability")
        if os.path.exists(os.path.join(dd, "stability_train.lmdb")):
            print("Data: already downloaded"); return
        tp = os.path.join(CACHE, "stability.tar.gz")
        if not os.path.exists(tp):
            print("Downloading TAPE stability dataset...")
            r = requests.get(URL, stream=True); r.raise_for_status()
            with open(tp, "wb") as f:
                for c in r.iter_content(8192): f.write(c)
            print(f"Downloaded {os.path.getsize(tp)/(1024*1024):.1f} MB")
        print("Extracting...")
        with tarfile.open(tp, "r:gz") as t: t.extractall(CACHE, filter="data")
        print("Done.")

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
                # Fallback: iterate all keys
                cursor = txn.cursor()
                for key, value in cursor:
                    if key != b"num_examples":
                        try: recs.append(pickle.loads(value))
                        except: pass
        env.close()
        return recs

    def enc(seq, maxlen):
        t = [START]
        for aa in seq[:maxlen]: t.append(AA_TO_IDX.get(aa, PAD))
        while len(t) < maxlen+1: t.append(PAD)
        return torch.tensor(t, dtype=torch.long)

    def build():
        download()
        dd = os.path.join(CACHE, "stability")

        # Try to load — inspect format first
        train_recs = load_lmdb(os.path.join(dd, "stability_train.lmdb"))
        valid_recs = load_lmdb(os.path.join(dd, "stability_valid.lmdb"))

        # Debug: inspect data format
        if train_recs:
            r0 = train_recs[0]
            print(f"Record keys: {list(r0.keys())}")
            for k, v in r0.items():
                print(f"  {k}: type={type(v).__name__}, val={str(v)[:100]}")

        def to_t(recs):
            ss, tt = [], []
            max_len = 0
            for r in recs:
                s = r.get("primary", "")
                if isinstance(s, bytes): s = s.decode()

                # Try different key names for stability target
                target = None
                for key in ["stability_score", "stability", "log_fluorescence", "score", "target"]:
                    if key in r:
                        target = r[key]
                        break
                if target is None:
                    # Try all float-valued keys
                    for k, v in r.items():
                        if k == "primary" or k == "protein_length":
                            continue
                        if isinstance(v, (float, int)):
                            target = v
                            break
                        if isinstance(v, (list, np.ndarray)):
                            target = float(v[0])
                            break
                if target is None:
                    continue

                if isinstance(target, (list, np.ndarray)):
                    target = float(target[0])
                else:
                    target = float(target)

                s = "".join(c for c in s if c in AA_LIST)
                if len(s) < 5: continue
                max_len = max(max_len, len(s))
                ss.append(s); tt.append(target)

            print(f"Max seq len: {max_len}")

            # Use actual max length + padding
            actual_maxlen = min(max_len + 5, 200)
            X = torch.stack([enc(s, actual_maxlen) for s in ss])
            Y = torch.tensor(tt, dtype=torch.float32)
            return X, Y, actual_maxlen

        Xt, Yt, maxlen = to_t(train_recs)
        Xv, Yv, _ = to_t(valid_recs)

        # Re-encode validation with same maxlen
        def re_enc(recs, maxlen):
            ss, tt = [], []
            for r in recs:
                s = r.get("primary", "")
                if isinstance(s, bytes): s = s.decode()
                target = None
                for key in ["stability_score", "stability", "log_fluorescence", "score", "target"]:
                    if key in r:
                        target = r[key]
                        break
                if target is None:
                    for k, v in r.items():
                        if k in ("primary", "protein_length"): continue
                        if isinstance(v, (float, int)): target = v; break
                        if isinstance(v, (list, np.ndarray)): target = float(v[0]); break
                if target is None: continue
                if isinstance(target, (list, np.ndarray)): target = float(target[0])
                else: target = float(target)
                s = "".join(c for c in s if c in AA_LIST)
                if len(s) < 5: continue
                ss.append(s); tt.append(target)
            return torch.stack([enc(s, maxlen) for s in ss]), torch.tensor(tt, dtype=torch.float32)

        Xv, Yv = re_enc(valid_recs, maxlen)

        print(f"Train: {len(Xt)}, Val: {len(Xv)}")
        print(f"Target range: [{Yt.min():.3f}, {Yt.max():.3f}], mean={Yt.mean():.3f}, std={Yt.std():.3f}")
        print(f"Seq maxlen: {maxlen}")
        return TensorDataset(Xt, Yt), TensorDataset(Xv, Yv), maxlen

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

    class Model(nn.Module):
        def __init__(self, maxlen):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, N_EMBD)
            self.pos = nn.Embedding(maxlen+1, N_EMBD)
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
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    train_ds, val_ds, maxlen = build()
    vol.commit()

    model = Model(maxlen).to(dev)
    print(f"Params: {model.np():,} ({model.np()/1e6:.2f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.HuberLoss(delta=1.0)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    vloader = DataLoader(val_ds, batch_size=64, shuffle=False)
    scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None

    best_sp = -1.0; best_st = None
    top_ckpts = []; TOP_K = 5

    def eval_model():
        model.eval()
        pa, la = [], []
        with torch.no_grad():
            for xv,yv in vloader:
                xv = xv.to(dev)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        pa.extend(model(xv).squeeze(-1).cpu().numpy().tolist())
                else:
                    pa.extend(model(xv).squeeze(-1).cpu().numpy().tolist())
                la.extend(yv.numpy().tolist())
        sp = spearmanr(pa,la)[0]
        mse = float(np.mean((np.array(pa)-np.array(la))**2))
        return sp, mse, pa, la

    tt = 0.0; step = 0; ep = 0; sl = 0.0
    model.train()

    while True:
        ep += 1
        for xb, yb in loader:
            t0 = time.time()
            xb, yb = xb.to(dev), yb.to(dev)
            if scaler:
                with torch.amp.autocast("cuda"):
                    p = model(xb).squeeze(-1)
                    loss = crit(p, yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                p = model(xb).squeeze(-1)
                loss = crit(p, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if dev.type == "cuda": torch.cuda.synchronize()
            dt = time.time() - t0
            if step > 5: tt += dt

            prog = min(tt / TIME_BUDGET, 1.0)
            if prog < 0.05: lr_m = prog / 0.05
            else:
                cp = (prog - 0.05) / 0.95
                lr_m = max(0.01, 0.5*(1+math.cos(math.pi*cp)))
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

    pk = torch.cuda.max_memory_allocated()/1024/1024 if dev.type=="cuda" else 0
    if ens_sp > sp_s:
        sp_f,mse_f,pr_f = ens_sp,ens_mse,ens_pr; mt="ensemble"
    else:
        sp_f,mse_f,pr_f = sp_s,mse_s,pr_s; mt="single"

    print(f"=> Using {mt}\n---")
    for k,v in [("val_spearman",sp_f),("val_mse",mse_f),("val_pearson",pr_f),
                ("val_samples",len(l_all)),("training_seconds",tt),
                ("peak_memory_mb",pk),("num_steps",step),("num_params_M",model.np()/1e6)]:
        print(f"{k}:     {v}")

    return {"val_spearman":sp_f, "val_mse":mse_f, "val_pearson":pr_f,
            "val_samples":len(l_all), "training_seconds":tt, "peak_memory_mb":pk,
            "num_steps":step, "num_params_M":model.np()/1e6, "method":mt}


@app.local_entrypoint()
def main():
    print("Launching Stability Exp1 (baseline)...")
    t0 = time.time()
    r = train_stability.remote()
    print(f"\n{'='*60}\nSTABILITY EXP1 COMPLETE ({time.time()-t0:.0f}s)\n{'='*60}")
    for k,v in r.items(): print(f"  {k}: {v}")
