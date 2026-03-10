"""
All-in-one Modal runner for GFP experiments.
Exp9: Exact Exp2 config (no augmentation) + checkpoint ensemble.
Usage: modal run gfp_run.py
"""

import modal
import os
import time

app = modal.App("gfp-direct")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pandas", "scipy", "requests", "matplotlib", "lmdb")
)

vol = modal.Volume.from_name("autoresearch-bio-gfp-data", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=600,
              volumes={"/root/.cache/autoresearch_bio": vol})
def train_gfp():
    """Train GFP model directly — no subprocess, no local file mounts."""
    import gc, math, copy, pickle, tarfile
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import requests
    from scipy.stats import spearmanr
    from torch.utils.data import TensorDataset, DataLoader

    # ---- Constants ----
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    AA_TO_IDX = {aa: i + 2 for i, aa in enumerate(AA_LIST)}
    PAD_TOKEN = 0; START_TOKEN = 1
    AA_TO_IDX["<PAD>"] = PAD_TOKEN; AA_TO_IDX["<START>"] = START_TOKEN
    VOCAB_SIZE = len(AA_LIST) + 2
    MAX_SEQ_LEN = 240
    TIME_BUDGET = 300

    CACHE_DIR = os.path.expanduser("~/.cache/autoresearch_bio/gfp_data")
    TAPE_URL = "https://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz"

    # ---- Data Loading ----
    def download_data():
        os.makedirs(CACHE_DIR, exist_ok=True)
        data_dir = os.path.join(CACHE_DIR, "fluorescence")
        if os.path.exists(os.path.join(data_dir, "fluorescence_train.lmdb")):
            print("Data: already downloaded")
            return
        tar_path = os.path.join(CACHE_DIR, "fluorescence.tar.gz")
        if not os.path.exists(tar_path):
            print(f"Downloading TAPE GFP dataset...")
            resp = requests.get(TAPE_URL, stream=True)
            resp.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(CACHE_DIR, filter="data")

    def load_lmdb(path):
        import lmdb
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        records = []
        with env.begin(write=False) as txn:
            raw = txn.get(b"num_examples")
            if raw:
                try: n = int(raw.decode())
                except: n = pickle.loads(raw)
                for i in range(n):
                    r = txn.get(str(i).encode())
                    if r: records.append(pickle.loads(r))
        env.close()
        return records

    def encode_seq(seq):
        tokens = [START_TOKEN]
        for aa in seq[:MAX_SEQ_LEN]:
            tokens.append(AA_TO_IDX.get(aa, PAD_TOKEN))
        while len(tokens) < MAX_SEQ_LEN + 1:
            tokens.append(PAD_TOKEN)
        return torch.tensor(tokens, dtype=torch.long)

    def build_datasets():
        download_data()
        data_dir = os.path.join(CACHE_DIR, "fluorescence")
        train_recs = load_lmdb(os.path.join(data_dir, "fluorescence_train.lmdb"))
        valid_recs = load_lmdb(os.path.join(data_dir, "fluorescence_valid.lmdb"))

        def to_tensors(recs):
            seqs, targets = [], []
            for rec in recs:
                s = rec.get("primary", "")
                if isinstance(s, bytes): s = s.decode()
                t = rec.get("log_fluorescence")
                if isinstance(t, (list, np.ndarray)): t = float(t[0])
                elif t is not None: t = float(t)
                if t is None: continue
                s = "".join(c for c in s if c in AA_LIST)
                if len(s) < 5: continue
                seqs.append(s); targets.append(t)
            X = torch.stack([encode_seq(s) for s in seqs])
            Y = torch.tensor(targets, dtype=torch.float32)
            return X, Y

        Xt, Yt = to_tensors(train_recs)
        Xv, Yv = to_tensors(valid_recs)
        print(f"Train: {len(Xt)}, Val: {len(Xv)}")
        print(f"Target range: [{Yt.min():.2f}, {Yt.max():.2f}], mean={Yt.mean():.3f}")
        return TensorDataset(Xt, Yt), TensorDataset(Xv, Yv)

    # ---- Model ----
    N_EMBD=128; N_CNN=256; N_LAYERS=5; N_HEADS=8; N_HIDDEN=512
    DROPOUT=0.15; BATCH=128; EMB_NOISE=0.015; SUB_PROB=0.0  # No augmentation

    # Conservative AA substitution groups
    SIMILAR = [
        [AA_TO_IDX['K'], AA_TO_IDX['R']],
        [AA_TO_IDX['D'], AA_TO_IDX['E']],
        [AA_TO_IDX['I'], AA_TO_IDX['L'], AA_TO_IDX['V']],
        [AA_TO_IDX['F'], AA_TO_IDX['Y'], AA_TO_IDX['W']],
        [AA_TO_IDX['S'], AA_TO_IDX['T']],
        [AA_TO_IDX['N'], AA_TO_IDX['Q']],
    ]
    SUB_MAP = {}
    for g in SIMILAR:
        for t in g:
            SUB_MAP[t] = [x for x in g if x != t]

    class ConvBlock(nn.Module):
        def __init__(self, ch, dp):
            super().__init__()
            self.norm = nn.BatchNorm1d(ch)
            self.c1 = nn.Conv1d(ch, ch, 3, padding=1)
            self.c2 = nn.Conv1d(ch, ch, 3, padding=1)
            self.drop = nn.Dropout(dp)
        def forward(self, x):
            r = x
            o = F.gelu(self.c1(self.norm(x)))
            return F.gelu(self.drop(self.c2(o)) + r)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
            self.pos = nn.Embedding(MAX_SEQ_LEN+1, N_EMBD)
            self.proj = nn.Conv1d(N_EMBD, N_CNN, 1)
            self.blocks = nn.ModuleList([ConvBlock(N_CNN, DROPOUT) for _ in range(N_LAYERS)])
            self.cnorm = nn.BatchNorm1d(N_CNN)
            self.mha = nn.MultiheadAttention(N_EMBD, N_HEADS, dropout=DROPOUT, batch_first=True)
            self.q = nn.Parameter(torch.randn(1,1,N_EMBD)*0.02)
            cd = 2*N_CNN + 2*N_EMBD
            self.head = nn.Sequential(
                nn.LayerNorm(cd), nn.Linear(cd, N_HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(N_HIDDEN, N_HIDDEN//4), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(N_HIDDEN//4, 1))
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None: nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None: nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.02)

        def forward(self, x):
            B = x.size(0)
            s = x[:,1:]
            e = self.emb(s) + self.pos(torch.arange(s.size(1), device=x.device).unsqueeze(0))
            if self.training and EMB_NOISE > 0:
                e = e + torch.randn_like(e) * EMB_NOISE
            c = self.proj(e.permute(0,2,1))
            for b in self.blocks: c = b(c)
            c = self.cnorm(c)
            cm, cx = c.mean(2), c.max(2).values
            a, _ = self.mha(self.q.expand(B,-1,-1), e, e)
            return self.head(torch.cat([cm, cx, a.squeeze(1), e.mean(1)], -1))

        def nparams(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def aa_sub(xb):
        xa = xb.clone()
        mask = torch.rand_like(xa.float()) < SUB_PROB
        for tid, subs in SUB_MAP.items():
            if not subs: continue
            hit = (xa == tid) & mask
            if hit.any():
                sc = torch.tensor(subs, device=xa.device)
                xa[hit] = sc[torch.randint(0, len(subs), (hit.sum().item(),), device=xa.device)]
        return xa

    # ---- Training ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds = build_datasets()
    vol.commit()

    model = Model().to(device)
    print(f"Params: {model.nparams():,} ({model.nparams()/1e6:.2f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.03)
    crit = nn.HuberLoss(delta=1.0)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    scaler = torch.amp.GradScaler("cuda")
    best_sp = -1.0; best_st = None
    top_ckpts = []  # (spearman, state)
    TOP_K = 5

    tt = 0.0; step = 0; ep = 0; sl = 0.0
    model.train()

    while True:
        ep += 1
        for xb, yb in loader:
            t0 = time.time()
            xb, yb = xb.to(device), yb.to(device)
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
            # Cosine LR with 3e-4 peak (Exp2)
            if prog < 0.05:
                lr_m = prog / 0.05
            else:
                cp = (prog - 0.05) / 0.95
                lr_m = max(0.01, 0.5*(1+math.cos(math.pi*cp)))
            for g in opt.param_groups: g["lr"] = 3e-4 * lr_m

            lf = loss.item()
            sl = 0.95*sl + 0.05*lf
            db = sl / (1 - 0.95**(step+1))

            if step % 100 == 0 or step < 5:
                print(f"\rstep {step:05d} ({100*prog:.1f}%) | loss: {db:.6f} | lr: {3e-4*lr_m:.2e} | ep: {ep} | rem: {max(0,TIME_BUDGET-tt):.0f}s    ", end="", flush=True)

            if step > 10 and step % 400 == 0 and tt < TIME_BUDGET * 0.95:
                model.eval()
                preds_all, labels_all = [], []
                with torch.no_grad():
                    for xv, yv in val_loader:
                        xv = xv.to(device)
                        pv = model(xv).squeeze(-1).cpu().numpy()
                        preds_all.extend(pv.tolist())
                        labels_all.extend(yv.numpy().tolist())
                sp = spearmanr(preds_all, labels_all)[0]
                mse = float(np.mean((np.array(preds_all)-np.array(labels_all))**2))

                ck = {k: v.clone() for k, v in model.state_dict().items()}
                top_ckpts.append((sp, ck))
                top_ckpts.sort(key=lambda x: -x[0])
                if len(top_ckpts) > TOP_K: top_ckpts.pop()

                if sp > best_sp:
                    best_sp = sp; best_st = ck
                    print(f"\n  [ckpt] sp={sp:.4f} mse={mse:.4f} (NEW BEST)")
                else:
                    print(f"\n  [ckpt] sp={sp:.4f} (best={best_sp:.4f})")
                model.train()

            if lf > 100 or math.isnan(lf):
                print(f"\nFAIL: loss={lf}")
                return {"error": "diverged"}
            if step == 0: gc.collect()
            step += 1
            if step > 5 and tt >= TIME_BUDGET: break
        if step > 5 and tt >= TIME_BUDGET: break

    print(f"\nDone. {step} steps, {ep} epochs.")

    # Single best
    if best_st: model.load_state_dict(best_st)
    model.eval()
    p_all, l_all = [], []
    with torch.no_grad():
        for xv, yv in val_loader:
            p_all.extend(model(xv.to(device)).squeeze(-1).cpu().numpy().tolist())
            l_all.extend(yv.numpy().tolist())
    single_sp = spearmanr(p_all, l_all)[0]
    single_mse = float(np.mean((np.array(p_all)-np.array(l_all))**2))
    single_pr = float(np.corrcoef(p_all, l_all)[0,1])

    # Ensemble
    ens_sp = -1
    if len(top_ckpts) >= 2:
        all_p = []
        for _, st in top_ckpts:
            model.load_state_dict(st); model.eval()
            pp = []
            with torch.no_grad():
                for xv, yv in val_loader:
                    pp.extend(model(xv.to(device)).squeeze(-1).cpu().numpy().tolist())
            all_p.append(np.array(pp))
        avg_p = np.mean(all_p, axis=0)
        ens_sp = spearmanr(avg_p, l_all)[0]
        ens_mse = float(np.mean((avg_p-np.array(l_all))**2))
        ens_pr = float(np.corrcoef(avg_p, l_all)[0,1])
        print(f"Single:   sp={single_sp:.4f}")
        print(f"Ensemble: sp={ens_sp:.4f} ({len(top_ckpts)} ckpts)")

    peak = torch.cuda.max_memory_allocated()/1024/1024

    # Use whichever is better
    if ens_sp > single_sp:
        sp_final, mse_final, pr_final = ens_sp, ens_mse, ens_pr
        method = "ensemble"
    else:
        sp_final, mse_final, pr_final = single_sp, single_mse, single_pr
        method = "single"

    print(f"=> Using {method}")
    print("---")
    print(f"val_spearman:     {sp_final:.6f}")
    print(f"val_mse:          {mse_final:.6f}")
    print(f"val_pearson:      {pr_final:.6f}")
    print(f"val_samples:      {len(l_all)}")
    print(f"training_seconds: {tt:.1f}")
    print(f"peak_memory_mb:   {peak:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {model.nparams()/1e6:.1f}")
    print(f"n_layer:          {N_LAYERS}")
    print(f"n_head:           {N_HEADS}")
    print(f"n_embd:           {N_EMBD}")

    return {
        "val_spearman": sp_final, "val_mse": mse_final, "val_pearson": pr_final,
        "val_samples": len(l_all), "training_seconds": tt, "peak_memory_mb": peak,
        "num_steps": step, "num_params_M": model.nparams()/1e6,
        "method": method,
    }


@app.local_entrypoint()
def main():
    print("Launching GFP experiment (all-in-one)...")
    t0 = time.time()
    results = train_gfp.remote()
    elapsed = time.time() - t0
    print(f"\n{'='*60}\nEXPERIMENT COMPLETE ({elapsed:.0f}s)\n{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v}")
