"""
GFP fluorescence prediction training script.
Exp8b: Conservative AA substitution augmentation + checkpoint ensemble.

Key ideas (simplified from failed Exp8):
1. Conservative AA substitutions (K↔R, D↔E, I↔L↔V, etc.) — biologically motivated
2. Checkpoint ensemble: average predictions from top-5 checkpoints
3. Slightly lower LR (2e-4) for smoother convergence
4. Back to standard AdamW (no SAM — too slow with 2x passes)

Usage: python train.py
"""

import os, gc, time, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    make_dataloader, evaluate_model, VOCAB_SIZE, MAX_SEQ_LEN, TIME_BUDGET,
    AA_TO_IDX,
)

# Device
if torch.cuda.is_available():
    device = torch.device("cuda"); device_type = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps"); device_type = "mps"
else:
    device = torch.device("cpu"); device_type = "cpu"
print(f"Device: {device}")

# Architecture — Exp2 base (proven best)
N_EMBD = 128
N_CNN_CHANNELS = 256
N_CNN_LAYERS = 5
N_ATTN_HEADS = 8
N_HIDDEN = 512
DROPOUT = 0.15
BATCH_SIZE = 128
EMB_NOISE = 0.015

# Conservative AA substitution
SUB_PROB = 0.08
SIMILAR_GROUPS = [
    [AA_TO_IDX['K'], AA_TO_IDX['R']],
    [AA_TO_IDX['D'], AA_TO_IDX['E']],
    [AA_TO_IDX['I'], AA_TO_IDX['L'], AA_TO_IDX['V']],
    [AA_TO_IDX['F'], AA_TO_IDX['Y'], AA_TO_IDX['W']],
    [AA_TO_IDX['S'], AA_TO_IDX['T']],
    [AA_TO_IDX['N'], AA_TO_IDX['Q']],
    [AA_TO_IDX['A'], AA_TO_IDX['G']],
]
SUB_LOOKUP = {}
for group in SIMILAR_GROUPS:
    for tok in group:
        SUB_LOOKUP[tok] = [t for t in group if t != tok]

# Checkpoint ensemble
TOP_K_CKPTS = 5

# Optimization
LEARNING_RATE = 2e-4   # slightly lower for smoother convergence
WEIGHT_DECAY = 0.03
WARMUP_RATIO = 0.05


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dropout, kernel_size=3):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.gelu(self.conv1(self.norm(x)))
        out = self.dropout(self.conv2(out))
        return F.gelu(out + residual)


class GFPPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.aa_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN + 1, N_EMBD)
        self.input_proj = nn.Conv1d(N_EMBD, N_CNN_CHANNELS, 1)
        self.cnn_blocks = nn.ModuleList([
            ResidualConvBlock(N_CNN_CHANNELS, DROPOUT) for _ in range(N_CNN_LAYERS)
        ])
        self.cnn_norm = nn.BatchNorm1d(N_CNN_CHANNELS)
        self.mha = nn.MultiheadAttention(
            embed_dim=N_EMBD, num_heads=N_ATTN_HEADS,
            dropout=DROPOUT, batch_first=True,
        )
        self.attn_query = nn.Parameter(torch.randn(1, 1, N_EMBD) * 0.02)
        combined_dim = 2 * N_CNN_CHANNELS + 2 * N_EMBD
        self.head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, N_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(N_HIDDEN, N_HIDDEN // 4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(N_HIDDEN // 4, 1),
        )
        self._init_weights()

    def _init_weights(self):
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
        B, T = x.shape
        seq = x[:, 1:]
        emb = self.aa_emb(seq) + self.pos_emb(torch.arange(seq.size(1), device=x.device).unsqueeze(0))
        if self.training and EMB_NOISE > 0:
            emb = emb + torch.randn_like(emb) * EMB_NOISE
        cnn_in = self.input_proj(emb.permute(0, 2, 1))
        for block in self.cnn_blocks:
            cnn_in = block(cnn_in)
        cnn_in = self.cnn_norm(cnn_in)
        cnn_mean = cnn_in.mean(dim=2)
        cnn_max = cnn_in.max(dim=2).values
        query = self.attn_query.expand(B, -1, -1)
        attn_out, _ = self.mha(query, emb, emb)
        attn_pooled = attn_out.squeeze(1)
        global_mean = emb.mean(dim=1)
        combined = torch.cat([cnn_mean, cnn_max, attn_pooled, global_mean], dim=-1)
        return self.head(combined)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def conservative_substitution(x_batch, sub_prob=0.05):
    """Apply conservative AA substitutions."""
    x_aug = x_batch.clone()
    B, T = x_aug.shape
    mask = torch.rand(B, T, device=x_aug.device) < sub_prob
    for tok_id, substitutes in SUB_LOOKUP.items():
        if not substitutes:
            continue
        is_tok = (x_aug == tok_id) & mask
        if is_tok.any():
            n_subs = is_tok.sum().item()
            sub_choices = torch.tensor(substitutes, device=x_aug.device)
            replacements = sub_choices[torch.randint(0, len(substitutes), (n_subs,), device=x_aug.device)]
            x_aug[is_tok] = replacements
    return x_aug


def ensemble_evaluate(model, checkpoints, device, batch_size=64):
    """Evaluate by averaging predictions from multiple checkpoints."""
    from prepare import _DATASETS
    from torch.utils.data import DataLoader
    import numpy as np
    from scipy.stats import spearmanr

    _, val_ds, _ = _DATASETS
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    all_preds_per_ckpt = []

    for ckpt_state in checkpoints:
        model.load_state_dict(ckpt_state)
        model.eval()
        preds = []
        with torch.no_grad():
            for X_b, Y_b in loader:
                X_b = X_b.to(device)
                p = model(X_b).squeeze(-1).cpu().numpy()
                preds.extend(p.tolist())
        all_preds_per_ckpt.append(np.array(preds))

    # Average predictions
    avg_preds = np.mean(all_preds_per_ckpt, axis=0)

    all_labels = []
    for X_b, Y_b in loader:
        all_labels.extend(Y_b.numpy().tolist())
    all_labels = np.array(all_labels)

    sp_corr, _ = spearmanr(avg_preds, all_labels)
    mse = float(np.mean((avg_preds - all_labels) ** 2))
    pearson = float(np.corrcoef(avg_preds, all_labels)[0, 1])

    return {
        "val_spearman": sp_corr,
        "val_mse": mse,
        "val_pearson": pearson,
        "n_samples": len(all_labels),
        "n_ckpts": len(checkpoints),
    }


def main():
    torch.manual_seed(42)
    if device_type == "cuda": torch.cuda.manual_seed(42)

    t_start = time.time()
    model = GFPPredictor().to(device)
    num_params = model.num_params()
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    use_amp = device_type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.HuberLoss(delta=1.0)
    train_loader = make_dataloader("train", BATCH_SIZE, shuffle=True)

    # Store top-K checkpoints for ensemble
    top_ckpts = []  # list of (spearman, state_dict)

    def get_lr(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        cp = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * cp)))

    print(f"Time budget: {TIME_BUDGET}s | Batch: {BATCH_SIZE} | AA sub prob: {SUB_PROB}")
    print(f"Ensemble top-K: {TOP_K_CKPTS} | LR: {LEARNING_RATE}")
    print()

    total_training_time = 0.0
    step = 0; epoch = 0; smooth_loss = 0.0
    best_val_spearman = -1.0; best_state = None

    model.train()
    while True:
        epoch += 1
        for X_batch, Y_batch in train_loader:
            t0 = time.time()
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # Apply conservative AA substitution augmentation
            if SUB_PROB > 0:
                X_batch = conservative_substitution(X_batch, SUB_PROB)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    preds = model(X_batch).squeeze(-1)
                    loss = criterion(preds, Y_batch)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(X_batch).squeeze(-1)
                loss = criterion(preds, Y_batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if device_type == "cuda": torch.cuda.synchronize()

            dt = time.time() - t0
            if step > 5: total_training_time += dt

            progress = min(total_training_time / TIME_BUDGET, 1.0) if TIME_BUDGET > 0 else 0.0
            lr_mult = get_lr(progress)
            for g in optimizer.param_groups: g["lr"] = LEARNING_RATE * lr_mult

            loss_f = loss.item()
            smooth_loss = 0.95 * smooth_loss + 0.05 * loss_f
            debiased = smooth_loss / (1 - 0.95 ** (step + 1))

            if step % 100 == 0 or step < 5:
                print(f"\rstep {step:05d} ({100*progress:.1f}%) | loss: {debiased:.6f} | lr: {LEARNING_RATE*lr_mult:.2e} | epoch: {epoch} | remaining: {max(0,TIME_BUDGET-total_training_time):.0f}s    ", end="", flush=True)

            if step > 10 and step % 400 == 0 and total_training_time < TIME_BUDGET * 0.95:
                model.eval()
                r = evaluate_model(model, device, batch_size=BATCH_SIZE)
                sp = r["val_spearman"]

                # Save checkpoint for ensemble
                ckpt = {k: v.clone() for k, v in model.state_dict().items()}
                top_ckpts.append((sp, ckpt))
                top_ckpts.sort(key=lambda x: -x[0])
                if len(top_ckpts) > TOP_K_CKPTS:
                    top_ckpts.pop()

                if sp > best_val_spearman:
                    best_val_spearman = sp
                    best_state = ckpt
                    print(f"\n  [ckpt] spearman={sp:.4f} mse={r['val_mse']:.4f} (NEW BEST)")
                else:
                    print(f"\n  [ckpt] spearman={sp:.4f} (best={best_val_spearman:.4f})")
                model.train()

            if loss_f > 100 or math.isnan(loss_f):
                print(f"\nFAIL: loss={loss_f}"); exit(1)
            if step == 0: gc.collect()
            step += 1
            if step > 5 and total_training_time >= TIME_BUDGET: break
        if step > 5 and total_training_time >= TIME_BUDGET: break

    print(f"\nTraining complete. {step} steps, {epoch} epochs.")

    # Evaluate single best
    if best_state is not None:
        model.load_state_dict(best_state)
    single_results = evaluate_model(model, device, batch_size=BATCH_SIZE)
    print(f"Single best:  spearman={single_results['val_spearman']:.4f}")

    # Evaluate ensemble of top-K checkpoints
    if len(top_ckpts) >= 2:
        ens_states = [s for _, s in top_ckpts]
        ens_results = ensemble_evaluate(model, ens_states, device, batch_size=BATCH_SIZE)
        print(f"Ensemble ({len(ens_states)}): spearman={ens_results['val_spearman']:.4f}")
    else:
        ens_results = None

    # Pick best
    if ens_results and ens_results["val_spearman"] > single_results["val_spearman"]:
        results = ens_results
        print("=> Using ensemble (better)")
    else:
        results = single_results
        if best_state: model.load_state_dict(best_state)
        print("=> Using single best (better)")

    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device_type == "cuda" else 0.0
    t_end = time.time()
    print("---")
    print(f"val_spearman:     {results['val_spearman']:.6f}")
    print(f"val_mse:          {results['val_mse']:.6f}")
    print(f"val_pearson:      {results['val_pearson']:.6f}")
    print(f"val_samples:      {results['n_samples']}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_memory_mb:   {peak_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"n_layer:          {N_CNN_LAYERS}")
    print(f"n_head:           {N_ATTN_HEADS}")
    print(f"n_embd:           {N_EMBD}")

if __name__ == "__main__":
    main()
