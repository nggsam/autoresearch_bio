"""
Autoresearch_bio training script. Single-file, single-device.
Experiment 8: explicit mutation-site features + CNN + attention.

Key idea: compare input sequence to wildtype to find the mutation
position, then extract (position, wt_aa, mut_aa, delta_embedding)
as explicit features alongside the CNN/attention features.

Usage: python train.py
"""

import os
import gc
import time
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    make_dataloader, evaluate_model, VOCAB_SIZE, SEQ_LEN, TIME_BUDGET,
    WILDTYPE_RBD, AA_TO_IDX, START_TOKEN,
)

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_type = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    device_type = "mps"
else:
    device = torch.device("cpu")
    device_type = "cpu"

print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

N_EMBD = 32          # AA embedding dimension
N_CNN_CHANNELS = 64  # CNN feature channels
N_CNN_LAYERS = 3     # CNN depth
N_HIDDEN = 128       # MLP hidden dimension
DROPOUT = 0.3
BATCH_SIZE = 64

# Optimization
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.05
ADAM_BETAS = (0.9, 0.999)

# Precompute wildtype token IDs (without START)
WT_TOKENS = torch.tensor(
    [AA_TO_IDX.get(aa, 0) for aa in WILDTYPE_RBD],
    dtype=torch.long,
)


class MutationAwareModel(nn.Module):
    """
    Mutation fitness predictor with explicit mutation-site features.

    The model:
    1. Compares input seq to wildtype → finds mutation position
    2. Extracts mutation-specific features (wt_emb, mut_emb, delta, position)
    3. Processes full sequence with 1D CNN for context
    4. Combines all features into MLP regression head

    This gives the model direct access to WHAT changed and WHERE,
    instead of having to discover it implicitly.
    """

    def __init__(self):
        super().__init__()

        # AA embeddings (shared between sequence encoding and mutation features)
        self.aa_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(SEQ_LEN + 1, N_EMBD)

        # 1D CNN for sequence context
        cnn_layers = []
        in_ch = N_EMBD
        for i in range(N_CNN_LAYERS):
            out_ch = N_CNN_CHANNELS
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Attention pooling query
        self.attn_query = nn.Linear(N_EMBD, 1)

        # Mutation-specific feature encoder
        # Features: wt_emb, mut_emb, delta_emb, position_emb = 4 * N_EMBD
        self.mutation_encoder = nn.Sequential(
            nn.Linear(4 * N_EMBD, N_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(N_HIDDEN, N_HIDDEN // 2),
        )

        # Combined features:
        # CNN (mean + max): 2 * N_CNN_CHANNELS
        # Attention-pooled: N_EMBD
        # Global mean: N_EMBD
        # Mutation features: N_HIDDEN // 2
        combined_dim = 2 * N_CNN_CHANNELS + 2 * N_EMBD + N_HIDDEN // 2

        self.head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, N_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(N_HIDDEN, N_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(N_HIDDEN // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)

    def forward(self, x):
        B, T = x.shape
        seq = x[:, 1:]  # skip START, (B, SEQ_LEN)

        # --- Mutation detection ---
        # Compare to wildtype to find mutation position
        wt = WT_TOKENS.to(seq.device).unsqueeze(0).expand(B, -1)  # (B, SEQ_LEN)
        # Pad wt if needed to match seq length
        if wt.size(1) < seq.size(1):
            wt = F.pad(wt, (0, seq.size(1) - wt.size(1)))
        elif wt.size(1) > seq.size(1):
            wt = wt[:, :seq.size(1)]

        diff_mask = (seq != wt)  # (B, SEQ_LEN) - True at mutation sites

        # Get mutation position (first differing position)
        # If no diff found (wildtype sequence), default to position 0
        mut_pos = diff_mask.float().argmax(dim=1)  # (B,)
        mut_pos_long = mut_pos.long()

        # Extract wildtype and mutant AA at mutation position
        wt_aa = wt.gather(1, mut_pos_long.unsqueeze(1)).squeeze(1)  # (B,)
        mut_aa = seq.gather(1, mut_pos_long.unsqueeze(1)).squeeze(1)  # (B,)

        # Mutation-specific features
        wt_emb_feat = self.aa_emb(wt_aa)        # (B, N_EMBD)
        mut_emb_feat = self.aa_emb(mut_aa)       # (B, N_EMBD)
        delta_emb = mut_emb_feat - wt_emb_feat   # (B, N_EMBD)
        pos_emb_feat = self.pos_emb(mut_pos_long) # (B, N_EMBD)

        mut_features = torch.cat([wt_emb_feat, mut_emb_feat, delta_emb, pos_emb_feat], dim=-1)
        mut_encoded = self.mutation_encoder(mut_features)  # (B, N_HIDDEN//2)

        # --- Sequence context features ---
        emb = self.aa_emb(seq)  # (B, SEQ_LEN, N_EMBD)

        # CNN
        cnn_in = emb.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_in)
        cnn_mean = cnn_out.mean(dim=2)
        cnn_max = cnn_out.max(dim=2).values

        # Attention-weighted pooling
        attn_logits = self.attn_query(emb).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_pooled = (emb * attn_weights.unsqueeze(-1)).sum(dim=1)

        # Global mean
        global_mean = emb.mean(dim=1)

        # --- Combine all features ---
        combined = torch.cat([
            cnn_mean,
            cnn_max,
            attn_pooled,
            global_mean,
            mut_encoded,
        ], dim=-1)

        return self.head(combined)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)

    t_start = time.time()

    model = MutationAwareModel()
    model = model.to(device)

    num_params = model.num_params()
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    if num_params > 10_000_000:
        print(f"WARNING: Model has {num_params/1e6:.1f}M params (limit: 10M)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=ADAM_BETAS,
    )

    criterion = nn.HuberLoss(delta=0.5)

    train_loader = make_dataloader("train", BATCH_SIZE, shuffle=True)

    def get_lr(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        else:
            cosine_progress = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * cosine_progress)))

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Starting training...")
    print()

    t_start_training = time.time()
    total_training_time = 0.0
    step = 0
    epoch = 0
    smooth_loss = 0.0
    best_val_spearman = -1.0
    best_state = None

    model.train()

    while True:
        epoch += 1
        for X_batch, Y_batch in train_loader:
            t0 = time.time()

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            preds = model(X_batch).squeeze(-1)
            loss = criterion(preds, Y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if device_type == "cuda":
                torch.cuda.synchronize()
            elif device_type == "mps":
                torch.mps.synchronize()

            t1 = time.time()
            dt = t1 - t0

            if step > 5:
                total_training_time += dt

            progress = min(total_training_time / TIME_BUDGET, 1.0) if TIME_BUDGET > 0 else 0.0
            lr_mult = get_lr(progress)
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_RATE * lr_mult

            loss_f = loss.item()
            ema_beta = 0.95
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
            debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
            pct_done = 100 * progress
            remaining = max(0, TIME_BUDGET - total_training_time)

            if step % 100 == 0 or step < 5:
                print(
                    f"\rstep {step:05d} ({pct_done:.1f}%) | "
                    f"loss: {debiased_loss:.6f} | "
                    f"lr: {LEARNING_RATE * lr_mult:.2e} | "
                    f"epoch: {epoch} | "
                    f"remaining: {remaining:.0f}s    ",
                    end="", flush=True,
                )

            # Mid-training eval every 600 steps
            if step > 10 and step % 600 == 0 and total_training_time < TIME_BUDGET * 0.95:
                model.eval()
                mid_results = evaluate_model(model, device, batch_size=BATCH_SIZE)
                sp = mid_results["val_spearman"]
                if sp > best_val_spearman:
                    best_val_spearman = sp
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    print(f"\n  [ckpt] spearman={sp:.4f} (NEW BEST)")
                else:
                    print(f"\n  [ckpt] spearman={sp:.4f} (best={best_val_spearman:.4f})")
                model.train()

            if loss_f > 100 or math.isnan(loss_f):
                print(f"\nFAIL: loss={loss_f}")
                exit(1)

            if step == 0:
                gc.collect()

            step += 1

            if step > 5 and total_training_time >= TIME_BUDGET:
                break

        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()
    print(f"Training complete. {step} steps, {epoch} epochs.")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint (val_spearman={best_val_spearman:.4f})")

    print("Running final validation evaluation...")
    results = evaluate_model(model, device, batch_size=BATCH_SIZE)

    if device_type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    elif device_type == "mps":
        try:
            peak_memory_mb = torch.mps.current_allocated_memory() / 1024 / 1024
        except AttributeError:
            peak_memory_mb = 0.0
    else:
        peak_memory_mb = 0.0

    t_end = time.time()

    print("---")
    print(f"val_spearman:     {results['val_spearman']:.6f}")
    print(f"val_mse:          {results['val_mse']:.6f}")
    print(f"val_pearson:      {results['val_pearson']:.6f}")
    print(f"val_samples:      {results['n_samples']}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_memory_mb:   {peak_memory_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"n_layer:          {N_CNN_LAYERS}")
    print(f"n_head:           0")
    print(f"n_embd:           {N_EMBD}")


if __name__ == "__main__":
    main()
