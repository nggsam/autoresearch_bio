"""
Autoresearch_bio training script. Single-file, single-device.
Scaled-up mutation-aware model (~5M params) for GPU training.

Architecture: Exp8 mutation-aware model scaled up:
- Embeddings: 128D (was 32D)
- CNN channels: 256 (was 64)
- CNN depth: 5 layers (was 3)
- Multi-head attention layer (4 heads)
- Larger MLP head with residual connections

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
# Model architecture — SCALED UP for GPU
# ---------------------------------------------------------------------------

N_EMBD = 128          # AA embedding dimension (4x larger)
N_CNN_CHANNELS = 256  # CNN feature channels (4x larger)
N_CNN_LAYERS = 5      # CNN depth (deeper)
N_ATTN_HEADS = 4      # multi-head attention
N_HIDDEN = 512        # MLP hidden dimension (4x larger)
DROPOUT = 0.25        # slightly less dropout — more params need less regularization
BATCH_SIZE = 128      # larger batch on GPU

# Optimization
LEARNING_RATE = 3e-4   # slightly higher LR for bigger model
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.05
ADAM_BETAS = (0.9, 0.999)

# Precompute wildtype token IDs
WT_TOKENS = torch.tensor(
    [AA_TO_IDX.get(aa, 0) for aa in WILDTYPE_RBD],
    dtype=torch.long,
)


class ResidualConvBlock(nn.Module):
    """Conv1d block with residual connection and pre-norm."""
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


class MutationAwareModel(nn.Module):
    """
    Scaled-up mutation fitness predictor (~5M params).

    Changes from 0.1M version:
    - 128D embeddings (was 32D)
    - 256-channel CNN with residual blocks (was 64-channel plain)
    - Multi-head attention pooling (4 heads)
    - Deeper MLP head with residual connection
    - Separate mutation encoder is larger
    """

    def __init__(self):
        super().__init__()

        # AA embeddings (larger)
        self.aa_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(SEQ_LEN + 1, N_EMBD)

        # Initial projection to CNN channel dimension
        self.input_proj = nn.Conv1d(N_EMBD, N_CNN_CHANNELS, 1)

        # Deep residual 1D CNN
        self.cnn_blocks = nn.ModuleList([
            ResidualConvBlock(N_CNN_CHANNELS, DROPOUT, kernel_size=3)
            for _ in range(N_CNN_LAYERS)
        ])
        # Also add a wider-kernel pass for regional context
        self.cnn_wide = nn.Sequential(
            nn.Conv1d(N_CNN_CHANNELS, N_CNN_CHANNELS // 2, kernel_size=11, padding=5),
            nn.BatchNorm1d(N_CNN_CHANNELS // 2),
            nn.GELU(),
        )

        # Multi-head attention pooling
        self.mha = nn.MultiheadAttention(
            embed_dim=N_EMBD, num_heads=N_ATTN_HEADS,
            dropout=DROPOUT, batch_first=True,
        )
        self.attn_query = nn.Parameter(torch.randn(1, 1, N_EMBD) * 0.02)

        # Mutation-specific feature encoder (larger)
        # Input: wt_emb + mut_emb + delta_emb + pos_emb = 4 * N_EMBD
        self.mutation_encoder = nn.Sequential(
            nn.Linear(4 * N_EMBD, N_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(N_HIDDEN, N_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(N_HIDDEN // 2, N_HIDDEN // 4),
        )

        # Combined features:
        # CNN mean + max: N_CNN_CHANNELS * 2
        # CNN wide mean: N_CNN_CHANNELS // 2
        # MHA pooled: N_EMBD
        # Global mean: N_EMBD
        # Mutation features: N_HIDDEN // 4
        combined_dim = (
            N_CNN_CHANNELS * 2
            + N_CNN_CHANNELS // 2
            + N_EMBD * 2
            + N_HIDDEN // 4
        )

        # Deep MLP head with residual
        self.head_norm = nn.LayerNorm(combined_dim)
        self.head_proj = nn.Linear(combined_dim, N_HIDDEN)
        self.head_residual = nn.Sequential(
            nn.LayerNorm(N_HIDDEN),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(N_HIDDEN, N_HIDDEN),
        )
        self.head_final = nn.Sequential(
            nn.LayerNorm(N_HIDDEN),
            nn.Linear(N_HIDDEN, N_HIDDEN // 4),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(N_HIDDEN // 4, 1),
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
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        seq = x[:, 1:]  # skip START

        # --- Mutation detection ---
        wt = WT_TOKENS.to(seq.device).unsqueeze(0).expand(B, -1)
        if wt.size(1) < seq.size(1):
            wt = F.pad(wt, (0, seq.size(1) - wt.size(1)))
        elif wt.size(1) > seq.size(1):
            wt = wt[:, :seq.size(1)]

        diff_mask = (seq != wt)
        mut_pos = diff_mask.float().argmax(dim=1).long()

        wt_aa = wt.gather(1, mut_pos.unsqueeze(1)).squeeze(1)
        mut_aa = seq.gather(1, mut_pos.unsqueeze(1)).squeeze(1)

        # Mutation features
        wt_emb_feat = self.aa_emb(wt_aa)
        mut_emb_feat = self.aa_emb(mut_aa)
        delta_emb = mut_emb_feat - wt_emb_feat
        pos_emb_feat = self.pos_emb(mut_pos)

        mut_features = torch.cat([wt_emb_feat, mut_emb_feat, delta_emb, pos_emb_feat], dim=-1)
        mut_encoded = self.mutation_encoder(mut_features)

        # --- Sequence features ---
        emb = self.aa_emb(seq)  # (B, SEQ_LEN, N_EMBD)
        pos = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        emb = emb + self.pos_emb(pos)

        # Deep residual CNN
        cnn_in = self.input_proj(emb.permute(0, 2, 1))  # (B, C, SEQ_LEN)
        for block in self.cnn_blocks:
            cnn_in = block(cnn_in)
        cnn_mean = cnn_in.mean(dim=2)
        cnn_max = cnn_in.max(dim=2).values

        # Wide CNN for regional context
        cnn_wide_out = self.cnn_wide(cnn_in)
        cnn_wide_mean = cnn_wide_out.mean(dim=2)

        # Multi-head attention pooling
        query = self.attn_query.expand(B, -1, -1)  # (B, 1, N_EMBD)
        attn_out, _ = self.mha(query, emb, emb)  # (B, 1, N_EMBD)
        attn_pooled = attn_out.squeeze(1)  # (B, N_EMBD)

        # Global mean
        global_mean = emb.mean(dim=1)

        # --- Combine ---
        combined = torch.cat([
            cnn_mean, cnn_max,
            cnn_wide_mean,
            attn_pooled,
            global_mean,
            mut_encoded,
        ], dim=-1)

        # Deep MLP head with residual
        h = F.gelu(self.head_proj(self.head_norm(combined)))
        h = h + self.head_residual(h)
        return self.head_final(h)

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

    # Compile model if on CUDA for speed
    if device_type == "cuda":
        model = torch.compile(model)

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
    print(f"n_head:           {N_ATTN_HEADS}")
    print(f"n_embd:           {N_EMBD}")


if __name__ == "__main__":
    main()
