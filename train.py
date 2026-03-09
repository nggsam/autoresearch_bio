"""
Autoresearch_bio training script. Single-file, single-device.
GPU Exp9: Mutation-aware model (Exp8 base) with GPU-specific optimizations.

Key insight: 4M params overfits on 4k samples. Keep ~0.5M params but
leverage GPU speed for: heavier augmentation, more eval checkpoints,
mixed precision training, and torch.compile.

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
# Model architecture — optimized for ~0.5M params
# ---------------------------------------------------------------------------

N_EMBD = 64           # AA embedding dimension
N_CNN_CHANNELS = 128  # CNN feature channels
N_CNN_LAYERS = 4      # CNN depth
N_ATTN_HEADS = 4      # multi-head attention
N_HIDDEN = 256        # MLP hidden dimension
DROPOUT = 0.35        # heavier dropout for small dataset
EMB_NOISE = 0.03      # Gaussian noise on embeddings
BATCH_SIZE = 64       # moderate batch
MASK_PROB = 0.05      # randomly mask positions during training

# Optimization
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.05
ADAM_BETAS = (0.9, 0.999)

# Precompute wildtype token IDs
WT_TOKENS = torch.tensor(
    [AA_TO_IDX.get(aa, 0) for aa in WILDTYPE_RBD],
    dtype=torch.long,
)


class ResidualConvBlock(nn.Module):
    """Conv1d with residual connection."""
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
    Mutation fitness predictor (~0.5M params, GPU-optimized).

    Architecture:
    - Mutation detection via wildtype comparison
    - Explicit mutation features (wt/mut/delta/position embeddings)
    - Deep residual 1D CNN for sequence context
    - Multi-head attention pooling (learned query)
    - Deep MLP regression head
    - Data augmentation: embedding noise + random position masking
    """

    def __init__(self):
        super().__init__()

        self.aa_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(SEQ_LEN + 1, N_EMBD)

        # CNN: project up then residual blocks
        self.input_proj = nn.Conv1d(N_EMBD, N_CNN_CHANNELS, 1)
        self.cnn_blocks = nn.ModuleList([
            ResidualConvBlock(N_CNN_CHANNELS, DROPOUT)
            for _ in range(N_CNN_LAYERS)
        ])
        self.cnn_norm = nn.BatchNorm1d(N_CNN_CHANNELS)

        # Multi-head attention pooling
        self.mha = nn.MultiheadAttention(
            embed_dim=N_EMBD, num_heads=N_ATTN_HEADS,
            dropout=DROPOUT, batch_first=True,
        )
        self.attn_query = nn.Parameter(torch.randn(1, 1, N_EMBD) * 0.02)

        # Mutation encoder
        self.mutation_encoder = nn.Sequential(
            nn.Linear(4 * N_EMBD, N_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(N_HIDDEN, N_HIDDEN // 4),
        )

        # Combined dim:
        # CNN mean + max: 2 * N_CNN_CHANNELS
        # MHA pooled: N_EMBD
        # Global mean: N_EMBD
        # Mutation: N_HIDDEN // 4
        combined_dim = 2 * N_CNN_CHANNELS + 2 * N_EMBD + N_HIDDEN // 4

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
                nn.init.normal_(module.weight, std=0.05)

    def forward(self, x):
        B, T = x.shape
        seq = x[:, 1:]

        # Mutation detection
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

        # Sequence embedding
        emb = self.aa_emb(seq)
        pos = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        emb = emb + self.pos_emb(pos)

        # Data augmentation during training
        if self.training:
            if EMB_NOISE > 0:
                emb = emb + torch.randn_like(emb) * EMB_NOISE
            if MASK_PROB > 0:
                mask = torch.rand(B, seq.size(1), 1, device=emb.device) > MASK_PROB
                emb = emb * mask

        # Residual CNN
        cnn_in = self.input_proj(emb.permute(0, 2, 1))
        for block in self.cnn_blocks:
            cnn_in = block(cnn_in)
        cnn_in = self.cnn_norm(cnn_in)
        cnn_mean = cnn_in.mean(dim=2)
        cnn_max = cnn_in.max(dim=2).values

        # Multi-head attention pooling
        query = self.attn_query.expand(B, -1, -1)
        attn_out, _ = self.mha(query, emb, emb)
        attn_pooled = attn_out.squeeze(1)

        # Global mean
        global_mean = emb.mean(dim=1)

        # Combine
        combined = torch.cat([
            cnn_mean, cnn_max, attn_pooled, global_mean, mut_encoded
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

    # Use AMP on CUDA for speed
    use_amp = device_type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

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
    print(f"AMP: {use_amp}")
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

            # Frequent evals on GPU (fast eval)
            if step > 10 and step % 400 == 0 and total_training_time < TIME_BUDGET * 0.95:
                model.eval()
                mid_results = evaluate_model(model, device, batch_size=BATCH_SIZE)
                sp = mid_results["val_spearman"]
                if sp > best_val_spearman:
                    best_val_spearman = sp
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    print(f"\n  [ckpt] spearman={sp:.4f} mse={mid_results['val_mse']:.4f} (NEW BEST)")
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
