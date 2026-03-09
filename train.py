"""
Autoresearch_bio training script. Single-file, single-device.
Adapted from karpathy/autoresearch for viral fitness prediction.

The agent modifies THIS FILE to try new architectures, optimizers,
hyperparameters, etc. Everything is fair game.

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
# Model architecture (AGENT: modify this!)
# ---------------------------------------------------------------------------

# Architecture hyperparameters
N_LAYER = 4          # number of transformer layers
N_HEAD = 4           # number of attention heads
N_EMBD = 128         # embedding dimension
DROPOUT = 0.1        # dropout rate
BATCH_SIZE = 64      # training batch size

# Optimization hyperparameters
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1   # fraction of time budget for LR warmup
ADAM_BETAS = (0.9, 0.999)


@dataclass
class BioConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len: int = SEQ_LEN + 1  # +1 for START token
    n_layer: int = N_LAYER
    n_head: int = N_HEAD
    n_embd: int = N_EMBD
    dropout: float = DROPOUT


class BioTransformer(nn.Module):
    """
    Transformer encoder for protein fitness prediction.
    Maps tokenized amino acid sequence → scalar fitness score.

    AGENT: Try swapping this for:
    - 1D dilated convolutions (WaveNet-style)
    - Mamba / S4 state-space model
    - Hybrid CNN + Attention
    - Different positional encodings (RoPE, ALiBi, sinusoidal)
    - Different pooling strategies (CLS token, max-pool, attention-pool)
    """

    def __init__(self, config: BioConfig):
        super().__init__()
        self.config = config

        # Token + positional embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.seq_len, config.n_embd)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=config.n_embd * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layer,
        )

        # Regression head: pool → project → scalar
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len) integer token IDs
        Returns:
            (B, 1) predicted fitness scores
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)

        tok = self.tok_emb(x)       # (B, T, n_embd)
        pos = self.pos_emb(pos)     # (1, T, n_embd)
        h = self.emb_dropout(tok + pos)

        # Transformer encoding
        h = self.transformer(h)     # (B, T, n_embd)

        # Mean pooling over sequence
        h = h.mean(dim=1)           # (B, n_embd)

        # Regression head
        h = self.ln_final(h)
        out = self.head(h)          # (B, 1)
        return out

    def num_params(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)

    t_start = time.time()

    # Build model
    config = BioConfig()
    model = BioTransformer(config)
    model = model.to(device)

    num_params = model.num_params()
    print(f"Model config: {asdict(config)}")
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    if num_params > 10_000_000:
        print(f"WARNING: Model has {num_params/1e6:.1f}M params (limit: 10M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=ADAM_BETAS,
    )

    # Loss function
    criterion = nn.MSELoss()

    # Data
    train_loader = make_dataloader("train", BATCH_SIZE, shuffle=True)

    # LR schedule
    def get_lr(progress):
        """Cosine schedule with linear warmup."""
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        else:
            cosine_progress = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
            return 0.5 * (1.0 + math.cos(math.pi * cosine_progress))

    # Training loop
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Starting training...")
    print()

    t_start_training = time.time()
    total_training_time = 0.0
    step = 0
    epoch = 0
    smooth_loss = 0.0

    model.train()

    while True:
        epoch += 1
        for X_batch, Y_batch in train_loader:
            t0 = time.time()

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Forward
            preds = model(X_batch).squeeze(-1)  # (B,)
            loss = criterion(preds, Y_batch)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Timing
            if device_type == "cuda":
                torch.cuda.synchronize()
            elif device_type == "mps":
                torch.mps.synchronize()

            t1 = time.time()
            dt = t1 - t0

            if step > 5:
                total_training_time += dt

            # LR schedule
            progress = min(total_training_time / TIME_BUDGET, 1.0) if TIME_BUDGET > 0 else 0.0
            lr_mult = get_lr(progress)
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_RATE * lr_mult

            # Logging
            loss_f = loss.item()
            ema_beta = 0.95
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
            debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
            pct_done = 100 * progress
            remaining = max(0, TIME_BUDGET - total_training_time)

            if step % 50 == 0 or step < 5:
                print(
                    f"\rstep {step:05d} ({pct_done:.1f}%) | "
                    f"loss: {debiased_loss:.6f} | "
                    f"lr: {LEARNING_RATE * lr_mult:.2e} | "
                    f"dt: {dt*1000:.0f}ms | "
                    f"epoch: {epoch} | "
                    f"remaining: {remaining:.0f}s    ",
                    end="", flush=True,
                )

            # Fast fail
            if loss_f > 100 or math.isnan(loss_f):
                print(f"\nFAIL: loss={loss_f}")
                exit(1)

            # GC management
            if step == 0:
                gc.collect()

            step += 1

            # Time's up
            if step > 5 and total_training_time >= TIME_BUDGET:
                break

        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()  # newline after \r log
    print(f"Training complete. {step} steps, {epoch} epochs.")
    print()

    # Final evaluation
    print("Running validation evaluation...")
    results = evaluate_model(model, device, batch_size=BATCH_SIZE)

    # Memory stats
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

    # Final summary (autoresearch format)
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
    print(f"n_layer:          {N_LAYER}")
    print(f"n_head:           {N_HEAD}")
    print(f"n_embd:           {N_EMBD}")


if __name__ == "__main__":
    main()
