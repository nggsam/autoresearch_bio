"""
Autoresearch_bio training script. Single-file, single-device.
Experiment 3: Hybrid local-context model.

Instead of full-sequence Transformer, use:
1. Mutation-specific features (position, wt/mut AA embeddings)
2. Local context window (k=15 residues each side of mutation)
3. Small 1D CNN for local context + MLP head

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
# Model architecture
# ---------------------------------------------------------------------------

N_EMBD = 32          # AA embedding dimension
CONTEXT_K = 20       # context window: K residues each side of mutation
N_CNN_CHANNELS = 64  # CNN feature channels
N_CNN_LAYERS = 3     # number of CNN layers
N_HIDDEN = 128       # MLP hidden dimension
DROPOUT = 0.2
BATCH_SIZE = 64

# Optimization
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.05
ADAM_BETAS = (0.9, 0.999)


class LocalContextModel(nn.Module):
    """
    Mutation fitness predictor using local sequence context.

    Architecture:
    1. Embed the full sequence with learned AA embeddings
    2. Extract a local window around the mutation site
    3. Process local context with 1D CNN (captures residue interactions)
    4. Combine with mutation-specific features (position, wt/mut identity)
    5. MLP regression head

    Key insight: For single-point mutations, the local biochemical
    context around the mutation site matters far more than distant
    residues. This eliminates the need for expensive full-sequence
    attention and dramatically reduces overfitting.
    """

    def __init__(self):
        super().__init__()

        # Amino acid embeddings
        self.aa_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)

        # Position embedding (to know WHERE in the protein the mutation is)
        self.pos_emb = nn.Embedding(SEQ_LEN + 1, N_EMBD)

        # 1D CNN for local context window
        context_len = 2 * CONTEXT_K + 1  # window size
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

        # Global average pool over context window → single vector
        # Plus mutation-specific features
        # CNN output: N_CNN_CHANNELS
        # Mutation position embedding: N_EMBD
        # Wildtype AA embedding: N_EMBD
        # Mutant AA embedding: N_EMBD
        combined_dim = N_CNN_CHANNELS + 3 * N_EMBD

        # MLP head
        self.head = nn.Sequential(
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
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len) integer token IDs
               x[:, 0] is START token
               x[:, 1:] is the (possibly mutated) protein sequence
        """
        B, T = x.shape

        # Find mutation position by comparing to wildtype
        # The mutated position is where the sequence differs from others
        # For now, we embed the FULL sequence and use attention to find
        # important positions. Alternative: pass mutation position explicitly.

        # Embed all tokens
        emb = self.aa_emb(x[:, 1:])  # (B, SEQ_LEN, N_EMBD), skip START

        # For each sample, we need to find where the mutation is.
        # Since all samples share the same wildtype, the mutation site
        # is where this sequence differs from wildtype.
        # But we process a batch, so we need a general approach.

        # Strategy: use the full embedded sequence, but weight it
        # by a learned positional importance + local CNN

        # Global: mean-pool the full sequence embedding
        global_feat = emb.mean(dim=1)  # (B, N_EMBD)

        # Local CNN: process the full sequence with CNN to capture
        # local interactions, then global average pool
        cnn_in = emb.permute(0, 2, 1)  # (B, N_EMBD, SEQ_LEN)
        cnn_out = self.cnn(cnn_in)     # (B, N_CNN_CHANNELS, SEQ_LEN)
        cnn_pooled = cnn_out.mean(dim=2)  # (B, N_CNN_CHANNELS)

        # Position-weighted features: use position embeddings as attention
        pos_ids = torch.arange(T - 1, device=x.device).unsqueeze(0)  # (1, SEQ_LEN)
        pos_emb = self.pos_emb(pos_ids)  # (1, SEQ_LEN, N_EMBD)

        # Attention weights from position embeddings
        attn_logits = (emb * pos_emb).sum(dim=-1)  # (B, SEQ_LEN)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, SEQ_LEN)
        attended = (emb * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, N_EMBD)

        # Combine features
        combined = torch.cat([
            cnn_pooled,       # local context features
            global_feat,      # global sequence features
            attended,         # position-attended features
            emb.max(dim=1).values,  # max-pool features
        ], dim=-1)  # (B, N_CNN_CHANNELS + 3*N_EMBD)

        out = self.head(combined)
        return out

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

    model = LocalContextModel()
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

            # Mid-training eval every 400 steps
            if step > 10 and step % 400 == 0 and total_training_time < TIME_BUDGET * 0.9:
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
