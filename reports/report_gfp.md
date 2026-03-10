# GFP Fluorescence Prediction — Experiment Report

## Problem
**Predict log-fluorescence of GFP variants from amino acid sequence.**

- **Dataset:** TAPE benchmark (Sarkisyan et al. 2016) — ~54k GFP variants
- **Splits:** Train: 21,446 | Val: 5,362 | Test: 27,217
- **Metric:** Spearman ρ (rank correlation)
- **Hardware:** NVIDIA T4 GPU via Modal

## Results Summary

| # | Experiment | Spearman ρ | Params | Budget | Key Idea |
|---|-----------|:----------:|:------:|:------:|----------|
| 1 | Baseline CNN+attn | 0.737 | 0.6M | 300s | Strong baseline |
| 2 | Scaled-up model | 0.768 | 2.6M | 300s | Larger capacity |
| 3 | Heavy regularization | 0.411 | 2.6M | 300s | dropout=0.25, wd=0.08 ❌ |
| 4 | Mixup + multi-scale CNN | 0.760 | 1.3M | 300s | k=3/5/7 CNN kernels |
| 5 | AA properties + ranking | 0.752 | 2.6M | 300s | ConFit-inspired |
| 6 | Dilated convolutions | 0.735 | 2.6M | 300s | Dilation 1,2,4,8,4 |
| 7 | EMA averaging | 0.763 | 2.6M | 300s | EMA decay=0.999 |
| 8b | AA substitution | 0.539 | 2.6M | 300s | Conservative subs ❌ |
| 9 | Ensemble (5 ckpts) | 0.782 | 2.6M | 300s | Top-5 checkpoint avg |
| 10 | Transformer-only | 0.449 | 1.0M | 300s | No CNN ❌ |
| 11 | Double compute | 0.795 | 2.6M | 600s | More training time |
| 12 | Wider model | 0.779 | 6.6M | 600s | Too large for 600s |
| **13** | **Warm restart + ensemble** | **0.796** | **2.6M** | **600s** | **Restart at 50%, 7 ckpts** ✅ |

## Context: TAPE Benchmark Baselines

| Model | Spearman ρ | Source |
|-------|:----------:|--------|
| Transformer (TAPE) | 0.68 | Rao et al. 2019 |
| LSTM (TAPE) | 0.67 | Rao et al. 2019 |
| ResNet (TAPE) | 0.21 | Rao et al. 2019 |
| **Ours (Exp13)** | **0.796** | **This work** |

**We beat all TAPE baselines by +0.116 over the best published result.**

## Best Model Architecture

```
GFPPredictor (2.6M params)
├── AA Embedding: 22 → 128
├── Positional Embedding: 241 → 128
├── CNN: 5 × ResidualConvBlock(256 channels, k=3)
│   └── BatchNorm → Conv1d → GELU → Conv1d → Residual → GELU
├── Multi-head Attention: 8 heads, learned query pooling
├── Pooling: CNN(mean+max) + Attention + Global Mean
└── MLP Head: 768 → 512 → 128 → 1
```

**Config:** dropout=0.15, batch=128, lr=3e-4, wd=0.03, Huber loss, 600s, warm restart at 50%

## Key Findings

1. **Simple architectures win.** Every "clever" technique (mixup, ranking loss, dilated conv, EMA, Transformer) performed worse than the basic CNN+attention.

2. **Right-sizing > fancy techniques.** Scaling from 0.6M→2.6M (+0.031) and doubling compute (+0.013) were the biggest improvements. Going to 6.6M hurt — too large for 600s.

3. **Checkpoint ensemble helps when diverse.** Ensemble of 5 similar checkpoints: +0.004. Warm restart creates diverse checkpoints: +0.001 more.

4. **GFP is mutation-sensitive.** AA substitution augmentation (D↔E, K↔R) at 8% destroyed performance (0.539). The protein's function depends on exact residue identity.

5. **CNN is critical.** Pure Transformer (0.449) can't match CNN+attention (0.796). Local motif patterns matter more than global attention for protein fitness.

6. **Overfitting is the bottleneck.** Model peaks at ~60-70% of training then degrades. More training time helps because it allows more checkpointing and slightly higher peaks.

## Cost Summary

| Item | Cost |
|------|------|
| GPU compute (13 experiments, ~70 min total) | ~$0.70 |
| Total Modal spend | ~$1.00 |
