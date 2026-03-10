# GFP Fluorescence Prediction — Experiment Report

## Problem
**Predict log-fluorescence of GFP variants from amino acid sequence.**

- **Dataset:** TAPE benchmark (Sarkisyan et al. 2016) — ~54k GFP variants
- **Splits:** Train: 21,446 | Val: 5,362 | Test: 27,217
- **Metric:** Spearman ρ (rank correlation)
- **Time budget:** 300s per experiment on T4 GPU
- **Hardware:** NVIDIA T4 GPU via Modal ($0.59/hr)

## Results

| # | Experiment | Spearman ρ | Pearson r | MSE | Params | Novel Technique | Key Finding |
|---|-----------|:----------:|:---------:|:---:|:------:|-----------------|-------------|
| 1 | Baseline CNN+attn | 0.737 | 0.928 | 0.098 | 0.6M | — | Strong baseline |
| **2** | **Scaled-up model** | **0.768** | **0.929** | **0.107** | **2.6M** | **Larger capacity** | **Best overall** |
| 3 | Heavy regularization | 0.411 | 0.775 | 0.296 | 2.6M | dropout=0.25, wd=0.08 | Underfitting |
| 4 | Mixup + multi-scale CNN | 0.760 | 0.931 | 0.107 | 1.3M | Embedding mixup, k=3/5/7 | Moderate improvement lost by capacity reduction |
| 5 | AA properties + ranking | 0.752 | 0.916 | 0.128 | 2.6M | Physicochemical features, ConFit-inspired | Model learns these implicitly |
| 6 | Dilated convolutions | 0.735 | 0.883 | 0.246 | 2.6M | Dilation pyramid 1,2,4,8,4 | Overfits severely |
| 7 | EMA model averaging | 0.763 | 0.925 | 0.125 | 2.6M | EMA decay=0.999 | Slight stabilization, doesn't help peak |

## Context: TAPE Benchmark Baselines

| Model | Spearman ρ | Source |
|-------|:----------:|--------|
| Transformer (TAPE) | 0.68 | Rao et al. 2019 |
| LSTM (TAPE) | 0.67 | Rao et al. 2019 |
| ResNet (TAPE) | 0.21 | Rao et al. 2019 |
| One-Hot (TAPE) | 0.14 | Rao et al. 2019 |
| **Ours (Exp2)** | **0.768** | **This work** |

Our best model beats all published TAPE baselines by a significant margin (+0.088 over Transformer).

## Best Model Architecture (Exp2)

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

**Config:** dropout=0.15, batch=128, lr=3e-4, wd=0.03, Huber loss

## Key Findings

1. **Simple architectures win.** The basic CNN+attention (Exp2) beats all fancy variants (multi-scale, dilated, mixup, ranking loss, AA properties, EMA).

2. **Right-sizing matters more than techniques.** Going from 0.6M → 2.6M by simply increasing embed dim and CNN channels gave the biggest improvement (+0.031).

3. **Overfitting is the main bottleneck.** All 2.6M models overfit after ~25 epochs. The best model peaks early and declines. But aggressive regularization (Exp3) destroys performance.

4. **Model learns AA properties implicitly.** Adding physicochemical features (hydrophobicity, charge, etc.) didn't help — the embedding layer already captures these from data.

5. **EMA helps stability but not peak.** EMA model (0.747) is worse than best checkpoint (0.763) but more stable over training.

6. **Pairwise ranking loss hurts MSE.** Adding a ranking objective improves ordering slightly but adds gradient noise that hurts overall fit.

## Cost Summary

| Item | Cost |
|------|------|
| GPU compute (7 experiments × ~6 min) | ~$0.40 |
| Data download | Free |
| Modal free tier | $30/month |
| **Total** | **~$0.40** |

## What Would Push Higher

- **Pretrained embeddings (ESM-2):** Using frozen ESM-2 embeddings would likely push to ρ > 0.85
- **Larger time budget:** 15-30 min training with learning rate restarts
- **Test-time augmentation:** Ensemble predictions from multiple checkpoints
- **Architecture search:** Systematic hyperparameter sweep (NAS-style)
