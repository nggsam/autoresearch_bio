# GFP Fluorescence Prediction — Experiment Report

## Problem

**Predict log-fluorescence of GFP variants from amino acid sequence.**

- **Dataset:** TAPE benchmark (Sarkisyan et al. 2016) — ~54k GFP variants
- **Splits:** Train: 21,446 | Val: 5,362 | Test: 27,217
- **Metric:** Spearman ρ (rank correlation)
- **Hardware:** NVIDIA T4 GPU via Modal (~$0.06/experiment)

## Results

| # | Experiment | ρ | Params | Budget | What We Learned |
|---|-----------|:---:|:------:|:------:|-----------------|
| 1 | Baseline CNN+attn | 0.737 | 0.6M | 300s | Strong starting point |
| 2 | Scaled-up model | 0.768 | 2.6M | 300s | Right-sizing matters most |
| 3 | Heavy regularization | 0.411 | 2.6M | 300s | Over-regularization destroys signal |
| 4 | Mixup + multi-scale CNN | 0.760 | 1.3M | 300s | Reduced capacity hurt more than augmentation helped |
| 5 | AA properties + ranking loss | 0.752 | 2.6M | 300s | Model learns these implicitly |
| 6 | Dilated convolutions | 0.735 | 2.6M | 300s | Severe overfitting |
| 7 | EMA averaging | 0.763 | 2.6M | 300s | Stabilizes but doesn't improve peak |
| 8b | AA substitution augmentation | 0.539 | 2.6M | 300s | GFP hypersensitive to even "conservative" mutations |
| 9 | Checkpoint ensemble (300s) | 0.782 | 2.6M | 300s | Averaging top-5 helps +0.004 |
| 10 | Transformer-only | 0.449 | 1.0M | 300s | Local motifs matter — CNNs essential |
| 11 | Double compute (600s) | 0.795 | 2.6M | 600s | More compute > fancy tricks |
| 12 | Wider model (6.6M) | 0.779 | 6.6M | 600s | Too large for 600s budget |
| **13** | **Warm restart + ensemble** | **0.796** | **2.6M** | **600s** | **Diverse checkpoints improve ensemble** |

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

## Novelty Assessment

### Not novel: absolute performance

Our best (**ρ = 0.796**) beats the TAPE baselines published in 2019:

| Model | ρ | Year |
|-------|:---:|:----:|
| TAPE Transformer | 0.68 | 2019 |
| TAPE LSTM | 0.67 | 2019 |
| TAPE ResNet | 0.21 | 2019 |
| **Ours** | **0.796** | **2026** |

But comparing against 7-year-old baselines is misleading. Modern protein language models (ESM-2, ProtTrans, FSFP) use billions of parameters pre-trained on millions of protein sequences and almost certainly exceed our results. We trained from scratch on 21k sequences.

### Not novel: architecture or techniques

CNN + attention, checkpoint ensembles, cosine warm restarts, EMA, Huber loss — all well-established. Nothing in our pipeline is new.

### What IS valuable

**1. Rapid iteration methodology.** 13 GPU experiments for ~$1 total in one afternoon. The all-in-one Modal pattern (embedding code directly in the remote function) enables true rapid iteration.

**2. Systematic negative results.** We quantified exactly how much each technique helps or hurts:
- AA substitution at 8% → -0.23 (catastrophic for GFP)
- Pure Transformer → -0.35 vs CNN+attention
- EMA, SWA, ranking loss → all slightly worse than vanilla training
- Model too large (6.6M vs 2.6M) → -0.016

**3. Scaling law for this problem.**
```
0.6M params, 300s  → 0.737
2.6M params, 300s  → 0.768  (+0.031 from params)
2.6M params, 600s  → 0.796  (+0.028 from compute)
6.6M params, 600s  → 0.779  (too large, underfits)
```

## Key Findings

1. **Simple architectures win.** Every "clever" technique performed worse than the basic CNN+attention.
2. **Right-sizing > fancy techniques.** The two biggest gains were scaling params (0.6M→2.6M: +0.031) and doubling compute (300s→600s: +0.028).
3. **GFP is mutation-sensitive.** Even "conservative" AA substitution augmentation (D↔E, K↔R) destroyed performance.
4. **CNN is critical for proteins.** Pure Transformer (0.449) can't match CNN+attention (0.796) — local motif patterns matter.
5. **Checkpoint ensemble helps when diverse.** Warm restarts create diverse checkpoints for better ensemble quality.

## What Would Be Actually Novel

1. **Beat PLM-based SOTA** — Fine-tune ESM-2 and achieve ρ > 0.85+
2. **Tackle unstudied problems** — Apply methodology to datasets without established baselines
3. **Discover a new technique** — e.g., protein-specific augmentation based on evolutionary conservation
4. **Demonstrate methodology on a hard problem** — Show rapid iteration can match months of manual experimentation

## Cost Summary

| Item | Amount |
|------|--------|
| 13 GFP experiments (~70 min GPU) | ~$0.70 |
| Earlier RBD DMS + AMP experiments | ~$0.30 |
| **Total project cost** | **~$1.00** |

## Bottom Line

This was a **methodology demo and learning exercise**, not a research contribution. We built a fast, cheap experiment pipeline and systematically explored a well-studied benchmark. The results are solid for from-scratch models but don't advance the state of the art. To achieve novelty, we'd need to either use pre-trained models or tackle less-explored problems.
