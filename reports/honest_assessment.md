# Autoresearch Bio — Honest Assessment

## What We Set Out To Do

Use an "autoresearch" methodology — rapid GPU experimentation via Modal — to iterate on protein fitness prediction problems. We chose the **TAPE GFP fluorescence prediction** benchmark (Sarkisyan et al. 2016) as our primary problem: predict log-fluorescence of ~54k GFP variants from amino acid sequence.

## What We Built

A pipeline for running from-scratch deep learning experiments on T4 GPUs at ~$0.06/experiment with ~6 minute turnaround. Over two rounds, we ran **13 experiments** testing different architectures, augmentation strategies, optimizers, and training schedules.

### Results Table

| # | Experiment | ρ | What We Learned |
|---|-----------|:---:|----------------|
| 1 | Baseline CNN+attn (0.6M) | 0.737 | Strong starting point |
| 2 | Scaled-up (2.6M) | 0.768 | Right-sizing matters most |
| 3 | Heavy regularization | 0.411 | Over-regularization destroys signal |
| 4 | Mixup + multi-scale CNN | 0.760 | Reduced capacity hurt more than augmentation helped |
| 5 | AA properties + ranking loss | 0.752 | Model learns these implicitly |
| 6 | Dilated convolutions | 0.735 | Severe overfitting |
| 7 | EMA averaging | 0.763 | Stabilizes but doesn't improve peak |
| 8b | AA substitution augmentation | 0.539 | GFP is hypersensitive to even "conservative" mutations |
| 9 | Checkpoint ensemble (300s) | 0.782 | Averaging top-5 helps +0.004 |
| 10 | Transformer-only | 0.449 | Local motifs matter — CNNs are essential |
| 11 | Double compute (600s) | 0.795 | More compute > fancy tricks |
| 12 | Wider model (6.6M, 600s) | 0.779 | Too large for 600s budget |
| **13** | **Warm restart + ensemble (600s)** | **0.796** | **Diverse checkpoints improve ensemble** |

## Is This Novel?

### No — in absolute performance

Our best (**ρ = 0.796**) beats the TAPE baselines published in 2019:

| Model | ρ | Year |
|-------|:---:|:----:|
| TAPE Transformer | 0.68 | 2019 |
| TAPE LSTM | 0.67 | 2019 |
| TAPE ResNet | 0.21 | 2019 |
| **Ours** | **0.796** | **2026** |

But comparing against 7-year-old baselines is misleading. Modern protein language models (ESM-2, ProtTrans, FSFP) use **billions of parameters pre-trained on millions of protein sequences** and almost certainly exceed our results on this exact benchmark. We trained from scratch on 21k sequences — a fundamentally different (and much easier) setting for pre-trained models to dominate.

### No — in architecture or techniques

CNN + attention, checkpoint ensembles, cosine warm restarts, EMA, Huber loss — all are well-established techniques. Nothing in our model architecture or training pipeline is new.

### No — in findings

"Simple models beat complex ones" and "more compute helps" are known. Our specific negative results (AA substitution hurts GFP, Transformer < CNN for short protein sequences) are interesting observations but not publishable findings on their own.

## What IS Valuable

### 1. The rapid iteration methodology
Running 13 GPU experiments for **~$1 total** in one afternoon is genuinely useful for prototyping. The all-in-one Modal pattern (embedding code directly in the remote function) eliminates image rebuild overhead and enables true rapid iteration.

### 2. Systematic negative results
We quantified exactly how much each technique helps or hurts on a well-studied benchmark:
- **AA substitution at 8%** → -0.23 (catastrophic for GFP)
- **Pure Transformer** → -0.35 vs CNN+attention (local patterns are critical)
- **EMA, SWA, ranking loss** → all slightly worse than vanilla training
- **Model too large (6.6M vs 2.6M)** → -0.016 (underfits in fixed time budget)

### 3. The ensemble + warm restart insight
Warm restarts create **diverse checkpoints** that improve ensemble quality beyond simple top-K selection from a single cosine schedule (+0.001 from Exp11→Exp13). Small but consistent.

### 4. The scaling law for this problem
```
0.6M params, 300s  → 0.737
2.6M params, 300s  → 0.768  (+0.031 from params)
2.6M params, 600s  → 0.796  (+0.028 from compute)
6.6M params, 600s  → 0.779  (too large, underfits)
```
The optimal model size scales with compute budget. With 600s, 2.6M is the sweet spot.

## What Would Be Actually Novel

1. **Beat PLM-based SOTA** — Fine-tune ESM-2 embeddings and achieve ρ > 0.85+ (then compare against published PLM results fairly)
2. **Tackle unstudied problems** — Apply the methodology to protein fitness datasets without established deep learning baselines
3. **Discover a new technique** — e.g., a protein-specific augmentation that actually works (our AA substitution failed, but a more principled approach based on evolutionary conservation might work)
4. **Demonstrate the methodology on a hard problem** — Show that rapid iteration can match months of manual experimentation

## Cost Summary

| Item | Amount |
|------|--------|
| 13 GFP experiments (~70 min GPU) | ~$0.70 |
| Earlier RBD DMS + AMP experiments | ~$0.30 |
| **Total project cost** | **~$1.00** |

## Bottom Line

This was a **methodology demo and learning exercise**, not a research contribution. We built a fast, cheap experiment pipeline and systematically explored a well-studied benchmark. The results are solid for from-scratch models but don't advance the state of the art. The honest takeaway: the autoresearch approach works well for rapid prototyping, but to achieve novelty we'd need to either use pre-trained models or tackle less-explored problems.
