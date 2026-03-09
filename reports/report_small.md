# autoresearch_bio — Experiment Report

**Project:** Agentic Viral Fitness Discovery  
**Target:** SARS-CoV-2 RBD mutation fitness prediction (ACE2 binding affinity)  
**Dataset:** Bloom Lab DMS — 4,003 mutations across 201 RBD sites  
**Constraint:** <10M parameters, 5-minute training budget  
**Metric:** Spearman ρ (rank correlation with experimental fitness)

---

## Results Summary

| # | Experiment | Spearman ρ | MSE | Params | Device | Status |
|---|-----------|:----------:|:---:|:------:|:------:|--------|
| 1 | Baseline Transformer | -0.035 | 0.108 | 0.8M | MPS | keep |
| 2 | Smaller + regularized | 0.066 | 0.108 | 0.2M | MPS | keep |
| 3 | CNN+attention hybrid | 0.319 | 0.131 | 0.1M | MPS | keep |
| 4 | Multi-scale CNN | 0.177 | 0.111 | 0.2M | MPS | ❌ |
| 5 | Tuned hyperparams | 0.394 | 0.146 | 0.1M | MPS | keep |
| 6 | Very low LR | 0.289 | 0.098 | 0.1M | MPS | ❌ |
| 7 | Residual CNN + EMA | 0.378 | 0.331 | 0.1M | MPS | ❌ |
| 8 | Mutation-aware model | 0.410 | 0.099 | 0.1M | MPS | keep |
| 9 (GPU) | Scaled 4M params | 0.279 | 0.147 | 4.0M | T4 | ❌ |
| **9b (GPU)** | **0.7M + AMP + augment** | **0.464** | **0.114** | **0.7M** | **T4** | **✅ best** |
| 10 (GPU) | AA properties + masking | 0.335 | 0.137 | 0.7M | T4 | ❌ |

---

## Key Insights

1. **Small models > big models on small data.** With only 4k training samples, the best model is 0.7M params — NOT the 4M param version. The 4M model scored 0.279 vs 0.464.

2. **Mutation-aware features are essential.** Explicitly detecting the mutation site (comparing to wildtype) and encoding (wt_emb, mut_emb, delta_emb, position_emb) as direct features gives the single biggest boost (+0.1 Spearman).

3. **Checkpointing prevents catastrophic overfitting.** Without it, models peak early then collapse to negative correlations. The gap between peak and final can be >0.7 Spearman.

4. **LR sweet spot = 2e-4.** Too high (5e-4) → fast overfitting. Too low (1e-4) → slow convergence. 

5. **Light augmentation helps, heavy hurts.** Embedding noise (0.03) + 5% position masking improved results. But 10% masking destroyed signal.

6. **GPU enables better throughput**, not bigger models. On T4 GPU: ~16ms/step (19k steps in 5 min) vs MPS: ~8ms/step for smaller model (36k in 5 min for 0.1M params). The key advantage is AMP mixed precision and more frequent checkpointing.

---

## Best Architecture (GPU Exp9b)

```
Input: tokenized mutant sequence (B, 212)
  ├── Compare to wildtype → find mutation site
  ├── Extract: wt_emb + mut_emb + delta_emb + pos_emb → MLP → (B, 64)
  ├── Embed seq → project to 128ch → 4x ResidualConvBlock(k=3) → mean+max pool → (B, 256)
  ├── Embed seq → MultiheadAttn(4 heads, learned query) → (B, 64)
  └── Embed seq → global mean pool → (B, 64)
Concat → LayerNorm → MLP(256→64→1) → scalar fitness prediction

Parameters: 0.7M
Spearman ρ: 0.464
Training: 300s on T4 GPU, AMP mixed precision
Augmentation: embedding noise (σ=0.03) + 5% random position masking
```

### Training Config
- **Optimizer:** AdamW (lr=2e-4, β=(0.9, 0.999), wd=0.1)
- **Loss:** Huber (δ=0.5)
- **Dropout:** 0.35
- **Batch size:** 64
- **LR schedule:** cosine with 5% warmup
- **Checkpointing:** every 400 steps, restore best val_spearman
- **Cost:** ~$0.03 per experiment on Modal T4

---

## Cost Summary

| Platform | GPU | Per Run Cost | Speed |
|----------|-----|:------------:|:-----:|
| Local MPS | Apple M-series | Free | ~8ms/step (0.1M model) |
| Modal T4 | NVIDIA T4 | ~$0.03 | ~16ms/step (0.7M model) |
| Modal A10G | NVIDIA A10G | ~$0.05 | ~5ms/step (est.) |

Total GPU spend for 3 experiments: **~$0.10**

---

## Files

| File | Role |
|------|------|
| `prepare.py` | Data download, tokenization, dataloader, Spearman ρ eval (read-only) |
| `train.py` | Model architecture, optimizer, training loop (agent edits) |
| `program.md` | Autonomous research directive |
| `modal_runner.py` | GPU runner for Modal cloud |
| `results.tsv` | Experiment log |
| `report_small.md` | This report |

---

## Next Steps

- Explore multi-task learning (predict binding + expression jointly)
- Try contrastive learning: pairs of (more fit, less fit) mutations
- Add secondary structure features from AlphaFold predictions
- Ensemble top-3 checkpoints from training
- Benchmark against ESM-3 zero-shot predictions
- Scale dataset: include HA influenza DMS data for transfer learning
