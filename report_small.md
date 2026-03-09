# autoresearch_bio — Experiment Report

**Project:** Agentic Viral Fitness Discovery  
**Target:** SARS-CoV-2 RBD mutation fitness prediction (ACE2 binding affinity)  
**Dataset:** Bloom Lab DMS — 4,003 mutations across 201 RBD sites  
**Constraint:** <10M parameters, 5-minute training budget  
**Metric:** Spearman ρ (rank correlation with experimental fitness)

---

## Results Summary

| # | Experiment | Spearman ρ | MSE | Params | Status | Key Change |
|---|-----------|:----------:|:---:|:------:|--------|------------|
| 1 | Baseline Transformer | -0.035 | 0.108 | 0.8M | keep | 4L/4H/128D, AdamW lr=1e-3 |
| 2 | Smaller + regularized | 0.066 | 0.108 | 0.2M | keep | 64D, dropout=0.3, Huber loss |
| 3 | CNN+attention hybrid | 0.319 | 0.131 | 0.1M | keep | 1D CNN + position-attention + multi-pool |
| 4 | Multi-scale CNN | 0.177 | 0.111 | 0.2M | ❌ discard | Early stopping too aggressive |
| 5 | Exp3 + tuned hyperparams | 0.394 | 0.146 | 0.1M | keep | LR=2e-4, wd=0.1, dropout=0.3 |
| 6 | Very low LR | 0.289 | 0.098 | 0.1M | ❌ discard | LR=1e-4 too slow to learn |
| 7 | Residual CNN + EMA | 0.378 | 0.331 | 0.1M | ❌ discard | EMA averaged over overfitting |
| **8** | **Mutation-aware model** | **0.410** | **0.099** | **0.1M** | **✅ best** | **Explicit wt/mut delta + position features** |

---

## Key Insights

1. **Transformers overfit badly** on small DMS datasets (4k samples). Simpler architectures (CNN+attention) are far better.
2. **Checkpointing is critical** — models peak early then collapse from overfitting. Best-model checkpointing saves 0.2–0.4 Spearman ρ.
3. **Explicit mutation features are the right inductive bias**: comparing input to wildtype, extracting (wt_emb, mut_emb, delta, position) as direct features gives the model what it needs without having to learn it implicitly.
4. **LR sweet spot is 2e-4** — 5e-4 overfits too fast, 1e-4 learns too slow.

---

## Best Architecture (Exp8)

```
Input: tokenized mutant sequence (B, 212)
  ├── Compare to wildtype → find mutation site
  ├── Extract: wt_emb, mut_emb, delta_emb, pos_emb → MLP → (B, 64)
  ├── Embed full seq → 3-layer 1D CNN → mean-pool + max-pool → (B, 128)
  ├── Embed full seq → attention-weighted pool → (B, 32)
  └── Embed full seq → global mean pool → (B, 32)
Concat all → LayerNorm → MLP → scalar fitness prediction

Parameters: 0.1M (well under 10M limit)
Spearman ρ: 0.410
```

### Training Config
- **Optimizer:** AdamW (lr=2e-4, β=(0.9, 0.999), wd=0.1)
- **Loss:** Huber (δ=0.5)
- **Dropout:** 0.3
- **Batch size:** 64
- **LR schedule:** cosine with 5% warmup
- **Checkpointing:** every 600 steps, restore best val_spearman

---

## Project Structure

| File | Role | Modified by Agent? |
|------|------|--------------------|
| `prepare.py` | Data download, tokenization, dataloader, Spearman ρ eval | ❌ Read-only |
| `train.py` | Model architecture, optimizer, training loop | ✅ Agent edits |
| `program.md` | Autonomous research directive | ❌ Human edits |
| `results.tsv` | Experiment log (tab-separated) | ✅ Agent appends |

---

## Dataset

- **Source:** Bloom Lab (`jbloomlab/SARS-CoV-2-RBD_DMS`)
- **Target:** ACE2 binding fitness (`bind_avg`)
- **Total mutations:** 4,003
- **Unique sites:** 201
- **Train/Val split:** 80/20 by site (no data leakage)
- **Binding score range:** [-4.84, 0.30]

---

## Next Steps

- Scale up model toward 10M param budget (currently only 0.1M — 100× headroom)
- Try multi-task learning (predict both binding + expression)
- Add physicochemical AA features (hydrophobicity, charge, volume)
- Run on GPU via Modal for faster iteration cycles
- Benchmark against ESM-3 zero-shot predictions on same dataset
