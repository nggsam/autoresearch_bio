# 🧬 autoresearch_bio

Autonomous discovery of small, high-precision architectures for predicting viral protein fitness — adapting [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for biology.

## What This Does

An autonomous agent iteratively designs, trains, and evaluates neural network architectures to predict how single-point mutations in the **SARS-CoV-2 Spike RBD** affect ACE2 binding affinity. The agent modifies only `train.py`, running 5-minute experiments in a loop, keeping what works and discarding what doesn't.

**Best result: Spearman ρ = 0.464** with a 0.7M parameter model (well under the 10M limit), trained in 5 minutes on a single T4 GPU.

## Architecture

The winning model uses **mutation-aware encoding** — it compares the input sequence to the wildtype to detect the mutation site, then extracts explicit features:

```
Input: tokenized mutant sequence (B, 212)
  ├── Wildtype comparison → mutation site detection
  ├── Mutation features: wt_emb + mut_emb + Δ_emb + pos_emb → MLP
  ├── 4-layer Residual 1D CNN → mean + max pooling
  ├── Multi-head attention pooling (4 heads, learned query)
  └── Global mean pooling
Concat all → LayerNorm → MLP → scalar fitness prediction
```

## Dataset

[Bloom Lab SARS-CoV-2 RBD Deep Mutational Scanning](https://github.com/jbloomlab/SARS-CoV-2-RBD_DMS) — 4,003 single-point mutations across 201 RBD sites, measuring ACE2 binding fitness.

## Quick Start

```bash
# Install dependencies
pip install torch numpy pandas scipy requests matplotlib

# Download data and verify pipeline
python prepare.py

# Train locally (MPS/CPU, 5 minutes)
python train.py

# Train on GPU via Modal (~$0.03/run)
pip install modal
modal setup
modal run modal_runner.py
```

## Project Structure

| File | Role | Agent Modifies? |
|------|------|:-:|
| `prepare.py` | Data download, AA tokenization, train/val split, Spearman ρ evaluation | ❌ |
| `train.py` | Model architecture, optimizer, training loop | ✅ |
| `program.md` | Autonomous research directive (agent reads this) | ❌ |
| `modal_runner.py` | Cloud GPU runner (Modal T4/A10G) | ❌ |
| `results.tsv` | Experiment log | ✅ |
| `reports/` | Experiment reports and analysis | — |

## Results

11 experiments across MPS (local) and T4 GPU (Modal):

| Experiment | Spearman ρ | Params | Device |
|:-----------|:----------:|:------:|:------:|
| Baseline Transformer | -0.035 | 0.8M | MPS |
| CNN + attention hybrid | 0.319 | 0.1M | MPS |
| Mutation-aware (MPS best) | 0.410 | 0.1M | MPS |
| **Mutation-aware + augment (best)** | **0.464** | **0.7M** | **T4 GPU** |

See [`reports/report_small.md`](reports/report_small.md) for the full breakdown.

## Key Findings

1. **Small models beat large models** on small datasets. 0.7M params >> 4M params on 4k samples.
2. **Mutation-aware encoding is essential** — explicitly detecting what changed and where gives +0.1 Spearman over implicit learning.
3. **Checkpointing prevents catastrophic overfitting** — models peak early then collapse. Best-of-run checkpointing saves 0.2–0.4 Spearman.
4. **Total GPU cost: ~$0.10** for all cloud experiments.

## How Autonomous Research Works

The agent follows `program.md` — an infinite loop:

1. Read the current `train.py` and `results.tsv`
2. Hypothesize an improvement (architecture, hyperparameter, loss function, etc.)
3. Edit `train.py` and commit
4. Run `python train.py` (5 minutes)
5. If val_spearman improves → keep. Otherwise → `git revert`.
6. Log to `results.tsv` and repeat.

## License

MIT
