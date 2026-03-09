# autoresearch_bio

This is an experiment to have an LLM do its own research on viral protein fitness prediction.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch_bio/` contains the DMS dataset. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The Problem

You are predicting the **fitness impact of single-point mutations** on the SARS-CoV-2 Spike receptor-binding domain (RBD). The dataset comes from the Bloom Lab's deep mutational scanning experiments, where every possible single amino acid change was measured for its effect on ACE2 binding affinity.

**Input**: Tokenized amino acid sequence with one mutation applied (shape: `(B, SEQ_LEN+1)`)
**Output**: Scalar fitness score (normalized ACE2 binding affinity)
**Vocab**: 22 tokens (20 amino acids + padding + start)
**Sequence length**: ~202 tokens (201 RBD residues + START)

## Experimentation

Each experiment runs on a single device (GPU or Apple MPS). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, pooling strategy, loss function, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies beyond what's in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_model` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_spearman** (Spearman rank correlation between predicted and experimental fitness). Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes.

**Parameter budget**: Stay under **10 million parameters**. This constraint forces you to find efficient architectures — the kind that could run in real-time inside a Virtual Cell simulation.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Research Directions

Here are some promising directions to explore (but don't limit yourself to these):

1. **Positional Encoding**: Compare RoPE vs ALiBi vs sinusoidal vs learnable for protein sequences. Proteins have strong local structure — local position may matter more than for text.

2. **Attention Patterns**: Test sliding window attention (captures local residue interactions), dilated attention, or learned sparse patterns. The RBD has known functional regions (receptor-binding motif) where local context is critical.

3. **Architecture Alternatives**:
   - 1D dilated convolutions (WaveNet-style) — great for capturing multi-scale sequence patterns
   - Mamba/S4 state-space models — efficient for long sequences
   - Hybrid CNN + Attention — local features via CNN, global context via attention
   - Simple MLP baseline — sometimes simpler is better for small datasets

4. **Embedding Strategies**: 
   - One-hot + learned embeddings (current default)
   - Physicochemical property features (hydrophobicity, charge, size)
   - Relative mutation encoding (encode the CHANGE, not just the mutant sequence)

5. **Optimizers**: Iterate on AdamW settings, try Muon (for matrix params), experiment with different warmup/cooldown schedules.

6. **Loss Functions**: Try Huber loss (robust to outliers), ranking losses (directly optimize Spearman), or combined MSE + ranking loss.

7. **Data Augmentation**: Try reverse-complement-like augmentations, noise injection, or multi-task learning (predict both binding and expression).

8. **Pooling Strategies**: Compare mean-pool vs CLS token vs attention-weighted pooling vs max-pool.

## Output format
Once the script finishes it prints a summary like this:

```
---
val_spearman:     0.650000
val_mse:          0.045000
val_pearson:      0.680000
val_samples:      750
training_seconds: 180.1
total_seconds:    195.0
peak_memory_mb:   1024.0
num_steps:        1500
num_params_M:     2.1
n_layer:          4
n_head:           4
n_embd:           128
```

You can extract the key metric from the log file:

```
grep "^val_spearman:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	val_spearman	val_mse	memory_mb	status	description
```

1. git commit hash (short, 7 chars)
2. val_spearman achieved (e.g. 0.650000) — use 0.000000 for crashes
3. val_mse (e.g. 0.045000) — use 0.000000 for crashes
4. peak memory in MB (round to .1f) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_spearman	val_mse	memory_mb	status	description
a1b2c3d	0.550000	0.082000	512.0	keep	baseline (4L 4H 128D transformer)
b2c3d4e	0.620000	0.065000	520.0	keep	increase depth to 6 layers
c3d4e5f	0.580000	0.070000	510.0	discard	switch to 1D CNN (worse)
d4e5f6g	0.000000	0.000000	0.0	crash	mamba architecture (import error)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar9`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly editing the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_spearman:\|^val_mse:\|^peak_memory_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_spearman improved (higher), you "advance" the branch, keeping the git commit
9. If val_spearman is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, etc.), use your judgment: if it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
