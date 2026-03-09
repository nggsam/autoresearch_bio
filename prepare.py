"""
Immutable data layer for autoresearch_bio.
Downloads Bloom Lab SARS-CoV-2 RBD deep mutational scanning data,
provides tokenization, dataloading, and evaluation utilities.

DO NOT MODIFY THIS FILE. The agent only modifies train.py.

Usage:
    python prepare.py               # download data + verify
    python prepare.py --stats       # print dataset statistics
"""

import os
import sys
import math
import argparse
import hashlib

import numpy as np
import pandas as pd
import requests
import torch
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Constants (DO NOT MODIFY)
# ---------------------------------------------------------------------------

# Amino acid vocabulary
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids
AA_TO_IDX = {aa: i + 2 for i, aa in enumerate(AA_LIST)}  # 2..21
PAD_TOKEN = 0
START_TOKEN = 1
VOCAB_SIZE = 22  # 20 AAs + PAD + START

# SARS-CoV-2 Spike RBD wildtype sequence (residues 331-531, Wuhan-Hu-1)
# This is the exact region covered by the Bloom Lab DMS
WILDTYPE_RBD = (
    "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSP"
    "TKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNS"
    "NNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQ"
    "SYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
).upper()

SEQ_LEN = len(WILDTYPE_RBD)  # ~201 residues
TIME_BUDGET = 300  # training time budget in seconds (5 minutes)

# Data source
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_bio")
DATA_DIR = os.path.join(CACHE_DIR, "data")
DATA_URL = (
    "https://media.githubusercontent.com/media/jbloomlab/SARS-CoV-2-RBD_DMS/"
    "master/results/single_mut_effects/single_mut_effects.csv"
)
DATA_FILENAME = "single_mut_effects.csv"

# Evaluation
RANDOM_SEED = 42
VAL_FRACTION = 0.2  # 20% validation split


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_dms_data():
    """Download Bloom Lab single mutation effects CSV. Idempotent."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, DATA_FILENAME)

    if os.path.exists(filepath):
        print(f"Data: already downloaded at {filepath}")
        return filepath

    print(f"Data: downloading from {DATA_URL}...")
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(DATA_URL, timeout=60)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Data: saved to {filepath} ({len(response.content)} bytes)")
            return filepath
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
    print("ERROR: Failed to download DMS data.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading and processing
# ---------------------------------------------------------------------------

def load_dms_dataset():
    """
    Load and process the DMS dataset.

    Returns a DataFrame with columns:
        - site: mutation position (1-indexed in original protein)
        - wildtype: wildtype amino acid at that position
        - mutation: mutant amino acid
        - site_idx: 0-indexed position in the RBD sequence
        - bind_avg: ACE2 binding fitness score (primary target)
        - expr_avg: RBD expression score (secondary target)
    """
    filepath = os.path.join(DATA_DIR, DATA_FILENAME)
    if not os.path.exists(filepath):
        filepath = download_dms_data()

    df = pd.read_csv(filepath)

    # The Bloom Lab CSV columns:
    # site_RBD, site_SARS2, wildtype, mutant, mutation, mutation_RBD,
    # bind_lib1, bind_lib2, bind_avg, expr_lib1, expr_lib2, expr_avg
    # - "mutant" = single-letter AA (e.g. "A")
    # - "mutation" = full mutation string (e.g. "N331A")
    # Rename to our canonical names
    rename_map = {}
    if "site_SARS2" in df.columns and "site" not in df.columns:
        rename_map["site_SARS2"] = "site"
    if "mutant" in df.columns:
        rename_map["mutant"] = "mut_aa"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Filter: only single AA mutations, drop stops and synonymous
    df = df[df["mut_aa"].isin(AA_LIST)].copy()
    df = df[df["wildtype"].isin(AA_LIST)].copy()

    # Drop rows with NaN fitness values
    df = df.dropna(subset=["bind_avg"])

    # Compute site_idx: 0-indexed position within RBD
    min_site = df["site"].min()
    df["site_idx"] = df["site"] - min_site

    # Filter to positions that exist in our RBD sequence
    df = df[df["site_idx"] < SEQ_LEN].copy()

    # Normalize binding scores to roughly [-1, 1] range
    bind_mean = df["bind_avg"].mean()
    bind_std = df["bind_avg"].std()
    df["bind_norm"] = (df["bind_avg"] - bind_mean) / (bind_std + 1e-8)
    df["bind_norm"] = df["bind_norm"].clip(-3, 3) / 3  # clip and rescale

    df = df.reset_index(drop=True)
    return df


def encode_sequence(wt_seq, position, mutant_aa):
    """
    Encode a mutated protein sequence as integer tokens.

    Args:
        wt_seq: wildtype sequence string
        position: 0-indexed mutation position
        mutant_aa: single-letter amino acid code for the mutant

    Returns:
        torch.LongTensor of shape (SEQ_LEN,) with START prepended
    """
    tokens = [START_TOKEN]
    for i, aa in enumerate(wt_seq):
        if i == position:
            tokens.append(AA_TO_IDX.get(mutant_aa, PAD_TOKEN))
        else:
            tokens.append(AA_TO_IDX.get(aa, PAD_TOKEN))

    # Pad or truncate to SEQ_LEN + 1 (for START token)
    target_len = SEQ_LEN + 1
    if len(tokens) < target_len:
        tokens.extend([PAD_TOKEN] * (target_len - len(tokens)))
    tokens = tokens[:target_len]

    return torch.tensor(tokens, dtype=torch.long)


# ---------------------------------------------------------------------------
# Train/val split and dataloading
# ---------------------------------------------------------------------------

def _split_dataset(df):
    """Deterministic 80/20 train/val split by site (no data leakage)."""
    rng = np.random.RandomState(RANDOM_SEED)
    sites = sorted(df["site"].unique())
    rng.shuffle(sites)
    n_val = max(1, int(len(sites) * VAL_FRACTION))
    val_sites = set(sites[:n_val])
    train_sites = set(sites[n_val:])

    train_df = df[df["site"].isin(train_sites)].reset_index(drop=True)
    val_df = df[df["site"].isin(val_sites)].reset_index(drop=True)
    return train_df, val_df


def _df_to_tensors(df):
    """Convert DataFrame to (X, Y) tensors."""
    X_list = []
    Y_list = []
    for _, row in df.iterrows():
        x = encode_sequence(WILDTYPE_RBD, int(row["site_idx"]), row["mut_aa"])
        y = float(row["bind_norm"])
        X_list.append(x)
        Y_list.append(y)

    X = torch.stack(X_list)  # (N, SEQ_LEN+1)
    Y = torch.tensor(Y_list, dtype=torch.float32)  # (N,)
    return X, Y


class DMSDataset:
    """Simple dataset wrapper for DMS mutation data."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_dataloader(split, batch_size, shuffle=None):
    """
    Create a DataLoader for the DMS dataset.

    Args:
        split: "train" or "val"
        batch_size: batch size
        shuffle: whether to shuffle (default: True for train, False for val)

    Returns:
        torch.utils.data.DataLoader
    """
    assert split in ["train", "val"], f"Invalid split: {split}"

    df = load_dms_dataset()
    train_df, val_df = _split_dataset(df)

    if split == "train":
        X, Y = _df_to_tensors(train_df)
        if shuffle is None:
            shuffle = True
    else:
        X, Y = _df_to_tensors(val_df)
        if shuffle is None:
            shuffle = False

    dataset = DMSDataset(X, Y)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=(split == "train"),
        num_workers=0,
        pin_memory=True,
    )
    return loader


# ---------------------------------------------------------------------------
# Evaluation (ground truth — DO NOT MODIFY)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, device, batch_size=64):
    """
    Evaluate model on the validation set.

    Returns dict with:
        - val_spearman: Spearman rank correlation (primary metric, higher=better)
        - val_mse: Mean squared error (secondary metric, lower=better)
        - val_pearson: Pearson correlation
        - n_samples: number of validation samples
    """
    model.eval()
    val_loader = make_dataloader("val", batch_size, shuffle=False)

    all_preds = []
    all_targets = []

    for X_batch, Y_batch in val_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).squeeze(-1)  # (B,)
        all_preds.append(preds.cpu())
        all_targets.append(Y_batch)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Spearman rank correlation (primary metric)
    spearman_r, spearman_p = scipy_stats.spearmanr(preds, targets)

    # Pearson correlation
    pearson_r, pearson_p = scipy_stats.pearsonr(preds, targets)

    # MSE
    mse = float(np.mean((preds - targets) ** 2))

    model.train()

    return {
        "val_spearman": float(spearman_r),
        "val_mse": mse,
        "val_pearson": float(pearson_r),
        "n_samples": len(targets),
    }


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def print_dataset_stats():
    """Print summary statistics of the DMS dataset."""
    df = load_dms_dataset()
    train_df, val_df = _split_dataset(df)

    print("=" * 60)
    print("SARS-CoV-2 RBD Deep Mutational Scanning Dataset")
    print("=" * 60)
    print(f"  Source:           Bloom Lab (jbloomlab/SARS-CoV-2-RBD_DMS)")
    print(f"  Target:           ACE2 binding fitness (bind_avg)")
    print(f"  Wildtype length:  {SEQ_LEN} residues")
    print(f"  Vocab size:       {VOCAB_SIZE} (20 AAs + PAD + START)")
    print(f"  Total mutations:  {len(df):,}")
    print(f"  Unique sites:     {df['site'].nunique()}")
    print(f"  Train mutations:  {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val mutations:    {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Train sites:      {train_df['site'].nunique()}")
    print(f"  Val sites:        {val_df['site'].nunique()}")
    print(f"  Binding score range: [{df['bind_avg'].min():.3f}, {df['bind_avg'].max():.3f}]")
    print(f"  Binding score mean:  {df['bind_avg'].mean():.3f}")
    print(f"  Binding score std:   {df['bind_avg'].std():.3f}")
    print(f"  Time budget:      {TIME_BUDGET}s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main: download + verify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DMS data for autoresearch_bio")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_dms_data()
    print()

    # Step 2: Verify and print stats
    print_dataset_stats()
    print()

    # Step 3: Quick sanity check — encode a mutation and verify shapes
    df = load_dms_dataset()
    row = df.iloc[0]
    x = encode_sequence(WILDTYPE_RBD, int(row["site_idx"]), row["mut_aa"])
    print(f"Sanity check:")
    print(f"  Encoded shape: {x.shape}")
    print(f"  First 10 tokens: {x[:10].tolist()}")
    print(f"  Target (bind_norm): {row['bind_norm']:.4f}")

    # Step 4: Verify dataloaders
    train_loader = make_dataloader("train", batch_size=32)
    val_loader = make_dataloader("val", batch_size=32)
    x_batch, y_batch = next(iter(train_loader))
    print(f"  Train batch X shape: {x_batch.shape}")
    print(f"  Train batch Y shape: {y_batch.shape}")
    x_batch, y_batch = next(iter(val_loader))
    print(f"  Val batch X shape:   {x_batch.shape}")
    print(f"  Val batch Y shape:   {y_batch.shape}")
    print()
    print("Done! Ready to train.")
