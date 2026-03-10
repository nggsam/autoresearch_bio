"""
GFP Fluorescence prediction — data preparation.
Regression: predict log-fluorescence of GFP variants from amino acid sequence.

Dataset: TAPE benchmark (Rao et al. 2019), originally from Sarkisyan et al. 2016.
~54k GFP variants with measured log-fluorescence.

This file is READ-ONLY. The agent must not modify data loading logic.
"""

import os
import json
import pickle
import hashlib
import random
import requests
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i + 2 for i, aa in enumerate(AA_LIST)}
PAD_TOKEN = 0
START_TOKEN = 1
AA_TO_IDX["<PAD>"] = PAD_TOKEN
AA_TO_IDX["<START>"] = START_TOKEN
VOCAB_SIZE = len(AA_LIST) + 2  # 22

GFP_SEQ_LEN = 237
MAX_SEQ_LEN = 240
SEQ_LEN = MAX_SEQ_LEN

TIME_BUDGET = 300  # 5 minutes

CACHE_DIR = os.path.expanduser("~/.cache/autoresearch_bio/gfp_data")
TAPE_DATA_URL = "https://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz"


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _download_tape_gfp():
    """Download TAPE GFP fluorescence dataset."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    data_dir = os.path.join(CACHE_DIR, "fluorescence")
    check_file = os.path.join(data_dir, "fluorescence_train.lmdb")

    if os.path.exists(check_file):
        print(f"Data: already downloaded at {data_dir}")
        return

    tar_path = os.path.join(CACHE_DIR, "fluorescence.tar.gz")

    if not os.path.exists(tar_path):
        print(f"Downloading TAPE GFP dataset from {TAPE_DATA_URL}...")
        resp = requests.get(TAPE_DATA_URL, stream=True)
        resp.raise_for_status()
        with open(tar_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {os.path.getsize(tar_path)} bytes")

    import tarfile
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(CACHE_DIR, filter="data")
    print(f"Data extracted to {data_dir}")


def _load_lmdb_dataset(lmdb_path):
    """Load a TAPE LMDB dataset."""
    import lmdb

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False,
                     readahead=False, meminit=False)
    records = []
    with env.begin(write=False) as txn:
        num_ex_raw = txn.get(b"num_examples")
        if num_ex_raw is not None:
            try:
                num_examples = int(num_ex_raw.decode())
            except (UnicodeDecodeError, ValueError):
                num_examples = pickle.loads(num_ex_raw)
            for i in range(num_examples):
                raw = txn.get(str(i).encode())
                if raw is not None:
                    records.append(pickle.loads(raw))
        else:
            # Fallback: iterate all keys
            cursor = txn.cursor()
            for key, value in cursor:
                if key != b"num_examples":
                    records.append(pickle.loads(value))
    env.close()
    return records


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def encode_sequence(seq):
    """Encode a protein sequence as a tensor of token IDs."""
    tokens = [START_TOKEN]
    for aa in seq[:MAX_SEQ_LEN]:
        tokens.append(AA_TO_IDX.get(aa, PAD_TOKEN))

    while len(tokens) < MAX_SEQ_LEN + 1:
        tokens.append(PAD_TOKEN)

    return torch.tensor(tokens, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_datasets():
    """Build train/val TensorDatasets from TAPE GFP data."""
    _download_tape_gfp()

    data_dir = os.path.join(CACHE_DIR, "fluorescence")

    train_records = _load_lmdb_dataset(os.path.join(data_dir, "fluorescence_train.lmdb"))
    valid_records = _load_lmdb_dataset(os.path.join(data_dir, "fluorescence_valid.lmdb"))
    test_path = os.path.join(data_dir, "fluorescence_test.lmdb")
    test_records = _load_lmdb_dataset(test_path) if os.path.exists(test_path) else []

    print(f"\nTAPE GFP splits: train={len(train_records)}, valid={len(valid_records)}, test={len(test_records)}")

    def records_to_tensors(records):
        seqs = []
        targets = []
        for rec in records:
            # TAPE LMDB records have 'primary' (sequence) and 'log_fluorescence'
            seq = rec.get("primary", "")
            if isinstance(seq, bytes):
                seq = seq.decode("utf-8")
            target = rec.get("log_fluorescence", None)
            if isinstance(target, (list, np.ndarray)):
                target = float(target[0])
            elif target is not None:
                target = float(target)
            if target is None:
                continue
            # Replace non-standard AAs
            seq = "".join(c for c in seq if c in AA_LIST)
            if len(seq) < 5:
                continue
            seqs.append(seq)
            targets.append(target)

        X = torch.stack([encode_sequence(s) for s in seqs])
        Y = torch.tensor(targets, dtype=torch.float32)
        return X, Y, seqs

    X_train, Y_train, seqs_train = records_to_tensors(train_records)
    X_val, Y_val, seqs_val = records_to_tensors(valid_records)

    return (
        TensorDataset(X_train, Y_train),
        TensorDataset(X_val, Y_val),
        {
            "n_train": len(seqs_train),
            "n_val": len(seqs_val),
            "n_test": len(test_records),
            "y_train_mean": Y_train.mean().item(),
            "y_train_std": Y_train.std().item(),
            "y_train_min": Y_train.min().item(),
            "y_train_max": Y_train.max().item(),
            "seq_len_median": np.median([len(s) for s in seqs_train]),
        },
    )


_DATASETS = None


def make_dataloader(split, batch_size, shuffle=False):
    """Get train or val dataloader."""
    global _DATASETS
    if _DATASETS is None:
        _DATASETS = _build_datasets()

    train_ds, val_ds, _ = _DATASETS
    ds = train_ds if split == "train" else val_ds
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=(split == "train"))


def evaluate_model(model, device, batch_size=64):
    """Evaluate model on validation set. Returns dict with Spearman ρ and MSE."""
    global _DATASETS
    if _DATASETS is None:
        _DATASETS = _build_datasets()

    _, val_ds, _ = _DATASETS

    model.eval()
    all_preds = []
    all_labels = []

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).squeeze(-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(Y_batch.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    sp_corr, _ = spearmanr(all_preds, all_labels)
    mse = float(np.mean((all_preds - all_labels) ** 2))
    pearson = float(np.corrcoef(all_preds, all_labels)[0, 1])

    return {
        "val_spearman": sp_corr,
        "val_mse": mse,
        "val_pearson": pearson,
        "n_samples": len(all_labels),
    }


# ---------------------------------------------------------------------------
# Main: sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Cache directory: {CACHE_DIR}")
    print()

    train_ds, val_ds, info = _build_datasets()

    print("=" * 60)
    print("GFP Fluorescence Prediction (TAPE Benchmark)")
    print("=" * 60)
    print(f"  Task:              Regression (predict log-fluorescence)")
    print(f"  Source:            Sarkisyan et al. 2016 / TAPE benchmark")
    print(f"  Train samples:     {info['n_train']:,}")
    print(f"  Val samples:       {info['n_val']:,}")
    print(f"  Test samples:      {info['n_test']:,}")
    print(f"  Seq length:        ~{info['seq_len_median']:.0f} AA (GFP=237)")
    print(f"  Target range:      [{info['y_train_min']:.2f}, {info['y_train_max']:.2f}]")
    print(f"  Target mean:       {info['y_train_mean']:.3f}")
    print(f"  Target std:        {info['y_train_std']:.3f}")
    print(f"  Vocab size:        {VOCAB_SIZE}")
    print(f"  Max seq length:    {MAX_SEQ_LEN}")
    print(f"  Metric:            Spearman ρ")
    print(f"  Time budget:       {TIME_BUDGET}s")
    print("=" * 60)

    X, Y = train_ds[0]
    print(f"\nSanity check:")
    print(f"  Encoded shape: {X.shape}")
    print(f"  First 10 tokens: {X[:10].tolist()}")
    print(f"  Target: {Y.item():.4f}")

    train_loader = make_dataloader("train", 32, shuffle=True)
    for X_b, Y_b in train_loader:
        print(f"  Train batch X: {X_b.shape}, Y: {Y_b.shape}")
        break

    print("\nDone! Ready to train.")
