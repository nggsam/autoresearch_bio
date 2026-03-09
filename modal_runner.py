"""
Modal GPU runner for autoresearch_bio.
Runs train.py on a cloud GPU (T4/A10G) and returns results.

Usage:
    # First time: authenticate with Modal
    modal setup

    # Run a single experiment on GPU
    modal run modal_runner.py

    # Run and return results
    modal run modal_runner.py --detach
"""

import modal
import os

# Define the Modal app
app = modal.App("autoresearch-bio")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "requests",
        "matplotlib",
    )
)

# Mount the local project directory into the container
project_mount = modal.Mount.from_local_dir(
    os.path.dirname(os.path.abspath(__file__)),
    remote_path="/app",
    condition=lambda path: not path.endswith((".pyc", ".log", ".tsv"))
    and "__pycache__" not in path
    and ".git" not in path,
)

# Persistent volume for caching downloaded data across runs
data_volume = modal.Volume.from_name("autoresearch-bio-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",  # Cheapest GPU option (~$0.17/hr). Use "A10G" for faster runs.
    timeout=600,  # 10 min max (experiment is 5 min + overhead)
    mounts=[project_mount],
    volumes={"/root/.cache/autoresearch_bio": data_volume},
)
def run_experiment():
    """Run a single training experiment on GPU and return results."""
    import subprocess
    import sys

    os.chdir("/app")

    # First ensure data is downloaded
    print("=" * 60)
    print("STEP 1: Preparing data...")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "prepare.py"],
        capture_output=True, text=True, cwd="/app"
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError("prepare.py failed")

    # Commit the data volume so it persists
    data_volume.commit()

    # Run training
    print("=" * 60)
    print("STEP 2: Training...")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, cwd="/app"
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"train.py failed with exit code {result.returncode}")

    # Parse results from output
    output = result.stdout
    results = {}
    for line in output.split("\n"):
        line = line.strip()
        if ":" in line and line.startswith(("val_", "training_", "total_", "peak_", "num_", "n_")):
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            try:
                results[key] = float(val)
            except ValueError:
                results[key] = val

    print("=" * 60)
    print("RESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    return results


@app.local_entrypoint()
def main():
    """Run from the command line: modal run modal_runner.py"""
    print("Launching experiment on GPU...")
    results = run_experiment.remote()
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Append to local results.tsv
    if "val_spearman" in results:
        sp = results.get("val_spearman", 0)
        mse = results.get("val_mse", 0)
        mem = results.get("peak_memory_mb", 0)
        print(f"\n  val_spearman: {sp:.6f}")
        print(f"  val_mse:      {mse:.6f}")
        print(f"  memory_mb:    {mem:.1f}")
