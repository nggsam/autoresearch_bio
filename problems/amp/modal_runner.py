"""
Modal GPU runner for AMP classification.
Usage: modal run modal_runner.py
"""

import modal
import os

app = modal.App("autoresearch-bio-amp")

project_dir = os.path.dirname(os.path.abspath(__file__))

# Image with deps + local files (add_local_dir with copy=False mounts at runtime)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "numpy", "pandas", "scipy", "requests",
        "scikit-learn", "matplotlib",
    )
    .add_local_dir(project_dir, remote_path="/app", ignore=["**/__pycache__", "**/*.pyc", "**/*.log", "**/*.tsv", "**/modal_runner*"])
)

data_volume = modal.Volume.from_name("autoresearch-bio-amp-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/root/.cache/autoresearch_bio": data_volume},
)
def run_experiment():
    import subprocess, sys
    os.chdir("/app")

    print("=" * 60)
    print("STEP 1: Preparing data...")
    print("=" * 60)
    result = subprocess.run([sys.executable, "prepare.py"], capture_output=True, text=True, cwd="/app")
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError("prepare.py failed")
    data_volume.commit()

    print("=" * 60)
    print("STEP 2: Training on GPU...")
    print("=" * 60)
    result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True, cwd="/app")
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"train.py failed: {result.returncode}")

    results = {}
    for line in result.stdout.split("\n"):
        line = line.strip()
        if ":" in line and line.startswith(("val_", "training_", "total_", "peak_", "num_", "n_")):
            key, val = line.split(":", 1)
            try: results[key.strip()] = float(val.strip())
            except ValueError: results[key.strip()] = val.strip()

    print("=" * 60)
    print("RESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    return results


@app.local_entrypoint()
def main():
    print("Launching AMP experiment on GPU...")
    results = run_experiment.remote()
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")
