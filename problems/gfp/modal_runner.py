"""Modal GPU runner for GFP fluorescence prediction. Usage: modal run modal_runner.py"""

import modal, os

app = modal.App("autoresearch-bio-gfp")
project_dir = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pandas", "scipy", "requests", "matplotlib", "lmdb")
    .add_local_dir(project_dir, remote_path="/app",
        ignore=["**/__pycache__", "**/*.pyc", "**/*.log", "**/*.tsv", "**/modal_runner*"])
)

data_volume = modal.Volume.from_name("autoresearch-bio-gfp-data", create_if_missing=True)

@app.function(image=image, gpu="T4", timeout=600,
    volumes={"/root/.cache/autoresearch_bio": data_volume})
def run_experiment():
    import subprocess, sys
    os.chdir("/app")

    print("=" * 60, "\nSTEP 1: Preparing data...\n", "=" * 60)
    r = subprocess.run([sys.executable, "prepare.py"], capture_output=True, text=True, cwd="/app")
    print(r.stdout)
    if r.returncode != 0: print("STDERR:", r.stderr); raise RuntimeError("prepare.py failed")
    data_volume.commit()

    print("=" * 60, "\nSTEP 2: Training on GPU...\n", "=" * 60)
    r = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True, cwd="/app")
    print(r.stdout)
    if r.returncode != 0: print("STDERR:", r.stderr); raise RuntimeError(f"train.py failed: {r.returncode}")

    results = {}
    for line in r.stdout.split("\n"):
        line = line.strip()
        if ":" in line and line.startswith(("val_", "training_", "total_", "peak_", "num_", "n_")):
            k, v = line.split(":", 1)
            try: results[k.strip()] = float(v.strip())
            except: results[k.strip()] = v.strip()
    print("=" * 60, "\nRESULTS:")
    for k, v in results.items(): print(f"  {k}: {v}")
    print("=" * 60)
    return results

@app.local_entrypoint()
def main():
    print("Launching GFP experiment on GPU...")
    results = run_experiment.remote()
    print("\n" + "=" * 60 + "\nEXPERIMENT COMPLETE\n" + "=" * 60)
    for k, v in results.items(): print(f"  {k}: {v}")
