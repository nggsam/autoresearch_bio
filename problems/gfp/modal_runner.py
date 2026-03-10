"""Modal GPU runner for GFP fluorescence prediction. Usage: modal run modal_runner.py"""

import modal, os

app = modal.App("autoresearch-bio-gfp")
project_dir = os.path.dirname(os.path.abspath(__file__))

# Separate image build (cached) from local files (runtime mount)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pandas", "scipy", "requests", "matplotlib", "lmdb")
)

data_volume = modal.Volume.from_name("autoresearch-bio-gfp-data", create_if_missing=True)

@app.function(image=image, gpu="T4", timeout=600,
    volumes={"/root/.cache/autoresearch_bio": data_volume})
def run_experiment(prepare_code: str, train_code: str):
    """Run GFP experiment. Code passed as strings to avoid image rebuilds."""
    import subprocess, sys, tempfile

    work_dir = "/tmp/gfp_exp"
    os.makedirs(work_dir, exist_ok=True)

    # Write code files
    with open(os.path.join(work_dir, "prepare.py"), "w") as f:
        f.write(prepare_code)
    with open(os.path.join(work_dir, "train.py"), "w") as f:
        f.write(train_code)

    print("=" * 60, "\nSTEP 1: Preparing data...\n", "=" * 60)
    r = subprocess.run([sys.executable, "prepare.py"], capture_output=True, text=True, cwd=work_dir)
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        raise RuntimeError("prepare.py failed")
    data_volume.commit()

    print("=" * 60, "\nSTEP 2: Training on GPU...\n", "=" * 60)
    r = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True, cwd=work_dir)
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        raise RuntimeError(f"train.py failed: {r.returncode}")

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
    # Read local code files and pass as arguments (avoids image rebuild)
    with open(os.path.join(project_dir, "prepare.py")) as f:
        prepare_code = f.read()
    with open(os.path.join(project_dir, "train.py")) as f:
        train_code = f.read()

    print("Launching GFP experiment on GPU...")
    results = run_experiment.remote(prepare_code, train_code)
    print("\n" + "=" * 60 + "\nEXPERIMENT COMPLETE\n" + "=" * 60)
    for k, v in results.items(): print(f"  {k}: {v}")
