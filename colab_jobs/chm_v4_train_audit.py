"""Standalone exact-commit Colab bootstrap: unchanged #1431 engineering audit.

No training, heldout scoring, scientific seed or Modal operation. Arguments are
supplied by the issue worker, not selected from runtime environment variables.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

REPOSITORY = "https://github.com/vinceackermann2-sys/tam-research.git"
TORCH_VERSION = "2.10.0+cpu"
NUMPY_VERSION = "2.5.4"


def command(args, log, cwd=None):
    log.write(json.dumps(args) + "\n")
    log.flush()
    process = subprocess.run(args, cwd=cwd, stdout=log, stderr=log,
                             text=True, timeout=110)
    if process.returncode:
        raise RuntimeError(f"{args[0]} exited {process.returncode}")


def bootstrap(commit, checkout, log):
    """Fetch the full SHA, never a moving branch; normal TLS verification."""
    checkout.mkdir()
    command(["git", "init", "--quiet"], log, checkout)
    command(["git", "remote", "add", "origin", REPOSITORY], log, checkout)
    command(["git", "fetch", "--quiet", "--depth", "1", "origin", commit], log, checkout)
    command(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], log, checkout)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout,
                                     text=True, timeout=10).strip()
    if actual != commit:
        raise RuntimeError("Bootstrapped repository commit differs from reviewed job")
    command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "--no-input", "torch==" + TORCH_VERSION,
             "--index-url", "https://download.pytorch.org/whl/cpu"], log)
    command([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "--no-input", "numpy==" + NUMPY_VERSION,
             "--index-url", "https://pypi.org/simple"], log)


def write_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def bundle_artifacts(folder, commit):
    files = {}
    for path in sorted(folder.iterdir()):
        if path.name in {"manifest.json", "bundle.zip"}:
            continue
        data = path.read_bytes()
        files[path.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    write_json(folder / "manifest.json", {"schema": 1, "commit": commit, "files": files})
    with zipfile.ZipFile(folder / "bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in [*files, "manifest.json"]:
            archive.write(folder / name, name)


def validate_result(result):
    expected = {"training_cases_only": 2048, "development_test_examples_created": 0,
                "trained_model_forward_calls": 0, "optimizer_steps": 0,
                "gpu_used": False, "new_scientific_attempt": False,
                "historical_seed_consumed": 2013161,
                "historical_status": "CHM_V3_100M_DAEC_STAGE_C_STOP"}
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Frozen TRAIN-only audit scope changed")


def run_job(commit, gpu, root=Path("/content")):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected full reviewed commit SHA")
    if gpu != "CPU":
        raise ValueError("This job is CPU-only; GPU allocation is not authorized")
    if sys.version_info < (3, 11):
        raise RuntimeError("Python >=3.11 required")
    folder = root / "tam-results"
    # Fresh runtimes must not accidentally return another attempt's artifacts.
    folder.mkdir(exist_ok=False)
    checkout = root / "tam-research-pinned"
    config = {"commit": commit, "repository": REPOSITORY, "gpu": gpu,
              "job": "chm-v4-1431-train-only-engineering-audit",
              "training_cases": 2048, "optimizer_steps": 0, "scientific_seeds": [],
              "heldout_scoring": False, "checkpoint_resume": False,
              "torch": TORCH_VERSION, "numpy": NUMPY_VERSION,
              "data_source": "unchanged repository TRAIN fixtures",
              "interpretation": "engineering verification; no breakthrough claim"}
    write_json(folder / "config.json", config)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "requested_hardware": gpu}
    result = {"status": "failed", "interpretation": config["interpretation"]}
    started = time.monotonic()
    failure = None
    previous_directory = Path.cwd()
    with (folder / "job.log").open("w") as log:
        log.write("Starting pinned TRAIN-only engineering audit\n")
        log.flush()
        write_json(folder / "result.json", {"status": "pending"})
        write_json(folder / "environment.json", environment)
        bundle_artifacts(folder, commit)
        try:
            if shutil.which("nvidia-smi"):
                hardware = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                                           "--format=csv,noheader"], capture_output=True,
                                          text=True, timeout=10)
                environment["detected_gpu"] = hardware.stdout.strip()
                if hardware.returncode == 0 and hardware.stdout.strip():
                    raise RuntimeError("Unexpected GPU on CPU-only job")
            bootstrap(commit, checkout, log)
            sys.path.insert(0, str(checkout))
            os.chdir(checkout)
            from tam_research.chm_v4_factorized_interventions_1431 import train_only_factorized_audit
            import torch
            import numpy
            if str(torch.__version__) != TORCH_VERSION or numpy.__version__ != NUMPY_VERSION:
                raise RuntimeError("Loaded dependencies do not match pinned requirements")
            environment["packages"] = sorted(
                [{"name": d.metadata.get("Name", ""), "version": d.version}
                 for d in importlib.metadata.distributions()],
                key=lambda x: (x["name"].lower(), x["version"]))
            result = train_only_factorized_audit()
            validate_result(result)
            result["status"] = "ok"
        except Exception as error:
            failure = error
            traceback.print_exc(file=log)
            result = {"status": "failed", "error_class": type(error).__name__,
                      "interpretation": config["interpretation"]}
        finally:
            os.chdir(previous_directory)
            result["elapsed_sec"] = time.monotonic() - started
            write_json(folder / "result.json", result)
            write_json(folder / "environment.json", environment)
    bundle_artifacts(folder, commit)
    print("COLAB_PUBLIC_RESULT=" + json.dumps({"status": result["status"], "gpu": "CPU",
          "elapsed_sec": result["elapsed_sec"],
          "note": "TRAIN-only engineering audit; no training or heldout scoring"}), flush=True)
    if failure:
        raise RuntimeError("Audit failed; retrieve bundled evidence before teardown") from failure
    return folder


def main():
    if len(sys.argv) != 3:
        raise ValueError("Expected full reviewed commit SHA and CPU")
    run_job(*sys.argv[1:])


if __name__ == "__main__":
    main()
