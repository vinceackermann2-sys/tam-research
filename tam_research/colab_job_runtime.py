"""Standard-library provenance and artifact contract for reviewed Colab jobs."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback
import zipfile

MODEL_BLOBS = {
    "tam_research/chm_v4_neural_hard_copy_train_1430.py": "78f9de5df6580cb85f557272af5d7f7cffccec55",
    "tam_research/chm_v4_independent_binding_benchmark_1421.py": "41fd46670a11f9db9615de020c85a80cc4a0db00",
    "tam_research/chm_v3_matched_tiny_models_1377.py": "ffe14b0701e18493a2bf9b45a1238b5acd5e3eab",
}


def source_audit(root):
    observed = {}
    for name, expected in MODEL_BLOBS.items():
        data = (root / name).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected:
            raise ValueError("Frozen model or data source changed: " + name)
        ast.parse(data.decode())
        observed[name] = actual
    return observed


def write_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")


def package_artifacts(root):
    """Complete bounded bundle, without extraction or credential directories."""
    allowed = {"configuration.json", "metrics.json", "run.log", "provenance.json", "checkpoint.pt"}
    paths = sorted(p for p in root.iterdir() if p.name in allowed and p.is_file())
    if not {"configuration.json", "metrics.json", "run.log", "provenance.json"}.issubset({p.name for p in paths}):
        raise ValueError("Incomplete research artifacts")
    if any(p.is_symlink() for p in paths) or sum(p.stat().st_size for p in paths) > 32 * 1024 * 1024:
        raise ValueError("Unsafe or oversize artifacts; separate approved storage required")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    target = root / "bundle.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p, p.name)
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
    with target.open("rb") as file:
        os.fsync(file.fileno())
    return {"sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "bytes": target.stat().st_size}


def run_job(args, root, output, *, training=False):
    """Fresh CPU-only jobs. No Modal import, checkpoint resume or heldout access."""
    if args.requested_gpu != "CPU":
        raise ValueError("These reviewed jobs require CPU; hardware changes need review")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != args.repo_commit:
        raise ValueError("Repository bootstrap commit mismatch")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"commit": commit, "issue": args.issue, "script": args.job_script,
                  "model_blobs": {}, "requested_gpu": "CPU",
                  "precision": "float32", "resume": False, "retries": 0}
    write_json(output / "provenance.json", provenance)
    write_json(output / "configuration.json", {"scope": "pending source validation",
                                              "training_requested": training})
    failure = None
    try:
        provenance["model_blobs"] = source_audit(root)
        write_json(output / "provenance.json", provenance)
        if training:
            from tam_research.colab_chm_v4_microfit_1436 import configuration, run_microfit
            write_json(output / "configuration.json", configuration())
            metrics = run_microfit(output)
        else:
            write_json(output / "configuration.json", {"scope": "source/bootstrap/artifact preflight",
                                                      "optimizer_steps": 0, "dependencies": "stdlib only"})
            metrics = {"status": "source-ready", "training_executed": False,
                       "heldout_generated": False, "source_blobs": provenance["model_blobs"]}
            (output / "run.log").write_text("Frozen source blobs and Python syntax verified. No model imported or trained.\n")
    except Exception as error:
        failure = error
        metrics = {"status": "failed", "error_type": type(error).__name__,
                   "checkpoint_available": (output / "checkpoint.pt").exists(),
                   "heldout_generated": False}
        with (output / "run.log").open("a") as log:
            traceback.print_exc(file=log)
    write_json(output / "metrics.json", metrics)
    descriptor = package_artifacts(output)
    print("COLAB_ARTIFACT=" + json.dumps(descriptor), flush=True)
    print("COLAB_PUBLIC_RESULT=" + json.dumps({"status": metrics["status"], "gpu": "CPU",
                                               "note": "TRAIN-only engineering" if training else "No-training source preflight"}), flush=True)
    if failure:
        raise RuntimeError("Job failed; bundled failure evidence is available for retrieval") from failure
