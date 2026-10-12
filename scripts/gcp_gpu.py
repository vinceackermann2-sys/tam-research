"""Bounded, single-VM Google Cloud GPU launcher (stdlib only)."""
from __future__ import annotations

import argparse
import base64
from decimal import Decimal, ROUND_CEILING
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

GPU_TYPES = {
    "L4": ("g2-standard-4", None),
    "A100": ("a2-highgpu-1g", None),
    "H100": ("a3-highgpu-1g", None),
    "T4": ("n1-standard-8", "nvidia-tesla-t4"),
}
INSTANCE = "tam-research-gpu"
ROOT = Path(__file__).resolve().parents[1]
# Conservative allowances, not a Google invoice or a promise of exact billing.
# Price/overhead review is required before enabling paid execution.
HOURLY_ALLOWANCE_USD = {"L4": 2, "T4": 2, "A100": 8, "H100": 20}
TOTAL_ALLOWANCE_CENTS = 26_000
SAFETY_RESERVE_CENTS = 2_000


def reservation_cents(config):
    hourly = Decimal(HOURLY_ALLOWANCE_USD[config["gpu"]])
    # Account for the full server-side TTL plus provisioning/cleanup and $2/job overhead.
    hours = Decimal(config["minutes"] + 5) / 60
    return int((hourly * hours * 100).to_integral_value(rounding=ROUND_CEILING)) + 200


def remaining_allowance(records):
    total = 0
    ids = set()
    for record in records:
        if (not isinstance(record, dict) or type(record.get("cents")) is not int
                or record["cents"] <= 0 or not isinstance(record.get("run_id"), str)
                or record["run_id"] in ids):
            raise ValueError("Invalid or duplicate reservation ledger; refusing paid compute")
        total += record["cents"]
        ids.add(record["run_id"])
    return TOTAL_ALLOWANCE_CENTS - SAFETY_RESERVE_CENTS - total


def check_credit_expiry(config, value, now=None):
    if not value:
        raise ValueError("Verified credit expiry is missing; refusing paid compute")
    expiry = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if expiry.tzinfo is None:
        raise ValueError("Credit expiry must include a timezone")
    now = now or datetime.now(timezone.utc)
    if now + timedelta(minutes=config["minutes"] + 10) >= expiry:
        raise ValueError("Credits expire before this job can safely finish")


def reserve_budget(config, bucket, run_id, out):
    # Bucket API lists objects consistently. GitHub serializes this sole authorized workflow.
    # Download errors, malformed ledger data and reused run IDs always fail closed.
    listing = gc("storage", "ls", f"gs://{bucket}/budget/reservations/", check=False)
    if listing.returncode:
        raise RuntimeError("Budget ledger cannot be read; refusing paid compute")
    records = []
    for uri in listing.stdout.splitlines():
        uri = uri.strip()
        if uri.endswith(".json"):
            records.append(json.loads(gc("storage", "cat", uri).stdout))
    if not records or not any(r.get("run_id") == "initial-safety-baseline" for r in records):
        raise RuntimeError("Missing initial budget baseline; refusing paid compute")
    cents = reservation_cents(config)
    remaining = remaining_allowance(records)
    if cents > remaining:
        raise RuntimeError(f"$260 lifetime allowance exhausted: job reserves ${cents/100:.2f}, remaining ${remaining/100:.2f}")
    record = {"run_id": run_id, "cents": cents, "config": config,
              "remaining_cents": remaining - cents, "total_limit_cents": TOTAL_ALLOWANCE_CENTS}
    path = out / "budget-reservation.json"
    path.write_text(json.dumps(record, indent=2))
    gc("storage", "cp", "--if-generation-match=0", str(path),
       f"gs://{bucket}/budget/reservations/{run_id}.json")
    # Reservations are never refunded automatically, even for failed provisioning.
    print(f"Reserved ${cents/100:.2f}; remaining job allowance ${(remaining-cents)/100:.2f}", flush=True)


def validate(config, root=ROOT):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    allowed = {"gpu", "minutes", "source_sha", "script", "args", "zone"}
    if set(config) - allowed:
        raise ValueError("Unknown configuration fields")
    gpu = config.get("gpu", "L4")
    minutes = config.get("minutes", 30)
    sha = config.get("source_sha", "")
    script = config.get("script", "scripts/gcp_smoke.py")
    args = config.get("args", [])
    zone = config.get("zone", "us-central1-a")
    if gpu not in GPU_TYPES:
        raise ValueError("GPU must be L4, T4, A100, or H100")
    if type(minutes) is not int or not 5 <= minutes <= 240:
        raise ValueError("minutes must be an integer between 5 and 240")
    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError("source_sha must be a full commit SHA")
    if not isinstance(script, str) or not re.fullmatch(r"[A-Za-z0-9_/-]+\.py", script):
        raise ValueError("script must be a repository Python file")
    path = (root / script).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("script must exist inside this repository")
    if path.name.startswith("modal_"):
        raise ValueError("Use the underlying experiment script, not a Modal wrapper")
    if not isinstance(args, list) or len(args) > 64 or any(
        not isinstance(x, str) or len(x) > 1024 or "\x00" in x for x in args
    ):
        raise ValueError("args must be at most 64 strings")
    if not isinstance(zone, str) or not re.fullmatch(r"us-central1-[abcf]", zone):
        raise ValueError("zone must be in us-central1")
    return dict(gpu=gpu, minutes=minutes, source_sha=sha, script=script, args=args, zone=zone)


def gc(*args, check=True):
    return subprocess.run(["gcloud", *args, "--quiet"], text=True, capture_output=True, check=check)


def startup(config, bucket, run_id):
    cfg64 = base64.b64encode(json.dumps(config).encode()).decode()
    return f'''#!/bin/bash
set -Eeuo pipefail
mkdir -p /opt/tam/results
export TAM_RESULTS_DIR=/opt/tam/results
export PYTHONUNBUFFERED=1
exec > >(tee -a /opt/tam/results/worker.log /dev/ttyS0) 2>&1
finish() {{
  code=$?
  trap - EXIT
  printf '{{"exit_code":%s,"source_sha":"{config['source_sha']}","gpu":"{config['gpu']}"}}\\n' "$code" > /opt/tam/status.json
  # Bound persisted output. Experiments must save portable results to TAM_RESULTS_DIR.
  if [ "$(du -sb /opt/tam/results | cut -f1)" -gt 104857600 ]; then
    code=90
    printf 'Result directory exceeds 100 MiB; upload refused\\n' > /opt/tam/oversize.log
    timeout 60 gcloud storage cp /opt/tam/oversize.log 'gs://{bucket}/runs/{run_id}/results/oversize.log' || true
  else
    timeout 120 gcloud storage cp --recursive /opt/tam/results 'gs://{bucket}/runs/{run_id}/' || true
  fi
  printf '{{"exit_code":%s,"source_sha":"{config['source_sha']}","gpu":"{config['gpu']}"}}\\n' "$code" > /opt/tam/status.json
  timeout 60 gcloud storage cp /opt/tam/status.json 'gs://{bucket}/runs/{run_id}/status.json' || true
  shutdown -h now
}}
trap finish EXIT
cd /opt/tam
curl --fail --location --retry 2 --max-time 120 'https://github.com/vinceackermann2-sys/tam-research/archive/{config['source_sha']}.tar.gz' -o source.tar.gz
mkdir repo
tar -xzf source.tar.gz -C repo --strip-components=1
cd repo
printf '%s' '{cfg64}' | base64 -d > /opt/tam/config.json
# The Google PyTorch image includes CUDA and NVIDIA drivers.
PYTHON=/opt/conda/bin/python
test -x "$PYTHON" || PYTHON=python3
"$PYTHON" -m pip install --disable-pip-version-check -e '.[modal]' > /opt/tam/results/install.log 2>&1
"$PYTHON" - <<'PY'
import json, os, subprocess, torch
cfg = json.load(open('/opt/tam/config.json'))
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Exactly one working CUDA GPU is required')
name = torch.cuda.get_device_name(0)
if cfg['gpu'] not in name:
    raise RuntimeError(f"Requested {{cfg['gpu']}}, got {{name}}")
print('Verified GPU:', name, flush=True)
subprocess.run(['timeout', '--signal=TERM', '--kill-after=30s',
                str(cfg['minutes'] * 60 - 180),
                os.sys.executable, cfg['script'], *cfg['args']], check=True)
PY
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    opts = parser.parse_args()
    config = validate(json.loads(os.environ["GCP_RUN_CONFIG"]))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], text=True).split()[0]
    if head != config["source_sha"] or remote != head:
        raise ValueError("source_sha must match the checkout and current main")
    if opts.validate_only:
        print(json.dumps(config, indent=2))
        return
    check_credit_expiry(config, os.environ.get("GCP_CREDIT_EXPIRES_AT", ""))
    project = os.environ["GCP_PROJECT_ID"]
    bucket = os.environ["GCP_RESULTS_BUCKET"]
    worker = os.environ["GCP_WORKER_SERVICE_ACCOUNT"]
    run_id = f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    zone = config["zone"]
    out = ROOT / "gcp-results"
    out.mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    # Fixed name is a cross-run mutex; never delete an existing job to acquire it.
    existing = json.loads(gc("compute", "instances", "list", f"--project={project}",
                            f"--filter=name={INSTANCE}", "--format=json").stdout)
    if existing:
        raise RuntimeError("A TAM GPU VM already exists; wait for cleanup before another job")
    quotas = json.loads(gc("compute", "project-info", "describe", f"--project={project}", "--format=json").stdout)
    global_gpu = next((q for q in quotas["quotas"] if q["metric"] == "GPUS_ALL_REGIONS"), None)
    if global_gpu is None or global_gpu["limit"] - global_gpu["usage"] < 1:
        raise RuntimeError("Google Cloud global GPU quota is zero or exhausted")
    reserve_budget(config, bucket, run_id, out)
    boot = out / "startup.sh"
    boot.write_text(startup(config, bucket, run_id), encoding="utf-8")
    machine, accelerator = GPU_TYPES[config["gpu"]]
    cmd = ["compute", "instances", "create", INSTANCE, f"--project={project}", f"--zone={zone}",
           f"--machine-type={machine}", "--provisioning-model=SPOT", "--maintenance-policy=TERMINATE",
           "--no-restart-on-failure", f"--max-run-duration={config['minutes']}m",
           "--instance-termination-action=DELETE", "--boot-disk-size=100GB", "--boot-disk-type=pd-balanced",
           "--image-project=deeplearning-platform-release",
           "--image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580",
           "--network=tam-research-compute", "--subnet=tam-research-compute",
           f"--service-account={worker}", "--scopes=https://www.googleapis.com/auth/devstorage.read_write",
           "--labels=managed-by=tam-research", f"--metadata-from-file=startup-script={boot}",
           "--metadata=block-project-ssh-keys=TRUE,enable-oslogin=TRUE",
           "--shielded-vtpm", "--shielded-integrity-monitoring"]
    if accelerator:
        cmd.append(f"--accelerator=type={accelerator},count=1")
    created = False
    try:
        result = gc(*cmd, check=False)
        (out / "create.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            # Never retry paid provisioning or fall back to another GPU automatically.
            raise RuntimeError(result.stderr)
        created = True
        deadline = time.monotonic() + config["minutes"] * 60 + 120
        while time.monotonic() < deadline:
            done = gc("storage", "cp", f"gs://{bucket}/runs/{run_id}/status.json", str(out / "status.json"), check=False)
            if done.returncode == 0:
                gc("storage", "cp", "--recursive", f"gs://{bucket}/runs/{run_id}/results", str(out), check=False)
                status = json.loads((out / "status.json").read_text())
                if status["exit_code"]:
                    raise RuntimeError(f"GPU task failed with exit code {status['exit_code']}")
                print(json.dumps(status, indent=2))
                return
            time.sleep(20)
        raise RuntimeError("VM deadline reached without a success report; inspect artifacts")
    finally:
        if created:
            cleanup = gc("compute", "instances", "delete", INSTANCE, f"--project={project}", f"--zone={zone}", check=False)
            (out / "cleanup.log").write_text(cleanup.stdout + cleanup.stderr)
            if cleanup.returncode:
                print("Cleanup call failed; Google's independent DELETE deadline remains active", flush=True)


if __name__ == "__main__":
    main()
