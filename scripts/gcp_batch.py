"""Owner-triggered, bounded CPU jobs on Google Cloud Batch / Compute Engine."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time


REPOSITORY = "vinceackermann2-sys/tam-research"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path("gcp-results")


def validate_config(config: dict, root: Path = ROOT) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Run configuration must be a JSON object")
    unknown = set(config) - {"script", "args", "timeout_seconds", "install_dependencies"}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    script = config.get("script", "scripts/gcp_compute_smoke.py")
    if not isinstance(script, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+\.py", script):
        raise ValueError("script must be a repository-relative Python file")
    path = PurePosixPath(script)
    if path.is_absolute() or ".." in path.parts or path.parts[0].startswith((".", "-")):
        raise ValueError("script must stay inside the repository")
    resolved = (root / script).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Script does not exist inside the checked-out commit: {script}")
    args = config.get("args", [])
    if not isinstance(args, list) or len(args) > 64 or any(
        not isinstance(arg, str) or len(arg) > 2048 or "\0" in arg for arg in args
    ):
        raise ValueError("args must be an array of at most 64 strings")
    duration = config.get("timeout_seconds", 600)
    if type(duration) is not int or not 60 <= duration <= 1800:
        raise ValueError("timeout_seconds must be an integer from 60 to 1800")
    install = config.get("install_dependencies", False)
    if type(install) is not bool:
        raise ValueError("install_dependencies must be a boolean")
    return {"script": script, "args": args, "timeout_seconds": duration,
            "install_dependencies": install}


def build_job(config: dict, sha: str, runtime_account: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("A full immutable commit SHA is required")
    # The official Python image includes git. No GitHub credential enters the VM.
    commands = [
        "set -euo pipefail",
        "git init /tmp/tam-research",
        "cd /tmp/tam-research",
        f"git remote add origin https://github.com/{REPOSITORY}.git",
        f"git fetch --depth=1 origin {sha}",
        "git checkout --detach FETCH_HEAD",
        f"test \"$(git rev-parse HEAD)\" = {sha}",
        f"export TAM_REPO_SHA={sha}",
        "export PYTHONPATH=/tmp/tam-research PYTHONUNBUFFERED=1",
    ]
    if config["install_dependencies"]:
        commands += [
            "python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu",
            "python -m pip install -e '.[test]' 'datasets>=4,<5' 'transformers>=4.55,<5' 'tokenizers>=0.21,<1' 'huggingface-hub>=0.34,<1'",
        ]
    commands.append(shlex.join(["python", config["script"], *config["args"]]))
    return {
        "taskGroups": [{"taskCount": "1", "parallelism": "1", "taskSpec": {
            "runnables": [{"container": {"imageUri": "python:3.11-bookworm",
                "entrypoint": "/bin/bash", "commands": ["-c", "\n".join(commands)]}}],
            "computeResource": {"cpuMilli": 4000, "memoryMib": 8192},
            "maxRetryCount": 0, "maxRunDuration": f"{config['timeout_seconds']}s",
        }}],
        "allocationPolicy": {
            "instances": [{"policy": {"machineType": "e2-standard-4",
                "provisioningModel": "STANDARD",
                "bootDisk": {"type": "pd-standard", "sizeGb": 30}}}],
            "serviceAccount": {"email": runtime_account},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"application": "tam-research", "source": "github-actions"},
    }


def gcloud(*args: str, check: bool = True) -> str:
    result = subprocess.run(["gcloud", *args, "--project", os.environ["GCP_PROJECT_ID"],
                             "--quiet"], text=True, capture_output=True, timeout=120)
    if result.returncode and check:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def prepare() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if os.environ["GITHUB_EVENT_NAME"] == "issues":
        config = json.loads(event["issue"].get("body") or "{}")
    else:
        config = json.loads(event.get("inputs", {}).get("config") or "{}")
    config = validate_config(config)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2))
    job = build_job(config, os.environ["GITHUB_SHA"], os.environ["GCP_RUNTIME_ACCOUNT"])
    (RESULTS / "job-request.json").write_text(json.dumps(job, indent=2))
    job_id = f"tam-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"job_id={job_id}\n")


def execute() -> None:
    job_id = os.environ["JOB_ID"]
    location = os.environ["GCP_REGION"]
    flags = [job_id, "--location", location]
    # Each gcloud invocation refreshes the short-lived federated credential.
    job = json.loads(gcloud("batch", "jobs", "submit", *flags,
                            "--config", str(RESULTS / "job-request.json"), "--format=json"))
    uid = job["uid"]
    (RESULTS / "job.json").write_text(json.dumps(job, indent=2))
    deadline = time.monotonic() + 2700
    terminal = False
    try:
        previous = None
        while time.monotonic() < deadline:
            job = json.loads(gcloud("batch", "jobs", "describe", *flags, "--format=json"))
            (RESULTS / "job.json").write_text(json.dumps(job, indent=2))
            state = job["status"]["state"]
            if state != previous:
                print(f"Google Cloud job {job_id}: {state}", flush=True)
                previous = state
            if state in {"SUCCEEDED", "FAILED", "DELETION_IN_PROGRESS"}:
                terminal = True
                break
            time.sleep(20)
        if not terminal:
            raise TimeoutError("Job exceeded the 45-minute scheduling/execution deadline")
        if state != "SUCCEEDED":
            raise RuntimeError(f"Google Cloud job ended in {state}")
    finally:
        # Queued or running jobs must not outlive a failed watcher.
        if not terminal:
            gcloud("batch", "jobs", "delete", *flags)
        logs = gcloud("logging", "read", f'labels.job_uid="{uid}"',
                      "--order=asc", "--limit=2000", "--format=json")
        (RESULTS / "logs.json").write_text(logs)
        entries = json.loads(logs or "[]")
        texts = [e.get("textPayload") or json.dumps(e.get("jsonPayload", {})) for e in entries]
        (RESULTS / "task.log").write_text("\n".join(texts))
        print("\n".join(texts)[-12000:])


if __name__ == "__main__":
    import sys
    {"prepare": prepare, "execute": execute}[sys.argv[1]]()
