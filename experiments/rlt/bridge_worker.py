from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tam_research.data import prepare_fineweb
from experiments.rlt.smoke import run_smoke
from experiments.rlt.train_rlt import train_rlt

OWNER = "vinceackermann2-sys"
REPO = "tam-research"
DEFAULT_BRANCH = "exp/rlt-colab"
JOBS_DIR = "experiments/rlt/bridge/jobs"
RESULTS_DIR = "experiments/rlt/bridge/results"
_JOB_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,80}$")


class GitHubBridge:
    def __init__(self, token: str, branch: str):
        self.token = token
        self.branch = branch

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/{path}"
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "tam-rlt-colab-bridge",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())

    def list_jobs(self) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode({"ref": self.branch})
        try:
            rows = self._request("GET", f"contents/{JOBS_DIR}?{query}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return []
            raise
        return sorted(
            [
                row
                for row in rows
                if row.get("type") == "file"
                and row.get("name", "").endswith(".json")
                and row.get("name") != "README.json"
            ],
            key=lambda row: row["name"],
        )

    def read_json(self, path: str) -> dict[str, Any]:
        query = urllib.parse.urlencode({"ref": self.branch})
        row = self._request("GET", f"contents/{path}?{query}")
        raw = base64.b64decode(row["content"]).decode()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path} must contain a JSON object")
        return value

    def exists(self, path: str) -> bool:
        query = urllib.parse.urlencode({"ref": self.branch})
        try:
            self._request("GET", f"contents/{path}?{query}")
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise

    def create_json(self, path: str, value: dict[str, Any], message: str) -> None:
        encoded = base64.b64encode(
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        ).decode()
        self._request(
            "PUT",
            f"contents/{path}",
            {
                "message": message,
                "content": encoded,
                "branch": self.branch,
            },
        )


def _bounded_int(
    job: dict[str, Any], key: str, default: int, low: int, high: int
) -> int:
    value = int(job.get(key, default))
    if not low <= value <= high:
        raise ValueError(f"{key} must be in [{low}, {high}], got {value}")
    return value


def validate_job(job: dict[str, Any], filename: str) -> dict[str, Any]:
    allowed_keys = {
        "schema",
        "job_id",
        "task",
        "seed",
        "profile",
        "token_budget",
        "seq_len",
        "micro_batch_size",
        "grad_accum_steps",
        "train_tokens",
        "val_tokens",
    }
    unknown = set(job) - allowed_keys
    if unknown:
        raise ValueError(f"unsupported job fields: {sorted(unknown)}")
    if int(job.get("schema", 1)) != 1:
        raise ValueError("only bridge schema 1 is supported")

    job_id = str(job.get("job_id", ""))
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("invalid job_id")
    if filename != f"{job_id}.json":
        raise ValueError("job_id must match the queue filename")

    task = str(job.get("task", ""))
    if task not in {"smoke", "train"}:
        raise ValueError("task must be 'smoke' or 'train'")

    normalized: dict[str, Any] = {
        "schema": 1,
        "job_id": job_id,
        "task": task,
        "seed": _bounded_int(job, "seed", 20260913, 0, 2_147_483_647),
    }
    if task == "train":
        profile = str(job.get("profile", "tiny")).lower()
        if profile != "tiny":
            raise ValueError("initial Colab bridge permits profile='tiny' only")
        normalized.update(
            {
                "profile": profile,
                "token_budget": _bounded_int(
                    job, "token_budget", 65_536, 65_536, 500_000
                ),
                "seq_len": _bounded_int(job, "seq_len", 64, 32, 128),
                "micro_batch_size": _bounded_int(
                    job, "micro_batch_size", 2, 1, 4
                ),
                "grad_accum_steps": _bounded_int(
                    job, "grad_accum_steps", 4, 1, 16
                ),
                "train_tokens": _bounded_int(
                    job, "train_tokens", 1_000_000, 1_000_000, 4_000_000
                ),
                "val_tokens": _bounded_int(
                    job, "val_tokens", 100_000, 100_000, 500_000
                ),
            }
        )
        if normalized["train_tokens"] < normalized["token_budget"] + 1024:
            raise ValueError("train_tokens must exceed token_budget by at least 1024")
    return normalized


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def execute_job(job: dict[str, Any]) -> dict[str, Any]:
    if job["task"] == "smoke":
        return run_smoke("cuda" if _cuda_available() else "cpu")

    if not _cuda_available():
        raise RuntimeError("train jobs require a Colab GPU runtime")

    data_dir = Path(os.environ.get("RLT_DATA_DIR", "/content/rlt-data"))
    run_root = Path(os.environ.get("RLT_RUN_ROOT", "/content/rlt-runs"))
    data_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    data_meta = prepare_fineweb(
        str(data_dir),
        train_tokens=job["train_tokens"],
        val_tokens=job["val_tokens"],
        seed=1234,
    )
    training = train_rlt(
        profile=job["profile"],
        seed=job["seed"],
        data_dir=str(data_dir),
        run_root=str(run_root / job["job_id"]),
        token_budget=job["token_budget"],
        seq_len=job["seq_len"],
        micro_batch_size=job["micro_batch_size"],
        grad_accum_steps=job["grad_accum_steps"],
    )
    return {"data": data_meta, "training": training}


def process_once(bridge: GitHubBridge) -> int:
    processed = 0
    for row in bridge.list_jobs():
        filename = row["name"]
        source_path = f"{JOBS_DIR}/{filename}"
        try:
            raw = bridge.read_json(source_path)
            queue_id = Path(filename).stem
            result_path = f"{RESULTS_DIR}/{queue_id}.json"
            if bridge.exists(result_path):
                continue

            started = time.time()
            try:
                job = validate_job(raw, filename)
                payload = {
                    "schema": 1,
                    "job_id": job["job_id"],
                    "status": "running",
                    "started_unix": started,
                    "worker": "google-colab",
                    "job": job,
                }
                output = execute_job(job)
                payload.update(
                    {
                        "status": "complete",
                        "finished_unix": time.time(),
                        "output": output,
                    }
                )
            except Exception as exc:
                payload = {
                    "schema": 1,
                    "job_id": queue_id,
                    "status": "failed",
                    "started_unix": started,
                    "finished_unix": time.time(),
                    "worker": "google-colab",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

            if not bridge.exists(result_path):
                bridge.create_json(
                    result_path,
                    payload,
                    f"rlt bridge: record {queue_id} result",
                )
            processed += 1
        except Exception as exc:
            print(f"[bridge] queue item {filename} could not be processed: {exc}", flush=True)
    return processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is required. In Colab, store it in the Secrets panel."
        )
    if args.poll_seconds < 10:
        raise ValueError("poll-seconds must be >= 10")

    bridge = GitHubBridge(token=token, branch=args.branch)
    print(
        f"[bridge] watching {OWNER}/{REPO}:{args.branch}/{JOBS_DIR}; "
        "only bounded smoke/train schema-1 jobs are accepted",
        flush=True,
    )
    while True:
        count = process_once(bridge)
        if args.once:
            return
        if count:
            print(f"[bridge] processed {count} queued job(s)", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
