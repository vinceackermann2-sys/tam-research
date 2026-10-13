#!/usr/bin/env python3
"""Execute approved GitHub issue jobs on personal Colab Pro (Linux/macOS/WSL)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

MARKER = "<!-- colab-pro-job:v1 -->"
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SCRIPT_PATTERN = re.compile(r"(?:colab_jobs/[A-Za-z0-9_/-]+|scripts/colab_bridge_gpu_probe)\.py")
ALLOWED_GPUS = {"CPU", "T4", "L4", "G4", "A100", "H100"}
PUBLIC_KEYS = {"status", "gpu", "cuda_available", "torch", "elapsed_sec", "loss", "note"}


def cmd(args, cwd=None, timeout=60, check=True):
    p = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"{args[0]} exited {p.returncode}: {p.stderr[-500:]}")
    return p


def gh_json(endpoint):
    return json.loads(cmd(["gh", "api", endpoint]).stdout)


def comment(repo, number, body):
    cmd(["gh", "issue", "comment", str(number), "-R", repo, "--body", body])


def parse_issue(issue, owner):
    if "pull_request" in issue or issue.get("user", {}).get("login", "").lower() != owner.lower():
        return None
    body = issue.get("body") or ""
    if not body.startswith(MARKER):
        return None
    fence = chr(96) * 3
    pattern = re.escape(fence) + r"json\s*(\{.*?\})\s*" + re.escape(fence)
    match = re.search(pattern, body, flags=re.DOTALL)
    if not match:
        return None
    try:
        job = json.loads(match.group(1))
    except ValueError:
        return None
    if set(job) != {"commit", "script", "gpu"}:
        return None
    if not all(isinstance(x, str) for x in job.values()):
        return None
    if not SHA_PATTERN.fullmatch(job["commit"]):
        return None
    if not SCRIPT_PATTERN.fullmatch(job["script"]) or ".." in job["script"]:
        return None
    if job["gpu"] not in ALLOWED_GPUS:
        return None
    return job


def source_from_main(repo_dir, job):
    cmd(["git", "fetch", "--quiet", "origin", "main"], cwd=repo_dir, timeout=120)
    check = cmd(["git", "merge-base", "--is-ancestor", job["commit"], "FETCH_HEAD"],
                cwd=repo_dir, check=False)
    if check.returncode:
        raise ValueError("Requested commit is not already merged into origin/main")
    return cmd(["git", "show", job["commit"] + ":" + job["script"]],
               cwd=repo_dir, timeout=30).stdout


def publishable_metrics(output):
    for line in reversed(output.splitlines()):
        if line.startswith("COLAB_PUBLIC_RESULT="):
            try:
                data = json.loads(line.split("=", 1)[1])
                if not isinstance(data, dict):
                    return None
                return {k: v[:160] if isinstance(v, str) else v
                        for k, v in data.items()
                        if k in PUBLIC_KEYS and type(v) in (str, int, float, bool)}
            except ValueError:
                return None
    return None


def reserve(state_dir, issue_number, job):
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"issue-{issue_number}.json"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w") as file:
        json.dump({"issue": issue_number, "job": job,
                   "state": "reserved", "retry_allowed": False}, file)
        file.flush()
        os.fsync(file.fileno())
    return path


def execute(args, issue, job, source):
    number = issue["number"]
    receipt = reserve(args.state_dir, number, job)
    if receipt is None:
        return False
    session = f"chatgpt-issue-{number}"
    log = args.state_dir / f"issue-{number}.log"
    output = ""
    error = None
    exit_code = None
    stop_ok = False
    try:
        comment(args.repo, number, "Colab bridge: one attempt reserved. No automatic retry.")
        with tempfile.TemporaryDirectory(prefix="colab-chatgpt-") as temporary:
            filename = Path(temporary) / "job.py"
            filename.write_text(source, encoding="utf-8")
            command = ["colab", "--auth=oauth2", "run", "--session", session]
            if job["gpu"] != "CPU":
                command += ["--gpu", job["gpu"]]
            command.append(str(filename))
            try:
                p = cmd(command, timeout=args.max_seconds, check=False)
                exit_code = p.returncode
                output = p.stdout + "\n" + p.stderr
            except subprocess.TimeoutExpired as ex:
                error = f"Timeout after {args.max_seconds}s. Attempt not retryable."
                raw = ex.stdout or b""
                output = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
    except Exception as ex:
        error = str(ex)[:250]
    finally:
        try:
            cmd(["colab", "--auth=oauth2", "stop", "--session", session],
                timeout=90, check=False)
            # 'colab run' normally auto-stops, so a second stop may exit nonzero.
            sessions = cmd(["colab", "--auth=oauth2", "sessions"],
                           timeout=60, check=False)
            stop_ok = sessions.returncode == 0 and session not in sessions.stdout
        except Exception:
            pass
        log.write_text(output[-150000:], encoding="utf-8")
        status = "SUCCESS" if exit_code == 0 and error is None else "FAILED"
        summary = (f"Colab Pro bridge: {status}; no automatic retries.\n"
                   f"Commit: {job['commit']}; script: {job['script']}; "
                   f"requested GPU: {job['gpu']}; exit: {exit_code}; stop confirmed: {stop_ok}.\n"
                   "Full logs are stored locally on the worker, not on public GitHub.")
        metrics = publishable_metrics(output)
        if metrics:
            summary += "\nPublic result: " + json.dumps(metrics, sort_keys=True)[:1200]
        if error:
            summary += "\nError: " + error.replace("\n", " ")
        try:
            comment(args.repo, number, summary)
        except Exception as ex:
            print(f"Failed to post status for issue #{number}: {ex}", file=sys.stderr)
        receipt.write_text(json.dumps({"issue": number, "job": job, "state": status,
                                        "exit_code": exit_code, "stop_ok": stop_ok,
                                        "local_log": str(log)}, indent=2) + "\n")
        print(f"Issue #{number}: {status}; local log: {log}; stop: {stop_ok}")
    return True


def sweep(args):
    """Consume at most args.max_attempts jobs; return attempts consumed."""
    owner = args.repo.split("/", 1)[0]
    endpoint = f"repos/{args.repo}/issues?state=open&labels=colab-approved&per_page=100"
    consumed = 0
    for issue in gh_json(endpoint):
        job = parse_issue(issue, owner)
        if not job:
            continue
        if job["gpu"] not in args.allowed_gpus:
            print(f"Issue #{issue['number']} skipped: GPU {job['gpu']} not allowed", file=sys.stderr)
            continue
        if (args.state_dir / f"issue-{issue['number']}.json").exists():
            continue
        try:
            source = source_from_main(args.repo_path, job)
        except Exception as ex:
            print(f"Issue #{issue['number']} preflight skipped: {ex}", file=sys.stderr)
            continue
        if execute(args, issue, job, source):
            consumed += 1
        if consumed >= args.max_attempts:
            break
    return consumed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="vinceackermann2-sys/tam-research")
    p.add_argument("--repo-path", type=Path, default=Path("."))
    p.add_argument("--state-dir", type=Path,
                   default=Path.home() / ".local/state/colab-pro-chatgpt")
    p.add_argument("--poll", action="store_true", help="Keep polling until one attempt")
    p.add_argument("--interval", type=int, default=45)
    p.add_argument("--max-seconds", type=int, default=180,
                   help="Wall-clock timeout including GPU allocation (max 300)")
    p.add_argument("--max-attempts", type=int, default=1,
                   help="Max attempts per worker invocation (default: 1)")
    p.add_argument("--allowed-gpus", default="CPU,T4",
                   help="Comma-separated accelerator allowlist; default CPU,T4")
    a = p.parse_args()
    if not 10 <= a.interval <= 3600 or not 30 <= a.max_seconds <= 300:
        p.error("interval must be 10..3600; max-seconds must be 30..300")
    if not 1 <= a.max_attempts <= 10:
        p.error("max-attempts must be 1..10; default is 1")
    a.allowed_gpus = {g.strip() for g in a.allowed_gpus.split(",") if g.strip()}
    if not a.allowed_gpus or not a.allowed_gpus.issubset(ALLOWED_GPUS):
        p.error("Allowed GPUs must be a subset of " + ",".join(sorted(ALLOWED_GPUS)))
    a.repo_path = a.repo_path.resolve()
    a.state_dir.mkdir(parents=True, exist_ok=True)
    # Prevent multiple workers on one computer from silently running in parallel.
    lock_file = open(a.state_dir / "worker.lock", "a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("Another Colab bridge worker is already running in this state directory")
    attempts = 0
    while attempts < a.max_attempts:
        try:
            attempts += sweep(a)
        except Exception as ex:
            print(f"Poll error: {ex}", file=sys.stderr)
            if not a.poll:
                raise
        if not a.poll:
            break
        if attempts < a.max_attempts:
            time.sleep(a.interval)



if __name__ == "__main__":
    main()
