#!/usr/bin/env python3
"""Execute approved GitHub issue jobs on personal Colab Pro (Linux/macOS/WSL)."""
import argparse
import ast
import fcntl
import hashlib
import math
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import zipfile

MARKER = "<!-- colab-pro-job:v1 -->"
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SCRIPT_PATTERN = re.compile(r"(?:colab_jobs/[A-Za-z0-9_/-]+|scripts/colab_bridge_gpu_probe)\.py")
ALLOWED_GPUS = {"CPU", "T4", "L4", "G4", "A100", "H100"}
ARTIFACT_REMOTE = "/content/tam-results/bundle.zip"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
ATTEMPT_MARKER = "<!-- colab-attempt-reserved:v1 -->"
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
                        if k in PUBLIC_KEYS and type(v) in (str, int, float, bool)
                        and (not isinstance(v, float) or math.isfinite(v))}
            except ValueError:
                return None
    return None


def reserve(state_dir, issue_number, job):
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state_dir, 0o700)
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
    directory = os.open(state_dir, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def account_snapshot():
    usage = cmd(["colab", "--auth=oauth2", "usage"]).stdout
    sessions = cmd(["colab", "--auth=oauth2", "sessions"]).stdout
    match = re.search(r"^Current balance: ([0-9]+(?:\.[0-9]+)?) compute units$",
                      usage, re.MULTILINE)
    assignments = re.search(r"^Active assignments: ([0-9]+)$", usage, re.MULTILINE)
    if not match or not assignments:
        raise ValueError("Cannot verify current Colab usage format; allocation blocked")
    balance = float(match.group(1))
    if not math.isfinite(balance):
        raise ValueError("Nonfinite compute balance")
    return {"checked_at": time.time(), "balance_units": balance,
            "active_assignments": int(assignments.group(1)), "sessions": sessions}


def remaining_seconds(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Allocation/execution/retrieval deadline expired")
    return remaining


def artifact_descriptor(output):
    for line in reversed(output.splitlines()):
        if line.startswith("COLAB_ARTIFACT="):
            data = json.loads(line.split("=", 1)[1])
            if (set(data) != {"sha256", "bytes"} or
                not isinstance(data["sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", data["sha256"]) or
                type(data["bytes"]) is not int or
                not 0 < data["bytes"] <= MAX_ARTIFACT_BYTES):
                raise ValueError("Invalid artifact descriptor")
            return data
    raise ValueError("No complete artifact descriptor received")


def verify_artifact(path, descriptor, job, issue_number):
    if path.stat().st_size != descriptor["bytes"]:
        raise ValueError("Artifact size mismatch")
    if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
        raise ValueError("Artifact digest mismatch")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [e.filename for e in entries]
        if (len(names) != len(set(names)) or len(names) > 64 or
            sum(e.file_size for e in entries) > MAX_ARTIFACT_BYTES or
            any(not re.fullmatch(r"[A-Za-z0-9_.-]+", n) for n in names)):
            raise ValueError("Unsafe artifact archive")
        manifest = json.loads(archive.read("manifest.json"))
        if set(manifest) != set(names) - {"manifest.json"}:
            raise ValueError("Incomplete artifact manifest")
        required = {"configuration.json", "metrics.json", "run.log", "provenance.json"}
        if job["script"].endswith("chm_v4_microfit_1436.py"):
            required.add("checkpoint.pt")
        if not required.issubset(manifest):
            raise ValueError("Missing required artifacts")
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Artifact member digest mismatch")
        provenance = json.loads(archive.read("provenance.json"))
        if (provenance.get("commit") != job["commit"] or
            provenance.get("issue") != issue_number or
            provenance.get("script") != job["script"]):
            raise ValueError("Artifact provenance mismatch")
    os.chmod(path, 0o600)
    with open(path, "rb") as file:
        os.fsync(file.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return descriptor["sha256"]


def remotely_consumed(repo, number):
    pages = json.loads(cmd(["gh", "api", "--paginate", "--slurp",
                          f"repos/{repo}/issues/{number}/comments?per_page=100"]).stdout)
    owner = repo.split("/", 1)[0].lower()
    return any(c.get("user", {}).get("login", "").lower() == owner and
               (ATTEMPT_MARKER in (c.get("body") or "") or
                "Colab bridge: one attempt reserved." in (c.get("body") or "") or
                "Colab Pro bridge:" in (c.get("body") or ""))
               for page in pages for c in page)


def research_identity(source):
    identities = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "COLAB_JOB_ID" for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                identities.append(node.value.value)
    if len(identities) != 1 or not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,100}", identities[0]):
        raise ValueError("Research script needs exactly one reviewed literal COLAB_JOB_ID")
    return identities[0]


def execute(args, issue, job, source):
    number = issue["number"]
    research = job["script"].startswith("colab_jobs/")
    identity = research_identity(source) if research else None
    # Read-only account check before reservation or provisioning. Never use an old balance.
    before = account_snapshot()
    if before["balance_units"] <= 0 or before["active_assignments"] != 0:
        raise ValueError("Allocation blocked: empty balance or existing active assignment")
    receipt = reserve(args.state_dir, number, job)
    if receipt is None:
        return False
    session = f"chatgpt-issue-{number}"
    log = args.state_dir / f"issue-{number}.log"
    archive = args.state_dir / f"issue-{number}-bundle.zip"
    output, error, exit_code = "", None, None
    stop_ok, artifact_ok = False, False
    after = None
    deadline = time.monotonic() + args.max_seconds
    try:
        if identity:
            # GitHub ref creation is atomic across issues AND worker machines.
            # Existing tags are never updated/deleted; conflict consumes this local receipt.
            cmd(["gh", "api", "--method", "POST", f"repos/{args.repo}/git/refs",
                 "-f", "ref=refs/tags/colab-attempts/" + identity, "-f", "sha=" + job["commit"]])
        # Remote create-once evidence also blocks a lost local receipt / changed machine.
        comment(args.repo, number, ATTEMPT_MARKER +
                "\nColab bridge: one attempt reserved. No automatic retry.")
        with tempfile.TemporaryDirectory(prefix="colab-chatgpt-") as temporary:
            filename = Path(temporary) / "job.py"
            filename.write_text(source, encoding="utf-8")
            command = ["colab", "--auth=oauth2", "run", "--session", session,
                       "--timeout", str(remaining_seconds(deadline))]
            if research:
                # Keep runtime until checksum-verified download is durably on the worker.
                command += ["--keep"]
            if job["gpu"] != "CPU":
                command += ["--gpu", job["gpu"]]
            command.append(str(filename))
            if research:
                command += ["--repo-commit", job["commit"], "--issue", str(number),
                            "--job-script", job["script"], "--requested-gpu", job["gpu"],
                            "--script-sha256", hashlib.sha256(source.encode("utf-8")).hexdigest()]
            try:
                p = cmd(command, timeout=remaining_seconds(deadline), check=False)
                exit_code = p.returncode
                output = p.stdout + "\n" + p.stderr
            except subprocess.TimeoutExpired as ex:
                raw = ex.stdout or b""
                output = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
                raise
            if research:
                descriptor = artifact_descriptor(output)
                cmd(["colab", "--auth=oauth2", "download", "--session", session,
                     ARTIFACT_REMOTE, str(archive)], timeout=remaining_seconds(deadline))
                verify_artifact(archive, descriptor, job, number)
                artifact_ok = True
    except BaseException as ex:
        # Never publish exception text or CLI stderr: they can contain local auth data.
        error = type(ex).__name__
    finally:
        try:
            cmd(["colab", "--auth=oauth2", "stop", "--session", session],
                timeout=90, check=False)
            after = account_snapshot()
            stop_ok = (after["active_assignments"] == 0 and
                       session not in after["sessions"])
        except Exception:
            pass
        log.write_text(output, encoding="utf-8")
        os.chmod(log, 0o600)
        with open(log, "rb") as file:
            os.fsync(file.fileno())
        status = ("SUCCESS" if exit_code == 0 and error is None and stop_ok and
                  after is not None and (artifact_ok or not research) else "FAILED")
        record = {"issue": number, "job": job, "state": status, "retry_allowed": False,
                  "exit_code": exit_code, "stop_ok": stop_ok, "artifact_ok": artifact_ok,
                  "before": before, "after": after, "error_type": error,
                  "job_id": identity, "local_log": str(log), "local_archive": str(archive) if artifact_ok else None}
        # Final receipt is durable even if public callback fails.
        atomic_json(receipt, record)
        summary = (f"Colab Pro bridge: {status}; no automatic retries.\n"
                   f"Commit: {job['commit']}; script: {job['script']}; "
                   f"requested GPU: {job['gpu']}; exit: {exit_code}; stop confirmed: {stop_ok}; "
                   f"artifact retrieval verified: {artifact_ok}.\n"
                   f"Balance before: {before['balance_units']} compute units; "
                   f"balance after: {after['balance_units'] if after else 'UNVERIFIED'}.\n"
                   "Full logs, receipts and retrieved artifacts remain on the local worker.")
        metrics = publishable_metrics(output)
        if metrics:
            summary += "\nPublic result: " + json.dumps(metrics, sort_keys=True)[:1200]
        if error:
            summary += "\nFailure type: " + error
        try:
            comment(args.repo, number, summary)
        except Exception:
            print(f"Failed to post status for issue #{number}; inspect local receipt", file=sys.stderr)
        print(f"Issue #{number}: {status}; stop: {stop_ok}; artifacts: {artifact_ok}")
    return True


def sweep(args, remaining):
    """Consume at most args.max_attempts jobs; return attempts consumed."""
    if remaining <= 0:
        return 0
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
            if remotely_consumed(args.repo, issue["number"]):
                continue
            source = source_from_main(args.repo_path, job)
        except Exception as ex:
            print(f"Issue #{issue['number']} preflight skipped: {type(ex).__name__}; inspect locally", file=sys.stderr)
            continue
        if execute(args, issue, job, source):
            consumed += 1
        if consumed >= remaining:
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
    p.add_argument("--status", action="store_true", help="Read-only worker/quota readiness")
    a = p.parse_args()
    if not 10 <= a.interval <= 3600 or not 30 <= a.max_seconds <= 300:
        p.error("interval must be 10..3600; max-seconds must be 30..300")
    if not 1 <= a.max_attempts <= 10:
        p.error("max-attempts must be 1..10; default is 1")
    a.allowed_gpus = {g.strip() for g in a.allowed_gpus.split(",") if g.strip()}
    if not a.allowed_gpus or not a.allowed_gpus.issubset(ALLOWED_GPUS):
        p.error("Allowed GPUs must be a subset of " + ",".join(sorted(ALLOWED_GPUS)))
    a.repo_path = a.repo_path.resolve()
    a.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(a.state_dir, 0o700)
    if a.status:
        status_lock = open(a.state_dir / "worker.lock", "a+")
        running = False
        try:
            fcntl.flock(status_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(status_lock.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            running = True
        finally:
            status_lock.close()
        snapshot = account_snapshot()
        snapshot.pop("sessions")
        heartbeat = a.state_dir / "worker-status.json"
        details = json.loads(heartbeat.read_text()) if heartbeat.exists() else None
        print(json.dumps({"worker_running": running, "worker": details,
                          "account": snapshot}, sort_keys=True))
        return
    # Prevent multiple workers on one computer from silently running in parallel.
    lock_file = open(a.state_dir / "worker.lock", "a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("Another Colab bridge worker is already running in this state directory")
    # Fail closed on a previous unconfirmed teardown/account check.
    for receipt_path in a.state_dir.glob("issue-*.json"):
        previous = json.loads(receipt_path.read_text())
        if previous.get("state") == "reserved" or not previous.get("stop_ok", False):
            p.error("Previous attempt has unverified teardown; inspect local receipts and sessions")
    attempts = 0
    while attempts < a.max_attempts:
        atomic_json(a.state_dir / "worker-status.json", {
            "pid": os.getpid(), "last_poll": time.time(), "max_seconds": a.max_seconds,
            "max_attempts": a.max_attempts, "allowed_gpus": sorted(a.allowed_gpus),
            "attempts_consumed": attempts,
        })
        try:
            attempts += sweep(a, a.max_attempts - attempts)
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
