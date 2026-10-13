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
CONSUMED_JOBS = {("vinceackermann2-sys/tam-research", 1443)}
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
    fd = os.open(temporary, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
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
        if not required.issubset(manifest):
            raise ValueError("Missing required artifacts")
        metrics = json.loads(archive.read("metrics.json"))
        if job["script"].endswith("chm_v4_microfit_1436.py") and metrics.get("status") != "failed":
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


def cloud_copy_verified(path, destination, deadline):
    """Retain on cloud only, and verify an independent readback of every byte."""
    with tempfile.TemporaryDirectory(prefix="colab-cloud-readback-") as temporary:
        returned = Path(temporary) / "readback"
        options = ["--retries", "1", "--low-level-retries", "1"]
        cmd(["rclone", "copyto", "--immutable", *options, str(path), destination],
            timeout=remaining_seconds(deadline))
        cmd(["rclone", "copyto", *options, destination, str(returned)],
            timeout=remaining_seconds(deadline))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (returned.stat().st_size != path.stat().st_size or
                hashlib.sha256(returned.read_bytes()).hexdigest() != digest):
            raise ValueError("Cloud artifact readback mismatch")
        return {"destination": destination, "sha256": digest, "bytes": path.stat().st_size}


def collect_cloud_artifacts(args, job, number, session, output, deadline):
    destination = args.artifact_remote.rstrip("/") + f"/issue-{number}/{job['commit']}/"
    evidence = {"complete": False}
    with tempfile.TemporaryDirectory(prefix="colab-transfer-") as temporary:
        root = Path(temporary)
        log = root / "worker.log"
        log.write_text(output, encoding="utf-8")
        # Even a failed bootstrap may have no bundle. Preserve transport first.
        evidence["transport_log"] = cloud_copy_verified(log, destination + "worker.log", deadline)
        try:
            descriptor = artifact_descriptor(output)
            archive = root / "bundle.zip"
            cmd(["colab", "--auth=oauth2", "download", "--session", session,
                 ARTIFACT_REMOTE, str(archive)], timeout=remaining_seconds(deadline))
            verify_artifact(archive, descriptor, job, number)
            with zipfile.ZipFile(archive) as bundle:
                evidence["job_failed"] = json.loads(bundle.read("metrics.json")).get("status") == "failed"
            evidence["bundle"] = cloud_copy_verified(archive, destination + "bundle.zip", deadline)
            evidence["complete"] = True
        except Exception as error:
            evidence["bundle_error_type"] = type(error).__name__
            # Contents API retrieval needs no running kernel. Preserve a current
            # partial training log if a interrupted script created one.
            evidence["partial_files"] = {}
            for name in ("run.log", "configuration.json", "provenance.json", "checkpoint.pt"):
                try:
                    partial = root / name
                    cmd(["colab", "--auth=oauth2", "download", "--session", session,
                         "/content/tam-results/" + name, str(partial)], timeout=remaining_seconds(deadline))
                    if partial.stat().st_size > MAX_ARTIFACT_BYTES:
                        raise ValueError("Partial file too large")
                    evidence["partial_files"][name] = cloud_copy_verified(partial, destination + "partial-" + name, deadline)
                except Exception as error:
                    evidence["partial_files"][name] = {"error_type": type(error).__name__}
                    if isinstance(error, TimeoutError):
                        break
    return evidence


def execute(args, issue, job, source):
    number = issue["number"]
    if (args.repo.lower(), number) in CONSUMED_JOBS:
        return False
    research = job["script"].startswith("colab_jobs/")
    identity = research_identity(source) if research else None
    # Read-only account check before reservation or provisioning. Never use an old balance.
    before = account_snapshot()
    if before["balance_units"] <= 0 or before["active_assignments"] != 0:
        raise ValueError("Allocation blocked: empty balance or existing active assignment")
    if research:
        if not getattr(args, "artifact_remote", None):
            raise ValueError("An owner-approved private cloud artifact remote is required")
        cmd(["rclone", "lsf", args.artifact_remote, "--max-depth", "1"])
        help_output = cmd(["colab", "--auth=oauth2", "run", "--help"]).stdout
        if "--keep" not in help_output or "--timeout" not in help_output:
            raise ValueError("Installed Colab CLI lacks required lifecycle options")
    receipt = reserve(args.state_dir, number, job)
    if receipt is None:
        return False
    session = f"chatgpt-issue-{number}"
    log = None if research else args.state_dir / f"issue-{number}.log"
    output, error, exit_code = "", None, None
    stop_ok, artifact_ok = False, False
    after = None
    evidence = None
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
                       "--timeout", str(max(0.01, remaining_seconds(deadline) - min(45, args.max_seconds // 3)))]
            if research:
                # Keep only until private cloud export/readback; then stop in finally.
                command += ["--keep"]
            if job["gpu"] != "CPU":
                command += ["--gpu", job["gpu"]]
            command.append(str(filename))
            if research:
                command += ["--repo-commit", job["commit"], "--issue", str(number),
                            "--job-script", job["script"], "--requested-gpu", job["gpu"],
                            "--script-sha256", hashlib.sha256(source.encode("utf-8")).hexdigest()]
            try:
                p = cmd(command, timeout=max(0.01, remaining_seconds(deadline) - min(45, args.max_seconds // 3)), check=False)
                exit_code = p.returncode
                output = p.stdout + "\n" + p.stderr
            except subprocess.TimeoutExpired as ex:
                raw = ex.stdout or b""
                output = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
                raw = ex.stderr or b""
                output += "\n" + (raw.decode(errors="replace") if isinstance(raw, bytes) else raw)
                error = type(ex).__name__
            if research:
                evidence = collect_cloud_artifacts(args, job, number, session, output, deadline)
                artifact_ok = evidence["complete"]
                if evidence.get("job_failed"):
                    error = "RuntimeJobFailed"
    except BaseException as ex:
        # Never publish exception text or CLI stderr: they can contain local auth data.
        error = type(ex).__name__
    finally:
        try:
            cmd(["colab", "--auth=oauth2", "stop", "--session", session],
                timeout=90, check=False)
        except Exception:
            pass
        try:
            after = account_snapshot()
            stop_ok = (after["active_assignments"] == 0 and
                       session not in after["sessions"])
        except Exception:
            pass
        if log is not None:
            log.write_text(output, encoding="utf-8")
            os.chmod(log, 0o600)
            with open(log, "rb") as file:
                os.fsync(file.fileno())
        status = ("SUCCESS" if exit_code == 0 and error is None and stop_ok and
                  after is not None and (artifact_ok or not research) else "FAILED")
        record = {"issue": number, "job": job, "state": status, "retry_allowed": False,
                  "exit_code": exit_code, "stop_ok": stop_ok, "artifact_ok": artifact_ok,
                  "before": before, "after": after, "error_type": error,
                  "job_id": identity, "local_log": str(log) if log else None,
                  "local_archive": None, "cloud_evidence": evidence}
        # Final receipt is durable even if public callback fails.
        atomic_json(receipt, record)
        summary = (f"Colab Pro bridge: {status}; no automatic retries.\n"
                   f"Commit: {job['commit']}; script: {job['script']}; "
                   f"requested GPU: {job['gpu']}; exit: {exit_code}; stop confirmed: {stop_ok}; "
                   f"artifact retrieval verified: {artifact_ok}.\n"
                   f"Fresh usage checked before/after: {after is not None}.\n"
                   "Research artifacts/logs use private cloud storage; local credentials, "
                   "attempt receipts and CLI diagnostics stay local.")
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
        if (args.repo.lower(), issue["number"]) in CONSUMED_JOBS:
            continue
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
            current = gh_json(f"repos/{args.repo}/issues/{issue['number']}")
            if (current.get("state") != "open" or
                "colab-approved" not in {x["name"] for x in current.get("labels", [])} or
                parse_issue(current, owner) != job):
                continue
        except Exception as ex:
            print(f"Issue #{issue['number']} preflight skipped: {type(ex).__name__}; inspect locally", file=sys.stderr)
            continue
        try:
            if execute(args, issue, job, source):
                consumed += 1
        except Exception:
            if (args.state_dir / f"issue-{issue['number']}.json").exists():
                # A final persistence failure cannot erase a consumed attempt.
                return consumed + 1
            raise
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
    p.add_argument("--artifact-remote", default=None,
                   help="Approved private rclone cloud folder; required for research jobs")
    a = p.parse_args()
    if not 10 <= a.interval <= 3600 or not 30 <= a.max_seconds <= 300:
        p.error("interval must be 10..3600; max-seconds must be 30..300")
    if not 1 <= a.max_attempts <= 10:
        p.error("max-attempts must be 1..10; default is 1")
    a.allowed_gpus = {g.strip() for g in a.allowed_gpus.split(",") if g.strip()}
    if not a.allowed_gpus or not a.allowed_gpus.issubset(ALLOWED_GPUS):
        p.error("Allowed GPUs must be a subset of " + ",".join(sorted(ALLOWED_GPUS)))
    a.repo_path = a.repo_path.resolve()
    if a.artifact_remote and not re.fullmatch(r"[A-Za-z0-9_-]+:.+", a.artifact_remote):
        p.error("artifact-remote must identify a cloud remote and folder")
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
            "cloud_artifacts_configured": bool(a.artifact_remote),
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
