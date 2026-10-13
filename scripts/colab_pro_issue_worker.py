#!/usr/bin/env python3
"""Execute approved GitHub issue jobs on personal Colab Pro (Linux/macOS/WSL)."""
import argparse
import fcntl
import hashlib
import json
import math
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
PUBLIC_KEYS = {"status", "gpu", "cuda_available", "torch", "elapsed_sec", "loss", "note"}
REMOTE_BUNDLE = "/content/tam-results/bundle.zip"
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
CONSUMED_JOBS = {("vinceackermann2-sys/tam-research", 1443)}


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
                        and (type(v) is not float or math.isfinite(v))}
            except ValueError:
                return None
    return None


def durable_json(path, data):
    """Atomic replacement; an interrupted final write must retain the receipt."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def balance():
    """Official CLI display contract; account details stay on the worker."""
    output = cmd(["colab", "--auth=oauth2", "usage"]).stdout
    match = re.search(r"^Current balance: (\d+(?:\.\d+)?) compute units$", output, re.M)
    assignments = re.search(r"^Active assignments: (\d+)$", output, re.M)
    if not match or not assignments:
        raise ValueError("Cannot verify current compute balance/assignments")
    value = float(match[1])
    if not math.isfinite(value):
        raise ValueError("Invalid compute balance")
    return {"compute_units": value, "active_assignments": int(assignments[1]),
            "checked_at": time.time()}


def verify_bundle(path, commit):
    """Verify downloaded bytes without extracting paths supplied by a runtime."""
    if not 0 < path.stat().st_size <= MAX_BUNDLE_BYTES:
        raise ValueError("Artifact bundle size outside small-job limit")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or len(names) > 64:
            raise ValueError("Duplicate/excess artifact entries")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+\.(json|log|txt|pt)", name)
               for name in names):
            raise ValueError("Unsafe artifact path")
        if sum(item.file_size for item in infos) > MAX_BUNDLE_BYTES:
            raise ValueError("Uncompressed artifact bundle exceeds limit")
        manifest = json.loads(archive.read("manifest.json"))
        files = manifest.get("files")
        if (manifest.get("schema") != 1 or manifest.get("commit") != commit
                or not isinstance(files, dict)
                or set(files) != set(names) - {"manifest.json"}
                or not {"config.json", "result.json", "environment.json", "job.log"} <= set(files)):
            raise ValueError("Artifact manifest/source identity mismatch")
        config = json.loads(archive.read("config.json"))
        if config.get("commit") != commit:
            raise ValueError("Configuration commit mismatch")
        result = json.loads(archive.read("result.json"))
        if result.get("status") not in {"ok", "failed", "pending"}:
            raise ValueError("Missing artifact job outcome")
        for name, entry in files.items():
            data = archive.read(name)
            if (not isinstance(entry, dict) or entry.get("bytes") != len(data)
                    or entry.get("sha256") != hashlib.sha256(data).hexdigest()):
                raise ValueError("Artifact integrity check failed")
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size, "files": sorted(files),
            "result_status": result["status"]}


def collect_artifacts(args, number, session, commit, deadline, output=""):
    """Transient staging only; retention is a configured cloud remote."""
    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Artifact retrieval deadline exceeded")
        return seconds

    destination = args.artifact_remote.rstrip("/") + f"/issue-{number}/{commit}/bundle.zip"
    with tempfile.TemporaryDirectory(prefix="colab-artifacts-") as temporary:
        bundle = Path(temporary) / "bundle.zip"
        retrieved = Path(temporary) / "cloud-readback.zip"
        # Retain transport evidence even when a timed-out job never made a bundle.
        transport_log = Path(temporary) / "worker.log"
        transport_log.write_text(output, encoding="utf-8")
        log_destination = destination.removesuffix("bundle.zip") + "worker.log"
        cmd(["rclone", "copyto", "--immutable", "--retries", "1", "--low-level-retries", "1",
             str(transport_log), log_destination], timeout=remaining())
        returned_log = Path(temporary) / "cloud-worker.log"
        cmd(["rclone", "copyto", "--retries", "1", "--low-level-retries", "1",
             log_destination, str(returned_log)], timeout=remaining())
        digest = hashlib.sha256(transport_log.read_bytes()).hexdigest()
        if hashlib.sha256(returned_log.read_bytes()).hexdigest() != digest:
            raise ValueError("Cloud transport log readback differs")
        verified = {"transport_log_sha256": digest, "transport_log_destination": log_destination,
                    "complete": False, "verified_before_teardown": True}
        try:
            cmd(["colab", "--auth=oauth2", "download", "--session", session,
                 REMOTE_BUNDLE, str(bundle)], timeout=remaining())
            manifest = verify_bundle(bundle, commit)
            cmd(["rclone", "copyto", "--immutable", "--retries", "1", "--low-level-retries", "1", str(bundle), destination],
                timeout=remaining())
            cmd(["rclone", "copyto", "--retries", "1", "--low-level-retries", "1", destination, str(retrieved)], timeout=remaining())
            if verify_bundle(retrieved, commit) != manifest:
                raise ValueError("Cloud artifact readback differs from runtime download")
            verified.update(manifest)
            verified["complete"] = manifest["result_status"] != "pending"
        except Exception as error:
            verified["bundle_error_class"] = type(error).__name__
        if "sha256" in verified:
            verified["cloud_destination"] = destination
        if not verified["complete"]:
            # The startup bundle may predate a stalled bootstrap. Retrieve its
            # current file through the Contents API without executing kernel code.
            try:
                snapshot = Path(temporary) / "job-partial.log"
                cmd(["colab", "--auth=oauth2", "download", "--session", session,
                     "/content/tam-results/job.log", str(snapshot)], timeout=remaining())
                if snapshot.stat().st_size > MAX_BUNDLE_BYTES:
                    raise ValueError("Partial log exceeds small-job limit")
                partial_destination = destination.removesuffix("bundle.zip") + "job-partial.log"
                cmd(["rclone", "copyto", "--immutable", "--retries", "1", "--low-level-retries", "1",
                     str(snapshot), partial_destination], timeout=remaining())
                returned = Path(temporary) / "cloud-partial.log"
                cmd(["rclone", "copyto", "--retries", "1", "--low-level-retries", "1",
                     partial_destination, str(returned)], timeout=remaining())
                partial_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
                if hashlib.sha256(returned.read_bytes()).hexdigest() != partial_hash:
                    raise ValueError("Cloud partial-log readback differs")
                verified["partial_log_destination"] = partial_destination
                verified["partial_log_sha256"] = partial_hash
            except Exception as error:
                verified["partial_log_error_class"] = type(error).__name__
        verified["verified_before_teardown"] = True
        return verified


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
    sync_directory(state_dir)
    return path


def execute(args, issue, job, source):
    number = issue["number"]
    research = job["script"].startswith("colab_jobs/")
    if (args.repo.lower(), number) in CONSUMED_JOBS:
        return False
    if (args.state_dir / "worker-blocked.json").exists():
        raise RuntimeError("Resolve previous unconfirmed teardown/balance before allocation")
    # These preflights never allocate a runtime or consume a scientific attempt.
    before = balance()
    if before["compute_units"] <= 0 or before["active_assignments"] != 0:
        raise RuntimeError("Positive balance and zero active assignments required")
    if research:
        if not getattr(args, "artifact_remote", None):
            raise RuntimeError("Configure an approved cloud artifact remote before allocation")
        cmd(["rclone", "lsf", args.artifact_remote, "--max-depth", "1"])
        help_output = cmd(["colab", "--auth=oauth2", "run", "--help"]).stdout
        if "--keep" not in help_output or "--timeout" not in help_output:
            raise RuntimeError("Installed Colab CLI lacks required lifecycle options")
    receipt = reserve(args.state_dir, number, job)
    if receipt is None:
        return False
    session = f"chatgpt-issue-{number}"
    log = None if research else args.state_dir / f"issue-{number}.log"
    output = ""
    error = None
    exit_code = None
    stop_ok = False
    artifacts = None
    after = None
    phase = "launch"
    deadline = time.monotonic() + args.max_seconds
    try:
        comment(args.repo, number, "Colab bridge: one attempt reserved. No automatic retry.")
        with tempfile.TemporaryDirectory(prefix="colab-chatgpt-") as temporary:
            filename = Path(temporary) / "job.py"
            filename.write_text(source, encoding="utf-8")
            command = ["colab", "--auth=oauth2", "run", "--session", session]
            if research:
                # Reserve part of the unchanged outer execution deadline for export.
                execution_seconds = args.max_seconds - min(45, args.max_seconds // 3)
                command += ["--keep", "--timeout", str(execution_seconds)]
            if job["gpu"] != "CPU":
                command += ["--gpu", job["gpu"]]
            command.append(str(filename))
            if research:
                command += [job["commit"], job["gpu"]]
            try:
                timeout = (execution_seconds if research else args.max_seconds)
                p = cmd(command, timeout=min(timeout, max(0.01, deadline - time.monotonic())), check=False)
                exit_code = p.returncode
                output = p.stdout + "\n" + p.stderr
            except subprocess.TimeoutExpired as ex:
                error = f"Timeout after {args.max_seconds}s. Attempt not retryable."
                raw = ex.stdout or b""
                output = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
                raw_error = ex.stderr or b""
                output += "\n" + (raw_error.decode(errors="replace") if isinstance(raw_error, bytes) else raw_error)
            if research:
                phase = "artifact-export"
                # Export failed-job evidence too, when the script produced its bundle.
                artifacts = collect_artifacts(args, number, session, job["commit"], deadline, output)
                if not artifacts.get("complete"):
                    error = "Artifact bundle incomplete; cloud transport evidence retained."
    except Exception as ex:
        # Raw CLI errors/logs may include account information; retain privately only.
        output += f"\n{phase}: {type(ex).__name__}: {ex}\n"
        error = f"{phase} failed ({type(ex).__name__}). Consult receipt/cloud evidence."
    finally:
        # New research logs were cloud-verified before cleanup. Preserve legacy
        # probe logging without changing historical local records.
        if log is not None:
            try:
                with log.open("w", encoding="utf-8") as file:
                    file.write(output)
                    file.flush()
                    os.fsync(file.fileno())
            except OSError:
                error = "Private log persistence failed."
        try:
            cmd(["colab", "--auth=oauth2", "stop", "--session", session],
                timeout=90, check=False)
            # 'colab run' normally auto-stops, so a second stop may exit nonzero.
            sessions = cmd(["colab", "--auth=oauth2", "sessions"],
                           timeout=60, check=False)
            stop_ok = sessions.returncode == 0 and session not in sessions.stdout
        except Exception:
            pass
        try:
            after = balance()
        except Exception:
            pass
        status = "SUCCESS" if (exit_code == 0 and error is None and stop_ok
                               and after is not None and after["active_assignments"] == 0
                               and (not research or (artifacts and artifacts.get("complete")
                                    and artifacts.get("result_status") == "ok"))) else "FAILED"
        if not stop_ok or after is None or after["active_assignments"] != 0:
            durable_json(args.state_dir / "worker-blocked.json", {
                "issue": number, "stop_ok": stop_ok, "balance_verified": after is not None,
                "instruction": "Check Colab sessions and usage; resolve before starting another worker."})
        summary = (f"Colab Pro bridge: {status}; no automatic retries.\n"
                   f"Commit: {job['commit']}; script: {job['script']}; "
                   f"requested GPU: {job['gpu']}; exit: {exit_code}; stop confirmed: {stop_ok}.\n"
                   + ("Job and transport logs use the private cloud artifact destination; "
                    "local authentication, attempt receipts and CLI diagnostics remain local." if research else
                    "Legacy probe logs are stored locally on the worker, not on public GitHub."))
        summary += (f"\nCloud artifact bundle complete: {bool(artifacts and artifacts.get('complete'))}; "
                    f"cloud evidence verified before teardown: {bool(artifacts)}; "
                    f"post-attempt balance checked: {after is not None}.")
        metrics = publishable_metrics(output)
        if metrics:
            summary += "\nPublic result: " + json.dumps(metrics, sort_keys=True)[:1200]
        if error:
            summary += "\nError: " + error.replace("\n", " ")
        try:
            comment(args.repo, number, summary)
        except Exception as ex:
            print(f"Failed to post status for issue #{number}: {ex}", file=sys.stderr)
        durable_json(receipt, {"issue": number, "job": job, "state": status,
                              "retry_allowed": False, "exit_code": exit_code, "stop_ok": stop_ok,
                              "balance_before": before, "balance_after": after,
                              "artifacts": artifacts, "local_log": str(log) if log else None})
        print(f"Issue #{number}: {status}; cloud artifacts: {bool(artifacts)}; stop: {stop_ok}")
    return True


def sweep(args, remaining):
    """Consume at most args.max_attempts jobs; return attempts consumed."""
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
            source = source_from_main(args.repo_path, job)
            # Approval and contents can change while source is fetched.
            current = gh_json(f"repos/{args.repo}/issues/{issue['number']}")
            if (current.get("state") != "open" or
                    "colab-approved" not in {x["name"] for x in current.get("labels", [])} or
                    parse_issue(current, owner) != job):
                continue
        except Exception as ex:
            print(f"Issue #{issue['number']} preflight skipped: {ex}", file=sys.stderr)
            continue
        receipt = args.state_dir / f"issue-{issue['number']}.json"
        try:
            if execute(args, issue, job, source):
                consumed += 1
        except Exception:
            if receipt.exists():
                # Count the durable reservation even if final persistence failed.
                # End this sweep; never lose paid attempts through an exception.
                print(f"Issue #{issue['number']}: reserved attempt failed; no retry.", file=sys.stderr)
                return consumed + 1
            raise
        if consumed >= remaining:
            break
    return consumed


def worker_status(state_dir):
    """Local liveness observation, never an assertion about a remote computer."""
    path = state_dir / "worker-status.json"
    if not path.exists():
        return {"running": False, "reason": "No worker heartbeat"}
    data = json.loads(path.read_text())
    lock_path = state_dir / "worker.lock"
    if not lock_path.exists():
        return {"running": False, "reason": "No worker lock"}
    with lock_path.open("r+") as file:
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = False
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            locked = True
    age = time.time() - data.get("heartbeat_at", 0)
    running = locked and data.get("state") == "running" and 0 <= age < 480
    return {"running": running, "heartbeat_age_seconds": round(age, 1),
            "settings": data.get("settings"), "attempts": data.get("attempts"),
            "blocked": (state_dir / "worker-blocked.json").exists()}


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
    p.add_argument("--artifact-remote", default=None,
                   help="Approved rclone cloud folder; required for colab_jobs (no local retention)")
    p.add_argument("--status", action="store_true",
                   help="Read local worker lock/heartbeat and current balance; never allocate")
    a = p.parse_args()
    if not 10 <= a.interval <= 3600 or not 30 <= a.max_seconds <= 300:
        p.error("interval must be 10..3600; max-seconds must be 30..300")
    if not 1 <= a.max_attempts <= 10:
        p.error("max-attempts must be 1..10; default is 1")
    a.allowed_gpus = {g.strip() for g in a.allowed_gpus.split(",") if g.strip()}
    if not a.allowed_gpus or not a.allowed_gpus.issubset(ALLOWED_GPUS):
        p.error("Allowed GPUs must be a subset of " + ",".join(sorted(ALLOWED_GPUS)))
    a.repo_path = a.repo_path.resolve()
    if a.artifact_remote and (not re.fullmatch(r"[A-Za-z0-9_-]+:.+", a.artifact_remote)
                              or a.artifact_remote.startswith("-")):
        p.error("artifact-remote must name a configured cloud remote and folder")
    if a.status:
        data = worker_status(a.state_dir)
        try:
            data["current_usage"] = balance()
        except Exception:
            data["current_usage"] = None
        print(json.dumps(data, sort_keys=True))
        return
    a.state_dir.mkdir(parents=True, exist_ok=True)
    # Prevent multiple workers on one computer from silently running in parallel.
    lock_file = open(a.state_dir / "worker.lock", "a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("Another Colab bridge worker is already running in this state directory")
    attempts = 0
    status_path = a.state_dir / "worker-status.json"
    settings = {"max_seconds": a.max_seconds, "max_attempts": a.max_attempts,
                "allowed_gpus": sorted(a.allowed_gpus),
                "cloud_artifacts_configured": bool(a.artifact_remote)}
    try:
        while attempts < a.max_attempts:
            durable_json(status_path, {"state": "running", "heartbeat_at": time.time(),
                                       "attempts": attempts, "settings": settings})
            if (a.state_dir / "worker-blocked.json").exists():
                raise RuntimeError("Unconfirmed cleanup/balance: inspect worker-blocked.json")
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
    finally:
        durable_json(status_path, {"state": "stopped", "heartbeat_at": time.time(),
                                   "attempts": attempts, "settings": settings})



if __name__ == "__main__":
    main()
