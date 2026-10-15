"""Public GitHub retention for one reviewed, nonsecret CPU readiness report.

This adapter never posts a ZIP, checkpoint, raw transport log or exception text.
The caller must hold the bridge's exclusive lock and global single-use attempt
reservation. GitHub comments remain editable by their owner; this adapter only
creates, never edits/deletes, and verifies a separate API readback. All command
and archive-verification capabilities are injected by the local worker.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import time
import zipfile

REPOSITORY = "vinceackermann2-sys/tam-research"
READINESS_SCRIPT = "colab_jobs/chm_v4_readiness.py"
COMMENT_MARKER = "<!-- colab-readiness-artifacts:v1 -->"
PUBLIC_CONSENT_MARKER = "<!-- colab-public-readiness-artifacts:v1 -->"
JOB_MARKER = "<!-- colab-pro-job:v1 -->"
MAX_BUNDLE_BYTES = 128 * 1024
MAX_MEMBER_BYTES = 32 * 1024
MAX_REPORT_BYTES = 24 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}")
MODEL_BLOBS = {
    "tam_research/chm_v4_neural_hard_copy_train_1430.py": "78f9de5df6580cb85f557272af5d7f7cffccec55",
    "tam_research/chm_v4_independent_binding_benchmark_1421.py": "41fd46670a11f9db9615de020c85a80cc4a0db00",
    "tam_research/chm_v3_matched_tiny_models_1377.py": "ffe14b0701e18493a2bf9b45a1238b5acd5e3eab",
}
CONFIGURATION = {"scope": "source/bootstrap/artifact preflight", "optimizer_steps": 0,
                 "dependencies": "stdlib only"}
METRICS = {"status": "source-ready", "training_executed": False,
           "heldout_generated": False, "source_blobs": MODEL_BLOBS}
CANONICAL_LOG = "Frozen source blobs and Python syntax verified. No model imported or trained.\n"
MEMBERS = {"configuration.json", "metrics.json", "provenance.json", "run.log"}
MEMBER_ENCODING = "runtime-json-sort-keys-indent2-newline-utf8;canonical-log-exact"
INCOMPLETE_REASONS = {"bootstrap-failed", "artifact-unavailable", "artifact-rejected", "runtime-failed"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json(data):
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError("Duplicate artifact JSON key")
            result[name] = value
        return result
    def invalid_constant(_value):
        raise ValueError("Nonfinite artifact JSON number")
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, UnicodeError, TypeError):
        raise ValueError("Invalid artifact JSON") from None


def _exact(observed, expected):
    # JSON equality distinguishes False from 0 and True from 1.
    if canonical(observed) != canonical(expected):
        raise ValueError("Readiness artifact exceeds public allowlist")


def require_readiness_job(job):
    """Call before reservation; this storage mode cannot accept training jobs."""
    if (type(job) is not dict or set(job) != {"commit", "script", "gpu"}
            or type(job.get("commit")) is not str
            or not re.fullmatch(r"[0-9a-f]{40}", job["commit"])
            or job["script"] != READINESS_SCRIPT or job["gpu"] != "CPU"):
        raise ValueError("GitHub artifacts support only the reviewed CPU readiness job")


def _identity(repo, number, job):
    require_readiness_job(job)
    if type(repo) is not str or repo.lower() != REPOSITORY:
        raise ValueError("Unexpected readiness repository")
    if type(number) is not int or number <= 0:
        raise ValueError("Invalid readiness issue identity")
    return REPOSITORY, REPOSITORY.split("/", 1)[0]


def _provenance(job, number):
    return {"commit": job["commit"], "issue": number, "script": READINESS_SCRIPT,
            "model_blobs": MODEL_BLOBS, "requested_gpu": "CPU", "precision": "float32",
            "resume": False, "retries": 0}


def _expected_members(job, number):
    values = {"configuration.json": CONFIGURATION, "metrics.json": METRICS,
              "provenance.json": _provenance(job, number)}
    result = {name: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
              for name, value in values.items()}
    result["run.log"] = CANONICAL_LOG.encode("utf-8")
    return result


def _remaining(deadline):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("Readiness artifact deadline expired")
    return left


def _api(command, arguments, deadline):
    try:
        response = command(["gh", "api", *arguments], timeout=_remaining(deadline), check=True)
        if response.returncode:
            raise ValueError("GitHub API command failed")
        return _json(response.stdout)
    except TimeoutError:
        raise
    except Exception:
        # CLI stderr, OAuth details and arbitrary API error bodies never escape.
        raise ValueError("GitHub readiness API operation failed") from None


def _consent_and_job(body, job):
    if type(body) is not str or not body.startswith(JOB_MARKER):
        raise ValueError("Readiness issue body is not source-bound")
    match = re.search(r"```json\s*(\{.*?\})\s*```", body, re.DOTALL)
    if not match:
        raise ValueError("Readiness issue body is not source-bound")
    _exact(_json(match.group(1)), job)
    consent = False
    fence = None
    for line in body.splitlines():
        if fence:
            if re.fullmatch(re.escape(fence[0]) + "{" + str(fence[1]) + r",}\s*", line.lstrip()):
                fence = None
        else:
            opening = re.match(r"\s*(`{3,}|~{3,})", line)
            if opening:
                fence = (opening[1][0], len(opening[1]))
            elif line == PUBLIC_CONSENT_MARKER:
                consent = True
    if not consent:
        raise ValueError("Explicit public readiness artifact consent is missing")


def github_preflight(repo, number, job, command, deadline, *, require_approval=True):
    """Read-only owner/public-repository/issue permissions before reservation.

    Repository permission metadata does not prove a token's actual issue-POST
    privilege. Publication and independent retention readback remain unverified
    until an explicitly authorized live attempt; there is no public sentinel.
    During artifact retention, a consumed attempt may have a closed issue or a
    removed compute label. require_approval=False still verifies its owner and
    public repository; it does not authorize another runtime or training job.
    """
    repo, owner = _identity(repo, number, job)
    actor = _api(command, ["user"], deadline)
    repository = _api(command, [f"repos/{repo}"], deadline)
    issue = _api(command, [f"repos/{repo}/issues/{number}"], deadline)
    if (type(actor) is not dict or type(actor.get("login")) is not str
            or actor["login"].lower() != owner
            or type(repository) is not dict
            or str(repository.get("full_name", "")).lower() != repo
            or repository.get("private") is not False
            or repository.get("archived") is not False
            or repository.get("has_issues") is not True
            or repository.get("permissions", {}).get("push") is not True
            or str(repository.get("owner", {}).get("login", "")).lower() != owner
            or type(issue) is not dict or "pull_request" in issue
            or issue.get("number") != number
            or str(issue.get("user", {}).get("login", "")).lower() != owner):
        raise ValueError("Public readiness artifact permissions are not verified")
    if require_approval and (issue.get("state") != "open" or
            "colab-approved" not in {item.get("name") for item in issue.get("labels", [])}):
        raise ValueError("Readiness issue is not approved for execution")
    _consent_and_job(issue.get("body"), job)
    return {"repository": repo, "owner": owner, "public": True, "issue": number,
            "post_privilege_live_verified": False}


def sanitized_report(path, descriptor, job, number, verify_callback):
    """Preserve allowlisted contents and original hashes, not original ZIP bytes."""
    _identity(REPOSITORY, number, job)
    if (type(descriptor) is not dict or set(descriptor) != {"sha256", "bytes"}
            or type(descriptor.get("sha256")) is not str
            or not SHA256.fullmatch(descriptor["sha256"])
            or type(descriptor.get("bytes")) is not int
            or not 0 < descriptor["bytes"] <= MAX_BUNDLE_BYTES):
        raise ValueError("Readiness bundle descriptor exceeds public allowlist")
    path = Path(path)
    if verify_callback(path, descriptor, job, number) != descriptor["sha256"]:
        raise ValueError("Readiness bundle was not verified")
    raw = path.read_bytes()
    if len(raw) != descriptor["bytes"] or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError("Readiness bundle digest mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if (len(names) != len(set(names)) or set(names) != MEMBERS | {"manifest.json"}
                or any(entry.file_size > MAX_MEMBER_BYTES for entry in entries)
                or sum(entry.file_size for entry in entries) > MAX_BUNDLE_BYTES):
            raise ValueError("Unexpected readiness artifact members")
        data = {name: archive.read(name) for name in names}
    manifest = _json(data["manifest.json"])
    expected_manifest = {name: hashlib.sha256(data[name]).hexdigest() for name in MEMBERS}
    _exact(manifest, expected_manifest)
    configuration = _json(data["configuration.json"])
    metrics = _json(data["metrics.json"])
    provenance = _json(data["provenance.json"])
    _exact(configuration, CONFIGURATION)
    _exact(metrics, METRICS)
    _exact(provenance, _provenance(job, number))
    expected_members = _expected_members(job, number)
    if any(data[name] != expected_members[name] for name in MEMBERS):
        raise ValueError("Readiness member bytes differ from reviewed runtime encoding")
    if data["manifest.json"] != json.dumps(expected_manifest, sort_keys=True).encode("utf-8"):
        raise ValueError("Readiness manifest bytes differ from reviewed runtime encoding")
    report = {"schema": 1, "status": "source-ready", "complete": True,
              "retention": "allowlisted-json-readiness-report", "original_zip_retained": False,
              "artifacts": {"configuration.json": configuration, "metrics.json": metrics,
                            "provenance.json": provenance, "run.log": CANONICAL_LOG},
              "log_representation": "validated-canonical-readiness-log",
              "member_encoding": MEMBER_ENCODING,
              "original_evidence": {"bundle": dict(descriptor), "member_sha256": manifest,
                                    "manifest_json_sha256": hashlib.sha256(data["manifest.json"]).hexdigest()}}
    validate_report(report, job, number)
    return report


def incomplete_report(job, number, reason="bootstrap-failed"):
    """Only fixed status and expected identity; no arbitrary failure payload."""
    _identity(REPOSITORY, number, job)
    if type(reason) is not str or reason not in INCOMPLETE_REASONS:
        raise ValueError("Failure status exceeds public allowlist")
    return {"schema": 1, "status": "incomplete", "complete": False, "reason": reason,
            "retention": "fixed-status-provenance-receipt", "original_zip_retained": False,
            "expected_provenance": {"commit": job["commit"], "issue": number,
                                    "script": READINESS_SCRIPT, "requested_gpu": "CPU"},
            "bundle_verified": False}


def validate_report(report, job, number):
    _identity(REPOSITORY, number, job)
    if type(report) is not dict:
        raise ValueError("Invalid readiness report")
    if report.get("status") == "incomplete":
        _exact(report, incomplete_report(job, number, report.get("reason")))
    else:
        if set(report) != {"schema", "status", "complete", "retention", "original_zip_retained",
                           "artifacts", "log_representation", "member_encoding", "original_evidence"}:
            raise ValueError("Readiness report exceeds public allowlist")
        expected = {"schema": 1, "status": "source-ready", "complete": True,
                    "retention": "allowlisted-json-readiness-report", "original_zip_retained": False,
                    "artifacts": {"configuration.json": CONFIGURATION, "metrics.json": METRICS,
                                  "provenance.json": _provenance(job, number), "run.log": CANONICAL_LOG},
                    "log_representation": "validated-canonical-readiness-log",
                    "member_encoding": MEMBER_ENCODING}
        _exact({key: value for key, value in report.items() if key != "original_evidence"}, expected)
        evidence = report["original_evidence"]
        if (type(evidence) is not dict or set(evidence) != {"bundle", "member_sha256", "manifest_json_sha256"}
                or type(evidence["bundle"]) is not dict or set(evidence["bundle"]) != {"sha256", "bytes"}
                or type(evidence["bundle"]["bytes"]) is not int
                or not 0 < evidence["bundle"]["bytes"] <= MAX_BUNDLE_BYTES
                or type(evidence["member_sha256"]) is not dict or set(evidence["member_sha256"]) != MEMBERS):
            raise ValueError("Invalid original readiness digest evidence")
        digests = [evidence["bundle"]["sha256"], evidence["manifest_json_sha256"],
                   *evidence["member_sha256"].values()]
        if any(type(digest) is not str or not SHA256.fullmatch(digest) for digest in digests):
            raise ValueError("Invalid original readiness digest evidence")
        reconstructed = _expected_members(job, number)
        expected_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in reconstructed.items()}
        _exact(evidence["member_sha256"], expected_hashes)
        if evidence["manifest_json_sha256"] != hashlib.sha256(json.dumps(expected_hashes, sort_keys=True).encode()).hexdigest():
            raise ValueError("Original readiness manifest digest mismatch")
    if len(canonical(report).encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("Readiness report exceeds comment limit")


def reconstruct_members(report, job, number):
    """Recover all original member bytes; ZIP metadata/bytes are not retained.

    JSON members use json.dumps(value, sort_keys=True, indent=2,
    allow_nan=False) + '\\n', encoded UTF-8. The source-ready log is exact.
    The original manifest uses json.dumps(member_sha256, sort_keys=True).
    """
    validate_report(report, job, number)
    if not report["complete"]:
        raise ValueError("Incomplete readiness receipt has no original artifacts")
    result = _expected_members(job, number)
    result["manifest.json"] = json.dumps(report["original_evidence"]["member_sha256"], sort_keys=True).encode("utf-8")
    return result


def publish_report(repo, number, job, report, command, deadline):
    """One create-only comment, independently read back before runtime teardown.

    Call only after explicit public-report authorization, under the worker's
    existing single-use global job tag and exclusive lock. A failed POST/readback
    is never retried here. It cannot claim successful complete artifact retention.
    """
    repo, owner = _identity(repo, number, job)
    validate_report(report, job, number)
    github_preflight(repo, number, job, command, deadline, require_approval=False)
    pages = _api(command, ["--paginate", "--slurp", f"repos/{repo}/issues/{number}/comments?per_page=100"], deadline)
    if type(pages) is not list or any(type(page) is not list for page in pages):
        raise ValueError("Cannot verify prior readiness retention")
    if any(str(item.get("user", {}).get("login", "")).lower() == owner
           and str(item.get("body", "")).startswith(COMMENT_MARKER)
           for page in pages for item in page):
        raise ValueError("Readiness artifact comment already exists; no retry")
    digest = hashlib.sha256(canonical(report).encode("utf-8")).hexdigest()
    envelope = {"report": report, "report_sha256": digest}
    body = COMMENT_MARKER + "\n```json\n" + canonical(envelope) + "\n```"
    with tempfile.TemporaryDirectory(prefix="colab-github-report-") as temporary:
        payload = Path(temporary) / "comment.json"
        payload.write_text(json.dumps({"body": body}, allow_nan=False), encoding="utf-8")
        posted = _api(command, ["--method", "POST", f"repos/{repo}/issues/{number}/comments",
                                "--input", str(payload)], deadline)
    if type(posted) is not dict or type(posted.get("id")) is not int or posted["id"] <= 0:
        raise ValueError("Cannot verify created readiness comment")
    identifier = posted["id"]
    readback = _api(command, [f"repos/{repo}/issues/comments/{identifier}"], deadline)
    if (type(readback) is not dict or readback.get("id") != identifier
            or str(readback.get("user", {}).get("login", "")).lower() != owner
            or readback.get("issue_url") != f"https://api.github.com/repos/{repo}/issues/{number}"
            or readback.get("body") != body):
        raise ValueError("GitHub readiness artifact readback mismatch")
    read_envelope = _json(readback["body"].split("\n```json\n", 1)[1].removesuffix("\n```"))
    if (read_envelope.get("report_sha256") != digest
            or hashlib.sha256(canonical(read_envelope.get("report")).encode()).hexdigest() != digest):
        raise ValueError("GitHub readiness report digest mismatch")
    return {"complete": report["complete"], "public_report": True, "verified_before_teardown": True,
            "report_sha256": digest, "comment_id": identifier,
            "destination": f"https://github.com/{repo}/issues/{number}#issuecomment-{identifier}",
            "original_zip_retained": False}
