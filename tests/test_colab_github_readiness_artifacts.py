"""CPU fixtures only: no GitHub writes, Colab calls, credentials or training."""
import hashlib
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("github_readiness", ROOT / "scripts/colab_github_readiness_artifacts.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
JOB = {"commit": "a" * 40, "script": adapter.READINESS_SCRIPT, "gpu": "CPU"}
OWNER = "vinceackermann2-sys"
NUMBER = 123
SECRET = "DO-NOT-PUBLISH-OAUTH-PRIVATE-LOG"


def verified(path, descriptor, job, number):
    data = Path(path).read_bytes()
    if len(data) != descriptor["bytes"] or hashlib.sha256(data).hexdigest() != descriptor["sha256"]:
        raise ValueError("fixture digest mismatch")
    return descriptor["sha256"]


def fixture(path, changes=None, additional=None):
    files = {
        "configuration.json": json.dumps(adapter.CONFIGURATION, sort_keys=True, indent=2).encode() + b"\n",
        "metrics.json": json.dumps(adapter.METRICS, sort_keys=True, indent=2).encode() + b"\n",
        "provenance.json": json.dumps(adapter._provenance(JOB, NUMBER), sort_keys=True, indent=2).encode() + b"\n",
        "run.log": adapter.CANONICAL_LOG.encode(),
    }
    files.update(changes or {})
    files.update(additional or {})
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


class FakeGitHub:
    """Emulate structured API responses and a distinct readback request."""
    def __init__(self, *, private=False, push=True, actor=OWNER, issue_owner=OWNER,
                 approved=True, duplicate=False, tamper=False, wrong_author=False,
                 consent=True, issue_job=None, consent_suffix=None, archived=False, has_issues=True):
        self.calls = []
        self.body = None
        self.payload_path = None
        self.private, self.push, self.actor, self.issue_owner = private, push, actor, issue_owner
        self.archived, self.has_issues = archived, has_issues
        self.approved, self.duplicate, self.tamper, self.wrong_author = approved, duplicate, tamper, wrong_author
        suffix = adapter.PUBLIC_CONSENT_MARKER if consent else ""
        self.issue_body = adapter.JOB_MARKER + "\n```json\n" + json.dumps(issue_job or JOB) + "\n```\n" + (consent_suffix if consent_suffix is not None else suffix)

    def __call__(self, arguments, **kwargs):
        self.calls.append(arguments)
        assert 0 < kwargs["timeout"] <= 180
        assert kwargs["check"] is True
        if "POST" in arguments:
            self.payload_path = Path(arguments[-1])
            self.body = json.loads(self.payload_path.read_text())["body"]
            data = {"id": 456}
        elif arguments[-1] == "user":
            data = {"login": self.actor}
        elif arguments[-1] == f"repos/{adapter.REPOSITORY}":
            data = {"full_name": adapter.REPOSITORY, "private": self.private,
                    "archived": self.archived, "has_issues": self.has_issues,
                    "owner": {"login": OWNER}, "permissions": {"push": self.push}}
        elif arguments[-1] == f"repos/{adapter.REPOSITORY}/issues/{NUMBER}":
            data = {"number": NUMBER, "state": "open", "user": {"login": self.issue_owner},
                    "body": self.issue_body,
                    "labels": [{"name": "colab-approved"}] if self.approved else []}
        elif "--paginate" in arguments:
            data = [[{"user": {"login": OWNER}, "body": adapter.COMMENT_MARKER + " existing"}]] if self.duplicate else [[]]
        elif arguments[-1] == f"repos/{adapter.REPOSITORY}/issues/comments/456":
            body = self.body + "changed" if self.tamper else self.body
            data = {"id": 456, "body": body, "user": {"login": "attacker" if self.wrong_author else OWNER},
                    "issue_url": f"https://api.github.com/repos/{adapter.REPOSITORY}/issues/{NUMBER}"}
        else:
            raise AssertionError("Unexpected fixture API request")
        return SimpleNamespace(returncode=0, stdout=json.dumps(data), stderr="")


class ReadinessReportTests(unittest.TestCase):
    def build(self, changes=None, additional=None):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.zip"
            descriptor = fixture(path, changes, additional)
            return adapter.sanitized_report(path, descriptor, JOB, NUMBER, verified), descriptor

    def test_preserves_allowlisted_contents_and_original_digest_evidence(self):
        report, descriptor = self.build()
        self.assertEqual(report["artifacts"]["configuration.json"], adapter.CONFIGURATION)
        self.assertEqual(report["artifacts"]["metrics.json"], adapter.METRICS)
        self.assertEqual(report["artifacts"]["provenance.json"], adapter._provenance(JOB, NUMBER))
        self.assertEqual(report["artifacts"]["run.log"], adapter.CANONICAL_LOG)
        self.assertEqual(report["original_evidence"]["bundle"], descriptor)
        self.assertEqual(set(report["original_evidence"]["member_sha256"]), adapter.MEMBERS)
        self.assertFalse(report["original_zip_retained"])
        self.assertTrue(report["complete"])

    def test_matches_actual_runtime_without_model_or_training_import(self):
        # Works on shallow CI checkouts; frozen ea062 success encoding is asserted
        # separately by member reconstruction and strict fixed-value fixtures.
        source = (ROOT / "tam_research/colab_job_runtime.py").read_text()
        runtime = ModuleType("original_ea_runtime")
        exec(compile(source, "original_ea_runtime", "exec"), runtime.__dict__)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        job = dict(JOB, commit=commit)
        args = SimpleNamespace(repo_commit=commit, issue=NUMBER, job_script=job["script"], requested_gpu="CPU")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "artifacts"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                runtime.run_job(args, ROOT, folder, training=False)
            descriptor = json.loads(next(line.split("=", 1)[1] for line in stdout.getvalue().splitlines() if line.startswith("COLAB_ARTIFACT=")))
            report = adapter.sanitized_report(folder / "bundle.zip", descriptor, job, NUMBER, verified)
            self.assertTrue(report["complete"])
            self.assertEqual(adapter.reconstruct_members(report, job, NUMBER)["run.log"], (folder / "run.log").read_bytes())

    def test_reconstruction_recovers_original_member_and_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.zip"
            descriptor = fixture(path)
            report = adapter.sanitized_report(path, descriptor, JOB, NUMBER, verified)
            reconstructed = adapter.reconstruct_members(report, JOB, NUMBER)
            with zipfile.ZipFile(path) as archive:
                for name, content in reconstructed.items():
                    self.assertEqual(content, archive.read(name))
            self.assertEqual(hashlib.sha256(reconstructed["manifest.json"]).hexdigest(), report["original_evidence"]["manifest_json_sha256"])

    def test_semantically_equal_arbitrary_json_encoding_rejected(self):
        altered = json.dumps(adapter.CONFIGURATION).encode()
        with self.assertRaisesRegex(ValueError, "runtime encoding"):
            self.build({"configuration.json": altered})

    def test_raw_log_secrets_and_exception_text_never_enter_report(self):
        with self.assertRaises(ValueError) as raised:
            self.build({"run.log": (adapter.CANONICAL_LOG + SECRET).encode()})
        self.assertNotIn(SECRET, str(raised.exception))
        report = adapter.incomplete_report(JOB, NUMBER)
        self.assertNotIn(SECRET, adapter.canonical(report))
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "incomplete")
        self.assertNotIn("artifacts", report)

    def test_unknown_json_fields_and_bool_integer_confusion_fail_closed(self):
        configuration = dict(adapter.CONFIGURATION, secret=SECRET)
        metrics = dict(adapter.METRICS, training_executed=0)
        for changes in ({"configuration.json": json.dumps(configuration).encode()},
                        {"metrics.json": json.dumps(metrics).encode()}):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.build(changes)

    def test_duplicate_json_keys_fail_closed(self):
        value = '{"scope":"source/bootstrap/artifact preflight","scope":"source/bootstrap/artifact preflight","optimizer_steps":0,"dependencies":"stdlib only"}'
        with self.assertRaises(ValueError):
            self.build({"configuration.json": value.encode()})

    def test_failed_runtime_metrics_require_incomplete_receipt(self):
        with self.assertRaises(ValueError):
            self.build({"metrics.json": json.dumps({"status": "failed", "error_type": SECRET}).encode()})
        receipt = adapter.incomplete_report(JOB, NUMBER, "runtime-failed")
        self.assertFalse(receipt["bundle_verified"])
        self.assertEqual(receipt["expected_provenance"]["commit"], JOB["commit"])
        with self.assertRaises(ValueError):
            adapter.incomplete_report(JOB, NUMBER, SECRET)

    def test_provenance_and_model_blob_drift_rejected(self):
        for changed in (dict(adapter._provenance(JOB, NUMBER), issue=124),
                        dict(adapter._provenance(JOB, NUMBER), model_blobs={})):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.build({"provenance.json": json.dumps(changed).encode()})

    def test_checkpoint_or_extra_archive_members_rejected(self):
        for name in ("checkpoint.pt", "transport.log", "../secret"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.build(additional={name: SECRET.encode()})

    def test_corrupted_digest_or_unverified_callback_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.zip"
            descriptor = fixture(path)
            with self.assertRaises(ValueError):
                adapter.sanitized_report(path, dict(descriptor, sha256="b" * 64), JOB, NUMBER, verified)
            with self.assertRaises(ValueError):
                adapter.sanitized_report(path, descriptor, JOB, NUMBER, lambda *args: None)

    def test_publication_validates_report_again(self):
        report, _ = self.build()
        report["artifacts"]["run.log"] = SECRET
        gh = FakeGitHub()
        with self.assertRaises(ValueError):
            adapter.publish_report(adapter.REPOSITORY, NUMBER, JOB, report, gh, time.monotonic() + 180)
        self.assertEqual(gh.calls, [])


class GitHubRetentionTests(unittest.TestCase):
    def test_preflight_is_read_only_and_rejects_training_before_any_command(self):
        gh = FakeGitHub()
        result = adapter.github_preflight(adapter.REPOSITORY, NUMBER, JOB, gh, time.monotonic() + 180)
        self.assertTrue(result["public"])
        self.assertEqual(len(gh.calls), 3)
        self.assertTrue(all("POST" not in call for call in gh.calls))
        for job in (dict(JOB, script="colab_jobs/chm_v4_microfit_1436.py"), dict(JOB, gpu="T4")):
            fresh = FakeGitHub()
            with self.assertRaises(ValueError):
                adapter.github_preflight(adapter.REPOSITORY, NUMBER, job, fresh, time.monotonic() + 180)
            self.assertEqual(fresh.calls, [])

    def test_wrong_auth_private_repo_missing_permission_or_approval_rejected(self):
        for options in ({"actor": "attacker"}, {"private": True}, {"push": False},
                        {"issue_owner": "attacker"}, {"approved": False},
                        {"archived": True}, {"has_issues": False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                adapter.github_preflight(adapter.REPOSITORY, NUMBER, JOB, FakeGitHub(**options), time.monotonic() + 180)

    def test_public_consent_and_same_job_required_even_for_retention(self):
        for options in ({"consent": False}, {"issue_job": dict(JOB, commit="b" * 40)},
                        {"consent_suffix": "```text\n" + adapter.PUBLIC_CONSENT_MARKER + "\n```"},
                        {"consent_suffix": "~~~text\n" + adapter.PUBLIC_CONSENT_MARKER + "\n~~~"}):
            gh = FakeGitHub(**options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                adapter.publish_report(adapter.REPOSITORY, NUMBER, JOB, adapter.incomplete_report(JOB, NUMBER), gh, time.monotonic() + 180)
            self.assertEqual(sum("POST" in call for call in gh.calls), 0)

    def test_one_owner_comment_with_independent_digest_readback_and_no_local_payload(self):
        report = adapter.incomplete_report(JOB, NUMBER, "bootstrap-failed")
        gh = FakeGitHub()
        evidence = adapter.publish_report(adapter.REPOSITORY, NUMBER, JOB, report, gh, time.monotonic() + 180)
        self.assertEqual(sum("POST" in call for call in gh.calls), 1)
        self.assertEqual(gh.calls[-1][-1], f"repos/{adapter.REPOSITORY}/issues/comments/456")
        self.assertTrue(evidence["verified_before_teardown"])
        self.assertFalse(evidence["complete"])
        self.assertFalse(evidence["original_zip_retained"])
        self.assertEqual(evidence["report_sha256"], hashlib.sha256(adapter.canonical(report).encode()).hexdigest())
        self.assertFalse(gh.payload_path.exists())
        self.assertNotIn(SECRET, gh.body)
        self.assertTrue(all("PATCH" not in call and "DELETE" not in call for call in gh.calls))

    def test_successful_report_publication_keeps_all_small_readiness_artifacts(self):
        report, _ = ReadinessReportTests().build()
        gh = FakeGitHub()
        evidence = adapter.publish_report(adapter.REPOSITORY, NUMBER, JOB, report, gh, time.monotonic() + 180)
        self.assertTrue(evidence["complete"])
        self.assertTrue(all(name in gh.body for name in adapter.MEMBERS))
        self.assertIn('"original_zip_retained":false', gh.body)

    def test_existing_comment_is_not_edited_recreated_or_retried(self):
        gh = FakeGitHub(duplicate=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            adapter.publish_report(adapter.REPOSITORY, NUMBER, JOB, adapter.incomplete_report(JOB, NUMBER), gh, time.monotonic() + 180)
        self.assertEqual(sum("POST" in call for call in gh.calls), 0)

    def test_readback_tampering_or_wrong_author_fail_without_retry(self):
        for options in ({"tamper": True}, {"wrong_author": True}):
            gh = FakeGitHub(**options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                adapter.publish_report(adapter.REPOSITORY, NUMBER, JOB, adapter.incomplete_report(JOB, NUMBER), gh, time.monotonic() + 180)
            self.assertEqual(sum("POST" in call for call in gh.calls), 1)

    def test_expired_deadline_does_not_call_api(self):
        gh = FakeGitHub()
        with self.assertRaises(TimeoutError):
            adapter.github_preflight(adapter.REPOSITORY, NUMBER, JOB, gh, time.monotonic() - 1)
        self.assertEqual(gh.calls, [])

    def test_api_exception_details_are_not_republished(self):
        def bad_command(*args, **kwargs):
            raise RuntimeError(SECRET)
        with self.assertRaises(ValueError) as raised:
            adapter.github_preflight(adapter.REPOSITORY, NUMBER, JOB, bad_command, time.monotonic() + 180)
        self.assertNotIn(SECRET, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
