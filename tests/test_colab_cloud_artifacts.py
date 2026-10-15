"""Cloud byte-retention and permission/attempt regressions; no live services."""
import hashlib
import importlib.util
import json
import shutil
import sys
import contextlib
import io
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cloud_worker", ROOT / "scripts/colab_pro_issue_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class CloudArtifactTests(unittest.TestCase):
    def test_missing_local_oauth_never_initiates_login_or_uses_adc(self):
        with patch.object(worker.shutil, "which", return_value="colab"), \
             patch.object(worker.Path, "is_file", return_value=False), \
             patch.object(worker, "cmd") as command:
            with self.assertRaises(PermissionError):
                worker.account_snapshot()
            command.assert_not_called()

    def test_status_reports_unknown_account_without_raw_auth_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            stdout = io.StringIO()
            with patch.object(sys, "argv", ["worker", "--status", "--state-dir", temporary]), \
                 patch.object(worker, "account_snapshot", side_effect=PermissionError("PRIVATE TOKEN")), \
                 contextlib.redirect_stdout(stdout):
                worker.main()
            status = json.loads(stdout.getvalue())
            self.assertFalse(status["worker_running"])
            self.assertFalse(status["account_verified"])
            self.assertIsNone(status["account"])
            self.assertEqual(status["account_error_type"], "PermissionError")
            self.assertNotIn("PRIVATE TOKEN", stdout.getvalue())

    def test_new_receipt_with_unknown_post_usage_blocks_worker_startup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker.atomic_json(root / "issue-12.json", {"schema": 2, "issue": 12,
                "state": "FAILED", "stop_ok": True, "after": None, "retry_allowed": False})
            with patch.object(sys, "argv", ["worker", "--state-dir", temporary]), \
                 patch.object(worker, "sweep") as sweep, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    worker.main()
                sweep.assert_not_called()

    def test_legacy_usage_unknown_is_reported_without_fabricating_old_balance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = {"issue": 1443, "state": "SUCCESS", "stop_ok": True, "retry_allowed": False}
            worker.atomic_json(root / "issue-1443.json", receipt)
            stdout = io.StringIO()
            with patch.object(sys, "argv", ["worker", "--status", "--state-dir", temporary]), \
                 patch.object(worker, "account_snapshot", return_value={"balance_units": 100, "active_assignments": 0, "sessions": ""}), \
                 contextlib.redirect_stdout(stdout):
                worker.main()
            status = json.loads(stdout.getvalue())
            self.assertEqual(status["legacy_post_usage_unknown"], [1443])
            self.assertEqual(json.loads((root / "issue-1443.json").read_text()), receipt)
            self.assertFalse(status["worker_running"])

    def test_local_backend_or_read_only_cloud_never_reaches_reservation(self):
        job = {"commit": "a" * 40, "script": "colab_jobs/chm_v4_readiness.py", "gpu": "CPU"}
        source = "COLAB_JOB_ID = 'fixture-job-cloud-preflight-v1'"
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(repo="owner/repo", state_dir=Path(temporary), artifact_remote="fixture:research")
            for failure in (ValueError("Local backend"), RuntimeError("Upload permission denied")):
                with patch.object(worker, "account_snapshot", return_value={"balance_units": 100, "active_assignments": 0}), \
                     patch.object(worker, "cloud_preflight", side_effect=failure), \
                     patch.object(worker, "reserve") as reserve, \
                     patch.object(worker, "cmd") as command:
                    with self.assertRaises(type(failure)):
                        worker.execute(args, {"number": 12}, job, source)
                    reserve.assert_not_called()
                    command.assert_not_called()

    def test_preflight_rejects_local_wrapped_unknown_and_unverifiable_backends(self):
        for backend in ("local", "alias", "crypt", "unknown"):
            with patch.object(worker, "cmd", return_value=SimpleNamespace(stdout="fixture: " + backend + "\n")) as command:
                with self.assertRaises(ValueError):
                    worker.cloud_preflight("fixture:research")
                self.assertEqual(command.call_count, 1)
        with patch.object(worker, "cmd", side_effect=[SimpleNamespace(stdout="fixture: drive\n"), SimpleNamespace(stdout='{"Features":{}}')]), \
             patch.object(worker, "cloud_copy_verified") as upload:
            with self.assertRaises(ValueError):
                worker.cloud_preflight("fixture:research")
            upload.assert_not_called()

    def test_cloud_probe_is_verified_and_deleted_before_preflight_succeeds(self):
        events = []
        cloud = {}
        def command(argv, **kwargs):
            if "listremotes" in argv:
                return SimpleNamespace(stdout="fixture: drive\n", returncode=0)
            if "features" in argv:
                return SimpleNamespace(stdout='{"Features":{"IsLocal":false}}', returncode=0)
            if "copyto" in argv:
                if "--immutable" in argv:
                    events.append("upload")
                    cloud[argv[-1]] = Path(argv[-2]).read_bytes()
                else:
                    events.append("readback")
                    Path(argv[-1]).write_bytes(cloud[argv[-2]])
            if "deletefile" in argv:
                events.append("delete")
                cloud.pop(argv[2])
            return SimpleNamespace(stdout="", returncode=0)
        with patch.object(worker, "cmd", side_effect=command):
            result = worker.cloud_preflight("fixture:research")
        self.assertTrue(result["write_readback_verified"])
        self.assertTrue(result["probe_cleanup_ok"])
        self.assertEqual(events, ["upload", "readback", "delete"])
        self.assertFalse(cloud)

    @unittest.skipUnless(shutil.which("rclone"), "rclone CLI not installed")
    def test_actual_rclone_metadata_rejects_misconfigured_local_remote(self):
        original = worker.cmd
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "fixture.conf"
            config.write_text("[fixture]\ntype = local\n")
            def command(argv, **kwargs):
                return original(["rclone", "--config", str(config), *argv[1:]], **kwargs)
            with patch.object(worker, "cmd", side_effect=command):
                with self.assertRaises(ValueError):
                    worker.cloud_preflight("fixture:" + temporary)

    @unittest.skipUnless(shutil.which("rclone"), "rclone CLI not installed")
    def test_actual_rclone_transfer_and_readback_use_only_temporary_fixture_files(self):
        original = worker.cmd
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "fixture.conf"
            config.write_text("[fixture]\ntype = local\n")
            source = root / "source.json"
            source.write_text('{"fixture":true}\n')
            destination = "fixture:" + str(root / "fake-cloud.json")
            def command(argv, **kwargs):
                return original(["rclone", "--config", str(config), *argv[1:]], **kwargs)
            with patch.object(worker, "cmd", side_effect=command):
                result = worker.cloud_copy_verified(source, destination, time.monotonic() + 30)
            self.assertEqual(result["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual((root / "fake-cloud.json").read_bytes(), source.read_bytes())

    def test_cloud_readback_detects_corruption_and_leaves_no_retained_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / "payload"
            local.write_bytes(b"configuration and private run logs")
            readbacks = []
            def command(argv, **kwargs):
                if "--immutable" not in argv:
                    returned = Path(argv[-1])
                    returned.write_bytes(b"corrupted cloud download")
                    readbacks.append(returned)
                return SimpleNamespace(returncode=0)
            with patch.object(worker, "cmd", side_effect=command):
                with self.assertRaises(ValueError):
                    worker.cloud_copy_verified(local, "drive:research/payload", time.monotonic() + 180)
            self.assertTrue(all(not path.exists() for path in readbacks))

    def test_timeout_without_bundle_keeps_transport_and_partial_logs_in_cloud(self):
        cloud = {}
        staged = []
        def command(argv, **kwargs):
            if "download" in argv:
                path = Path(argv[-1])
                path.write_bytes(b"live bootstrap log")
                staged.append(path)
            elif "--immutable" in argv:
                cloud[argv[-1]] = Path(argv[-2]).read_bytes()
            else:
                path = Path(argv[-1])
                path.write_bytes(cloud[argv[-2]])
                staged.append(path)
            return SimpleNamespace(returncode=0)
        args = SimpleNamespace(artifact_remote="drive:research")
        job = {"commit": "a" * 40, "script": "colab_jobs/chm_v4_readiness.py", "gpu": "CPU"}
        with patch.object(worker, "cmd", side_effect=command):
            evidence = worker.collect_cloud_artifacts(args, job, 12, "session", "partial transport", time.monotonic() + 180)
        self.assertFalse(evidence["complete"])
        self.assertEqual(set(cloud.values()), {b"partial transport", b"live bootstrap log"})
        self.assertTrue(all(not path.exists() for path in staged))

    def test_partial_metrics_retained_and_foreign_provenance_not_treated_as_current(self):
        cloud = {}
        def command(argv, **kwargs):
            if "download" in argv:
                path = Path(argv[-1])
                if path.name == "provenance.json":
                    path.write_text(json.dumps({"commit": "b" * 40, "issue": 999, "script": "other.py", "requested_gpu": "T4"}))
                else:
                    path.write_bytes(b"available partial bytes")
            elif "--immutable" in argv:
                cloud[argv[-1]] = Path(argv[-2]).read_bytes()
            else:
                Path(argv[-1]).write_bytes(cloud[argv[-2]])
            return SimpleNamespace(returncode=0)
        job = {"commit": "a" * 40, "script": "colab_jobs/chm_v4_readiness.py", "gpu": "CPU"}
        with patch.object(worker, "cmd", side_effect=command):
            evidence = worker.collect_cloud_artifacts(SimpleNamespace(artifact_remote="drive:research"),
                job, 12, "session", "no final descriptor", time.monotonic() + 180)
        self.assertFalse(evidence["complete"])
        self.assertFalse(evidence["partial_source_identity_verified"])
        self.assertEqual(evidence["partial_source_identity"], "mismatched")
        self.assertIn("metrics.json", evidence["partial_files"])
        self.assertFalse(evidence["partial_files"]["checkpoint.pt"]["loadability_verified"])
        self.assertFalse(evidence["partial_files"]["checkpoint.pt"]["resume_allowed"])

    def test_missing_cloud_remote_prevents_reservation_and_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(repo="owner/repo", state_dir=Path(temporary))
            job = {"commit": "a" * 40, "script": "colab_jobs/chm_v4_readiness.py", "gpu": "CPU"}
            with patch.object(worker, "account_snapshot", return_value={"balance_units": 100, "active_assignments": 0}), \
                 patch.object(worker, "cmd") as command:
                with self.assertRaises(ValueError):
                    worker.execute(args, {"number": 12}, job, "COLAB_JOB_ID = 'test-job-12-v1'")
                command.assert_not_called()
            self.assertFalse((Path(temporary) / "issue-12.json").exists())

    def test_atomic_replacement_failure_keeps_consumed_receipt_private(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = worker.reserve(root, 12, {})
            with patch.object(worker.os, "replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    worker.atomic_json(receipt, {"state": "SUCCESS"})
            self.assertIsNone(worker.reserve(root, 12, {}))
            self.assertEqual(json.loads(receipt.read_text())["state"], "reserved")
            worker.atomic_json(receipt, {"state": "FAILED", "retry_allowed": False})
            self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)

    def test_reserved_exception_cannot_disappear_from_lifetime_attempt_count(self):
        def issue(number):
            return {"number": number, "state": "open", "user": {"login": "owner"},
                    "labels": [{"name": "colab-approved"}], "body": worker.MARKER + '\n```json\n' +
                    json.dumps({"commit": "a" * 40, "script": "colab_jobs/chm_v4_readiness.py", "gpu": "CPU"}) + '\n```'}
        first, second = issue(12), issue(13)
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(repo="owner/repo", state_dir=Path(temporary), repo_path=Path(temporary), allowed_gpus={"CPU"})
            def execute(args, issue, job, source):
                worker.reserve(args.state_dir, issue["number"], job)
                raise OSError("Final persistence error")
            with patch.object(worker, "gh_json", side_effect=[[first, second], first]), \
                 patch.object(worker, "remotely_consumed", return_value=False), \
                 patch.object(worker, "source_from_main", return_value="reviewed source"), \
                 patch.object(worker, "execute", side_effect=execute) as attempted:
                self.assertEqual(worker.sweep(args, 1), 1)
                self.assertEqual(attempted.call_count, 1)
            self.assertFalse((Path(temporary) / "issue-13.json").exists())

    def test_consumed_1443_case_variants_are_never_scanned(self):
        args = SimpleNamespace(repo="VinceAckermann2-sys/Tam-Research")
        with patch.object(worker, "gh_json", return_value=[{"number": 1443}]), \
             patch.object(worker, "execute") as execute:
            self.assertEqual(worker.sweep(args, 1), 0)
            execute.assert_not_called()

    def test_removed_approval_during_source_fetch_prevents_execution(self):
        body = worker.MARKER + '\n```json\n' + json.dumps({"commit": "a" * 40,
            "script": "colab_jobs/chm_v4_readiness.py", "gpu": "CPU"}) + '\n```'
        issue = {"number": 12, "state": "open", "body": body, "user": {"login": "owner"}, "labels": []}
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(repo="owner/repo", repo_path=Path(temporary), state_dir=Path(temporary), allowed_gpus={"CPU"})
            with patch.object(worker, "gh_json", side_effect=[[issue], issue]), \
                 patch.object(worker, "remotely_consumed", return_value=False), \
                 patch.object(worker, "source_from_main", return_value="source"), \
                 patch.object(worker, "execute") as execute:
                self.assertEqual(worker.sweep(args, 1), 0)
                execute.assert_not_called()
