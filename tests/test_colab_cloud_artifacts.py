"""Cloud byte-retention and permission/attempt regressions; no live services."""
import hashlib
import importlib.util
import json
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
