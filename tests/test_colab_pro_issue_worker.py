"""Zero-GPU unit tests for the Colab Pro issue bridge."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import hashlib
import subprocess
import time
from unittest.mock import patch
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / "scripts/colab_pro_issue_worker.py"
spec = importlib.util.spec_from_file_location("colab_bridge", SOURCE)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def sample_issue(job=None, user="vinceackermann2-sys"):
    job = job or {"commit": "a" * 40,
                  "script": "scripts/colab_bridge_gpu_probe.py", "gpu": "T4"}
    fence = chr(96) * 3
    body = (bridge.MARKER + "\n" + fence + "json\n"
            + json.dumps(job) + "\n" + fence)
    return {"number": 900001, "user": {"login": user}, "body": body}


class ColabProBridgeTests(unittest.TestCase):
    def test_owner_only(self):
        self.assertIsNotNone(bridge.parse_issue(sample_issue(), "vinceackermann2-sys"))
        self.assertIsNone(bridge.parse_issue(sample_issue(user="random"), "vinceackermann2-sys"))

    def test_exact_marker(self):
        issue = sample_issue()
        issue["body"] = "extra line\n" + issue["body"]
        self.assertIsNone(bridge.parse_issue(issue, "vinceackermann2-sys"))

    def test_reject_other_paths_and_invalid_sha(self):
        for script in ("scripts/modal_app.py", "../escape.py", "colab_jobs/../../escape.py"):
            with self.subTest(script=script):
                issue = sample_issue({"commit": "a" * 40, "script": script, "gpu": "T4"})
                self.assertIsNone(bridge.parse_issue(issue, "vinceackermann2-sys"))
        issue = sample_issue({"commit": "invalid", "script": "scripts/colab_bridge_gpu_probe.py", "gpu": "T4"})
        self.assertIsNone(bridge.parse_issue(issue, "vinceackermann2-sys"))

    def test_reject_unlisted_gpu_and_fields(self):
        issue = sample_issue({"commit": "a" * 40, "script": "scripts/colab_bridge_gpu_probe.py", "gpu": "B200"})
        self.assertIsNone(bridge.parse_issue(issue, "vinceackermann2-sys"))

    def test_publish_filtered_metrics(self):
        output = 'noise\nCOLAB_PUBLIC_RESULT={"gpu":"T4","loss":0.8,"secret":"PRIVATE"}\n'
        self.assertEqual(bridge.publishable_metrics(output), {"gpu": "T4", "loss": 0.8})

    def test_no_retry_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            job = {"commit": "a" * 40, "script": "scripts/colab_bridge_gpu_probe.py", "gpu": "T4"}
            self.assertIsNotNone(bridge.reserve(state, 12, job))
            self.assertIsNone(bridge.reserve(state, 12, job))


    def test_sweep_respects_gpu_allowlist_and_one_attempt(self):
        blocked = sample_issue({"commit": "a" * 40,
                                "script": "scripts/colab_bridge_gpu_probe.py",
                                "gpu": "H100"})
        blocked["number"] = 42
        allowed = sample_issue()
        allowed["number"] = 43
        second = sample_issue()
        second["number"] = 44
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(repo="vinceackermann2-sys/tam-research",
                                   state_dir=Path(tmp), repo_path=Path(tmp),
                                   allowed_gpus={"T4"})
            current = dict(allowed, state="open", labels=[{"name": "colab-approved"}])
            with patch.object(bridge, "gh_json", side_effect=[[blocked, allowed, second], current]), \
                 patch.object(bridge, "source_from_main", return_value="print(1)"), \
                 patch.object(bridge, "execute", return_value=True) as execute:
                attempts = bridge.sweep(args, remaining=1)
            self.assertEqual(attempts, 1)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(execute.call_args[0][1]["number"], 43)

    def test_approval_removed_during_fetch_does_not_execute(self):
        issue = sample_issue()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(repo="vinceackermann2-sys/tam-research", state_dir=Path(tmp),
                                   repo_path=Path(tmp), allowed_gpus={"T4"})
            with patch.object(bridge, "gh_json", side_effect=[[issue], dict(issue, state="open", labels=[])]), \
                 patch.object(bridge, "source_from_main", return_value="print(1)"), \
                 patch.object(bridge, "execute") as execute:
                self.assertEqual(bridge.sweep(args, 1), 0)
                execute.assert_not_called()

    def test_issue_1443_is_consumed_even_without_local_receipt(self):
        issue = dict(sample_issue(), number=1443)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(repo="VinceAckermann2-sys/tam-research", state_dir=Path(tmp))
            with patch.object(bridge, "gh_json", return_value=[issue]), \
                 patch.object(bridge, "execute") as execute:
                self.assertEqual(bridge.sweep(args, 1), 0)
                execute.assert_not_called()

    def test_unmerged_commit_rejected(self):
        job = bridge.parse_issue(sample_issue(), "vinceackermann2-sys")
        with patch.object(bridge, "cmd", side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)]) as command:
            with self.assertRaises(ValueError):
                bridge.source_from_main(Path("."), job)
            self.assertEqual(command.call_count, 2)

    def test_atomic_final_receipt_keeps_attempt_consumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            receipt = bridge.reserve(state, 12, {})
            with patch.object(bridge.os, "replace", side_effect=OSError("disk fault")):
                with self.assertRaises(OSError):
                    bridge.durable_json(receipt, {"state": "SUCCESS"})
            self.assertIsNone(bridge.reserve(state, 12, {}))
            self.assertEqual(json.loads(receipt.read_text())["state"], "reserved")
            bridge.durable_json(receipt, {"state": "FAILED", "retry_allowed": False})
            self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)

    def test_stale_heartbeat_or_unheld_lock_is_not_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            (state / "worker.lock").touch()
            bridge.durable_json(state / "worker-status.json", {"state": "running", "heartbeat_at": time.time()})
            self.assertFalse(bridge.worker_status(state)["running"])
            with (state / "worker.lock").open("r+") as lock:
                bridge.fcntl.flock(lock, bridge.fcntl.LOCK_EX)
                self.assertTrue(bridge.worker_status(state)["running"])
                bridge.durable_json(state / "worker-status.json", {"state": "running", "heartbeat_at": 0})
                self.assertFalse(bridge.worker_status(state)["running"])

    def test_execute_exports_cloud_before_stop_and_checks_balance(self):
        self.exercise_execution()

    def test_timeout_consumes_attempt_but_still_exports_and_cleans_up(self):
        self.exercise_execution(timeout=True)

    def test_cloud_export_failure_is_failed_and_still_tears_down(self):
        self.exercise_execution(export_failure=True)

    def test_unconfirmed_cleanup_blocks_next_allocation(self):
        self.exercise_execution(cleanup_failure=True)

    def exercise_execution(self, timeout=False, export_failure=False, cleanup_failure=False):
        issue = sample_issue({"commit": "a" * 40, "script": "colab_jobs/chm_v4_train_audit.py", "gpu": "CPU"})
        job = bridge.parse_issue(issue, "vinceackermann2-sys")
        events = []
        def command(argv, **kwargs):
            if "run" in argv and "--help" in argv:
                return SimpleNamespace(stdout="--keep --timeout", stderr="", returncode=0)
            if "run" in argv:
                events.append("run")
                self.assertIn("--keep", argv)
                self.assertEqual(argv[-2:], [job["commit"], "CPU"])
                if timeout:
                    raise subprocess.TimeoutExpired(argv, 135, output=b"private partial log")
            if "stop" in argv:
                events.append("stop")
            return SimpleNamespace(stdout="chatgpt-issue-900001" if cleanup_failure and "sessions" in argv else "", stderr="", returncode=0)
        def collect(*args):
            events.append("export")
            if export_failure:
                raise ValueError("PRIVATE cloud detail")
            return {"verified_before_teardown": True, "complete": True, "result_status": "ok"}
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(repo="vinceackermann2-sys/tam-research", state_dir=Path(tmp),
                                   max_seconds=180, artifact_remote="drive:research")
            with patch.object(bridge, "cmd", side_effect=command), \
                 patch.object(bridge, "balance", return_value={"compute_units": 100, "active_assignments": 0}) as balance, \
                 patch.object(bridge, "comment") as comment, \
                 patch.object(bridge, "collect_artifacts", side_effect=collect):
                self.assertTrue(bridge.execute(args, issue, job, "print(1)"))
                self.assertEqual(balance.call_count, 2)
                self.assertEqual(events, ["run", "export", "stop"])
                receipt = json.loads((Path(tmp) / "issue-900001.json").read_text())
                self.assertFalse(receipt["retry_allowed"])
                self.assertEqual(receipt["state"], "FAILED" if timeout or export_failure or cleanup_failure else "SUCCESS")
                self.assertNotIn("PRIVATE cloud detail", comment.call_args[0][2])
                self.assertFalse(list(Path(tmp).glob("*.zip")))
                self.assertFalse(list(Path(tmp).glob("*.log")))
                self.assertIsNone(receipt["local_log"])
                if cleanup_failure:
                    self.assertTrue((Path(tmp) / "worker-blocked.json").exists())
                    with self.assertRaises(RuntimeError):
                        bridge.execute(args, issue, job, "print(1)")

    def test_no_cloud_destination_or_unavailable_balance_allocates_nothing(self):
        issue = sample_issue({"commit": "a" * 40, "script": "colab_jobs/job.py", "gpu": "CPU"})
        job = bridge.parse_issue(issue, "vinceackermann2-sys")
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(repo="repo/name", state_dir=Path(tmp), artifact_remote=None)
            with patch.object(bridge, "balance", return_value={"compute_units": 100, "active_assignments": 0}), \
                 patch.object(bridge, "cmd") as command:
                with self.assertRaises(RuntimeError):
                    bridge.execute(args, issue, job, "print(1)")
                command.assert_not_called()
                self.assertFalse((Path(tmp) / "issue-900001.json").exists())
            with patch.object(bridge, "balance", side_effect=ValueError("unavailable")), \
                 patch.object(bridge, "cmd") as command:
                with self.assertRaises(ValueError):
                    bridge.execute(args, issue, job, "print(1)")
                command.assert_not_called()

    def test_manifest_corruption_and_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle.zip"
            write_test_bundle(path, "a" * 40)
            self.assertIn("sha256", bridge.verify_bundle(path, "a" * 40))
            with self.assertRaises(ValueError):
                bridge.verify_bundle(path, "b" * 40)
            write_test_bundle(path, "a" * 40, corrupt=True)
            with self.assertRaises(ValueError):
                bridge.verify_bundle(path, "a" * 40)
            write_test_bundle(path, "a" * 40)
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                bridge.verify_bundle(path, "a" * 40)

    def test_cloud_roundtrip_is_verified_and_transient_files_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            remotes = {}
            paths = []
            def command(argv, **kwargs):
                if "download" in argv:
                    path = Path(argv[-1])
                    write_test_bundle(path, "a" * 40)
                    paths.append(path)
                elif argv[0] == "rclone" and "--immutable" in argv:
                    remotes[argv[-1]] = Path(argv[-2]).read_bytes()
                else:
                    path = Path(argv[-1])
                    path.write_bytes(remotes[argv[-2]])
                    paths.append(path)
                return SimpleNamespace(returncode=0)
            args = SimpleNamespace(artifact_remote="drive:research")
            with patch.object(bridge, "cmd", side_effect=command):
                result = bridge.collect_artifacts(args, 12, "session", "a" * 40, time.monotonic() + 180, "transport log")
            self.assertTrue(result["verified_before_teardown"])
            self.assertEqual(len(remotes), 2)
            self.assertTrue(all(not path.exists() for path in paths))

    def test_reserved_execution_exception_still_counts_toward_one_attempt(self):
        first = dict(sample_issue(), state="open", labels=[{"name": "colab-approved"}])
        second = dict(first, number=900002)
        def failing_execute(args, issue, job, source):
            bridge.reserve(args.state_dir, issue["number"], job)
            raise OSError("Final receipt persistence fault after attempted execution")
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(repo="vinceackermann2-sys/tam-research", state_dir=Path(temporary),
                                   repo_path=Path(temporary), allowed_gpus={"T4"})
            with patch.object(bridge, "gh_json", side_effect=[[first, second], first]), \
                 patch.object(bridge, "source_from_main", return_value="print(1)"), \
                 patch.object(bridge, "execute", side_effect=failing_execute) as execute:
                self.assertEqual(bridge.sweep(args, remaining=1), 1)
                self.assertEqual(execute.call_count, 1)
                self.assertFalse((Path(temporary) / "issue-900002.json").exists())

    def test_public_nonfinite_metric_not_published(self):
        self.assertEqual(bridge.publishable_metrics('COLAB_PUBLIC_RESULT={"loss":NaN,"status":"ok"}'), {"status": "ok"})

    def test_missing_bundle_still_retains_cloud_transport_and_live_partial_log(self):
        remotes = {}
        def command(argv, **kwargs):
            if "download" in argv:
                if argv[-2].endswith("bundle.zip"):
                    raise RuntimeError("Bootstrap interrupted before bundle")
                Path(argv[-1]).write_bytes(b"git/pip partial bootstrap output")
            elif "--immutable" in argv:
                remotes[argv[-1]] = Path(argv[-2]).read_bytes()
            else:
                Path(argv[-1]).write_bytes(remotes[argv[-2]])
            return SimpleNamespace(returncode=0)
        args = SimpleNamespace(artifact_remote="drive:research")
        with patch.object(bridge, "cmd", side_effect=command):
            result = bridge.collect_artifacts(args, 12, "session", "a" * 40,
                                              time.monotonic() + 180, "private transport output")
        self.assertFalse(result["complete"])
        self.assertTrue(result["verified_before_teardown"])
        self.assertIn("partial_log_sha256", result)
        self.assertNotIn("cloud_destination", result)
        self.assertEqual(set(remotes.values()), {b"private transport output", b"git/pip partial bootstrap output"})

    def test_live_balance_parser_rejects_unverifiable_usage(self):
        with patch.object(bridge, "cmd", return_value=SimpleNamespace(stdout="Current balance: 108.89 compute units\nActive assignments: 0\n")):
            self.assertEqual(bridge.balance()["compute_units"], 108.89)
        with patch.object(bridge, "cmd", return_value=SimpleNamespace(stdout="usage unavailable")):
            with self.assertRaises(ValueError):
                bridge.balance()


def write_test_bundle(path, commit, corrupt=False):
    files = {"config.json": json.dumps({"commit": commit}).encode(), "result.json": b'{"status":"ok"}',
             "environment.json": b"{}", "job.log": b"audit log"}
    manifest = {"schema": 1, "commit": commit,
                "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in files.items()}}
    if corrupt:
        files["job.log"] = b"changed log"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()
