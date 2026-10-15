"""Mocked source-readiness lifecycle; no network, OAuth or scientific execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load("github_lifecycle_worker", ROOT / "scripts/colab_pro_issue_worker.py")
fixtures = load("github_lifecycle_fixtures", ROOT / "tests/test_colab_github_readiness_artifacts.py")


class GitHubWorkerLifecycleTests(unittest.TestCase):
    def lifecycle(self, mode="success"):
        job = fixtures.JOB
        github = fixtures.FakeGitHub(tamper=mode == "readback-corrupt")
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "remote-fixture.zip"
            changes = {"run.log": fixtures.SECRET.encode()} if mode == "private-log" else None
            descriptor = fixtures.fixture(archive, changes=changes)
            args = SimpleNamespace(repo=fixtures.adapter.REPOSITORY, state_dir=root,
                                   max_seconds=180, artifact_remote=None,
                                   artifact_github_readiness=True)

            def command(argv, **kwargs):
                if argv[:2] == ["gh", "api"]:
                    if any("/git/refs" in part for part in argv):
                        events.append("reserve-global-tag")
                        return SimpleNamespace(returncode=0, stdout="", stderr="")
                    if "POST" in argv:
                        events.append("publish")
                    if argv[-1].endswith("/issues/comments/456"):
                        events.append("readback")
                    kwargs.setdefault("check", True)
                    kwargs.setdefault("timeout", 60)
                    return github(argv, **kwargs)
                self.assertEqual(argv[0], "colab")
                if "--help" in argv:
                    return SimpleNamespace(returncode=0, stdout="--keep --timeout", stderr="")
                if "run" in argv:
                    events.append("run")
                    self.assertIn("--keep", argv)
                    self.assertNotIn("--gpu", argv)
                    return SimpleNamespace(returncode=0, stdout="COLAB_ARTIFACT=" + json.dumps(descriptor), stderr="")
                if "download" in argv:
                    events.append("download")
                    Path(argv[-1]).write_bytes(archive.read_bytes())
                if "stop" in argv:
                    events.append("stop")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def account():
                events.append("account")
                return {"balance_units": 100., "active_assignments": 0, "sessions": ""}

            with patch.object(worker, "cmd", side_effect=command), \
                 patch.object(worker, "account_snapshot", side_effect=account), \
                 patch.object(worker, "comment"), \
                 patch.object(worker, "cloud_preflight") as private_cloud:
                self.assertTrue(worker.execute(args, {"number": fixtures.NUMBER}, job,
                                "COLAB_JOB_ID = 'fixture-github-readiness-v1'"))
                private_cloud.assert_not_called()
            receipt = json.loads((root / f"issue-{fixtures.NUMBER}.json").read_text())
            self.assertFalse(receipt["retry_allowed"])
            self.assertTrue(receipt["stop_ok"])
            self.assertEqual(receipt["state"], "SUCCESS" if mode == "success" else "FAILED")
            self.assertEqual(events.count("run"), 1)
            self.assertEqual(events.count("publish"), 1)
            self.assertEqual(events.count("account"), 3)
            self.assertLess(events.index("download"), events.index("publish"))
            self.assertLess(events.index("readback"), events.index("stop"))
            self.assertLess(events.index("run"), events.index("stop"))
            self.assertEqual(events[-1], "account")
            self.assertIsNone(receipt["local_archive"])
            self.assertIsNone(receipt["local_log"])
            self.assertNotIn(fixtures.SECRET, github.body)
            if mode == "success":
                self.assertTrue(receipt["artifact_ok"])
                evidence = receipt["cloud_evidence"]
                self.assertTrue(evidence["verified_before_teardown"])
                self.assertFalse(evidence["original_zip_retained"])
            elif mode == "private-log":
                self.assertFalse(receipt["cloud_evidence"]["complete"])
                self.assertIn('"status":"incomplete"', github.body)
            else:
                self.assertFalse(receipt["artifact_ok"])
            self.assertFalse(github.payload_path.exists())

    def test_public_readiness_retained_and_read_back_before_stop(self):
        self.lifecycle()

    def test_unexpected_private_log_is_replaced_by_incomplete_receipt(self):
        self.lifecycle("private-log")

    def test_failed_github_readback_consumes_attempt_and_still_stops(self):
        self.lifecycle("readback-corrupt")

    def test_github_mode_rejects_microfit_and_probe_before_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(repo=fixtures.adapter.REPOSITORY, state_dir=Path(temporary),
                                   artifact_remote=None, artifact_github_readiness=True)
            for job in (dict(fixtures.JOB, script="colab_jobs/chm_v4_microfit_1436.py"),
                        dict(fixtures.JOB, script="scripts/colab_bridge_gpu_probe.py", gpu="T4")):
                with self.subTest(script=job["script"]), \
                     patch.object(worker, "account_snapshot") as account, \
                     patch.object(worker, "cmd") as command, patch.object(worker, "reserve") as reserve:
                    with self.assertRaises(ValueError):
                        worker.execute(args, {"number": fixtures.NUMBER}, job,
                                       "COLAB_JOB_ID = 'fixture-github-forbidden-training-v1'")
                    account.assert_not_called()
                    reserve.assert_not_called()
                    command.assert_not_called()

    def test_github_worker_skips_incompatible_queue_jobs_without_consuming_them(self):
        def issue(number, job):
            return {"number": number, "state": "open", "user": {"login": fixtures.OWNER},
                    "labels": [{"name": "colab-approved"}],
                    "body": worker.MARKER + "\n```json\n" + json.dumps(job) + "\n```"}
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(repo=fixtures.adapter.REPOSITORY, state_dir=Path(temporary),
                                   repo_path=ROOT, allowed_gpus={"CPU", "T4"}, artifact_github_readiness=True)
            queue = [issue(121, dict(fixtures.JOB, script="scripts/colab_bridge_gpu_probe.py", gpu="T4")),
                     issue(122, dict(fixtures.JOB, script="colab_jobs/chm_v4_microfit_1436.py")),
                     issue(123, fixtures.JOB)]
            with patch.object(worker, "gh_json", return_value=queue), \
                 patch.object(worker, "remotely_consumed", return_value=False), \
                 patch.object(worker, "source_from_main", return_value="reviewed source") as source, \
                 patch.object(worker, "current_approval", return_value=True), \
                 patch.object(worker, "execute", return_value=True) as execute:
                self.assertEqual(worker.sweep(args, 1), 1)
                self.assertEqual(source.call_count, 1)
                self.assertEqual(execute.call_count, 1)
                self.assertEqual(execute.call_args.args[1]["number"], 123)


if __name__ == "__main__":
    unittest.main()
