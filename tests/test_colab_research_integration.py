"""CPU-only contract tests; fake CLI/receipts, never execute the microfit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("worker", ROOT / "scripts/colab_pro_issue_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
runtime_spec = importlib.util.spec_from_file_location("runtime", ROOT / "tam_research/colab_job_runtime.py")
runtime = importlib.util.module_from_spec(runtime_spec)
runtime_spec.loader.exec_module(runtime)
package_artifacts, source_audit, write_json = runtime.package_artifacts, runtime.source_audit, runtime.write_json


def fixture_bundle(output, job, number):
    output.mkdir()
    write_json(output / "configuration.json", {"steps": 0})
    write_json(output / "metrics.json", {"status": "source-ready"})
    write_json(output / "provenance.json", {"issue": number, **job})
    (output / "run.log").write_text("cpu fixture\n")
    return package_artifacts(output)


class ArtifactContractTests(unittest.TestCase):
    def test_bundle_roundtrip_and_wrong_provenance_or_corruption(self):
        job = {"commit": "a" * 40, "script": "colab_jobs/chm_v4_readiness.py", "gpu": "CPU"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            descriptor = fixture_bundle(root, job, 123)
            self.assertEqual(worker.verify_artifact(root / "bundle.zip", descriptor, job, 123), descriptor["sha256"])
            with self.assertRaises(ValueError):
                worker.verify_artifact(root / "bundle.zip", descriptor, job, 124)
            (root / "bundle.zip").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                worker.verify_artifact(root / "bundle.zip", descriptor, job, 123)

    def test_train_bundle_requires_checkpoint(self):
        job = {"commit": "a" * 40, "script": "colab_jobs/chm_v4_microfit_1436.py", "gpu": "CPU"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            descriptor = fixture_bundle(root, job, 123)
            with self.assertRaisesRegex(ValueError, "required"):
                worker.verify_artifact(root / "bundle.zip", descriptor, job, 123)

    def test_archive_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../secret", "bad")
            desc = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                worker.verify_artifact(path, desc, {}, 123)

    def test_source_lock_and_no_training_import_on_preflight(self):
        self.assertEqual(len(source_audit(ROOT)), 3)
        source = (ROOT / "colab_jobs/chm_v4_readiness.py").read_text()
        self.assertIn("training=False", source)
        self.assertNotIn('"pip", "install"', source)
        for script in ROOT.glob("colab_jobs/*.py"):
            self.assertIn('actual != a.repo_commit', script.read_text())
            self.assertIn('hashlib.sha256(expected).hexdigest() != a.script_sha256', script.read_text())
            self.assertNotIn('Path(__file__)', script.read_text())

    def test_descriptor_rejects_invalid_and_missing(self):
        for line in ('', 'COLAB_ARTIFACT={"sha256":"bad","bytes":12}',
                     'COLAB_ARTIFACT={"sha256":"' + 'a'*64 + '","bytes":999999999}'):
            with self.assertRaises(ValueError):
                worker.artifact_descriptor(line)


class WorkerLifecycleTests(unittest.TestCase):
    def lifecycle(self, mode="success"):
        job = {"commit": "a" * 40, "script": "colab_jobs/chm_v4_readiness.py", "gpu": "CPU"}
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            desc = fixture_bundle(root / "output", job, 123)
            output = 'COLAB_ARTIFACT=' + json.dumps(desc)
            args = SimpleNamespace(repo="vinceackermann2-sys/tam-research", state_dir=root, max_seconds=180)
            def fake_cmd(command, **kwargs):
                calls.append(command)
                if any("/git/refs" in part for part in command):
                    if mode == "duplicate":
                        raise RuntimeError("existing immutable tag")
                if "run" in command:
                    self.assertIn("--keep", command)
                    self.assertIn("--timeout", command)
                    self.assertIn("--repo-commit", command)
                    if mode == "timeout":
                        raise subprocess.TimeoutExpired(command, 180)
                    return SimpleNamespace(returncode=0, stdout=output, stderr="")
                if "download" in command:
                    Path(command[-1]).write_bytes((root / "output/bundle.zip").read_bytes() if mode != "corrupt" else b"bad")
                if "stop" in command and mode == "success":
                    # Retrieval and fsync verification must have happened while runtime was kept.
                    self.assertTrue((root / "issue-123-bundle.zip").exists())
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            before = {"balance_units": 100., "active_assignments": 0, "sessions": ""}
            after = dict(before, balance_units=99.9)
            with patch.object(worker, "cmd", side_effect=fake_cmd), \
                 patch.object(worker, "account_snapshot", side_effect=[before, after]), \
                 patch.object(worker, "comment"):
                self.assertTrue(worker.execute(args, {"number": 123}, job, "COLAB_JOB_ID = 'fixture-job-123-v1'"))
            receipt = json.loads((root / "issue-123.json").read_text())
            self.assertFalse(receipt["retry_allowed"])
            self.assertTrue(receipt["stop_ok"])
            self.assertEqual(receipt["artifact_ok"], mode == "success")
            self.assertEqual(receipt["state"], "SUCCESS" if mode == "success" else "FAILED")
            self.assertEqual(sum("run" in c for c in calls), 0 if mode == "duplicate" else 1)
            self.assertEqual(sum("stop" in c for c in calls), 1)
            if mode == "success":
                self.assertLess(next(i for i,c in enumerate(calls) if "download" in c),
                                next(i for i,c in enumerate(calls) if "stop" in c))

    def test_download_verified_before_teardown(self):
        self.lifecycle()

    def test_corruption_still_tears_down_and_consumes_attempt(self):
        self.lifecycle("corrupt")

    def test_duplicate_global_job_tag_never_launches(self):
        self.lifecycle("duplicate")

    def test_timeout_still_tears_down_without_retry(self):
        self.lifecycle("timeout")

    def test_usage_parser_and_fail_closed(self):
        with patch.object(worker, "cmd", side_effect=[
            SimpleNamespace(stdout="Current balance: 108.89 compute units\nActive assignments: 0"),
            SimpleNamespace(stdout="No sessions")]):
            self.assertEqual(worker.account_snapshot()["balance_units"], 108.89)
        with patch.object(worker, "cmd", return_value=SimpleNamespace(stdout="unknown")):
            with self.assertRaises(ValueError):
                worker.account_snapshot()

    def test_remote_receipt_blocks_consumed_probe_on_new_machine(self):
        comments = [[{"user": {"login": "vinceackermann2-sys"},
                      "body": "Colab bridge: one attempt reserved. No automatic retry."}]]
        with patch.object(worker, "cmd", return_value=SimpleNamespace(stdout=json.dumps(comments))):
            self.assertTrue(worker.remotely_consumed("vinceackermann2-sys/tam-research", 1443))

    def test_empty_balance_never_provisions_or_reserves(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(repo="owner/repo", state_dir=Path(temporary))
            with patch.object(worker, "account_snapshot", return_value={"balance_units": 0, "active_assignments": 0}), \
                 patch.object(worker, "cmd") as call:
                with self.assertRaises(ValueError):
                    worker.execute(args, {"number": 12}, {"script": "scripts/colab_bridge_gpu_probe.py"}, "")
                call.assert_not_called()
            self.assertFalse((Path(temporary) / "issue-12.json").exists())

    def test_zero_remaining_never_scans_or_launches(self):
        with patch.object(worker, "gh_json") as gh:
            self.assertEqual(worker.sweep(SimpleNamespace(), 0), 0)
            gh.assert_not_called()

class JobIdentityTests(unittest.TestCase):
    def test_reviewed_literal_identity_required(self):
        self.assertEqual(worker.research_identity("COLAB_JOB_ID = 'fixture-job-v1'"), "fixture-job-v1")
        for source in ("print(1)", "COLAB_JOB_ID = input()", "COLAB_JOB_ID = '../unsafe'"):
            with self.assertRaises(ValueError):
                worker.research_identity(source)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch checked by original repo CPU CI")
    def test_microfit_configuration_and_exact_train_only_stream_without_training(self):
        from tam_research.colab_chm_v4_microfit_1436 import configuration, training_cases
        from tam_research.chm_v4_neural_hard_copy_train_1430 import encode_visible_v4
        from tam_research.chm_v4_independent_binding_benchmark_1421 import seal_input
        config = configuration()
        cases = training_cases()
        self.assertEqual(config["steps"], 256)
        self.assertEqual(config["model_init_seed"], 1377005)
        self.assertTrue(all(e.split == "train" for e in cases))
        self.assertEqual(len(cases), 32)
        self.assertEqual([(e.family, e.variant) for e in cases],
                         [(f, v) for f in config["families"] for v in range(8)])
        tokens = sum(len(encode_visible_v4(seal_input(e)).model_tokens.token_ids) for e in cases) * 8
        self.assertLessEqual(tokens, config["max_input_tokens"])

class StandalonePreflightTests(unittest.TestCase):
    def test_actual_source_preflight_without_site_packages_or_torch(self):
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifacts"
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            job = {"commit": commit, "script": "colab_jobs/chm_v4_readiness.py", "gpu": "CPU"}
            code = f'''
import importlib.util
from pathlib import Path
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location("runtime", {str(ROOT / 'tam_research/colab_job_runtime.py')!r})
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
args = SimpleNamespace(repo_commit={commit!r}, issue=123, job_script={job['script']!r}, requested_gpu="CPU")
runtime.run_job(args, Path({str(ROOT)!r}), Path({str(output)!r}), training=False)
import sys
assert "torch" not in sys.modules
'''
            result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, check=True)
            descriptor = worker.artifact_descriptor(result.stdout)
            worker.verify_artifact(output / "bundle.zip", descriptor, job, 123)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertFalse(metrics["training_executed"])
            self.assertFalse(metrics["heldout_generated"])

class ReadinessTests(unittest.TestCase):
    def test_status_checks_actual_worker_lock_not_stale_heartbeat(self):
        import contextlib
        import fcntl
        import io
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker.atomic_json(root / "worker-status.json", {"last_poll": 0, "pid": 99999999})
            for active in (False, True):
                with (root / "worker.lock").open("a+") as lock:
                    if active:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    stdout = io.StringIO()
                    with patch.object(sys, "argv", ["worker", "--status", "--state-dir", temporary]), \
                         patch.object(worker, "account_snapshot", return_value={"balance_units": 99., "sessions": "PRIVATE", "active_assignments": 0}), \
                         contextlib.redirect_stdout(stdout):
                        worker.main()
                    data = json.loads(stdout.getvalue())
                    self.assertEqual(data["worker_running"], active)
                    self.assertNotIn("PRIVATE", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
