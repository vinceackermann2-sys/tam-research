"""CPU validation; never invokes Colab, consumes a scientific seed, or trains."""
import importlib.util
import importlib.metadata
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("chm_colab_job", ROOT / "colab_jobs/chm_v4_train_audit.py")
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


class StandaloneChmJobTests(unittest.TestCase):
    def test_real_train_only_audit_and_complete_bundle(self):
        def bootstrap(commit, checkout, log):
            checkout.symlink_to(ROOT, target_is_directory=True)
            log.write("CPU test uses existing reviewed checkout/dependencies\n")
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(job, "bootstrap", side_effect=bootstrap), \
             patch.object(job, "NUMPY_VERSION", importlib.metadata.version("numpy")), \
             patch.object(job.shutil, "which", return_value=None):
            folder = job.run_job("a" * 40, "CPU", Path(temporary))
            result = json.loads((folder / "result.json").read_text())
            self.assertEqual(result["training_cases_only"], 2048)
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertEqual(result["trained_model_forward_calls"], 0)
            self.assertEqual(result["development_test_examples_created"], 0)
            self.assertEqual(result["historical_seed_consumed"], 2013161)
            self.assertEqual(result["status"], "ok")
            with zipfile.ZipFile(folder / "bundle.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"manifest.json", "config.json", "result.json", "environment.json", "job.log"})
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["commit"], "a" * 40)

    def test_bootstrap_verifies_full_commit_before_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(job, "command") as command, \
                 patch.object(job.subprocess, "check_output", return_value="b" * 40 + "\n"):
                with self.assertRaises(RuntimeError):
                    job.bootstrap("a" * 40, Path(temporary) / "repo", io.StringIO())
                self.assertEqual(command.call_count, 4)
                self.assertEqual(command.call_args_list[2].args[0],
                                 ["git", "fetch", "--quiet", "--depth", "1", "origin", "a" * 40])
            with patch.object(job, "command") as command, \
                 patch.object(job.subprocess, "check_output", return_value="a" * 40 + "\n"):
                job.bootstrap("a" * 40, Path(temporary) / "repo2", io.StringIO())
                installs = [call.args[0] for call in command.call_args_list[4:]]
                self.assertIn("torch==2.10.0+cpu", installs[0])
                self.assertIn("numpy==2.5.4", installs[1])
                self.assertIn("https://download.pytorch.org/whl/cpu", installs[0])

    def test_bootstrap_failure_still_produces_evidence_bundle(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(job, "bootstrap", side_effect=RuntimeError("bootstrap failed")), \
             patch.object(job.shutil, "which", return_value=None):
            with self.assertRaises(RuntimeError):
                job.run_job("a" * 40, "CPU", Path(temporary))
            folder = Path(temporary) / "tam-results"
            self.assertEqual(json.loads((folder / "result.json").read_text())["status"], "failed")
            self.assertTrue((folder / "bundle.zip").exists())

    def test_gpu_or_scope_change_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                job.run_job("a" * 40, "T4", Path(temporary))
            self.assertFalse((Path(temporary) / "tam-results").exists())
        with self.assertRaises(RuntimeError):
            job.validate_result({"optimizer_steps": 1})
