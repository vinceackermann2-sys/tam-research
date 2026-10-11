"""Connection boundary tests that require no Torch or Google Cloud credentials."""

import importlib.util
from pathlib import Path
import shlex
import unittest

spec = importlib.util.spec_from_file_location("gcp_batch", Path(__file__).parents[1] / "scripts/gcp_batch.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CloudJobTests(unittest.TestCase):
    def test_command_arguments_are_not_executable_shell(self):
        config = module.validate_config({"args": ["$(touch /tmp/injected)", "a; exit 1", "line\nbreak"]})
        job = module.build_job(config, "a" * 40, "runtime@example.iam.gserviceaccount.com")
        shell = job["taskGroups"][0]["taskSpec"]["runnables"][0]["container"]["commands"][1]
        self.assertEqual(shlex.split(shell.split("export PYTHONPATH=/tmp/tam-research PYTHONUNBUFFERED=1\n")[1]),
                         ["python", "scripts/gcp_compute_smoke.py", *config["args"]])

    def test_no_unbounded_or_gpu_jobs(self):
        for config in [{"timeout_seconds": 0}, {"timeout_seconds": 1801}, {"timeout_seconds": True},
                       {"machine_type": "a3-highgpu-1g"}, {"install_dependencies": "false"}]:
            with self.assertRaises(ValueError):
                module.validate_config(config)

    def test_paths_cannot_escape_repo(self):
        for script in ["../other.py", "/tmp/other.py", "scripts/../../other.py", "-c.py", "missing.py"]:
            with self.assertRaises(ValueError):
                module.validate_config({"script": script})

    def test_no_runtime_git_credentials_or_retries(self):
        job = module.build_job(module.validate_config({}), "b" * 40, "limited@example.iam.gserviceaccount.com")
        self.assertEqual(job["taskGroups"][0]["taskSpec"]["maxRetryCount"], 0)
        self.assertEqual(job["allocationPolicy"]["serviceAccount"]["email"], "limited@example.iam.gserviceaccount.com")
        self.assertNotIn("GITHUB_TOKEN", str(job))
        self.assertNotIn("secrets", str(job))

    def test_source_must_be_an_immutable_sha(self):
        with self.assertRaises(ValueError):
            module.build_job(module.validate_config({}), "main", "runtime@example.iam.gserviceaccount.com")


if __name__ == "__main__":
    unittest.main()
