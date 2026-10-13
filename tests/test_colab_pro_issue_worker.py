"""Zero-GPU unit tests for the Colab Pro issue bridge."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
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
                 patch.object(bridge, "remotely_consumed", return_value=False), \
                 patch.object(bridge, "source_from_main", return_value="print(1)"), \
                 patch.object(bridge, "execute", return_value=True) as execute:
                attempts = bridge.sweep(args, remaining=1)
            self.assertEqual(attempts, 1)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(execute.call_args[0][1]["number"], 43)


if __name__ == "__main__":
    unittest.main()
