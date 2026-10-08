from pathlib import Path
import ast
import unittest
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "modal_cortex_attention8_v2_successor_readonly_status_v1.py"
WORKFLOW = ROOT / ".github/workflows/modal-cortex-attention8-v2-successor-readonly-status-v1.yml"

class ReadOnlyStatusTests(unittest.TestCase):
    def test_contract(self):
        source = SCRIPT.read_text(encoding="utf-8")
        ast.parse(source)
        for required in ("27a3a15cb8456523798163f4cfa5a829844477ed",
                         "027e0c1bbc845783d547f935f378889d298c9dab",
                         "5d9e1a170d9bbb93e4a02de6a47467049df10d55",
                         "SCIENTIFIC_SEED = 60232", "create_if_missing=False",
                         "volume.reload()", '"PAIR1_DISPATCH_RESERVED.json"', '"RESULT.json"',
                         "retries=0", '"writes_performed": False',
                         '"new_seed_consumed": False', '"gpu_allocated": False'):
            self.assertIn(required, source)
    def test_zero_gpu_and_one_shot(self):
        source = SCRIPT.read_text(encoding="utf-8")
        workflow = WORKFLOW.read_text(encoding="utf-8")
        for forbidden in ("gpu=", ".commit(", ".write_text(", ".write_bytes(",
                          ".mkdir(", ".unlink(", "create_if_missing=True",
                          "h100_train_one.remote(", "reserve_pair_dispatch.remote("):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("workflow_dispatch:", workflow.split("\njobs:", 1)[0])
        self.assertIn('test "$RUN_ATTEMPT" = "1"', workflow)
        self.assertIn('"target_issue": 1300', workflow)
        self.assertIn('"target_run": 37748939335', workflow)
        self.assertEqual(workflow.count("modal run --detach --timestamps"), 1)
if __name__ == "__main__":
    unittest.main()
