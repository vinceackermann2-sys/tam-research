import importlib.util
from pathlib import Path
import unittest
from datetime import datetime, timezone

spec = importlib.util.spec_from_file_location("gcp_gpu", Path(__file__).parents[1] / "scripts/gcp_gpu.py")
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


def config():
    return {"source_sha": "a" * 40, "script": "scripts/gcp_smoke.py", "gpu": "H100", "minutes": 30}


class SafetyTests(unittest.TestCase):
    def test_expired_or_missing_credits_block_compute(self):
        now = datetime(2026, 10, 12, tzinfo=timezone.utc)
        for value in ("", "2026-10-11T00:00:00Z", "2026-10-12T00:05:00Z", "2026-12-01"):
            with self.assertRaises(ValueError):
                gpu.check_credit_expiry(config(), value, now=now)
        gpu.check_credit_expiry(config(), "2026-12-01T00:00:00Z", now=now)

    def test_reject_unbounded_or_unsafe_inputs(self):
        for field, value in [
            ("gpu", "H100:8"), ("minutes", 241), ("minutes", True),
            ("source_sha", "main"), ("script", "../outside.py"),
            ("script", "modal_app.py"), ("zone", "europe-west4-a"), ("args", "--foo"),
        ]:
            with self.subTest(field=field, value=value):
                cfg = config()
                cfg[field] = value
                with self.assertRaises(ValueError):
                    gpu.validate(cfg)


    def test_valid_gpu_selection_and_literal_arguments(self):
        for name in gpu.GPU_TYPES:
            cfg = config() | {"gpu": name, "args": ["$(touch /tmp/injection)"]}
            self.assertEqual(gpu.validate(cfg)["args"], cfg["args"])


    def test_no_job_can_overdraw_lifetime_allowance(self):
        balance = gpu.remaining_allowance([{"run_id": "prior", "cents": 24000}])
        self.assertEqual(balance, 0)
        self.assertGreater(gpu.reservation_cents(config()), balance)


    def test_ledger_rejects_credit_or_duplicate_reservations(self):
        for records in ([{"run_id": "prior", "cents": -1}],
                        [{"run_id": "prior", "cents": 100}] * 2):
            with self.assertRaises(ValueError):
                gpu.remaining_allowance(records)


    def test_startup_has_no_github_credentials_or_shell_eval(self):
        text = gpu.startup(gpu.validate(config()), "test-bucket", "123-1")
        self.assertNotIn("GITHUB_TOKEN", text)
        self.assertIn("shutdown -h now", text)
        self.assertIn("subprocess.run(['timeout'", text)
        self.assertIn("cfg['script'], *cfg['args']", text)


if __name__ == "__main__":
    unittest.main()
