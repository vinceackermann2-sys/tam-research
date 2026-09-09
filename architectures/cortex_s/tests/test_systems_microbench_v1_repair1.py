from __future__ import annotations

from pathlib import Path


def _repo() -> Path:
    return Path(__file__).resolve().parents[3]


def test_repair1_workflow_uses_fresh_trigger_and_never_imports_cortex_before_modal_handoff():
    workflow = (_repo() / ".github/workflows/modal-cortex-s-100m-systems-microbench-v1-repair1.yml").read_text(
        encoding="utf-8"
    )
    assert "[modal-cortex-s-100m-systems-microbench-v1-repair1]" in workflow
    assert "phase == 'microbench-v1-repair1'" in workflow
    assert "workflow_dispatch" not in workflow
    assert "github.event.issue.user.login == github.repository_owner" in workflow
    assert "git rev-parse origin/main" in workflow
    assert "Fail closed with stdlib-only static protocol check" in workflow
    # Regression for consumed issue #781: the original launcher imported the
    # architecture package before PyTorch existed on the vanilla runner.
    assert "from architectures." not in workflow
    assert "import torch" not in workflow
    assert "modal_cortex_s_100m_systems_microbench_v1_repair1.py" in workflow


def test_repair1_modal_app_has_fresh_result_namespace_and_no_full_training_path():
    app = (_repo() / "modal_cortex_s_100m_systems_microbench_v1_repair1.py").read_text(
        encoding="utf-8"
    )
    assert 'APP_NAME = "cortex-s-v0-100m-systems-microbench-v1-repair1"' in app
    assert 'RESULT_ROOT = "/vol/cortex-s-v0/100m-systems-microbench-v1-repair1"' in app
    assert "ENGINEERING_SEED = 2_026_090_901" in app
    assert "H100_TIMEOUT_SECONDS = 15 * 60" in app
    assert 'phase: str = "microbench-v1-repair1"' in app
    assert "H100_DISPATCH_CONSUMED.json" in app
    assert "def full_2b(" not in app
    assert "train_full_2b" not in app
    assert "full_training_authorized\"] = False" in app
    assert "next_stage_authorized\"] = False" in app


def test_repair1_prereg_records_pre_modal_failure_and_seed_nonconsumption():
    doc = (_repo() / "architectures/cortex_s/SYSTEMS_MICROBENCH_100M_V1_REPAIR1.md").read_text(
        encoding="utf-8"
    )
    assert "34324639156" in doc
    assert "ModuleNotFoundError: No module named 'torch'" in doc
    assert "engineering seed `2026090901` remains unconsumed" in doc
    assert "issue #781" in doc
    assert "must never be rerun or reused" in doc
    assert "/vol/cortex-s-v0/100m-systems-microbench-v1-repair1" in doc
    assert "cannot authorize the 2B run" in doc
