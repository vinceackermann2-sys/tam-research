from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO_ROOT / "modal_cortex_s_affine_scan_triton_integrated_cuda_equivalence_v1.py"
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/modal-cortex-s-affine-scan-triton-integrated-cuda-equivalence-v1.yml"


def test_integrated_cuda_equivalence_runner_is_bounded_and_non_scientific() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    ast.parse(source)

    assert 'APP_NAME = "cortex-s-affine-scan-triton-integrated-cuda-equivalence-v1"' in source
    assert 'RESULT_ROOT = "/vol/cortex-s-v0/affine-scan-triton-integrated-cuda-equivalence-v1"' in source
    assert "ENGINEERING_SEED = 2_026_091_001" in source
    assert 'FROZEN_SOURCE_SHA = "46b29efb5cc0d975b4f4e6b3c397248723cacf3f"' in source
    assert 'FROZEN_SOURCE_TREE = "dc485e134eba99b709fcfa302a69616928286d7e"' in source
    assert 'INTEGRATION_BLOB = "c3d05f8b4d1e3e8ea7bcbad5fdeca5d7091651c7"' in source
    assert 'TRITON_CANDIDATE_BLOB = "5c6bf7df95f49c306b4ffadd6a8a09b9f71344f9"' in source
    assert "PRODUCTION_SHAPE = (64, 512, 512, 128)" in source
    assert "FP32_PROBE_SHAPE = (4, 64, 64, 16)" in source
    assert "EXPECTED_CORTEX_PARAMS = 101_778_112" in source
    assert "H100_TIMEOUT_SECONDS = 10 * 60" in source
    assert source.count('gpu="H100!"') == 1

    assert "ProductionScanPersistentWorldState" in source
    assert "convert_world_state_to_triton_scan" in source
    assert "_should_use_triton" in source
    assert "affine_scan_triton_candidate" in source
    assert "candidate_call_count" in source
    assert "grad_input" in source
    assert "grad_initial" in source
    assert "parameter_grads" in source
    assert "build_memory_lean_grouped_triton_scan_cortex_100m" in source
    assert "MemoryLeanPhysicalPaddedGroupedSparseMoE" in source
    assert '"timing_measured": False' in source
    assert '"optimizer_step_executed": False' in source
    assert '"corpus_accessed": False' in source
    assert '"production_preflight_authorized": False' in source
    assert '"full_training_authorized": False' in source
    assert '"scientific_claim_authorized": False' in source

    # The remote functions return JSON strings, so the local Modal client never
    # needs torch merely to deserialize a successful durable result.
    assert source.count("return json.dumps(result, sort_keys=True)") >= 2

    for forbidden in (
        "train_full_2b",
        "run_h100_calibration",
        "torch.optim",
        "scientific_train",
        "48131",
        "48132",
        "48133",
    ):
        assert forbidden not in source


def test_issue_workflow_binds_frozen_harness_and_pr821_source() -> None:
    source = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "[modal-cortex-s-affine-scan-triton-integrated-cuda-equivalence-v1]" in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "workflow_dispatch" not in source
    assert "schedule:" not in source

    assert "harness_sha" in source
    assert "git -C harness merge-base --is-ancestor" in source
    assert "16c299ecc409a1f9b97dcb396193596aca151922" in source

    assert "46b29efb5cc0d975b4f4e6b3c397248723cacf3f" in source
    assert "dc485e134eba99b709fcfa302a69616928286d7e" in source
    assert "c3d05f8b4d1e3e8ea7bcbad5fdeca5d7091651c7" in source
    assert "5c6bf7df95f49c306b4ffadd6a8a09b9f71344f9" in source
    assert "2026091001" in source

    assert "Install Modal only" in source
    assert "pip install 'modal>=1.1,<2'" in source
    assert "cp harness/modal_cortex_s_affine_scan_triton_integrated_cuda_equivalence_v1.py source/" in source
    assert "modal run --detach --timestamps modal_cortex_s_affine_scan_triton_integrated_cuda_equivalence_v1.py" in source
    assert "production preflight" in source
    assert "2B training" in source


def test_harness_does_not_modify_or_require_main_production_integration() -> None:
    # The harness branch is intentionally based on current main while PR #821
    # remains separate. The future issue workflow checks out #821 as source.
    integration_path = REPO_ROOT / "architectures/cortex_s/production_scan_integration_v1.py"
    assert not integration_path.exists()
