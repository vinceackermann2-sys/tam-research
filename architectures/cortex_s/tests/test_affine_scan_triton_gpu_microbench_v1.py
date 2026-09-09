from __future__ import annotations

import ast
from pathlib import Path

import torch

from architectures.cortex_s.experiments import affine_scan_triton_gpu_microbench_v1 as bench


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_scan_gpu_microbench_protocol_is_fresh_and_non_authorizing() -> None:
    protocol = bench.validate_protocol()
    assert protocol["classification"] == "ENGINEERING_GPU_CORRECTNESS_AND_MICROBENCH_ONLY"
    assert protocol["engineering_seed"] == 2_026_090_907
    assert protocol["source_parent"] == "15a2cc2cdddfa9f22a632a2c72ad4bf95d94ed75"
    assert protocol["production_shape"] == [64, 512, 128]
    assert protocol["trigger_title"] == "[modal-cortex-s-affine-scan-triton-gpu-microbench-v1]"
    assert 2_026_090_906 in protocol["consumed_engineering_seeds"]
    assert 8_100 in protocol["forbidden_scientific_seeds"]
    assert 48_131 in protocol["forbidden_scientific_seeds"]
    assert protocol["gpu_dispatch_authorized"] is False
    assert protocol["production_integration_authorized"] is False
    assert protocol["full_training_authorized"] is False


def test_small_cpu_correctness_probe_exercises_semantic_fallback() -> None:
    result = bench._correctness_probe(
        shape=(2, 11, 3),
        device=torch.device("cpu"),
        dtype=torch.float32,
        atol=1e-5,
        rtol=1e-4,
        seed_offset=900,
    )
    assert result["pass"] is True
    assert result["forward"]["finite_left"] is True
    assert result["forward"]["finite_right"] is True
    assert result["grad_a"]["allclose"] is True
    assert result["grad_b"]["allclose"] is True
    assert result["grad_initial"]["allclose"] is True


def test_launcher_has_single_bounded_h100_entrypoint_and_no_training() -> None:
    path = REPO_ROOT / "modal_cortex_s_affine_scan_triton_gpu_microbench_v1.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'APP_NAME = "cortex-s-affine-scan-triton-gpu-microbench-v1"' in source
    assert 'RESULT_ROOT = "/vol/cortex-s-v0/affine-scan-triton-gpu-microbench-v1"' in source
    assert "ENGINEERING_SEED = 2_026_090_907" in source
    assert "H100_TIMEOUT_SECONDS = 10 * 60" in source
    assert source.count('gpu="H100!"') == 1
    assert "train_full_2b" not in source
    assert "full_2b" not in source


def test_issue_workflow_is_exact_title_source_bound_and_not_automatic() -> None:
    path = REPO_ROOT / ".github/workflows/modal-cortex-s-affine-scan-triton-gpu-microbench-v1.yml"
    source = path.read_text(encoding="utf-8")
    assert "[modal-cortex-s-affine-scan-triton-gpu-microbench-v1]" in source
    assert "github.event.issue.user.login == github.repository_owner" in source
    assert "git rev-parse origin/main" in source
    assert "source_sha" in source
    assert "2026090907" in source
    assert "workflow_dispatch" not in source
    assert "schedule:" not in source


def test_candidate_remains_unwired_from_production() -> None:
    production = (REPO_ROOT / "architectures/cortex_s/language_model.py").read_text(encoding="utf-8")
    assert "affine_scan_triton_candidate" not in production
    assert "affine_scan_triton_gpu_microbench" not in production
