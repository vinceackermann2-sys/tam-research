from __future__ import annotations

from pathlib import Path

import torch

from architectures.cortex_s.language_model import TrulySparseMoE, parameter_count
from architectures.cortex_s.experiments.scale100m_2b.systems_microbench_v1_repair4 import (
    CONSUMED_ENGINEERING_SEEDS,
    ENGINEERING_SEED,
    PhysicalPaddedBF16GroupedSparseMoE,
    _padded_hidden,
    validate_microbench_protocol,
    zero_gpu_physical_padding_contract_probe,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_repair4_physical_hidden_is_real_aligned_width() -> None:
    assert _padded_hidden(338) == 344
    probe = zero_gpu_physical_padding_contract_probe()
    assert probe["status"] == "PASS"
    assert probe["logical_hidden"] == 338
    assert probe["physical_hidden"] == 344
    assert probe["padding_channels"] == 6
    assert probe["w1_shape"] == [8, 512, 344]
    assert probe["w1_stride"][-2:] == [344, 1]
    assert probe["w2_shape"] == [8, 344, 512]
    assert probe["w2_stride"][-2:] == [512, 1]
    assert probe["activation_stride"] == [344, 1]
    assert probe["all_grouped_layouts_valid"] is True
    assert probe["runtime_padding_adds_parameters"] is False


def test_repair4_padded_cpu_reference_preserves_moe_semantics_and_params() -> None:
    torch.manual_seed(2026090904)
    legacy = TrulySparseMoE(d_model=16, num_experts=4, top_k=2, hidden=10).eval()
    candidate = PhysicalPaddedBF16GroupedSparseMoE(
        d_model=16, num_experts=4, top_k=2, hidden=10
    ).eval()
    candidate.copy_from_legacy(legacy)

    assert candidate.physical_hidden == 16
    assert parameter_count(candidate) == parameter_count(legacy)

    x = torch.randn(3, 7, 16)
    with torch.no_grad():
        legacy_out = legacy(x)
        candidate_out = candidate(x)
    assert torch.allclose(candidate_out, legacy_out, rtol=1e-5, atol=1e-6)


def test_repair4_seed_and_authority_are_fresh_and_fail_closed() -> None:
    protocol = validate_microbench_protocol()
    assert ENGINEERING_SEED == 2026090904
    assert 2026090903 in CONSUMED_ENGINEERING_SEEDS
    assert ENGINEERING_SEED not in CONSUMED_ENGINEERING_SEEDS
    assert protocol["parent_repair3_run_id"] == 34335827718
    assert protocol["parent_repair3_job_id"] == 102414857556
    assert protocol["logical_hidden"] == 338
    assert protocol["physical_hidden"] == 344
    assert protocol["full_training_authorized"] is False
    assert protocol["next_stage_authorized"] is False
    assert 8100 in protocol["forbidden_seeds"]
    assert 48131 in protocol["forbidden_seeds"]
    assert 48132 in protocol["forbidden_seeds"]
    assert 48133 in protocol["forbidden_seeds"]


def test_repair4_launcher_has_unique_namespace_json_transport_and_no_full_run() -> None:
    source = (
        REPO_ROOT / "modal_cortex_s_100m_systems_microbench_v1_repair4.py"
    ).read_text(encoding="utf-8")
    assert 'APP_NAME = "cortex-s-v0-100m-systems-microbench-v1-repair4"' in source
    assert 'RESULT_ROOT = "/vol/cortex-s-v0/100m-systems-microbench-v1-repair4"' in source
    assert 'ENGINEERING_SEED = 2_026_090_904' in source
    assert 'gpu="H100!"' in source
    assert 'H100_TIMEOUT_SECONDS = 15 * 60' in source
    assert "_json_safe" in source
    assert "json.loads(json.dumps(value))" in source
    assert 'phase: str = "microbench-v1-repair4"' in source
    assert "def full_2b(" not in source
    assert "train_full_2b" not in source


def test_repair4_workflow_has_single_use_trigger_and_exact_source_binding() -> None:
    workflow = (
        REPO_ROOT / ".github/workflows/modal-cortex-s-100m-systems-microbench-v1-repair4.yml"
    ).read_text(encoding="utf-8")
    assert "[modal-cortex-s-100m-systems-microbench-v1-repair4]" in workflow
    assert "phase == 'microbench-v1-repair4'" in workflow
    assert "Verify checkout and live main are exactly the frozen source" in workflow
    assert "modal_cortex_s_100m_systems_microbench_v1_repair4.py" in workflow
    assert "2026090904" in workflow
    assert "[modal-cortex-s-100m-systems-microbench-v1-repair3]" not in workflow
