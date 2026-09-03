from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

from tam_research import aera_hardware_core_v26_9_ficem_read_identity_weight_visibility as v269


ROOT = Path(__file__).resolve().parents[1]


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def test_issue600_frozen_lineage_blobs() -> None:
    expected = {
        "tam_research/aera_hardware_core_v26_8_ficem_read_mixed_strength_precision.py": "3575c58d1cd730be77649f087908c51dbf3e6088",
        "tam_research/aera_hardware_core_v26_7_ficem_read_mixed_dtype.py": "d8133c6b204b1ee5f23955255fb2fb09d09bd723",
        "tam_research/aera_hardware_core_v26_3_ficem_read_triton.py": "263f68eb1186a8ac14a08fc4b4df1fc5b292c711",
        "tam_research/aera_hardware_core_v26_6_ficem_write_materialize_cast.py": "d45c262314a0b4691f26812a279937a225043ad9",
        "tam_research/aera_hardware_core_v26.py": "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7",
        "tam_research/aera_hardware_core_v25_1_compact.py": "4e336b6e1a6238dac782fa320751d68281493ee1",
        "tam_research/aera_hardware_core_v25.py": "f8cce87fa4dcae69fd171ba95fcbdab50e743a2f",
        "tam_research/aera_v26_8_issue558_ficem_read_mixed_strength_precision_probe.py": "99ab8252f2b594404aae1ca86752eaa902eb80a5",
    }
    assert {_path: _git_blob(ROOT / _path) for _path in expected} == expected


def test_issue600_reference_literal_uses_identity_source_dtype() -> None:
    source = (ROOT / "tam_research/aera_hardware_core_v25.py").read_text()
    assert "torch.softmax(safe_logits.float(), dim=-1).to(identity_source.dtype)" in source


def test_issue600_dispatch_and_visibility_matrix() -> None:
    expected = {
        (torch.float32, torch.bfloat16, torch.float32, torch.float32): (
            "mixed-identity-weight-visibility-v26.9",
            "float32",
        ),
        (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32): (
            "mixed-identity-weight-visibility-v26.9",
            "bfloat16",
        ),
        (torch.float32, torch.float32, torch.bfloat16, torch.bfloat16): (
            "mixed-identity-weight-visibility-v26.9",
            "float32",
        ),
        (torch.float32, torch.float32, torch.float32, torch.float32): (
            "historical-repair5",
            "historical-repair5",
        ),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16): (
            "historical-repair5",
            "historical-repair5",
        ),
        (torch.float16, torch.float16, torch.float16, torch.float16): (
            "historical-repair5",
            "historical-repair5",
        ),
    }
    observed = {
        layout: (
            v269.read_dispatch_kind(*layout),
            v269.initial_weight_visibility_kind(*layout),
        )
        for layout in expected
    }
    assert observed == expected


def test_issue600_source_has_one_mixed_kernel_and_separate_controls() -> None:
    source = inspect.getsource(v269)
    assert source.count("@triton.jit") == 1
    assert "IS_BF16_COMPUTE: tl.constexpr" in source
    assert "DURABLE_IS_BF16: tl.constexpr" in source
    assert "WEIGHT_VISIBLE_BF16: tl.constexpr" in source
    assert "WEIGHT_VISIBLE_BF16=identity_dtype is torch.bfloat16" in source
    assert "WEIGHT_VISIBLE_BF16=similarity.dtype" not in source
    assert "if WEIGHT_VISIBLE_BF16:" in source
    assert "if IS_BF16_COMPUTE:" in source
    assert "if DURABLE_IS_BF16:" in source


def test_issue600_same_dtype_repair5_and_no_pytorch_tail_fallback() -> None:
    source = inspect.getsource(v269.fused_ficem_read_tail_v26_9)
    assert 'dispatch == "historical-repair5"' in source
    assert "return fused_ficem_read_tail(" in source
    whole = inspect.getsource(v269)
    for forbidden in ("torch.topk", "torch.softmax", ".gather(", "torch.stack", "torch.cat"):
        assert forbidden not in whole


def test_issue600_backend_passes_original_identity_dtype_and_inherits_write() -> None:
    source = inspect.getsource(v269.IdentityWeightVisibilityTritonFICEMReadWriteBackend.read)
    assert "identity_dtype=identity_source.dtype" in source
    assert "recalled=memory.out(recalled)" in source
    assert issubclass(
        v269.IdentityWeightVisibilityTritonFICEMReadWriteBackend,
        v269.StrengthPrecisionTritonFICEMReadWriteBackend,
    )
    assert "def update(" not in inspect.getsource(v269.IdentityWeightVisibilityTritonFICEMReadWriteBackend)
    assert "def update_from_projected(" not in inspect.getsource(v269.IdentityWeightVisibilityTritonFICEMReadWriteBackend)


def test_issue600_protocol_preserves_history_and_authorizes_nothing_higher() -> None:
    protocol = v269.identity_weight_visibility_v26_9_protocol()
    assert protocol["research_issue"] == 600
    assert protocol["source_main_issue600"] == "d4128a1b4e021ef998491e45ab2355586ea07b04"
    assert protocol["v26_8_backend_blob"] == "3575c58d1cd730be77649f087908c51dbf3e6088"
    assert protocol["issue558_preserved_as_authoritative_pass"] is True
    assert protocol["issue558_identity_context_followed_compute_dtype"] is True
    assert protocol["issue558_covered_fp32_identity_bf16_similarity_fp32_durable"] is False
    assert protocol["softmax_weight_visibility_controlled_by_compute_dtype"] is False
    assert protocol["softmax_weight_visibility_controlled_by_identity_dtype"] is True
    assert protocol["mixed_strength_precision_semantics_changed_by_v26_9"] is False
    assert protocol["write_backend_changed_by_v26_9"] is False
    assert protocol["training_backend_changed_by_v26_9"] is False
    for key in (
        "gpu_authorized_by_issue600",
        "mixed_dtype_read_gpu_gate_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "independent_replication_credit",
        "100m_authorized",
        "breakthrough_proven",
        "scientific_seed_consumed",
    ):
        assert protocol[key] is False


def test_issue600_cpu_preflight_is_nonexecuting() -> None:
    result = v269.cpu_contract_preflight_issue600()
    assert result["gpu_authorized_by_cpu_preflight"] is False
    assert result["model_construction_performed"] is False
    assert result["checkpoint_loaded"] is False
    assert result["scientific_seed_consumed"] is False
