from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_4_ficem_write_probe as probe
from tam_research import aera_hardware_core_v26_4_ficem_write_triton as write_backend

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py"
V26 = ROOT / "tam_research" / "aera_hardware_core_v26.py"
STABLE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"
READ = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
PROBE = ROOT / "tam_research" / "aera_v26_4_ficem_write_probe.py"
LAUNCHER = ROOT / "modal_aera_v26_4_ficem_write_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-4-ficem-write-l4.yml"

SOURCE_MAIN = "c0ee36ba66e11d24bb9990787e125e986171a46e"
BACKEND_BLOB = "5d703bbba296328ca2f49407e56192d10541349d"
V26_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
READ_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
PROBE_BLOB = "7d8c2c4990beb4c7b4a719d02d009ffefe94671f"
LAUNCHER_BLOB = "663b30c406d676fe9574cbde6eb53f012e87de68"
WORKFLOW_BLOB = "f1caf2c8bca61df8db5aa5695cbbd801a841bdb2"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue488_freezes_all_production_reference_and_harness_blobs() -> None:
    assert _blob(BACKEND) == BACKEND_BLOB
    assert _blob(V26) == V26_BLOB
    assert _blob(STABLE) == STABLE_BLOB
    assert _blob(READ) == READ_BLOB
    assert _blob(PROBE) == PROBE_BLOB
    assert _blob(LAUNCHER) == LAUNCHER_BLOB
    assert _blob(WORKFLOW) == WORKFLOW_BLOB


def test_issue488_frozen_design_geometry_fixtures_and_thresholds() -> None:
    assert probe.DESIGN_SEED == 408_487
    assert (probe.D_MODEL, probe.WRITE_COUNT, probe.CAPACITY, probe.MEMORY_DIM) == (200, 16, 48, 50)
    assert probe.DUPLICATE_THRESHOLD == 0.95
    assert probe.BATCH_SIZES == (8, 64)
    assert probe.DTYPE_NAMES == ("float32", "bfloat16")
    assert probe.VALIDITY_KINDS == ("mixed", "full")
    assert probe.EDGE_FIXTURES == (
        "empty_old_all_new_valid",
        "mixed_incoming_validity",
        "incoming_duplicate_newest_wins",
        "threshold_inclusive_and_below_control",
        "surviving_new_suppresses_old",
        "shadowed_new_does_not_suppress_old",
        "over_capacity_truncation",
        "invalid_retained_storage_order",
    )
    assert (probe.WARMUP_CALLS, probe.TIMED_ROUNDS, probe.CALLS_PER_ROUND) == (10, 5, 100)
    assert (probe.FP32_ATOL, probe.FP32_RTOL) == (1e-5, 1e-5)
    assert (probe.BF16_ATOL, probe.BF16_RTOL) == (1e-2, 1e-2)
    assert probe.MAX_GEOMEAN_LATENCY_RATIO == 0.90
    assert probe.MAX_ROW_LATENCY_RATIO == 1.05
    assert probe.MAX_TAIL_EVENT_RATIO == 0.75


def test_issue488_probe_has_production_projected_and_isolated_tail_oracles() -> None:
    source = PROBE.read_text()
    assert "TorchFICEMReferenceBackend" in source
    assert "TritonFICEMReadWriteBackend" in source
    assert "reference_backend.update_from_projected(" in source
    assert "candidate_backend.update_from_projected(" in source
    assert "memory._stable_compact_state(" in source
    assert "fused_ficem_write_tail(" in source
    assert "projected_correctness" in source
    assert "tail_correctness" in source
    assert "latency_ratio_candidate_over_reference" in source
    assert "tail_cuda_event_ratio_candidate_over_reference" in source
    assert "geomean_latency_ratio_by_dtype" in source
    assert "generator = torch.Generator().manual_seed(DESIGN_SEED)" in source
    for forbidden in ("resample", "rejection", "candidate_ordinal", "eligible_case", "nudge"):
        assert forbidden not in source.lower()


def test_issue488_probe_edge_semantics_are_explicit_before_gpu() -> None:
    source = PROBE.read_text()
    assert 'x.incoming_similarity[:, 0, 1] = DUPLICATE_THRESHOLD' in source
    assert 'x.incoming_similarity[:, 2, 3] = 0.949' in source
    assert 'x.old_similarity[:, 7, 3] = 1.0' in source
    assert 'x.old_similarity[:, 0, 4] = 1.0' in source
    assert "surviving_new = inputs.new_valid & ~shadowed" in source
    assert "keep_old = inputs.state.valid & ~duplicate_old" in source
    assert "inputs.new_keys.flip(1)" in source
    assert "inputs.new_valid.flip(1)" in source
    assert "valid_exact" in source
    assert "source_unchanged" in source
    assert "dtype_device_shape_exact" in source
    assert "finite" in source


def test_issue488_candidate_topology_and_shared_pytorch_boundary_are_frozen() -> None:
    backend_source = BACKEND.read_text()
    assert backend_source.count("@triton.jit") == 2
    assert "def _write_adjudicate_map_kernel(" in backend_source
    assert "def _write_materialize_kernel(" in backend_source
    assert "new_values = torch.tanh(memory.v(payload))" in backend_source
    assert 'torch.einsum("bkd,bjd->bkj", new_keys, new_keys)' in backend_source
    assert 'torch.einsum("bkd,bsd->bks", new_keys, normalized_old)' in backend_source
    tail = backend_source[backend_source.index("def fused_ficem_write_tail(") : backend_source.index("class TritonFICEMReadWriteBackend")]
    for forbidden in ("torch.cat", "torch.stack", "torch.cumsum", "scatter_add", "torch.einsum", "for ", "while "):
        assert forbidden not in tail

    probe_source = PROBE.read_text()
    assert '"_write_adjudicate_map_kernel"' in probe_source
    assert '"_write_materialize_kernel"' in probe_source
    for op in ("aten::cat", "aten::stack", "aten::cumsum", "aten::scatter_add"):
        assert op in probe_source


def test_issue488_launcher_is_duplicate_safe_one_l4_one_probe_and_persists_first() -> None:
    source = LAUNCHER.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-4-issue488-ficem-write"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue488-ficem-write/result.json"' in source
    assert "MAX_GPU_SECONDS = 300" in source
    assert source.count('gpu="L4"') == 1
    assert source.count("result = run_ficem_write_probe()") == 1
    assert source.count("result_path.exists()") >= 2
    assert "refusing duplicate issue488 WRITE run because result exists" in source
    write = source.index("result_path.write_text(durable_json)")
    commit = source.index("volume.commit()")
    marker = source.index("AERA_V26_ISSUE488_FICEM_WRITE_RESULT_JSON=")
    summary = source.index("AERA_V26_ISSUE488_FICEM_WRITE_SUMMARY_JSON=")
    assert write < commit < marker < summary
    assert SOURCE_MAIN in source
    assert BACKEND_BLOB in source
    assert PROBE_BLOB in source
    assert "33618950619" in source
    assert "33620850681" in source


def test_issue488_workflow_is_unique_attempt1_bound_and_has_no_retry_path() -> None:
    source = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "[aera-v26-4-ficem-write-l4]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert 'startswith("🔬 **AERA-v26.4 #488 FICEM WRITE result**")' in source
    assert 'test "${report_count}" = "0"' in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert source.count("modal run modal_aera_v26_4_ficem_write_app.py") == 1
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source
    assert "automatic retry" in source
    assert "redispatch" in source
    assert "alternate trigger" in source
    assert "timeout increase" in source
    for blob in (BACKEND_BLOB, V26_BLOB, STABLE_BLOB, READ_BLOB, PROBE_BLOB, LAUNCHER_BLOB):
        assert blob in source


def test_issue488_no_science_or_higher_authorization() -> None:
    protocol = probe.issue488_protocol()
    assert protocol["design_seed_is_scientific_seed"] is False
    for key in (
        "model_loaded", "checkpoint_loaded", "corpus_accessed", "training_performed",
        "optimizer_created", "backward_performed", "scientific_seed_consumed",
        "end_to_end_systems_authorized", "architecture_freeze_authorized", "s2_authorized",
        "fresh_scientific_seed_authorized", "100m_authorized", "breakthrough_proven",
    ):
        assert protocol[key] is False
    source = (PROBE.read_text() + LAUNCHER.read_text()).lower()
    for forbidden in ("torch.optim", ".backward(", "seed8471", "checkpoint.pt", "transformer.pt"):
        assert forbidden not in source
