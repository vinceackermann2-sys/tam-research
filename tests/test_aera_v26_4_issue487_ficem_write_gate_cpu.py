from __future__ import annotations

import hashlib
from pathlib import Path

from tam_research import aera_v26_4_ficem_write_probe as probe
from tam_research.aera_hardware_core_v26_4_ficem_write_triton import (
    fused_ficem_read_write_v26_4_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
WRITE_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_4_ficem_write_triton.py"
READ_BACKEND = ROOT / "tam_research" / "aera_hardware_core_v26_3_ficem_read_triton.py"
V26_INTERFACE = ROOT / "tam_research" / "aera_hardware_core_v26.py"
STABLE_REFERENCE = ROOT / "tam_research" / "aera_hardware_core_v25_1_compact.py"
PROBE = ROOT / "tam_research" / "aera_v26_4_ficem_write_probe.py"
LAUNCHER = ROOT / "modal_aera_v26_4_ficem_write_app.py"
WORKFLOW = ROOT / ".github" / "workflows" / "aera-v26-4-ficem-write-l4.yml"

SOURCE_MAIN = "c0ee36ba66e11d24bb9990787e125e986171a46e"
WRITE_BACKEND_BLOB = "5d703bbba296328ca2f49407e56192d10541349d"
READ_BACKEND_BLOB = "263f68eb1186a8ac14a08fc4b4df1fc5b292c711"
V26_INTERFACE_BLOB = "268644ac4edee15a4cc4e29d3fed7f61eeb3caa7"
STABLE_REFERENCE_BLOB = "4e336b6e1a6238dac782fa320751d68281493ee1"
PROBE_BLOB = "8497bb6e5f077c5cfd190aa7b95d1632b2e4fa1a"
LAUNCHER_BLOB = "6cdc191286594800db8160fc3ce073eeb615f9f0"
WORKFLOW_BLOB = "c8440a34ddd9408696d85cc6b34adcdecb38b26a"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_issue487_freezes_candidate_and_all_semantic_reference_blobs() -> None:
    assert _blob(WRITE_BACKEND) == WRITE_BACKEND_BLOB
    assert _blob(READ_BACKEND) == READ_BACKEND_BLOB
    assert _blob(V26_INTERFACE) == V26_INTERFACE_BLOB
    assert _blob(STABLE_REFERENCE) == STABLE_REFERENCE_BLOB
    assert _blob(PROBE) == PROBE_BLOB
    assert _blob(LAUNCHER) == LAUNCHER_BLOB
    assert _blob(WORKFLOW) == WORKFLOW_BLOB

    protocol = fused_ficem_read_write_v26_4_protocol()
    assert protocol["write_count"] == 16
    assert protocol["capacity"] == 48
    assert protocol["memory_dim"] == 50
    assert protocol["duplicate_similarity"] == 0.95
    assert protocol["write_tail_triton_launches_target"] == 2
    for key in (
        "read_backend_changed_by_v26_4",
        "write_similarity_einsums_changed",
        "write_value_projection_changed",
        "write_strength_semantics_changed",
        "write_duplicate_semantics_changed",
        "write_incoming_order_changed",
        "write_stable_compaction_semantics_changed",
        "write_invalid_storage_semantics_changed",
        "write_training_backend_changed",
        "write_persistent_state_changed",
        "write_persistent_cache",
        "write_gpu_gate_authorized",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert protocol[key] is False


def test_issue487_probe_freezes_geometry_fixtures_and_pass_thresholds() -> None:
    contract = probe.cpu_contract_preflight()["protocol"]
    assert probe.DESIGN_SEED == 487_485
    assert probe.DESIGN_SEED == contract["design_seed"]
    assert contract["design_seed_is_scientific_seed"] is False
    assert (probe.D_MODEL, probe.MEMORY_DIM, probe.CAPACITY, probe.WRITE_K) == (
        200,
        50,
        48,
        16,
    )
    assert probe.BATCH_SIZES == (8, 64)
    assert probe.DTYPE_NAMES == ("float32", "bfloat16")
    assert probe.VALIDITY_KINDS == ("mixed", "full")
    assert probe.STRESS_KINDS == (
        "empty_old",
        "incoming_duplicate_heavy",
        "old_duplicate_heavy",
        "combined_duplicates",
        "near_threshold",
    )
    assert (probe.WARMUP_CALLS, probe.TIMED_ROUNDS, probe.CALLS_PER_ROUND) == (
        10,
        5,
        100,
    )
    assert probe.MAX_GEOMEAN_LATENCY_RATIO == 0.90
    assert probe.MAX_ROW_LATENCY_RATIO == 1.05
    assert probe.MAX_TAIL_EVENT_RATIO == 0.25
    assert (probe.NEAR_THRESHOLD_LOW, probe.NEAR_THRESHOLD_HIGH) == (0.94, 0.96)
    assert contract["full_call_is_update_from_projected"] is True
    assert contract["bit_exact_complete_durable_state_required"] is True
    assert contract["invalid_retained_storage_bit_exact_required"] is True


def test_issue487_probe_requires_exact_state_and_audits_production_adjudication_map() -> None:
    source = PROBE.read_text()
    for field in (
        '"keys": torch.equal(ref.keys, cand.keys)',
        '"values": torch.equal(ref.values, cand.values)',
        '"strengths": torch.equal(ref.strengths, cand.strengths)',
        '"valid": torch.equal(ref.valid, cand.valid)',
        '"candidate_source_map_exact": map_exact',
        '"source_unchanged": source_unchanged',
        '"dtype_device_shape_exact": meta_exact',
    ):
        assert field in source
    assert "_write_adjudicate_map_kernel" in source
    assert "_reference_source_map" in source
    assert "old_valid.zero_()" in source
    assert "NEAR_THRESHOLD_LOW" in source and "NEAR_THRESHOLD_HIGH" in source
    assert "projected[:, 1:4] = projected[:, 0:1]" in source
    assert "old_keys[:, :16] = projected[:, :16]" in source
    assert "write_strength[:, 2] = 0" in source
    assert "write_strength[:, 10] = 0" in source
    assert "_ordinary_update_correctness" in source


def test_issue487_timing_is_full_projected_write_and_tail_profile_is_separate() -> None:
    source = PROBE.read_text()
    assert "backend.update_from_projected(" in source
    assert "_timed_summaries(calls)" in source
    assert "reference_tail = lambda" in source
    assert "candidate_tail = lambda" in source
    assert "_reference_tail_from_precomputed" in source
    assert "_candidate_tail_from_precomputed" in source
    assert "fused_ficem_write_tail(" in source
    assert '"isolated_tail_event_ratio_candidate_over_reference"' in source
    assert '"candidate_tail_kernel_exact"' in source
    assert '"candidate_tail_ops_clean"' in source
    assert 'tail_profiles["candidate"]["triton_adjudicate_events"] == 1' in source
    assert 'tail_profiles["candidate"]["triton_materialize_events"] == 1' in source
    assert 'for token in ("cumsum", "scatter_add", "cat", "stack", "topk")' in source
    assert "MAX_TAIL_EVENT_RATIO" in source
    assert "MAX_GEOMEAN_LATENCY_RATIO" in source
    assert "MAX_ROW_LATENCY_RATIO" in source


def test_issue487_probe_has_no_scientific_training_or_checkpoint_path() -> None:
    source = PROBE.read_text()
    for forbidden in (
        "torch.load(",
        "load_state_dict(",
        "torch.optim",
        ".backward(",
        "optimizer.step(",
        "save_checkpoint",
        "seed8471",
        "aera.pt",
        "transformer.pt",
    ):
        assert forbidden not in source
    contract = probe.issue487_protocol()
    for key in (
        "model_loaded",
        "checkpoint_loaded",
        "corpus_accessed",
        "training_performed",
        "optimizer_created",
        "backward_performed",
        "scientific_seed_consumed",
        "end_to_end_systems_authorized",
        "architecture_freeze_authorized",
        "s2_authorized",
        "fresh_scientific_seed_authorized",
        "100m_authorized",
        "breakthrough_proven",
    ):
        assert contract[key] is False


def test_issue487_launcher_is_duplicate_safe_one_l4_one_probe_and_persists_first() -> None:
    source = LAUNCHER.read_text()
    assert 'APP_NAME = "tam-research-aera-v26-4-issue487-ficem-write"' in source
    assert 'VOLUME_NAME = "tam-research-data"' in source
    assert 'RESULT_PATH = "/vol/aera-v26/issue487-ficem-write/result.json"' in source
    assert "MAX_GPU_SECONDS = 300" in source
    assert source.count('gpu="L4"') == 1
    assert source.count("def run_probe()") == 1
    assert source.count("result_path.exists()") >= 2
    assert "refusing duplicate issue487 FICEM write run because result exists" in source
    assert source.count("result = run_ficem_write_probe()") == 1
    assert "run_ficem_write_probe(" not in source.replace(
        "result = run_ficem_write_probe()", ""
    )
    write = source.index("result_path.write_text(durable_json)")
    commit = source.index("volume.commit()")
    digest = source.index("result_sha256 = hashlib.sha256")
    marker = source.index("AERA_V26_ISSUE487_FICEM_WRITE_RESULT_JSON=")
    summary = source.index("AERA_V26_ISSUE487_FICEM_WRITE_SUMMARY_JSON=")
    assert write < commit < digest < marker < summary
    assert SOURCE_MAIN in source
    assert WRITE_BACKEND_BLOB in source
    assert PROBE_BLOB in source


def test_issue487_workflow_is_bound_attempt1_single_invocation_and_no_retry() -> None:
    source = WORKFLOW.read_text()
    assert "issues:\n    types: [opened]" in source
    assert "workflow_dispatch" not in source
    assert "[aera-v26-4-ficem-write-l4]" in source
    assert 'test "${GITHUB_RUN_ATTEMPT}" = "1"' in source
    assert 'test "${trigger_count}" = "1"' in source
    assert 'test "${report_count}" = "0"' in source
    assert 'startswith("🔬 **AERA-v26.4 #487 FICEM write result**")' in source
    assert "Bind main:" in source
    assert 'test "$(git rev-parse HEAD)" = "${bound_main}"' in source
    assert SOURCE_MAIN in source
    assert "33620850681" in source
    assert "100217278171" in source
    assert "33618950619" in source
    assert "100211244996" in source
    assert WRITE_BACKEND_BLOB in source
    assert READ_BACKEND_BLOB in source
    assert V26_INTERFACE_BLOB in source
    assert STABLE_REFERENCE_BLOB in source
    assert PROBE_BLOB in source
    assert LAUNCHER_BLOB in source
    assert source.count("modal run modal_aera_v26_4_ficem_write_app.py") == 1
    assert "AERA_V26_ISSUE487_FICEM_WRITE_RESULT_JSON=" in source
    assert "AERA_V26_ISSUE487_FICEM_WRITE_SUMMARY_JSON=" in source
    assert "gh run rerun" not in source
    assert "modal deploy" not in source
    assert "cancel-in-progress: false" in source
    assert "automatic retry" in source
    assert "redispatch" in source
    assert "alternate trigger" in source
    assert "timeout increase" in source


def test_issue487_workflow_permissions_are_narrow_and_reporting_best_effort() -> None:
    source = WORKFLOW.read_text()
    permissions = source.split("permissions:\n", 1)[1].split("\n\nconcurrency:", 1)[0]
    assert "actions: read" in permissions
    assert "contents: read" in permissions
    assert "issues: write" in permissions
    assert "pull-requests: read" in permissions
    assert "actions: write" not in permissions
    assert "contents: write" not in permissions
    assert "pull-requests: write" not in permissions
    assert source.count("continue-on-error: true") == 3
    assert "Durable Modal result + authoritative marker are the experiment record" in source
    assert "No end-to-end/freeze/S2/scientific-seed/100M/breakthrough authorization" in source
