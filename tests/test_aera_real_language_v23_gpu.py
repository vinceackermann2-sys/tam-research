import torch

from tam_research import aera_real_language_v18_gpu as v18gpu
from tam_research import aera_real_language_v23_efficiency as v23
from tam_research import aera_real_language_v23_gpu as v23gpu


def test_v23_real_language_development_protocol_is_frozen_to_seed8461():
    summary = v23gpu.frozen_protocol_summary()
    assert summary["research_issue"] == 338
    assert summary["seed"] == 8461
    assert summary["development_only"] is True
    assert summary["memory_dim"] == 50
    assert summary["chunk_size"] == 256
    assert summary["candidates_per_chunk"] == 255
    assert summary["selected_writes_per_chunk"] == 16
    assert summary["memory_aux_events_per_microbatch"] == 256
    assert summary["memory_aux_events_per_optimizer_step"] == 1024
    assert summary["thresholds_identical_to_issue_324_v18"] is True
    assert summary["gpu_authorized_by_module"] is False
    assert summary["100m_authorized"] is False

    assert v23gpu.QUALITY_GAP_MAX_NLL == v18gpu.QUALITY_GAP_MAX_NLL == 0.50
    assert (
        v23gpu.MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL
        == v18gpu.MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL
        == 0.005
    )
    assert v23gpu.BATCH8_MIN_SPEED_RATIO == v18gpu.BATCH8_MIN_SPEED_RATIO == 0.25
    assert v23gpu.BATCH64_MIN_SPEED_RATIO == v18gpu.BATCH64_MIN_SPEED_RATIO == 1.25


def test_v23_protocol_decoration_preserves_sparse_science_and_aux_budget():
    protocol = v23gpu._decorate_protocol({"development_seed": 8461})
    architecture = protocol["architecture"]
    objective = protocol["memory_training_objective"]
    safety = protocol["v23_specific_safety"]

    assert protocol["counts_toward_independent_replication"] is False
    assert architecture["memory_dim"] == 50
    assert architecture["event_pair_candidates_per_chunk"] == 255
    assert architecture["physically_selected_writes_per_chunk"] == 16
    assert architecture["dual_delta_equations_changed_from_v22"] is False
    assert architecture["new_learned_parameters_vs_v22"] == 0
    assert objective["address_temperature"] == 0.10
    assert objective["address_contrastive_weight"] == 1.0
    assert objective["payload_token_weight"] == 1.0
    assert objective["latent_payload_weight"] == 0.0
    assert objective["max_sampled_adjacent_events_per_microbatch"] == 256
    assert objective["gradient_accumulation_microbatches"] == 4
    assert objective["max_sampled_adjacent_events_per_optimizer_step"] == 1024
    assert safety["exact_sparse_write_geometry_required"] == [16, 255]
    assert safety["memory_state_bytes_include_M_and_P"] is True
    assert safety["inverse_key_covariance_symmetry_max_abs"] == 1e-4


def test_v23_development_pass_still_cannot_claim_freeze_replication_or_scale(tmp_path):
    result = {
        "protocol": {"development_seed": 8461},
        "v18_memory_eval": {
            "inverse_key_covariance": {
                "all_finite": True,
                "max_symmetry_error": 0.0,
            },
            "session_isolation_exact": True,
            "memory_state_bytes_per_session": 200,
            "matrix_state_bytes_per_session": 100,
            "inverse_covariance_state_bytes_per_session": 100,
            "sparse_write_execution": {
                "all_completed_measured_stages_exact_16_of_255": True,
            },
        },
        "v18_heldout_adaptivity": {},
        "v18_systems_eval": {},
        "v18_development_checks": {"inherited_gate": True},
        "v18_development_pass": True,
    }
    finalized = v23gpu._remap_and_finalize(result, str(tmp_path))
    assert finalized["v23_development_pass"] is True
    assert finalized["claims"]["real_language_memory_advantage_proven_in_development"] is True
    assert finalized["claims"]["architecture_freeze_boundary_may_be_preregistered"] is True
    assert finalized["claims"]["architecture_frozen"] is False
    assert finalized["claims"]["counts_toward_independent_replication"] is False
    assert finalized["claims"]["independent_replication_complete"] is False
    assert finalized["claims"]["100m_authorized"] is False
    assert finalized["claims"]["breakthrough_proven"] is False


def test_v23_sparse_reporting_stats_are_deterministic():
    values = [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])]
    first = v23gpu._tensor_stats(values)
    second = v23gpu._tensor_stats(values)
    assert first == second
    assert first["count"] == 4.0
    assert abs(first["mean"] - 0.25) < 1e-6


def test_v23_training_objective_keeps_corrected_microbatch_cap():
    assert v23.MAX_MEMORY_AUX_EVENTS_PER_MICROBATCH == 256
    assert v23.MAX_MEMORY_AUX_EVENTS_PER_OPTIMIZER_STEP == 1024
