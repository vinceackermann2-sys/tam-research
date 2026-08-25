from tam_research.aera_hardware_core_v24 import (
    episodic_state_bytes_per_session,
    vectorized_contextual_episodic_protocol,
)


def test_v24_protocol_freeze_metadata_matches_issue347():
    p = vectorized_contextual_episodic_protocol()
    assert p['context_window_previous_events'] == 8
    assert p['capacity_slots_per_stage'] == 48
    assert p['duplicate_similarity_threshold'] == 0.95
    assert p['within_incoming_block_newest_wins'] is True
    assert p['read_top_k'] == 4
    assert p['read_temperature'] == 0.10
    assert p['controlled_selected_writes'] == 2
    assert p['real_language_selected_writes'] == 16
    assert p['sequential_delta_recurrence'] is False
    assert p['inverse_covariance_state'] is False
    assert p['extra_learned_parameters'] == 0
    assert p['gpu_authorized'] is False
    assert episodic_state_bytes_per_session(n_stages=4, memory_dim=50) == 77_760
