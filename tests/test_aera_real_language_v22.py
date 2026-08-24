import torch

from tam_research import aera_real_language_v18_gpu as v18gpu
from tam_research import aera_real_language_v22 as v22
from tam_research import aera_real_language_v22_gpu as v22gpu
from tam_research.aera_hardware_core import HardwareAERAConfig
from tam_research.aera_hardware_core_v22 import HardwareAwareAERATextLMV22


def _tiny_model() -> HardwareAwareAERATextLMV22:
    cfg = HardwareAERAConfig(
        vocab_size=97,
        d_model=24,
        n_stages=4,
        n_heads=4,
        chunk_size=v22.CHUNK_SIZE,
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=16,
        max_reason_steps=2,
        block_size=2,
    )
    return HardwareAwareAERATextLMV22(cfg)


def test_v22_real_language_protocol_keeps_frozen_budget_and_thresholds():
    summary = v22gpu.frozen_protocol_summary()
    assert summary["seed"] == 8391
    assert summary["development_only"] is True
    assert summary["memory_dim"] == 50
    assert summary["thresholds_identical_to_v18"] is True
    assert summary["gpu_authorized_by_module"] is False
    assert v22.MAX_MEMORY_AUX_EVENTS == 1024
    assert v22.ADDRESS_TEMPERATURE == 0.10
    assert v22.ADDRESS_CONTRASTIVE_WEIGHT == 1.0
    assert v22.PAYLOAD_TOKEN_WEIGHT == 1.0
    assert v22.LATENT_PAYLOAD_WEIGHT == 0.0
    assert v22gpu.QUALITY_GAP_MAX_NLL == v18gpu.QUALITY_GAP_MAX_NLL == 0.50
    assert (
        v22gpu.MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL
        == v18gpu.MEMORY_SECOND_CHUNK_MIN_ADVANTAGE_NLL
        == 0.005
    )
    assert v22gpu.BATCH8_MIN_SPEED_RATIO == v18gpu.BATCH8_MIN_SPEED_RATIO == 0.25
    assert v22gpu.BATCH64_MIN_SPEED_RATIO == v18gpu.BATCH64_MIN_SPEED_RATIO == 1.25


def test_stratified_memory_aux_sampling_is_deterministic_bounded_and_rotates():
    a = v22._stratified_indices(16_320, step=7, limit=1024, device=torch.device("cpu"))
    b = v22._stratified_indices(16_320, step=7, limit=1024, device=torch.device("cpu"))
    c = v22._stratified_indices(16_320, step=8, limit=1024, device=torch.device("cpu"))
    assert torch.equal(a, b)
    assert a.numel() == 1024
    assert a.unique().numel() == 1024
    assert int(a.min()) >= 0 and int(a.max()) < 16_320
    assert not torch.equal(a, c)


def test_sampled_conflict_free_auxiliary_only_trains_existing_memory_code():
    torch.manual_seed(8390)
    model = _tiny_model().train()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, v22.CHUNK_SIZE))
    model.zero_grad(set_to_none=True)
    terms = v22.memory_auxiliary_terms(model, tokens, step=3, max_events=64)
    total = terms["memory_address_contrastive_loss"] + terms["memory_payload_token_loss"]
    assert torch.isfinite(total)
    assert int(terms["memory_aux_events"].item()) == 64
    total.backward()
    memory = model.stages[0].memory
    for param in (memory.q.weight, memory.k.weight, memory.v.weight, memory.out.weight):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert float(torch.linalg.vector_norm(param.grad.float())) > 0.0
    # Event representations and the tied decoder are explicitly detached for aux.
    assert model.token_emb.weight.grad is None or float(model.token_emb.weight.grad.abs().max()) == 0.0
    assert model.stages[0].norm.weight.grad is None or float(model.stages[0].norm.weight.grad.abs().max()) == 0.0


def test_v22_protocol_decoration_keeps_development_only_claims_and_state_accounting():
    protocol = v22gpu._decorate_protocol({"development_seed": 8391})
    assert protocol["counts_toward_independent_replication"] is False
    assert protocol["architecture"]["memory_dim"] == 50
    assert protocol["architecture"]["inverse_key_covariance_state"] is True
    assert protocol["memory_training_objective"]["max_sampled_adjacent_events_per_step"] == 1024
    assert protocol["v22_specific_safety"]["memory_state_bytes_include_M_and_P"] is True
    assert protocol["v22_specific_safety"]["inverse_key_covariance_symmetry_max_abs"] == 1e-4


def test_v22_full_real_language_cpu_preflight_contract():
    result = v22.cpu_preflight()
    assert result["gpu_authorized"] is False
    assert result["memory_dim"] == 50
    assert result["memory_dim_changed_from_prior_real_language"] is False
    assert result["real_language_memory_dim_preserved_at_50"] is True
    assert result["memory_off_logits_and_stream_bit_exact_v19"] is True
    assert result["initialization"]["inherited_values_bit_exact"] is True
    assert abs(result["parameter_accounting"]["stored_parameter_delta_fraction"]) <= 0.05
    for name in ("q", "k", "v", "out"):
        assert result["memory_auxiliary"]["gradient_norms"][name] > 0.0
