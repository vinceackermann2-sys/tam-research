import torch

from tam_research.aera_full import (
    AERATextLM,
    BlockVerifier,
    FullAERAConfig,
    LocalCausalAttention,
    ModalityAdapterBank,
    ReplayRecord,
    SurpriseEventPatcher,
    VerifiedReplayBuffer,
    aera_parameter_accounting,
)


def cfg() -> FullAERAConfig:
    return FullAERAConfig(
        vocab_size=67,
        d_model=32,
        n_stages=2,
        n_heads=4,
        local_window=4,
        max_seq_len=16,
        n_experts=4,
        top_k_experts=1,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=3,
        block_size=3,
    )


def test_local_attention_is_causal():
    torch.manual_seed(1)
    attn = LocalCausalAttention(32, 4, window=4).eval()
    x = torch.randn(2, 8, 32)
    y = attn(x)
    x2 = x.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:])
    y2 = attn(x2)
    torch.testing.assert_close(y[:, :5], y2[:, :5], rtol=1e-5, atol=1e-5)


def test_full_text_model_forward_loss_and_backward():
    torch.manual_seed(2)
    model = AERATextLM(cfg())
    tokens = torch.randint(0, model.cfg.vocab_size, (3, 12))
    out = model(tokens, update_memory=False, return_block_logits=True)
    assert out["logits"].shape == (3, 12, model.cfg.vocab_size)
    assert out["block_logits"].shape == (
        3,
        12,
        model.cfg.block_size,
        model.cfg.vocab_size,
    )
    losses = model.loss(tokens, out)
    assert all(torch.isfinite(v) for v in losses.values())
    losses["total"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_hard_reasoning_respects_selected_budget():
    torch.manual_seed(3)
    model = AERATextLM(cfg()).eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 9))
    model(tokens, hard=True, update_memory=False)
    for stage in model.compute_stats()["stages"]:
        stats = stage["reasoning"]
        assert stats["mode"] == "hard"
        assert 1 <= stats["min"] <= stats["max"] <= model.cfg.max_reason_steps


def test_stream_and_fast_memory_are_explicit_per_session():
    torch.manual_seed(4)
    model = AERATextLM(cfg()).eval()
    history = torch.randint(0, model.cfg.vocab_size, (1, 8))
    query = torch.randint(0, model.cfg.vocab_size, (1, 5))
    learned = model(history, update_memory=True)["state"]
    with_state = model(query, learned, update_memory=False)["logits"]
    fresh_a = model(query, None, update_memory=False)["logits"]
    fresh_b = model(query, None, update_memory=False)["logits"]
    torch.testing.assert_close(fresh_a, fresh_b)
    assert not torch.allclose(with_state, fresh_a)


def test_surprise_patcher_allocates_smaller_events_near_surprise():
    patcher = SurpriseEventPatcher(min_patch=1, max_patch=6, threshold=0.6)
    low = torch.full((24,), 0.1)
    high = low.clone()
    high[2::3] = 0.95
    low_spans = patcher.spans(low)
    high_spans = patcher.spans(high)
    low_mean = sum(b - a for a, b in low_spans) / len(low_spans)
    high_mean = sum(b - a for a, b in high_spans) / len(high_spans)
    assert low_mean > high_mean
    assert sum(b - a for a, b in high_spans) == 24


def test_modality_adapters_share_output_space():
    torch.manual_seed(5)
    bank = ModalityAdapterBank(16, {"text": 7, "image": 11, "action": 3})
    assert bank("text", torch.randn(4, 7)).shape == (4, 16)
    assert bank("image", torch.randn(4, 11)).shape == (4, 16)
    assert bank("action", torch.randn(4, 3)).shape == (4, 16)


def test_verified_replay_latest_verified_wins_and_sessions_isolate():
    replay = VerifiedReplayBuffer()
    replay.add(ReplayRecord("a", 1, 5, True, 1.0, "verifier"))
    replay.add(ReplayRecord("a", 1, 2, False, 10.0, "untrusted"))
    replay.add(ReplayRecord("b", 1, 9, True, 1.0, "verifier"))
    assert replay.current_verified_value("a", 1) == 5
    replay.add(ReplayRecord("a", 1, 7, True, 2.0, "verifier-correction"))
    assert replay.current_verified_value("a", 1) == 7
    assert replay.current_verified_value("b", 1) == 9
    assert all(r.verified for r in replay.prioritized_sample(3, seed=1))


def test_block_verifier_rejects_low_confidence_positions():
    verifier = BlockVerifier(min_confidence=0.8)
    confidence = torch.tensor([[0.99, 0.40, 0.91], [0.85, 0.80, 0.10]])
    accepted = verifier.accept_mask(confidence)
    assert accepted.tolist() == [[True, False, True], [True, True, False]]
    assert verifier.accepted_per_call(confidence) == 2.0


def test_parameter_accounting_reports_stored_and_active():
    model = AERATextLM(cfg())
    stats = aera_parameter_accounting(model)
    assert stats["stored_parameters"] > 0
    assert stats["expert_parameters_stored"] > 0
    assert stats["estimated_active_parameters_per_event"] < stats["stored_parameters"]
    assert 0 < stats["estimated_active_fraction"] < 1
