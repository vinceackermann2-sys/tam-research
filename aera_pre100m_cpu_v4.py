from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from aera_integration_diagnostic import DenseDiagnosticLM
from aera_pre100m_cpu import (
    adaptive_budget_probe,
    controller_routing_probe,
    event_patching_probe,
    memory_stress_probe,
    replay_probe,
    retrieval_precision_controller_probe,
    seed_all,
    stream_state_probe,
)
from aera_pre100m_cpu_v2 import multimodal_world_probe_v2
from tam_research.aera_full import BlockVerifier, FullAERAConfig
from tam_research.aera_integrated import (
    IntegratedAERATextLM,
    integrated_parameter_accounting,
)


def pair_copy_sequences(batch: int, length: int, vocab: int) -> torch.Tensor:
    """Context-dependent alternating pair task: a,b,a,b,...

    For t>=1 the next token is the previous token, so the current token alone is
    insufficient when b is independently sampled. This cheaply tests use of recent
    context/state without requiring tiny models to learn modular arithmetic.
    """

    a = torch.randint(0, vocab, (batch,))
    b = torch.randint(0, vocab, (batch,))
    seq = [a, b]
    for t in range(2, length):
        seq.append(seq[t - 2])
    return torch.stack(seq, dim=1)


def conditional_metrics(logits: torch.Tensor, tokens: torch.Tensor) -> tuple[float, float]:
    # Exclude the first a->b transition: b is independently sampled and therefore
    # intentionally unpredictable. From position 1 onward the task is deterministic.
    pred_logits = logits[:, 1:-1]
    target = tokens[:, 2:]
    nll = float(
        F.cross_entropy(
            pred_logits.float().reshape(-1, logits.size(-1)), target.reshape(-1)
        )
    )
    accuracy = float((pred_logits.argmax(dim=-1) == target).float().mean())
    return nll, accuracy


def dense_task_validity_gate(cfg: FullAERAConfig) -> dict[str, float | bool]:
    """The architecture gate is invalid unless a simple dense causal control solves it."""

    seed_all(5001)
    model = DenseDiagnosticLM(
        vocab=cfg.vocab_size,
        d_model=cfg.d_model,
        heads=cfg.n_heads,
        length=cfg.max_seq_len,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.01)
    fixed = pair_copy_sequences(512, cfg.max_seq_len, cfg.vocab_size)
    for _ in range(220):
        tok = pair_copy_sequences(64, cfg.max_seq_len, cfg.vocab_size)
        logits = model(tok)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size), tok[:, 1:].reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        nll, accuracy = conditional_metrics(model(fixed), fixed)
    passed = nll <= 0.20 and accuracy >= 0.95
    if not passed:
        raise AssertionError({"invalid_smoke_task_dense_control": {"nll": nll, "accuracy": accuracy}})
    return {"pass": True, "context_conditioned_nll": nll, "context_conditioned_accuracy": accuracy}


def integrated_context_and_block_probe_v4() -> dict[str, object]:
    seed_all(5002)
    cfg = FullAERAConfig(
        vocab_size=31,
        d_model=32,
        n_stages=1,
        n_heads=4,
        local_window=8,
        max_seq_len=20,
        n_experts=4,
        top_k_experts=1,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=3,
    )

    dense_gate = dense_task_validity_gate(cfg)
    model = IntegratedAERATextLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.01)
    fixed = pair_copy_sequences(512, cfg.max_seq_len, cfg.vocab_size)

    with torch.no_grad():
        initial_out = model(fixed, update_memory=False, return_block_logits=True)
        initial_logits = initial_out["logits"]
        assert isinstance(initial_logits, torch.Tensor)
        initial_nll, initial_accuracy = conditional_metrics(initial_logits, fixed)

    for _ in range(320):
        tok = pair_copy_sequences(64, cfg.max_seq_len, cfg.vocab_size)
        out = model(tok, update_memory=False, return_block_logits=True)
        losses = model.objective(
            tok,
            out,
            event_weight=0.03,
            compute_weight=0.001,
            balance_weight=0.01,
            block_weight=0.35,
        )
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        out = model(fixed, hard=False, update_memory=False, return_block_logits=True)
        logits = out["logits"]
        block_logits = out["block_logits"]
        assert isinstance(logits, torch.Tensor)
        assert isinstance(block_logits, torch.Tensor)
        final_nll, final_accuracy = conditional_metrics(logits, fixed)

        # Evaluate block drafts only from positions with both members of the pair in
        # causal history, so every target in the block is actually predictable.
        valid = fixed.size(1) - cfg.block_size
        all_correct = []
        all_confidence = []
        correct_units = 0
        total_units = 0
        for offset in range(cfg.block_size):
            pred_logits = block_logits[:, 1:valid, offset]
            prob = F.softmax(pred_logits.float(), dim=-1)
            confidence, pred = prob.max(dim=-1)
            target = fixed[:, offset + 2 : offset + 1 + valid]
            correct = pred == target
            all_correct.append(correct)
            all_confidence.append(confidence)
            correct_units += int(correct.sum())
            total_units += target.numel()
        block_accuracy = correct_units / total_units
        confidence = torch.stack(all_confidence, dim=-1)
        correctness = torch.stack(all_correct, dim=-1)
        verifier = BlockVerifier(min_confidence=0.80)
        accepted = verifier.accept_mask(confidence)
        accepted_units = int(accepted.sum())
        accepted_fraction = accepted_units / max(accepted.numel(), 1)
        accepted_precision = (
            float(correctness[accepted].float().mean()) if accepted_units else 0.0
        )

        # Exercise true conditional inference after training.
        model(fixed[:16], hard=True, update_memory=False)
        stats = model.stats()
        route_stats = stats["stages"][0]["experts"]
        reason_stats = stats["stages"][0]["reasoning"]

    accounting = integrated_parameter_accounting(model)
    passed = (
        final_nll <= 0.20
        and final_accuracy >= 0.95
        and block_accuracy >= 0.90
        and accepted_fraction >= 0.10
        and accepted_precision >= 0.95
        and route_stats is not None
        and reason_stats is not None
        and route_stats["active_fraction_of_experts_per_event"] == 0.25
        and reason_stats["max"] <= cfg.max_reason_steps
    )
    if not passed:
        raise AssertionError(
            {
                "dense_task_gate": dense_gate,
                "initial_context_nll": initial_nll,
                "initial_context_accuracy": initial_accuracy,
                "final_context_nll": final_nll,
                "final_context_accuracy": final_accuracy,
                "block_accuracy": block_accuracy,
                "accepted_fraction": accepted_fraction,
                "accepted_precision": accepted_precision,
                "route": route_stats,
                "reason": reason_stats,
            }
        )

    return {
        "pass": True,
        "dense_task_validity_gate": dense_gate,
        "initial_context_conditioned_nll": initial_nll,
        "initial_context_conditioned_accuracy": initial_accuracy,
        "final_context_conditioned_nll": final_nll,
        "final_context_conditioned_accuracy": final_accuracy,
        "block_accuracy": block_accuracy,
        "verified_accepted_fraction": accepted_fraction,
        "verified_accepted_precision": accepted_precision,
        "expert_routing": route_stats,
        "reasoning": reason_stats,
        "parameter_accounting": accounting,
    }


def main() -> None:
    results = {
        "status": "pass",
        "scope": "AERA_pre100M_CPU_mechanism_and_integration_validation_v4",
        "learned_sparse_routing": controller_routing_probe(),
        "learned_adaptive_compute": adaptive_budget_probe(),
        "delta_fast_memory_stress": memory_stress_probe(),
        "stream_state_across_chunks": stream_state_probe(),
        "variable_event_patching": event_patching_probe(),
        "multimodal_and_world_transition": multimodal_world_probe_v2(),
        "verified_replay": replay_probe(),
        "retrieval_and_precision_controller": retrieval_precision_controller_probe(),
        "integrated_context_and_block_generation": integrated_context_and_block_probe_v4(),
        "claims": {
            "100m_authorized": False,
            "gpu_sparse_speedup_proven": False,
            "real_language_superiority_proven": False,
            "dynamic_precision_hardware_savings_proven": False,
            "full_multimodal_scale_proven": False,
            "breakthrough_proven": False,
        },
    }
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
