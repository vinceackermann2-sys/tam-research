from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F

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


def integrated_language_and_block_probe_v3() -> dict[str, object]:
    """Strict integrated learnability gate for the canonical AERA text core.

    Unlike the v1/v2 probe, this gate requires absolute quality above chance, not
    merely a relative loss decrease from a pathological initialization. It also
    requires useful verified block drafts and healthy sparse/hard-compute paths.
    """

    seed_all(2007)
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
    model = IntegratedAERATextLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.01)

    def sequences(batch: int, length: int) -> torch.Tensor:
        # Second-order modular recurrence: the next unit is not a unigram lookup.
        a = torch.randint(0, cfg.vocab_size, (batch,))
        b = torch.randint(0, cfg.vocab_size, (batch,))
        seq = [a, b]
        for _ in range(2, length):
            seq.append((seq[-1] + 2 * seq[-2] + 1) % cfg.vocab_size)
        return torch.stack(seq, dim=1)

    fixed_eval = sequences(512, cfg.max_seq_len)
    chance_nll = math.log(cfg.vocab_size)

    with torch.no_grad():
        initial_out = model(fixed_eval, update_memory=False, return_block_logits=True)
        initial = float(model.objective(fixed_eval, initial_out)["next_token"])

    # Synthetic integration smoke only; enough optimization to establish that the
    # composed architecture can learn and that router/block gradients are useful.
    for _ in range(420):
        tok = sequences(64, cfg.max_seq_len)
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
        out = model(fixed_eval, hard=False, update_memory=False, return_block_logits=True)
        losses = model.objective(fixed_eval, out)
        final = float(losses["next_token"])
        logits = out["logits"]
        block_logits = out["block_logits"]
        assert isinstance(logits, torch.Tensor)
        assert isinstance(block_logits, torch.Tensor)

        next_pred = logits[:, :-1].argmax(dim=-1)
        next_target = fixed_eval[:, 1:]
        next_accuracy = float((next_pred == next_target).float().mean())

        valid = fixed_eval.size(1) - cfg.block_size
        all_correct = []
        all_confidence = []
        block_correct = 0
        block_total = 0
        for offset in range(cfg.block_size):
            pred_logits = block_logits[:, :valid, offset]
            prob = F.softmax(pred_logits.float(), dim=-1)
            confidence, pred = prob.max(dim=-1)
            target = fixed_eval[:, offset + 1 : offset + 1 + valid]
            correct = pred == target
            all_correct.append(correct)
            all_confidence.append(confidence)
            block_correct += int(correct.sum())
            block_total += target.numel()
        block_accuracy = block_correct / block_total

        confidence = torch.stack(all_confidence, dim=-1)
        correctness = torch.stack(all_correct, dim=-1)
        verifier = BlockVerifier(min_confidence=0.80)
        accepted = verifier.accept_mask(confidence)
        accepted_units = int(accepted.sum())
        accepted_total = accepted.numel()
        accepted_fraction = accepted_units / max(accepted_total, 1)
        accepted_precision = float(correctness[accepted].float().mean()) if accepted_units else 0.0

        # Exercise the genuinely conditional inference path after training.
        model(fixed_eval[:16], hard=True, update_memory=False)
        stats = model.stats()
        route_stats = stats["stages"][0]["experts"]
        reason_stats = stats["stages"][0]["reasoning"]

    accounting = integrated_parameter_accounting(model)
    loss_drop = (initial - final) / max(initial, 1e-9)

    # Absolute gates prevent chance-level models from passing via relative loss.
    passed = (
        final <= 1.25
        and final <= chance_nll - 1.5
        and next_accuracy >= 0.75
        and block_accuracy >= 0.65
        and accepted_fraction >= 0.10
        and accepted_precision >= 0.90
        and route_stats is not None
        and reason_stats is not None
        and route_stats["active_fraction_of_experts_per_event"] == 0.25
        and reason_stats["max"] <= cfg.max_reason_steps
    )
    if not passed:
        raise AssertionError(
            {
                "initial_nll": initial,
                "final_nll": final,
                "chance_nll": chance_nll,
                "next_accuracy": next_accuracy,
                "block_accuracy": block_accuracy,
                "accepted_fraction": accepted_fraction,
                "accepted_precision": accepted_precision,
                "route": route_stats,
                "reason": reason_stats,
            }
        )

    return {
        "pass": True,
        "initial_next_token_nll": initial,
        "final_next_token_nll": final,
        "chance_nll": chance_nll,
        "relative_nll_drop": loss_drop,
        "next_token_accuracy": next_accuracy,
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
        "scope": "AERA_pre100M_CPU_mechanism_and_integration_validation_v3",
        "learned_sparse_routing": controller_routing_probe(),
        "learned_adaptive_compute": adaptive_budget_probe(),
        "delta_fast_memory_stress": memory_stress_probe(),
        "stream_state_across_chunks": stream_state_probe(),
        "variable_event_patching": event_patching_probe(),
        "multimodal_and_world_transition": multimodal_world_probe_v2(),
        "verified_replay": replay_probe(),
        "retrieval_and_precision_controller": retrieval_precision_controller_probe(),
        "integrated_text_and_block_generation": integrated_language_and_block_probe_v3(),
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
