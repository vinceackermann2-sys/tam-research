from __future__ import annotations

import json
import random

import torch
import torch.nn.functional as F

from aera_integration_diagnostic import DenseDiagnosticLM
from tam_research.aera_full import BlockVerifier
from tam_research.aera_hardware_core import (
    ChunkController,
    HardwareAERAConfig,
    HardwareAwareAERATextLM,
    hardware_parameter_accounting,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pair_copy_sequences(batch: int, length: int, vocab: int) -> torch.Tensor:
    a = torch.randint(0, vocab, (batch,))
    b = torch.randint(0, vocab, (batch,))
    seq = [a, b]
    for t in range(2, length):
        seq.append(seq[t - 2])
    return torch.stack(seq, dim=1)


def conditional_metrics(logits: torch.Tensor, tokens: torch.Tensor) -> tuple[float, float]:
    pred = logits[:, 1:-1]
    target = tokens[:, 2:]
    nll = float(F.cross_entropy(pred.float().reshape(-1, logits.size(-1)), target.reshape(-1)))
    acc = float((pred.argmax(dim=-1) == target).float().mean())
    return nll, acc


def boundary_second_token_accuracy(logits: torch.Tensor, tokens: torch.Tensor, chunk: int) -> float:
    positions = [p for p in range(chunk, tokens.size(1) - 1, chunk)]
    pred = torch.stack([logits[:, p].argmax(dim=-1) for p in positions], dim=1)
    target = torch.stack([tokens[:, p + 1] for p in positions], dim=1)
    return float((pred == target).float().mean())


def dense_validity_gate(vocab: int, d_model: int, heads: int, length: int) -> dict[str, float | bool]:
    seed_all(7101)
    model = DenseDiagnosticLM(vocab=vocab, d_model=d_model, heads=heads, length=length)
    opt = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.01)
    fixed = pair_copy_sequences(512, length, vocab)
    for _ in range(220):
        tok = pair_copy_sequences(64, length, vocab)
        logits = model(tok)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), tok[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        nll, acc = conditional_metrics(model(fixed), fixed)
    passed = nll <= 0.20 and acc >= 0.95
    if not passed:
        raise AssertionError({"dense_task_invalid": {"nll": nll, "accuracy": acc}})
    return {"pass": True, "context_conditioned_nll": nll, "context_conditioned_accuracy": acc}


def controller_policy_probe() -> dict[str, object]:
    """Verify the new chunk controller can jointly learn expert/count/depth decisions."""
    seed_all(7102)
    cfg = HardwareAERAConfig(d_model=16, n_stages=1, n_heads=4, chunk_size=7, n_experts=4, max_reason_steps=3)
    controller = ChunkController(cfg)
    opt = torch.optim.AdamW(controller.parameters(), lr=0.02)

    def make(n: int):
        event = torch.randn(n, cfg.d_model)
        stream = torch.randn(n, cfg.d_model)
        expert = ((event[:, 0] > 0).long() + 2 * (event[:, 1] > 0).long()).clamp_max(cfg.n_experts - 1)
        count = (event[:, 2] + stream[:, 0] > 0).long()  # 0=>top1, 1=>top2
        depth = torch.bucketize(event[:, 3] + stream[:, 1], torch.tensor([-0.5, 0.5])).long()
        return event, stream, expert, count, depth

    for _ in range(220):
        event, stream, expert, count, depth = make(256)
        out = controller(event, stream)
        loss = (
            F.cross_entropy(out["expert_logits"], expert)
            + F.cross_entropy(out["expert_count_logits"], count)
            + F.cross_entropy(out["depth_logits"], depth)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    event, stream, expert, count, depth = make(4096)
    with torch.no_grad():
        out = controller(event, stream)
        expert_acc = float((out["expert_logits"].argmax(-1) == expert).float().mean())
        count_acc = float((out["expert_count_logits"].argmax(-1) == count).float().mean())
        depth_acc = float((out["depth_logits"].argmax(-1) == depth).float().mean())
    passed = min(expert_acc, count_acc, depth_acc) >= 0.95
    if not passed:
        raise AssertionError({"controller_policy": [expert_acc, count_acc, depth_acc]})
    return {
        "pass": True,
        "expert_accuracy": expert_acc,
        "expert_count_accuracy": count_acc,
        "depth_accuracy": depth_acc,
    }


def hardware_language_probe() -> dict[str, object]:
    seed_all(7103)
    cfg = HardwareAERAConfig(
        vocab_size=31,
        d_model=32,
        n_stages=1,
        n_heads=4,
        chunk_size=7,  # odd length forces pair phase to cross chunk boundaries
        n_experts=4,
        max_active_experts=2,
        expert_mult=2,
        memory_dim=8,
        max_reason_steps=2,
        block_size=3,
    )
    length = 28
    dense_gate = dense_validity_gate(cfg.vocab_size, cfg.d_model, cfg.n_heads, length)
    model = HardwareAwareAERATextLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.01)
    fixed = pair_copy_sequences(512, length, cfg.vocab_size)

    with torch.no_grad():
        initial = model(fixed, hard=False, update_memory=False, return_block_logits=True)
        initial_nll, initial_acc = conditional_metrics(initial["logits"], fixed)

    for _ in range(380):
        tok = pair_copy_sequences(64, length, cfg.vocab_size)
        out = model(tok, hard=False, update_memory=False, return_block_logits=True)
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
        final_nll, final_acc = conditional_metrics(logits, fixed)
        carried_boundary_acc = boundary_second_token_accuracy(logits, fixed, cfg.chunk_size)

        # Reset state independently for every chunk. The first token of each later
        # odd-length chunk cannot recover its paired partner from local context alone.
        reset_logits = []
        for start in range(0, length, cfg.chunk_size):
            chunk = fixed[:, start : start + cfg.chunk_size]
            reset_logits.append(model(chunk, hard=False, update_memory=False)["logits"])
        reset_logits = torch.cat(reset_logits, dim=1)
        reset_boundary_acc = boundary_second_token_accuracy(reset_logits, fixed, cfg.chunk_size)

        valid = fixed.size(1) - cfg.block_size
        correct_parts = []
        confidence_parts = []
        for offset in range(cfg.block_size):
            pred_logits = block_logits[:, 1:valid, offset]
            prob = F.softmax(pred_logits.float(), dim=-1)
            confidence, pred = prob.max(dim=-1)
            target = fixed[:, offset + 2 : offset + 1 + valid]
            correct_parts.append(pred == target)
            confidence_parts.append(confidence)
        correctness = torch.stack(correct_parts, dim=-1)
        confidence = torch.stack(confidence_parts, dim=-1)
        block_accuracy = float(correctness.float().mean())
        verifier = BlockVerifier(min_confidence=0.80)
        accepted = verifier.accept_mask(confidence)
        accepted_n = int(accepted.sum())
        accepted_fraction = accepted_n / accepted.numel()
        accepted_precision = float(correctness[accepted].float().mean()) if accepted_n else 0.0

        # Exercise hard conditional inference after learning.
        model(fixed[:16], hard=True, update_memory=False)
        stats = model.stats()
        expert_stats = stats["stages"][0]["experts"]
        reason_stats = stats["stages"][0]["reasoning"]

    accounting = hardware_parameter_accounting(model, mean_active_experts=1.5)
    state_advantage = carried_boundary_acc - reset_boundary_acc
    passed = (
        final_nll <= 0.20
        and final_acc >= 0.95
        and carried_boundary_acc >= 0.90
        and state_advantage >= 0.40
        and block_accuracy >= 0.90
        and accepted_fraction >= 0.10
        and accepted_precision >= 0.95
        and expert_stats is not None
        and reason_stats is not None
        and 1.0 <= expert_stats["mean_active_experts"] <= 2.0
        and reason_stats["max"] <= cfg.max_reason_steps
    )
    result = {
        "pass": passed,
        "dense_task_validity_gate": dense_gate,
        "initial_context_nll": initial_nll,
        "initial_context_accuracy": initial_acc,
        "final_context_nll": final_nll,
        "final_context_accuracy": final_acc,
        "carried_boundary_accuracy": carried_boundary_acc,
        "reset_boundary_accuracy": reset_boundary_acc,
        "state_advantage": state_advantage,
        "block_accuracy": block_accuracy,
        "verified_accepted_fraction": accepted_fraction,
        "verified_accepted_precision": accepted_precision,
        "expert_stats": expert_stats,
        "reasoning_stats": reason_stats,
        "parameter_accounting": accounting,
    }
    if not passed:
        raise AssertionError(result)
    return result


def main() -> None:
    result = {
        "status": "pass",
        "scope": "AERA_hardware_aware_CPU_learned_integration_gate",
        "controller_policy": controller_policy_probe(),
        "language_state_block": hardware_language_probe(),
        "claims": {
            "100m_authorized": False,
            "gpu_end_to_end_efficiency_proven": False,
            "real_language_superiority_proven": False,
            "breakthrough_proven": False,
        },
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
