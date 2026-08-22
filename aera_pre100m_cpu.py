from __future__ import annotations

import copy
import json
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from tam_research.aera import StreamState
from tam_research.aera_delta_memory import DeltaFastMemory
from tam_research.aera_full import (
    AERATextLM,
    BlockVerifier,
    BudgetedLatentReasoner,
    FullAERAConfig,
    FullComputeController,
    ModalityAdapterBank,
    ReplayRecord,
    SurpriseEventPatcher,
    VerifiedReplayBuffer,
    aera_parameter_accounting,
)
from tam_research.aera_world import WorldTransitionHead


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def controller_routing_probe() -> dict[str, object]:
    seed_all(2001)
    cfg = FullAERAConfig(d_model=16, n_experts=4, top_k_experts=1, max_reason_steps=4)
    controller = FullComputeController(cfg)
    opt = torch.optim.AdamW(controller.parameters(), lr=0.03)

    def batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.arange(n) % 4
        labels = labels[torch.randperm(n)]
        x = 0.05 * torch.randn(n, cfg.d_model)
        x[torch.arange(n), labels] += 2.0
        return x[:, None, :], labels

    for _ in range(160):
        x, labels = batch(128)
        logits = controller(x)["expert_logits"][:, 0]
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    x, labels = batch(4096)
    with torch.no_grad():
        pred = controller(x)["expert_logits"][:, 0].argmax(dim=-1)
    accuracy = float((pred == labels).float().mean())
    counts = torch.bincount(pred, minlength=4)
    max_fraction = float(counts.max()) / int(counts.sum())
    passed = accuracy >= 0.95 and max_fraction <= 0.30
    if not passed:
        raise AssertionError((accuracy, counts.tolist()))
    return {
        "pass": passed,
        "heldout_route_accuracy": accuracy,
        "route_counts": counts.tolist(),
        "max_expert_traffic_fraction": max_fraction,
    }


def adaptive_budget_probe() -> dict[str, object]:
    seed_all(2002)
    cfg = FullAERAConfig(d_model=16, n_experts=4, top_k_experts=1, max_reason_steps=4)
    controller = FullComputeController(cfg)
    reasoner = BudgetedLatentReasoner(cfg.d_model, cfg.max_reason_steps)
    opt = torch.optim.AdamW(controller.parameters(), lr=0.03)

    def batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.arange(n) % 4
        labels = labels[torch.randperm(n)]
        x = 0.05 * torch.randn(n, cfg.d_model)
        x[torch.arange(n), labels] += 2.0
        return x[:, None, :], labels

    for _ in range(160):
        x, labels = batch(128)
        logits = controller(x)["depth_logits"][:, 0]
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    x, labels = batch(4096)
    with torch.no_grad():
        depth_logits = controller(x)["depth_logits"]
        pred = depth_logits[:, 0].argmax(dim=-1)
        reasoner(x, depth_logits, hard=True)
    accuracy = float((pred == labels).float().mean())
    steps = reasoner.last_steps
    assert steps is not None
    means = []
    for label in range(4):
        means.append(float(steps[labels == label].float().mean()))
    monotonic = all(a < b for a, b in zip(means, means[1:]))
    mean_steps = float(steps.float().mean())
    passed = accuracy >= 0.95 and monotonic and mean_steps < cfg.max_reason_steps
    if not passed:
        raise AssertionError((accuracy, means, mean_steps))
    return {
        "pass": passed,
        "budget_class_accuracy": accuracy,
        "mean_hard_steps_by_difficulty": means,
        "overall_mean_steps": mean_steps,
        "fixed_max_baseline_steps": cfg.max_reason_steps,
    }


def memory_stress_probe() -> dict[str, object]:
    seed_all(2003)
    d_model = 32
    memory_dim = 16
    memory = DeltaFastMemory(d_model, memory_dim, lr=1.0, decay=1.0).eval()
    with torch.no_grad():
        for p in memory.parameters():
            p.zero_()
        memory.q.weight[:, :memory_dim] = torch.eye(memory_dim)
        memory.k.weight[:, :memory_dim] = torch.eye(memory_dim)
        memory.v.weight[:, memory_dim:] = torch.eye(memory_dim)
        memory.out.weight[memory_dim:, :] = torch.eye(memory_dim)

    base_before = [p.detach().clone() for p in memory.parameters()]
    state = memory.empty_state(1, torch.device("cpu"), torch.float32)
    session_b = memory.empty_state(1, torch.device("cpu"), torch.float32)
    current = {k: k for k in range(memory_dim)}

    def event(key: int, value: int | None) -> torch.Tensor:
        x = torch.zeros(1, 1, d_model)
        x[0, 0, key] = 1.0
        if value is not None:
            x[0, 0, memory_dim + value] = 1.0
        return x

    one = torch.ones(1, 1, 1)
    for key, value in current.items():
        state = memory.local_update(event(key, value), one, state)

    rng = random.Random(2003)
    overwrites = 1000
    for _ in range(overwrites):
        key = rng.randrange(memory_dim)
        value = rng.randrange(memory_dim)
        current[key] = value
        state = memory.local_update(event(key, value), one, state)

    correct = 0
    stale = 0
    for key, value in current.items():
        recall = memory.read(event(key, None), state)[0, 0, memory_dim:]
        pred = int(recall.argmax())
        correct += pred == value
        stale += pred != value
    accuracy = correct / memory_dim

    # Fresh independent session must remain empty and differ from learned state.
    leaked_norm = float(memory.read(event(0, None), session_b).abs().max())
    base_unchanged = all(torch.equal(a, b) for a, b in zip(base_before, memory.parameters()))
    stale_rate = stale / memory_dim
    passed = accuracy >= 0.95 and stale_rate <= 0.05 and leaked_norm == 0.0 and base_unchanged
    if not passed:
        raise AssertionError((accuracy, stale_rate, leaked_norm, base_unchanged))
    return {
        "pass": passed,
        "keys": memory_dim,
        "random_overwrites": overwrites,
        "current_value_accuracy": accuracy,
        "stale_resurrection_rate": stale_rate,
        "fresh_session_max_recall": leaked_norm,
        "base_parameters_unchanged": base_unchanged,
    }


def stream_state_probe() -> dict[str, object]:
    seed_all(2004)
    d_model = 12
    stream = StreamState(d_model)
    head = nn.Linear(d_model, 2)
    params = list(stream.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=0.02)

    def make_batch(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        bits = torch.randint(0, 2, (batch, length))
        x = torch.zeros(batch, length, d_model)
        x.scatter_(2, bits[..., None], 1.0)
        target = bits.cumsum(dim=1) % 2
        return x, target

    for _ in range(300):
        x, target = make_batch(64, 16)
        initial = torch.zeros(x.size(0), d_model)
        out, _ = stream(x, initial)
        loss = F.cross_entropy(head(out).reshape(-1, 2), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    x, target = make_batch(256, 64)
    with torch.no_grad():
        state = torch.zeros(x.size(0), d_model)
        carry_logits = []
        reset_logits = []
        for start in range(0, 64, 16):
            chunk = x[:, start : start + 16]
            out, state = stream(chunk, state)
            carry_logits.append(head(out))
            reset_out, _ = stream(chunk, torch.zeros_like(state))
            reset_logits.append(head(reset_out))
        carry = torch.cat(carry_logits, dim=1).argmax(dim=-1)
        reset = torch.cat(reset_logits, dim=1).argmax(dim=-1)
        carry_acc = float((carry == target).float().mean())
        reset_acc = float((reset == target).float().mean())
        state_norm = float(state.norm(dim=-1).mean())
    passed = carry_acc >= 0.90 and carry_acc >= reset_acc + 0.15 and math.isfinite(state_norm)
    if not passed:
        raise AssertionError((carry_acc, reset_acc, state_norm))
    return {
        "pass": passed,
        "carried_state_accuracy": carry_acc,
        "reset_each_chunk_accuracy": reset_acc,
        "advantage": carry_acc - reset_acc,
        "final_state_mean_norm": state_norm,
    }


def event_patching_probe() -> dict[str, object]:
    patcher = SurpriseEventPatcher(min_patch=1, max_patch=8, threshold=0.6)
    predictable = torch.full((256,), 0.05)
    difficult = torch.full((256,), 0.05)
    difficult[1::2] = 0.95
    p = patcher.spans(predictable)
    d = patcher.spans(difficult)
    p_mean = sum(b - a for a, b in p) / len(p)
    d_mean = sum(b - a for a, b in d) / len(d)
    preserved = sum(b - a for a, b in p) == 256 and sum(b - a for a, b in d) == 256
    passed = p_mean >= 2.0 * d_mean and preserved
    if not passed:
        raise AssertionError((p_mean, d_mean, preserved))
    return {
        "pass": passed,
        "predictable_mean_patch": p_mean,
        "high_surprise_mean_patch": d_mean,
        "raw_order_and_coverage_preserved": preserved,
    }


def multimodal_world_probe() -> dict[str, object]:
    seed_all(2005)
    n_classes = 48
    d_model = 16
    text_dim, image_dim, action_dim = 10, 14, 6
    bank = ModalityAdapterBank(d_model, {"text": text_dim, "image": image_dim, "action": action_dim})
    target = F.normalize(torch.randn(n_classes, d_model), dim=-1)
    text_raw = torch.randn(n_classes, text_dim)
    image_raw = torch.randn(n_classes, image_dim)
    action_raw = torch.randn(n_classes, action_dim)
    opt = torch.optim.AdamW(bank.parameters(), lr=0.03)

    for _ in range(300):
        ids = torch.randint(0, n_classes, (96,))
        loss = 0.0
        for name, raw in (("text", text_raw), ("image", image_raw), ("action", action_raw)):
            z = F.normalize(bank(name, raw[ids]), dim=-1)
            loss = loss + F.mse_loss(z, target[ids])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        zt = F.normalize(bank("text", text_raw), dim=-1)
        zi = F.normalize(bank("image", image_raw), dim=-1)
        sim = zt @ zi.T
        retrieval = float((sim.argmax(dim=-1) == torch.arange(n_classes)).float().mean())

    # Controlled action-conditioned latent dynamics.
    world = WorldTransitionHead(d_model, action_dim)
    world_opt = torch.optim.AdamW(world.parameters(), lr=0.02)
    action_effect = torch.randn(action_dim, d_model) * 0.25
    for _ in range(300):
        s = torch.randn(128, d_model)
        a = F.one_hot(torch.randint(0, action_dim, (128,)), num_classes=action_dim).float()
        y = s + a @ action_effect
        pred = world(s, a)
        loss = F.mse_loss(pred, y)
        world_opt.zero_grad(set_to_none=True)
        loss.backward()
        world_opt.step()
    with torch.no_grad():
        s = torch.randn(512, d_model)
        a = F.one_hot(torch.randint(0, action_dim, (512,)), num_classes=action_dim).float()
        y = s + a @ action_effect
        mse = float(F.mse_loss(world(s, a), y))
    passed = retrieval >= 0.90 and mse <= 0.03
    if not passed:
        raise AssertionError((retrieval, mse))
    return {
        "pass": passed,
        "text_to_image_retrieval_accuracy": retrieval,
        "action_conditioned_state_mse": mse,
    }


def replay_probe() -> dict[str, object]:
    replay = VerifiedReplayBuffer(capacity=100)
    replay.add(ReplayRecord("alice", 7, 2, True, 1.0, "verifier-v1"))
    replay.add(ReplayRecord("alice", 7, 9, False, 100.0, "unverified-hallucination"))
    before = replay.current_verified_value("alice", 7)
    replay.add(ReplayRecord("alice", 7, 5, True, 3.0, "verifier-correction"))
    after = replay.current_verified_value("alice", 7)
    replay.add(ReplayRecord("bob", 7, 11, True, 1.0, "bob-verifier"))
    sample = replay.prioritized_sample(3, seed=9)
    passed = before == 2 and after == 5 and replay.current_verified_value("bob", 7) == 11 and all(r.verified for r in sample)
    if not passed:
        raise AssertionError((before, after, sample))
    return {
        "pass": passed,
        "initial_verified_value": before,
        "corrected_verified_value": after,
        "unverified_record_cannot_override": True,
        "session_isolation": replay.current_verified_value("bob", 7) == 11,
        "replay_sample_all_verified": all(r.verified for r in sample),
    }


def retrieval_precision_controller_probe() -> dict[str, object]:
    seed_all(2006)
    cfg = FullAERAConfig(d_model=16, n_experts=4, max_reason_steps=4)
    controller = FullComputeController(cfg)
    opt = torch.optim.AdamW(controller.parameters(), lr=0.03)

    for _ in range(180):
        label = torch.randint(0, 2, (256, 1, 1)).float()
        x = 0.05 * torch.randn(256, 1, cfg.d_model)
        x[:, :, 0:1] += 2.0 * label
        x[:, :, 1:2] += 2.0 * (1.0 - label)
        out = controller(x)
        retrieval = out["retrieval_need"]
        precision = out["precision_budget"]
        loss = F.binary_cross_entropy(retrieval, label) + F.binary_cross_entropy(precision, label)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    label = torch.randint(0, 2, (4096, 1, 1)).float()
    x = 0.05 * torch.randn(4096, 1, cfg.d_model)
    x[:, :, 0:1] += 2.0 * label
    x[:, :, 1:2] += 2.0 * (1.0 - label)
    with torch.no_grad():
        out = controller(x)
        retrieval_acc = float(((out["retrieval_need"] >= 0.5) == label.bool()).float().mean())
        precision_acc = float(((out["precision_budget"] >= 0.5) == label.bool()).float().mean())
    passed = retrieval_acc >= 0.95 and precision_acc >= 0.95
    if not passed:
        raise AssertionError((retrieval_acc, precision_acc))
    return {
        "pass": passed,
        "retrieval_decision_accuracy": retrieval_acc,
        "precision_budget_signal_accuracy": precision_acc,
        "hardware_dynamic_precision_proven": False,
    }


def integrated_language_and_block_probe() -> dict[str, object]:
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
    model = AERATextLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.01)

    def sequences(batch: int, length: int) -> torch.Tensor:
        a = torch.randint(0, cfg.vocab_size, (batch,))
        b = torch.randint(0, cfg.vocab_size, (batch,))
        seq = [a, b]
        for _ in range(2, length):
            seq.append((seq[-1] + 2 * seq[-2] + 1) % cfg.vocab_size)
        return torch.stack(seq, dim=1)

    fixed_eval = sequences(256, 20)
    with torch.no_grad():
        initial = float(model.loss(fixed_eval, model(fixed_eval, update_memory=False))["next_token"])

    for _ in range(220):
        tok = sequences(48, 20)
        out = model(tok, update_memory=False, return_block_logits=True)
        losses = model.loss(tok, out, event_weight=0.05, compute_weight=0.003, balance_weight=0.02)
        block_logits = out["block_logits"]
        assert isinstance(block_logits, torch.Tensor)
        block_loss = 0.0
        valid_positions = tok.size(1) - cfg.block_size
        for offset in range(cfg.block_size):
            pred = block_logits[:, :valid_positions, offset]
            target = tok[:, offset + 1 : offset + 1 + valid_positions]
            block_loss = block_loss + F.cross_entropy(pred.reshape(-1, cfg.vocab_size), target.reshape(-1))
        total = losses["total"] + 0.25 * block_loss / cfg.block_size
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        out = model(fixed_eval, hard=False, update_memory=False, return_block_logits=True)
        final = float(model.loss(fixed_eval, out)["next_token"])
        block_logits = out["block_logits"]
        assert isinstance(block_logits, torch.Tensor)
        valid_positions = fixed_eval.size(1) - cfg.block_size
        correct = 0
        total_units = 0
        confidences = []
        for offset in range(cfg.block_size):
            pred_logits = block_logits[:, :valid_positions, offset]
            prob = F.softmax(pred_logits, dim=-1)
            confidence, pred = prob.max(dim=-1)
            target = fixed_eval[:, offset + 1 : offset + 1 + valid_positions]
            correct += int((pred == target).sum())
            total_units += target.numel()
            confidences.append(confidence)
        block_accuracy = correct / total_units
        confidence = torch.stack(confidences, dim=-1)
        verifier = BlockVerifier(min_confidence=0.80)
        accepted = verifier.accept_mask(confidence)
        accepted_units = int(accepted.sum())
        accepted_per_call = verifier.accepted_per_call(confidence)
        # Precision among accepted units across offsets.
        accepted_correct = 0
        accepted_total = 0
        for offset in range(cfg.block_size):
            pred = block_logits[:, :valid_positions, offset].argmax(dim=-1)
            target = fixed_eval[:, offset + 1 : offset + 1 + valid_positions]
            mask = accepted[..., offset]
            accepted_correct += int(((pred == target) & mask).sum())
            accepted_total += int(mask.sum())
        accepted_precision = accepted_correct / max(accepted_total, 1)

        # Hard inference path must execute and obey bounded depth.
        model(fixed_eval[:8], hard=True, update_memory=False)
        hard_stats = model.compute_stats()
        route_stats = hard_stats["stages"][0]["experts"]
        reason_stats = hard_stats["stages"][0]["reasoning"]

    accounting = aera_parameter_accounting(model)
    loss_drop = (initial - final) / max(initial, 1e-9)
    passed = (
        final < initial * 0.75
        and block_accuracy >= 0.70
        and accepted_precision >= 0.90
        and route_stats is not None
        and reason_stats is not None
        and reason_stats["max"] <= cfg.max_reason_steps
    )
    if not passed:
        raise AssertionError((initial, final, block_accuracy, accepted_precision, route_stats, reason_stats))
    return {
        "pass": passed,
        "initial_next_token_nll": initial,
        "final_next_token_nll": final,
        "relative_nll_drop": loss_drop,
        "block_accuracy": block_accuracy,
        "verified_accepted_units": accepted_units,
        "verified_accepted_units_per_position_call": accepted_per_call,
        "verified_accepted_precision": accepted_precision,
        "expert_routing": route_stats,
        "reasoning": reason_stats,
        "parameter_accounting": accounting,
    }


def main() -> None:
    results = {
        "status": "pass",
        "scope": "AERA_pre100M_CPU_mechanism_and_integration_validation",
        "learned_sparse_routing": controller_routing_probe(),
        "learned_adaptive_compute": adaptive_budget_probe(),
        "delta_fast_memory_stress": memory_stress_probe(),
        "stream_state_across_chunks": stream_state_probe(),
        "variable_event_patching": event_patching_probe(),
        "multimodal_and_world_transition": multimodal_world_probe(),
        "verified_replay": replay_probe(),
        "retrieval_and_precision_controller": retrieval_precision_controller_probe(),
        "integrated_text_and_block_generation": integrated_language_and_block_probe(),
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
