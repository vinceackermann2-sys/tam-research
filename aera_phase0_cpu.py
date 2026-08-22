from __future__ import annotations

import json
import math

import torch

from tam_research.aera import (
    AERAConfig,
    AdaptiveLatentReasoner,
    FastWeightMemory,
    SparseExpertLayer,
)


def sparse_probe() -> dict[str, object]:
    torch.manual_seed(100)
    cfg = AERAConfig(
        d_model=16,
        n_experts=8,
        top_k_experts=2,
        expert_mult=2,
        memory_dim=8,
    )
    layer = SparseExpertLayer(cfg).eval()
    x = torch.randn(4, 32, cfg.d_model)
    logits = torch.randn(4, 32, cfg.n_experts)
    y = layer(x, logits)
    stats = layer.routing_stats()
    if stats is None:
        raise RuntimeError("missing sparse routing stats")
    expected = x.size(0) * x.size(1) * cfg.top_k_experts
    if stats["assignments"] != expected:
        raise AssertionError((stats["assignments"], expected))
    return {
        "shape_ok": list(y.shape) == list(x.shape),
        "stored_experts": cfg.n_experts,
        "top_k": cfg.top_k_experts,
        "expert_assignment_count": stats["assignments"],
        "active_expert_fraction_per_event": stats["active_fraction_of_experts_per_token"],
        "per_expert_assignments": stats["per_expert"],
    }


def memory_probe() -> dict[str, object]:
    torch.manual_seed(101)
    cfg = AERAConfig(
        d_model=16,
        n_experts=2,
        top_k_experts=1,
        memory_dim=8,
        fast_memory_lr=1.0,
        fast_memory_decay=0.0,
    )
    memory = FastWeightMemory(cfg).eval()
    with torch.no_grad():
        memory.q.weight.zero_()
        memory.k.weight.zero_()
        memory.v.weight.zero_()
        memory.out.weight.zero_()
        memory.q.weight[:, :8] = torch.eye(8)
        memory.k.weight[:, :8] = torch.eye(8)
        memory.v.weight[:, 8:] = torch.eye(8)
        memory.out.weight[8:, :] = torch.eye(8)

    state = memory.empty_state(1, torch.device("cpu"), torch.float32)

    def event(key: int, value: int | None) -> torch.Tensor:
        x = torch.zeros(1, 1, 16)
        x[0, 0, key] = 1.0
        if value is not None:
            x[0, 0, 8 + value] = 1.0
        return x

    key = 2
    first_value = 5
    replacement_value = 1
    strength = torch.ones(1, 1, 1)

    state1 = memory.local_update(event(key, first_value), strength, state)
    query = event(key, None)
    recall1 = memory.read(query, state1)[0, 0, 8:]
    predicted1 = int(recall1.argmax())

    state2 = memory.local_update(event(key, replacement_value), strength, state1)
    recall2 = memory.read(query, state2)[0, 0, 8:]
    predicted2 = int(recall2.argmax())

    if predicted1 != first_value or predicted2 != replacement_value:
        raise AssertionError((predicted1, predicted2))

    return {
        "first_write_recalled": predicted1,
        "replacement_write_recalled": predicted2,
        "stale_value_replaced": predicted2 == replacement_value and predicted2 != first_value,
        "base_parameter_update_required": False,
    }


def reasoning_probe() -> dict[str, object]:
    torch.manual_seed(102)
    cfg = AERAConfig(
        d_model=8,
        n_experts=2,
        top_k_experts=1,
        memory_dim=4,
        max_reason_steps=4,
        halt_threshold=0.55,
    )
    reasoner = AdaptiveLatentReasoner(cfg).eval()
    with torch.no_grad():
        reasoner.halt.weight.zero_()
        reasoner.halt.bias.fill_(math.log(0.7 / 0.3))

    x = torch.randn(1, 2, cfg.d_model)
    difficulty = torch.tensor([[[0.0], [1.0]]])
    reasoner(x, difficulty)
    steps = reasoner.last_steps
    if steps is None:
        raise RuntimeError("missing reasoning step stats")
    easy_steps = int(steps[0])
    hard_steps = int(steps[1])
    if not easy_steps < hard_steps:
        raise AssertionError((easy_steps, hard_steps))
    return {
        "easy_steps": easy_steps,
        "hard_steps": hard_steps,
        "hard_gets_more_compute": hard_steps > easy_steps,
        "max_step_budget": cfg.max_reason_steps,
    }


def main() -> None:
    result = {
        "status": "pass",
        "scope": "phase0_functional_mechanism_probe_not_quality_benchmark",
        "sparse_conditional_compute": sparse_probe(),
        "local_fast_memory": memory_probe(),
        "adaptive_reasoning": reasoning_probe(),
        "claims": {
            "gpu_speedup_proven": False,
            "language_quality_proven": False,
            "breakthrough_proven": False,
        },
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
