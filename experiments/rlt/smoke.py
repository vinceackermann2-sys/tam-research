from __future__ import annotations

from dataclasses import replace
import math

import torch
import torch.nn.functional as F

from tam_research.models import ModelConfig

from experiments.rlt.model import RecurrentLoopedLM, parameter_count


def run_smoke(device: str | torch.device | None = None) -> dict[str, object]:
    resolved = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(20260913)

    base_cfg = ModelConfig(
        vocab_size=257,
        d_model=64,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        architecture="transformer",
    )
    short = RecurrentLoopedLM(base_cfg).to(resolved)
    deep = RecurrentLoopedLM(replace(base_cfg, n_layers=12)).to(resolved)
    short_params = parameter_count(short)
    deep_params = parameter_count(deep)
    if short_params != deep_params:
        raise AssertionError("RLT parameter count changed with loop count")

    short.eval()
    tokens = torch.randint(0, base_cfg.vocab_size, (2, 16), device=resolved)
    changed = tokens.clone()
    changed[:, 8:] = torch.randint(
        0, base_cfg.vocab_size, (2, 8), device=resolved
    )
    with torch.no_grad():
        prefix_a = short(tokens)[:, :8]
        prefix_b = short(changed)[:, :8]
    causal_max_diff = float((prefix_a - prefix_b).abs().max().cpu())
    if causal_max_diff > 2e-5:
        raise AssertionError(f"causality check failed: max diff={causal_max_diff}")

    short.train()
    logits = short(tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, base_cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()
    grads = [
        p.grad for p in short.shared_block.parameters() if p.grad is not None
    ]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise AssertionError("shared block did not receive finite gradients")
    grad_l1 = sum(float(g.detach().abs().sum().cpu()) for g in grads)

    learn_cfg = ModelConfig(
        vocab_size=32,
        d_model=32,
        n_layers=3,
        n_heads=4,
        max_seq_len=32,
        architecture="transformer",
    )
    learner = RecurrentLoopedLM(learn_cfg).to(resolved)
    optimizer = torch.optim.AdamW(learner.parameters(), lr=3e-3, weight_decay=0.0)
    pattern = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8] * 4, device=resolved)
    batch = torch.stack([pattern.roll(i % 8) for i in range(16)])

    def sequence_loss() -> torch.Tensor:
        out = learner(batch[:, :-1])
        return F.cross_entropy(
            out.reshape(-1, learn_cfg.vocab_size),
            batch[:, 1:].reshape(-1),
        )

    learner.train()
    initial_loss = float(sequence_loss().detach().cpu())
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        train_loss = sequence_loss()
        train_loss.backward()
        optimizer.step()
    final_loss = float(sequence_loss().detach().cpu())
    if not math.isfinite(final_loss) or final_loss >= initial_loss * 0.5:
        raise AssertionError(
            f"optimization smoke failed: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    return {
        "status": "pass",
        "device": str(resolved),
        "short_loops": base_cfg.n_layers,
        "deep_loops": 12,
        "parameters_short": short_params,
        "parameters_deep": deep_params,
        "parameter_invariance": short_params == deep_params,
        "causal_max_diff": causal_max_diff,
        "shared_block_grad_l1": grad_l1,
        "optimization_initial_nll": initial_loss,
        "optimization_final_nll": final_loss,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_smoke(), indent=2))
