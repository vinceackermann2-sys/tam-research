from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from experiments.rlt.model import RLTConfig, RecurrentLoopedTransformer, parameter_count


def run_smoke(device: str | torch.device | None = None) -> dict[str, object]:
    resolved = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(20260913)

    cfg = RLTConfig(
        vocab_size=257,
        d_model=64,
        n_heads=4,
        n_stages=2,
        max_seq_len=32,
        swa_window=4,
    )
    model = RecurrentLoopedTransformer(cfg).to(resolved).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 12), device=resolved)
    changed = tokens.clone()
    changed[:, 6:] = torch.randint(0, cfg.vocab_size, (2, 6), device=resolved)

    with torch.no_grad():
        prefix_a = model(tokens)[:, :6]
        prefix_b = model(changed)[:, :6]
    causal_max_diff = float((prefix_a - prefix_b).abs().max().cpu())
    if causal_max_diff > 2e-5:
        raise AssertionError(f"causality check failed: max diff={causal_max_diff}")
    if not model.last_cache_lengths or max(model.last_cache_lengths) > cfg.swa_window:
        raise AssertionError("decoder SWA cache exceeded its configured window")

    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(tokens)
    final_target = tokens[:, -1]
    final_loss = F.cross_entropy(logits[:, -2, :], final_target)
    final_loss.backward()
    start_grad = model.start_state.grad
    if start_grad is None or not torch.isfinite(start_grad).all():
        raise AssertionError("full-BPTT path to recurrent start state is missing")
    start_grad_l1 = float(start_grad.detach().abs().sum().cpu())
    if start_grad_l1 == 0.0:
        raise AssertionError("recurrent start state received zero gradient")

    learn_cfg = RLTConfig(
        vocab_size=16,
        d_model=24,
        n_heads=4,
        n_stages=1,
        max_seq_len=16,
        swa_window=4,
    )
    learner = RecurrentLoopedTransformer(learn_cfg).to(resolved)
    optimizer = torch.optim.AdamW(learner.parameters(), lr=4e-3, weight_decay=0.0)
    pattern = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8] * 2, device=resolved)
    batch = torch.stack([pattern.roll(i % 8) for i in range(8)])

    def sequence_loss() -> torch.Tensor:
        out = learner(batch[:, :-1])
        return F.cross_entropy(
            out.reshape(-1, learn_cfg.vocab_size),
            batch[:, 1:].reshape(-1),
        )

    learner.train()
    optimization_initial = float(sequence_loss().detach().cpu())
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        loss = sequence_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
        optimizer.step()
    optimization_final = float(sequence_loss().detach().cpu())
    if not math.isfinite(optimization_final) or optimization_final >= optimization_initial * 0.5:
        raise AssertionError(
            f"optimization smoke failed: {optimization_initial:.4f} -> "
            f"{optimization_final:.4f}"
        )

    depth_cfg = RLTConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_stages=48,
        max_seq_len=4,
        swa_window=4,
    )
    depth_probe = RecurrentLoopedTransformer(depth_cfg).to(resolved).eval()
    with torch.no_grad():
        depth_logits = depth_probe(
            torch.randint(0, depth_cfg.vocab_size, (1, 4), device=resolved)
        )
    if depth_logits.shape != (1, 4, depth_cfg.vocab_size):
        raise AssertionError("48-stage depth probe produced the wrong shape")

    return {
        "status": "pass",
        "device": str(resolved),
        "architecture": "causal encoder + token-recurrent decoder",
        "parameters_smoke_model": parameter_count(model),
        "encoder_stages": cfg.n_stages,
        "decoder_stages": cfg.n_stages,
        "stage_weights_shared_across_encoder_decoder": True,
        "causal_max_diff": causal_max_diff,
        "swa_window": cfg.swa_window,
        "decoder_cache_lengths": list(model.last_cache_lengths),
        "recurrent_start_state_grad_l1": start_grad_l1,
        "optimization_initial_nll": optimization_initial,
        "optimization_final_nll": optimization_final,
        "paper_depth_probe_stages": 48,
        "paper_depth_probe_parameters": parameter_count(depth_probe),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_smoke(), indent=2))
