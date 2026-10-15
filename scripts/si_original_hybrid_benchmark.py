"""Bounded CPU pilot with predeclared paired seeds; NO GPU launcher."""
import argparse
import json
import random
import time

import numpy as np
import torch

from tam_research.si_original_hybrid_v0 import (
    OriginalSIHybrid, TransformerControl, SIConfig, count_parameters,
    generate_world_batch, prediction_loss,
)


def evaluate(model, seed, modes=("stable", "switch", "nonlinear")):
    model.eval()
    metrics = {}
    with torch.no_grad():
        for mode in modes:
            vals, persistence = [], []
            for b in range(3):
                x, act, target = generate_world_batch(32, 16, seed + b, mode)
                y = model(x, act)["prediction"]
                vals.append((y - target).square().mean().item())
                persistence.append((x - target).square().mean().item())
            metrics[mode] = {"mse": float(np.mean(vals)), "persistence_mse": float(np.mean(persistence))}
    return metrics


def run(seed, steps, batch, no_moe=False, no_refine=False, no_recurrence=False):
    torch.set_num_threads(2)
    config = SIConfig()
    results = {"seed": seed, "steps": steps, "batch": batch, "models": {}}
    models = {
        "si_original_hybrid": lambda: OriginalSIHybrid(config, no_moe=no_moe, no_refine=no_refine, no_recurrence=no_recurrence),
        "transformer": lambda: TransformerControl(config),
    }
    for name, model_factory in models.items():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = model_factory()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        start = time.monotonic()
        for j in range(steps):
            x, action, target = generate_world_batch(batch, 16, seed * 100000 + j)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(x, action)
            loss, _ = prediction_loss(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        elapsed = time.monotonic() - start
        metrics = evaluate(model, seed=seed * 100000 + 85000)
        results["models"][name] = {
            "params": count_parameters(model), "train_wall_seconds": elapsed,
            "metrics": metrics, "last_train_loss": loss.item(),
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=93011)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--no-moe", action="store_true")
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--no-recurrence", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.steps, args.batch, args.no_moe, args.no_refine, args.no_recurrence), indent=2))
