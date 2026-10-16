"""CPU-only no-history test: is attention over earlier timesteps useful at all?

Uses frozen SI adaptive-horizon world generator and same model class to
avoid changing the backbone. A one-token mask permits current state/action
and position, but no prior observations or actions. No paid GPU work.
"""
import argparse
import json
import random
import time

import numpy as np
import torch

from scripts.si_original_adaptive_horizon_v1 import (
    Experiment, HorizonWorldModel, make_batch, objective, evaluation,
)


def run(seed: int, steps: int = 300, batch: int = 16):
    torch.set_num_threads(2)
    report = {"seed": seed, "steps": steps, "batch": batch, "candidates": {}}
    for name, length, mode in (
        ("no_history", 1, "local"),
        ("local_6", 6, "local"),
        ("full_history", 6, "global"),
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cfg = Experiment(window=length, steps=steps, batch=batch)
        model = HorizonWorldModel(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
        start = time.monotonic()
        for j in range(steps):
            obs, acts, target = make_batch(batch, cfg.length, seed * 1000000 + j, "mixed")
            model.train()
            opt.zero_grad(set_to_none=True)
            loss, _ = objective(model(obs, acts, mode), target, mode, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        report["candidates"][name] = {
            "parameters_instantiated": sum(p.numel() for p in model.parameters()),
            "train_seconds": time.monotonic() - start,
            "evaluation": evaluation(model, mode, cfg, seed * 1000000 + 87000),
        }
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=95211)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    print(json.dumps(run(args.seed, args.steps, args.batch), indent=2))
