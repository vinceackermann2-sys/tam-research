"""Controlled local-vs-global causal attention mask ablation in the SI scaffold."""
import argparse
import json
import random
import time

import numpy as np
import torch

from tam_research.si_original_hybrid_v0 import OriginalSIHybrid, TransformerControl, SIConfig, count_parameters, prediction_loss
from scripts.si_original_action_probe import generate_action_world, evaluate


def run(seed, steps=300, batch=16):
    torch.set_num_threads(2)
    factories = {
        "local_dense": lambda: OriginalSIHybrid(SIConfig(window=6), no_recurrence=True, no_moe=True, no_refine=True),
        "global_dense": lambda: OriginalSIHybrid(SIConfig(window=None), no_recurrence=True, no_moe=True, no_refine=True),
        "original_hybrid": lambda: OriginalSIHybrid(SIConfig()),
        "transformer": lambda: TransformerControl(SIConfig()),
    }
    result = {"seed": seed, "steps": steps, "batch": batch, "models": {}}
    for name, factory in factories.items():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = factory()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        start = time.monotonic()
        for j in range(steps):
            obs, action, target = generate_action_world(batch, 24, seed*1000000+j)
            model.train()
            opt.zero_grad(set_to_none=True)
            loss, _ = prediction_loss(model(obs, action), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        result["models"][name] = {
            "parameters": count_parameters(model),
            "train_seconds": time.monotonic()-start,
            "metrics": evaluate(model, seed*1000000+85000),
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=94311)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    print(json.dumps(run(args.seed, args.steps, args.batch), indent=2))
