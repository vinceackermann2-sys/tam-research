"""Exploratory CPU-only original-SI mechanism ablations on fresh engineering seeds.

Ablations have unequal parameter and FLOP counts; these are screens, not
controlled causal attribution or SI evidence.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
from tam_research.si_original_hybrid_v0 import (
    OriginalSIHybrid, TransformerControl, SIConfig,
    count_parameters, prediction_loss,
)
from scripts.si_original_action_probe import generate_action_world, evaluate


def screen(seed, steps=300, batch=16):
    torch.set_num_threads(2)
    config = SIConfig()
    factories = {
        "original_hybrid": lambda: OriginalSIHybrid(config),
        "no_recurrence": lambda: OriginalSIHybrid(config, no_recurrence=True),
        "no_moe": lambda: OriginalSIHybrid(config, no_moe=True),
        "no_refine": lambda: OriginalSIHybrid(config, no_refine=True),
        "transformer": lambda: TransformerControl(config),
    }
    report = {"seed": seed, "steps": steps, "batch": batch, "models": {}}
    for name, factory in factories.items():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = factory()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        begin = time.monotonic()
        for step in range(steps):
            state, action, target = generate_action_world(batch, 24, seed*1000000 + step)
            model.train()
            opt.zero_grad(set_to_none=True)
            prediction = model(state, action)
            loss, _ = prediction_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        report["models"][name] = {
            "params": count_parameters(model),
            "train_seconds": time.monotonic() - begin,
            "metrics": evaluate(model, seed*1000000 + 85000),
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=94211)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(screen(args.seed, args.steps, args.batch), indent=2))
