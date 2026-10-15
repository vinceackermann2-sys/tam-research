"""Action-dominant hidden-dynamics world test for original SI hybrid (CPU only)."""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from tam_research.si_original_hybrid_v0 import (
    OriginalSIHybrid, TransformerControl, SIConfig,
    count_parameters, prediction_loss,
)


def generate_action_world(batch, length, seed, mode="stable"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=g)
    transition = 0.35 * rand(batch, 3, 3) + 0.25 * torch.eye(3)[None]
    alternate = 0.35 * rand(batch, 3, 3) + 0.25 * torch.eye(3)[None]
    action_effect = 1.1 * rand(batch, 4, 3)
    alternate_effect = 1.1 * rand(batch, 4, 3)
    current = rand(batch, 3) * 0.45
    observations, actions, targets = [], [], []
    for t in range(length):
        choice = torch.randint(0, 4, (batch,), generator=g)
        action = F.one_hot(choice, 4).float()
        changed = mode == "switch" and t >= length // 2
        a = alternate if changed else transition
        b = alternate_effect if changed else action_effect
        proposal = torch.bmm(current[:, None, :], a).squeeze(1) + torch.bmm(action[:, None, :], b).squeeze(1)
        if mode == "nonlinear":
            proposal = proposal + 0.5 * current.roll(1, -1) * current.roll(2, -1)
        nxt = 0.10 * current + 0.90 * torch.tanh(proposal)
        observations.append(current)
        actions.append(action)
        targets.append(nxt)
        current = nxt
    return torch.stack(observations, 1), torch.stack(actions, 1), torch.stack(targets, 1)


def evaluate(model, seed, length=24):
    out = {}
    model.eval()
    with torch.no_grad():
        for mode in ("stable", "switch", "nonlinear"):
            all_mse, late_mse, persist_mse, zero_mse = [], [], [], []
            for index in range(4):
                obs, act, target = generate_action_world(32, length, seed + index, mode)
                prediction = model(obs, act)["prediction"]
                all_mse.append(F.mse_loss(prediction, target).item())
                late_mse.append(F.mse_loss(prediction[:, length//2:], target[:, length//2:]).item())
                persist_mse.append(F.mse_loss(obs[:, length//2:], target[:, length//2:]).item())
                zero_mse.append(target[:, length//2:].square().mean().item())
            out[mode] = {
                "mse": float(np.mean(all_mse)),
                "late_mse": float(np.mean(late_mse)),
                "late_persistence": float(np.mean(persist_mse)),
                "late_zero": float(np.mean(zero_mse)),
            }
    return out


def run(seed, steps, batch):
    torch.set_num_threads(2)
    config = SIConfig(max_seq_len=64)
    result = {"seed": seed, "steps": steps, "batch": batch, "models": {}}
    for name, cls in (("original_hybrid", OriginalSIHybrid), ("transformer", TransformerControl)):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = cls(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        started = time.monotonic()
        for step in range(steps):
            obs, actions, target = generate_action_world(batch, 24, seed*1000000 + step)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            result_model = model(obs, actions)
            loss, mse = prediction_loss(result_model, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        elapsed = time.monotonic() - started
        result["models"][name] = {
            "params": count_parameters(model),
            "train_seconds": elapsed,
            "last_batch_mse": float(mse),
            "metrics": evaluate(model, seed*1000000 + 85000),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=94111)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.steps, args.batch), indent=2))
