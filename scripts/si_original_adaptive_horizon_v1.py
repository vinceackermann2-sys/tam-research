"""Original SI track: CPU-only learned causal horizon test, not an SI model.

Tests whether a learned, per-token causal local/full attention gate improves
an action-conditioned world predictor over fixed attention masks. Oracle
switch labels are NEVER provided to the networks. Gate mixes TWO full-cost
SDPA calls, so this prototype does not claim conditional compute efficiency.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Experiment:
    length: int = 24
    window: int = 6
    width: int = 32
    heads: int = 4
    layers: int = 2
    batch: int = 16
    steps: int = 300
    lr: float = 0.001
    global_penalty: float = 0.002


def make_batch(batch: int, length: int, seed: int, mode: str):
    """Per-episode latent dynamics; random unknown switch time and regime.

    Output observations/actions/next-state. Models observe only first two.
    No switch labels, change times or transition matrices reach a model.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = lambda *shape: torch.randn(*shape, generator=g)
    m1 = 0.35 * rnd(batch, 3, 3) + .25 * torch.eye(3)[None]
    m2 = 0.35 * rnd(batch, 3, 3) + .25 * torch.eye(3)[None]
    b1 = 1.1 * rnd(batch, 4, 3)
    b2 = 1.1 * rnd(batch, 4, 3)
    state = 0.45 * rnd(batch, 3)
    switch_at = torch.randint(8, 19, (batch,), generator=g)
    if mode == "mixed":
        switch = torch.rand(batch, generator=g) < 0.5
    elif mode == "switch":
        switch = torch.ones(batch, dtype=torch.bool)
    else:
        switch = torch.zeros(batch, dtype=torch.bool)
    observations, actions, targets = [], [], []
    for t in range(length):
        a = F.one_hot(torch.randint(0, 4, (batch,), generator=g), num_classes=4).float()
        changed = (switch & (t >= switch_at))[:, None, None]
        m = torch.where(changed, m2, m1)
        b = torch.where(changed, b2, b1)
        raw = torch.bmm(state[:, None, :], m).squeeze(1) + torch.bmm(a[:, None, :], b).squeeze(1)
        if mode == "nonlinear":
            raw = raw + 0.5 * state.roll(1, -1) * state.roll(2, -1)
        new_state = .1 * state + .9 * torch.tanh(raw)
        observations.append(state)
        actions.append(a)
        targets.append(new_state)
        state = new_state
    return torch.stack(observations, 1), torch.stack(actions, 1), torch.stack(targets, 1)


class HorizonAttention(nn.Module):
    def __init__(self, width: int, heads: int, window: int):
        super().__init__()
        self.width, self.heads, self.window = width, heads, window
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.gate = nn.Sequential(nn.Linear(2 * width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))

    def forward(self, x: torch.Tensor, mode: str):
        b, t, d = x.shape
        q, k, v = [z.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for z in self.qkv(x).chunk(3, -1)]
        index = torch.arange(t, device=x.device)
        causal = index[None, :] <= index[:, None]
        local = causal & (index[None, :] > (index[:, None] - self.window))
        if mode == "local":
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=local)
            fraction = x.new_zeros(b, t, 1)
        elif mode == "global":
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            fraction = x.new_ones(b, t, 1)
        elif mode == "adaptive":
            short = F.scaled_dot_product_attention(q, k, v, attn_mask=local)
            long = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), 1)
            fraction = torch.sigmoid(self.gate(torch.cat([x, x-previous], -1)))
            y = short * (1 - fraction[:, None]) + long * fraction[:, None]
        else:
            raise ValueError(mode)
        y = self.out(y.transpose(1, 2).reshape(b, t, d))
        return y, fraction


class Block(nn.Module):
    def __init__(self, c: Experiment):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.width)
        self.attn = HorizonAttention(c.width, c.heads, c.window)
        self.norm2 = nn.LayerNorm(c.width)
        self.ffn = nn.Sequential(nn.Linear(c.width, 2*c.width), nn.GELU(), nn.Linear(2*c.width, c.width))

    def forward(self, x, mode):
        y, global_fraction = self.attn(self.norm1(x), mode)
        x = x + y
        return x + self.ffn(self.norm2(x)), global_fraction


class HorizonWorldModel(nn.Module):
    def __init__(self, c: Experiment):
        super().__init__()
        self.c = c
        self.encoder = nn.Linear(7, c.width)
        self.position = nn.Embedding(c.length, c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width)
        self.pred = nn.Linear(c.width, 3)
        self.log_var = nn.Linear(c.width, 3)
        self.verify = nn.Linear(c.width, 1)

    def forward(self, obs, actions, mode):
        b, t, _ = obs.shape
        x = self.encoder(torch.cat((obs, actions), -1)) + self.position(torch.arange(t, device=obs.device))[None]
        gates = []
        for block in self.blocks:
            x, frac = block(x, mode)
            gates.append(frac)
        x = self.norm(x)
        return {
            "prediction": self.pred(x),
            "log_variance": self.log_var(x).clamp(-6, 3),
            "verified_error": F.softplus(self.verify(x)),
            "mean_global_fraction": torch.stack(gates).mean(),
        }


def objective(out, target, mode, c):
    diff = out["prediction"] - target
    mse = diff.square().mean()
    gaussian = 0.5 * (out["log_variance"] + diff.square() * (-out["log_variance"]).exp()).mean()
    calibrated = (out["verified_error"] - diff.detach().square().mean(-1, keepdim=True)).square().mean()
    pen = c.global_penalty * out["mean_global_fraction"] if mode == "adaptive" else mse.new_zeros(())
    return mse + .05 * gaussian + .05 * calibrated + pen, mse


def evaluation(model, mode, c, seed):
    model.eval()
    report = {}
    with torch.no_grad():
        for world in ("stable", "switch", "nonlinear"):
            acc = []
            for j in range(4):
                obs, actions, target = make_batch(32, c.length, seed + 5*j, world)
                out = model(obs, actions, mode)
                acc.append((float(F.mse_loss(out["prediction"], target)),
                            float(F.mse_loss(out["prediction"][:, 12:], target[:, 12:])),
                            float(F.mse_loss(obs[:, 12:], target[:, 12:])),
                            float(target[:, 12:].square().mean()),
                            float(out["mean_global_fraction"])))
            m = np.array(acc).mean(axis=0)
            report[world] = dict(mse=float(m[0]), late_mse=float(m[1]),
                                 persistence_late=float(m[2]), zero_late=float(m[3]),
                                 mean_global_fraction=float(m[4]))
    return report


def run(seed: int, c: Experiment):
    torch.set_num_threads(2)
    report = {"seed": seed, "config": asdict(c), "models": {}}
    for mode in ("local", "global", "adaptive"):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = HorizonWorldModel(c)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=.01)
        start = time.monotonic()
        for step in range(c.steps):
            obs, act, target = make_batch(c.batch, c.length, seed * 1000000 + step, "mixed")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(obs, act, mode)
            loss, mse = objective(out, target, mode, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        elapsed = time.monotonic() - start
        report["models"][mode] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_seconds": round(elapsed, 3),
            "train_last_mse": float(mse),
            "evaluation": evaluation(model, mode, c, seed * 1000000 + 87000),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=95100)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, Experiment(batch=args.batch, steps=args.steps)), indent=2))


if __name__ == "__main__":
    main()
