"""SI-original attention-horizon submechanism: strictly causal surprise router.
CPU-only scientific-engineering feasibility study; this is NOT a complete SI model.
No hidden regime labels or future states are given to the predictor/router.
"""
from __future__ import annotations
import argparse
import json
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def world(batch, length, seed, mode="mixed", family="base"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g)
    m0 = .35 * rand(batch, 3, 3) + .25 * torch.eye(3)[None]
    m1 = .35 * rand(batch, 3, 3) + .25 * torch.eye(3)[None]
    b0, b1 = 1.1 * rand(batch, 4, 3), 1.1 * rand(batch, 4, 3)
    state = .45 * rand(batch, 3)
    switch_at = torch.randint(8, 19, (batch,), generator=g)
    switched = (torch.rand(batch, generator=g) < .5 if mode == "mixed" else
                torch.ones(batch, dtype=torch.bool) if mode == "switch" else
                torch.zeros(batch, dtype=torch.bool))
    prev_action = torch.zeros(batch, 4)
    obs, actions, targets = [], [], []
    for t in range(length):
        action = F.one_hot(torch.randint(0, 4, (batch,), generator=g), num_classes=4).float()
        changed = (switched & (t >= switch_at))[:, None, None]
        m = torch.where(changed, m1, m0)
        b = torch.where(changed, b1, b0)
        raw = torch.bmm(state[:, None, :], m).squeeze(1) + torch.bmm(action[:, None, :], b).squeeze(1)
        if family == "delayed":
            # A separate (unseen in training) controlled world family.
            raw = .35 * raw + 1.2 * torch.bmm(prev_action[:, None, :], b).squeeze(1)
        if mode == "nonlinear":
            raw += .5 * state.roll(1, -1) * state.roll(2, -1)
        future = .1 * state + .9 * torch.tanh(raw)
        obs.append(state)
        actions.append(action)
        targets.append(future)
        state, prev_action = future, action
    return torch.stack(obs, 1), torch.stack(actions, 1), torch.stack(targets, 1)


def causal_surprise_mask(obs, actions, window=6, threshold=.09):
    """Select local attention only when a prefix-fit model fails recently.

    At query t, fits on older transitions i<t-5 and evaluates only on the
    last five *completed* transitions (i<t); each target is obs[i+1].
    Thus no obs[t+1], hidden switch labels or future actions enter selection.
    """
    b, tmax, _ = obs.shape
    pos = torch.arange(tmax, device=obs.device)
    full = pos[None, :] <= pos[:, None]
    local = full & (pos[None, :] > pos[:, None] - window)
    mask = full[None].expand(b, -1, -1).clone()
    flags = torch.zeros(b, tmax, dtype=torch.bool, device=obs.device)
    x = torch.cat([obs, actions], -1)
    eye = torch.eye(x.size(-1), device=obs.device)
    with torch.no_grad():
        for t in range(12, tmax):
            old_start = max(0, t - 18)
            split = t - 5
            xold, yold = x[:, old_start:split], obs[:, old_start + 1:split + 1]
            xrecent, yrecent = x[:, split:t], obs[:, split + 1:t + 1]
            xt = xold.transpose(1, 2)
            w = torch.linalg.solve(xt @ xold + .25 * eye[None], xt @ yold)
            old_mse = (xold @ w - yold).square().mean((1, 2))
            new_mse = (xrecent @ w - yrecent).square().mean((1, 2))
            flags[:, t] = (new_mse - old_mse) > threshold
            mask[:, t] = torch.where(flags[:, t, None], local[t][None], full[t][None])
    return mask[:, None], flags


class HorizonAttention(nn.Module):
    def __init__(self, width=32, heads=4):
        super().__init__()
        self.width, self.heads = width, heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x, mask):
        b, t, d = x.shape
        q, k, v = [z.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for z in self.qkv(x).chunk(3, -1)]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class WorldModel(nn.Module):
    def __init__(self, width=32, layers=2):
        super().__init__()
        self.encoder = nn.Linear(7, width)
        self.position = nn.Embedding(24, width)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "norm1": nn.LayerNorm(width), "attn": HorizonAttention(width),
            "norm2": nn.LayerNorm(width),
            "ffn": nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)),
        }) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.pred = nn.Linear(width, 3)
        self.log_var = nn.Linear(width, 3)
        self.verify = nn.Linear(width, 1)

    def forward(self, obs, actions, mode):
        b, t, _ = obs.shape
        assert t <= 24
        index = torch.arange(t, device=obs.device)
        full = index[None, :] <= index[:, None]
        local = full & (index[None, :] > index[:, None] - 6)
        if mode == "global":
            mask, fraction = full, 0.
        elif mode == "local":
            mask, fraction = local, 1.
        elif mode == "surprise":
            mask, flags = causal_surprise_mask(obs, actions)
            fraction = float(flags[:, 12:].float().mean()) if t > 12 else 0.
        elif mode == "no_history":
            mask, fraction = torch.eye(t, dtype=torch.bool, device=obs.device), 1.
        else:
            raise ValueError(mode)
        x = self.encoder(torch.cat([obs, actions], -1)) + self.position(index)[None]
        for blk in self.blocks:
            x = x + blk["attn"](blk["norm1"](x), mask)
            x = x + blk["ffn"](blk["norm2"](x))
        x = self.norm(x)
        return {"prediction": self.pred(x),
                "log_variance": self.log_var(x).clamp(-6, 3),
                "verified_error": F.softplus(self.verify(x)),
                "local_route_fraction": fraction}


def loss_fn(output, targets):
    err = output["prediction"] - targets
    mse = err.square().mean()
    var_loss = .5 * (output["log_variance"] + err.square() * (-output["log_variance"]).exp()).mean()
    verify_loss = (output["verified_error"] - err.detach().square().mean(-1, keepdim=True)).square().mean()
    return mse + .05 * var_loss + .05 * verify_loss, mse


def evaluate(model, mode, seed):
    model.eval()
    scores = {}
    with torch.no_grad():
        for family in ("base", "delayed"):
            for regime in ("stable", "switch"):
                metrics = []
                for j in range(3):
                    obs, a, tgt = world(32, 24, seed + 7*j, regime, family)
                    o = model(obs, a, mode)
                    mse = F.mse_loss(o["prediction"], tgt).item()
                    late = F.mse_loss(o["prediction"][:, 12:], tgt[:, 12:]).item()
                    base = F.mse_loss(obs[:, 12:], tgt[:, 12:]).item()
                    metrics.append((mse, late, base, o["local_route_fraction"]))
                m = np.array(metrics, dtype=float).mean(axis=0)
                scores[f"{family}_{regime}"] = {
                    "mse": float(m[0]), "late_mse": float(m[1]),
                    "persistence_late": float(m[2]), "local_route_fraction": float(m[3])}
    return scores


def run(seed, steps=150, batch=16):
    torch.set_num_threads(2)
    results = {"seed": seed, "steps": steps, "batch": batch, "models": {}}
    for mode in ("no_history", "local", "global", "surprise"):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = WorldModel()
        optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        start = time.perf_counter()
        for step in range(steps):
            obs, actions, targets = world(batch, 24, seed * 1000000 + step, "mixed")
            model.train()
            optim.zero_grad(set_to_none=True)
            out = model(obs, actions, mode)
            loss, _ = loss_fn(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        results["models"][mode] = {
            "params": sum(p.numel() for p in model.parameters()),
            "cpu_train_seconds": round(time.perf_counter() - start, 3),
            "eval": evaluate(model, mode, seed * 1000000 + 94000)}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=96311)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    print(json.dumps(run(args.seed, args.steps, args.batch), indent=2))


if __name__ == "__main__":
    main()
