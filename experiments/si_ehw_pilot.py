"""Executable Hypothesis Workspace (EHW): SI architecture phase-0 CPU pilot.

Every episode has an unseen action-conditioned transition law.
The model sees context (state, action, next_state) triples then predicts
one held-out transition. This is a structural-bias feasibility probe,
NOT an AGI or superintelligence benchmark.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Config:
    dim: int = 4
    shots: int = 12
    steps: int = 120
    batch: int = 32
    seed: int = 8117
    hidden: int = 64
    device: str = "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_worlds(batch: int, dim: int, shots: int, device: str, mode: str = "iid"):
    # Each episode gets a fresh, hidden transition law. The models get observations only.
    theta = torch.randn(batch, 2 * dim, dim, device=device) * 0.26
    theta[:, :dim, :] *= (torch.rand(batch, dim, dim, device=device) < 0.55)
    theta[:, :dim, :] += 0.55 * torch.eye(dim, device=device).unsqueeze(0)
    theta[:, dim:, :] *= 0.7
    theta2 = torch.randn_like(theta) * 0.26
    theta2[:, :dim, :] *= (torch.rand(batch, dim, dim, device=device) < 0.55)
    theta2[:, :dim, :] += 0.55 * torch.eye(dim, device=device).unsqueeze(0)
    theta2[:, dim:, :] *= 0.7
    if mode == "ood_strong":
        theta[:, :dim, :] *= 1.35
        theta[:, dim:, :] *= 1.35
    if mode == "ood_switch":
        theta2 = theta2 * 1.1

    x = torch.empty(batch, shots + 1, dim, device=device).uniform_(-1., 1.)
    act_idx = torch.randint(dim, (batch, shots + 1), device=device)
    act = F.one_hot(act_idx, num_classes=dim).to(x.dtype)
    feat = torch.cat([x, act], dim=-1)
    raw = torch.einsum("bti,bij->btj", feat, theta)
    if mode == "ood_switch":
        alt = torch.einsum("bti,bij->btj", feat, theta2)
        raw[:, shots // 2:] = alt[:, shots // 2:]
    if mode == "ood_quadratic":
        raw += .45 * (x.roll(1, -1) * x.roll(2, -1))
    y = torch.tanh(raw)
    if mode == "iid":
        y = y + .005 * torch.randn_like(y)
    return feat[:, :shots], y[:, :shots], feat[:, shots], y[:, shots]


class EHW(nn.Module):
    """Episode-local hypothesis fit + shared learned nonlinear correction.

    This pilot deliberately builds in a linear system-identification prior;
    the baseline does not have that prior. A win is NOT a fair general-architecture
    comparison, and a ridge-only control is required.
    """
    def __init__(self, dim: int = 4, hidden: int = 64):
        super().__init__()
        self.log_ridge = nn.Parameter(torch.tensor(-2.0))
        self.readout = nn.Sequential(
            nn.Linear(4 * dim + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, hist_x, hist_y, query):
        d = hist_x.size(-1)
        lam = self.log_ridge.exp().clamp(1e-4, 10.)
        xt = hist_x.transpose(1, 2)
        weights = torch.linalg.solve(
            xt @ hist_x + lam * torch.eye(d, device=hist_x.device),
            xt @ hist_y,
        )
        hypothesis = torch.bmm(query.unsqueeze(1), weights).squeeze(1)
        residual = hist_y - torch.bmm(hist_x, weights)
        err = residual.square().mean(dim=(1, 2)).unsqueeze(-1)
        summary = hist_y.mean(dim=1)
        full = torch.cat([query, hypothesis, summary, err], dim=-1)
        return hypothesis + self.readout(full)


class TransformerControl(nn.Module):
    """Observed transitions and query encoded as an unordered token set."""
    def __init__(self, dim: int = 4, hidden: int = 64):
        super().__init__()
        self.token = nn.Linear(4 * dim + 1, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, 4, 2 * hidden, 0., batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim))

    def forward(self, hist_x, hist_y, query):
        b, k, _ = hist_x.shape
        d = hist_y.shape[-1]
        context = torch.cat([
            hist_x, hist_y,
            torch.zeros(b, k, d, device=query.device),
            torch.zeros(b, k, 1, device=query.device),
        ], dim=-1)
        q = torch.cat([
            query, torch.zeros(b, 2 * d, device=query.device),
            torch.ones(b, 1, device=query.device),
        ], dim=-1).unsqueeze(1)
        tokens = self.token(torch.cat([context, q], dim=1))
        return self.head(self.encoder(tokens)[:, -1, :])


def params(model):
    return sum(p.numel() for p in model.parameters())


def run(cfg: Config):
    set_seed(cfg.seed)
    if cfg.device == "cpu":
        torch.set_num_threads(2)
    models = {
        "ehw": EHW(cfg.dim, cfg.hidden).to(cfg.device),
        "transformer": TransformerControl(cfg.dim, cfg.hidden).to(cfg.device),
    }
    scores = {"config": vars(cfg), "models": {}}
    stream_seed = cfg.seed + 1500
    for name, model in models.items():
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=.01)
        set_seed(stream_seed)
        started = time.monotonic()
        model.train()
        for _ in range(cfg.steps):
            batch = sample_worlds(cfg.batch, cfg.dim, cfg.shots, cfg.device)
            prediction = model(*batch[:3])
            loss = F.mse_loss(prediction, batch[3])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        train_time = time.monotonic() - started
        model.eval()
        evals = {}
        with torch.no_grad():
            for mode in ["iid", "ood_strong", "ood_quadratic", "ood_switch"]:
                set_seed(cfg.seed + 9000)
                errors = []
                for _ in range(8):
                    batch = sample_worlds(
                        128, cfg.dim, cfg.shots, cfg.device, mode=mode
                    )
                    errors.append(F.mse_loss(model(*batch[:3]), batch[3]).item())
                evals[mode] = round(float(np.mean(errors)), 6)
        scores["models"][name] = {
            "parameters": params(model),
            "train_seconds": round(train_time, 2),
            "train_loss_last": round(float(loss.item()), 6),
            "eval_mse": evals,
        }
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=8117)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(run(Config(
        steps=args.steps, batch=args.batch,
        seed=args.seed, hidden=args.hidden,
    )), indent=2))


if __name__ == "__main__":
    main()
