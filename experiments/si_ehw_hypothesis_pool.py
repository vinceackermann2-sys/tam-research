"""Exploratory EHW multi-hypothesis pool, not a preregistered SI result.

Candidates infer transition laws from 12, 8 or 6 recent context
transitions. Fit each on data excluding the last two observations; choose
the hypothesis with lowest last-two prediction error plus a complexity
penalty; refit selected hypothesis on the full candidate window.

This is fixed-function system identification, NOT learned program induction.
"""
import argparse
import json

import torch
import torch.nn.functional as F

from si_ehw_pilot import sample_worlds, set_seed


def fit(x, y, q, lam=0.14):
    xt = x.transpose(1, 2)
    weights = torch.linalg.solve(
        xt @ x + lam * torch.eye(x.size(-1), device=x.device), xt @ y
    )
    return torch.bmm(q.unsqueeze(1), weights).squeeze(1)


def predict(hist, y, q, penalty=0.001):
    k = hist.size(1)
    outs, scores = [], []
    val_start = k - 2
    for window in (12, 8, 6):
        start = k - window
        trainx, trainy = hist[:, start:val_start], y[:, start:val_start]
        val = torch.stack(
            [fit(trainx, trainy, hist[:, j]) for j in range(val_start, k)],
            dim=1,
        )
        loss = (val - y[:, val_start:]).square().mean((1, 2))
        outs.append(fit(hist[:, start:], y[:, start:], q))
        scores.append(loss + penalty * (k / window) ** 0.5)
    idx = torch.stack(scores, dim=1).argmin(dim=1)
    out = torch.stack(outs, dim=1)
    return out.gather(
        1, idx[:, None, None].expand(-1, 1, y.size(-1))
    ).squeeze(1)


def evaluate(seed):
    torch.set_num_threads(2)
    report = {}
    for mode in ("iid", "ood_strong", "ood_quadratic", "ood_switch"):
        set_seed(seed + 9000)
        pool_mse, ridge_mse = [], []
        with torch.no_grad():
            for _ in range(8):
                hist, y, query, target = sample_worlds(
                    128, 4, 12, "cpu", mode
                )
                pool_mse.append(F.mse_loss(
                    predict(hist, y, query), target
                ).item())
                ridge_mse.append(F.mse_loss(
                    fit(hist, y, query), target
                ).item())
        report[mode] = {
            "pool_mse": round(sum(pool_mse) / len(pool_mse), 6),
            "ridge_all_mse": round(sum(ridge_mse) / len(ridge_mse), 6),
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=8401)
    args = parser.parse_args()
    print(json.dumps({"seed": args.seed, "results": evaluate(args.seed)}, indent=2))
