"""Training-free structural-prior ablation for the SI EHW pilot."""
from __future__ import annotations
import argparse
import json
import torch
import torch.nn.functional as F
from si_ehw_pilot import sample_worlds, set_seed


def evaluate(seed: int, dim: int = 4, shots: int = 12, ridge: float = 0.14):
    result = {}
    for mode in ("iid", "ood_strong", "ood_quadratic", "ood_switch"):
        set_seed(seed + 9000)
        errors = []
        with torch.no_grad():
            for _ in range(8):
                hist, y, query, target = sample_worlds(128, dim, shots, "cpu", mode)
                xt = hist.transpose(1, 2)
                weights = torch.linalg.solve(
                    xt @ hist + ridge * torch.eye(2 * dim), xt @ y
                )
                pred = torch.bmm(query.unsqueeze(1), weights).squeeze(1)
                errors.append(F.mse_loss(pred, target).item())
        result[mode] = round(sum(errors) / len(errors), 6)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=8117)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps({
        "seed": args.seed,
        "ridge_only_eval_mse": evaluate(args.seed),
    }, indent=2))
