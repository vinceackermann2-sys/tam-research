from __future__ import annotations

"""Post-run causal predictability audit for AERA-v14 seed8271.

The v14 development gate can pass its mean-compute target without proving that
compute is allocated to the examples that actually need it.  This diagnostic
loads the frozen checkpoint and asks two narrower questions without retraining the
model:

1. Can the declared difficulty labels be predicted from the exact causal feature
   vector available to each optional stage router (current chunk first event +
   carried stream state)?
2. How unstable are those labels under re-batching, given that v12-v14 define
   difficulty by ranking each example against the other examples in its minibatch?

A fresh held-out linear probe uses the same linear capacity class as the actual
StageRouteGate.  The model itself is frozen.  No GPU or optimizer step on AERA is
required.
"""

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .aera_real_language import SEQ_LEN
from .aera_real_language_v12 import per_chunk_language_loss, validate_production_data
from .aera_real_language_v14 import build_aera
from .data import TokenBin

AUDIT_SEED = 82_711
DEFAULT_BATCHES = 24
DEFAULT_BATCH_SIZE = 8
PROBE_STEPS = 300
PROBE_LR = 0.05


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().float().flatten()
    labels = labels.detach().bool().flatten()
    pos = scores[labels]
    neg = scores[~labels]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    # Pairwise definition handles ties exactly and is cheap at this audit scale.
    cmp = (pos[:, None] > neg[None, :]).float()
    ties = (pos[:, None] == neg[None, :]).float()
    return float((cmp + 0.5 * ties).mean())


def balanced_accuracy(scores: torch.Tensor, labels: torch.Tensor, *, threshold: float = 0.5) -> float:
    pred = scores.detach().float().flatten() >= threshold
    truth = labels.detach().bool().flatten()
    pos = truth
    neg = ~truth
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    tpr = (pred[pos] == truth[pos]).float().mean()
    tnr = (pred[neg] == truth[neg]).float().mean()
    return float(0.5 * (tpr + tnr))


def fit_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    train_x = train_x.float()
    test_x = test_x.float()
    train_y = train_y.float().flatten()
    test_y = test_y.float().flatten()

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    weight = torch.zeros(train_x.size(1), requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam((weight, bias), lr=PROBE_LR)
    positives = train_y.sum().clamp_min(1.0)
    negatives = (1.0 - train_y).sum().clamp_min(1.0)
    pos_weight = (negatives / positives).detach()

    for _ in range(PROBE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = train_x @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = test_x @ weight + bias
        test_prob = torch.sigmoid(test_logits)
    return {
        "auc": binary_auc(test_prob, test_y),
        "balanced_accuracy": balanced_accuracy(test_prob, test_y),
        "positive_rate": float(test_y.mean()),
    }


def _split_sequences(n_sequences: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n_sequences < 8:
        raise ValueError("causal audit needs at least 8 sequences")
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_sequences, generator=g)
    n_train = max(4, int(0.70 * n_sequences))
    n_train = min(n_train, n_sequences - 2)
    return order[:n_train], order[n_train:]


def _rebatch_targets_from_losses(
    losses: torch.Tensor,
    *,
    run_rates: tuple[float, ...],
    batch_size: int,
    repeats: int,
    seed: int,
) -> torch.Tensor:
    """Return [repeats,N,stages] labels after random peer regrouping."""
    if losses.ndim != 1:
        raise ValueError("losses must be [N]")
    n = losses.numel()
    if n % batch_size:
        raise ValueError("N must divide exactly by batch_size for rebatching")
    g = torch.Generator().manual_seed(seed)
    all_labels: list[torch.Tensor] = []
    for _ in range(repeats):
        order = torch.randperm(n, generator=g)
        labels = torch.empty((n, len(run_rates)), dtype=torch.float32)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            values = losses.index_select(0, idx)
            rank_order = torch.argsort(values, stable=True)
            ranks = torch.empty_like(values)
            ranks[rank_order] = torch.arange(batch_size, dtype=values.dtype)
            hardness = (ranks + 0.5) / float(batch_size)
            group_targets = torch.stack(
                [(hardness >= 1.0 - float(rate)).float() for rate in run_rates],
                dim=1,
            )
            labels.index_copy_(0, idx, group_targets)
        all_labels.append(labels)
    return torch.stack(all_labels, dim=0)


def rebatching_flip_metrics(
    losses: torch.Tensor,
    *,
    run_rates: tuple[float, ...],
    batch_size: int,
    repeats: int = 32,
    seed: int = AUDIT_SEED,
) -> list[dict[str, float]]:
    labels = _rebatch_targets_from_losses(
        losses.float(),
        run_rates=run_rates,
        batch_size=batch_size,
        repeats=repeats,
        seed=seed,
    )
    metrics: list[dict[str, float]] = []
    for stage in range(labels.size(2)):
        s = labels[:, :, stage]
        per_example_rate = s.mean(dim=0)
        unstable = ((per_example_rate > 0.0) & (per_example_rate < 1.0)).float().mean()
        # Two independently sampled batch contexts for the same example disagree
        # with probability 2*p*(1-p).
        pairwise_flip = (2.0 * per_example_rate * (1.0 - per_example_rate)).mean()
        metrics.append(
            {
                "stage": float(stage + 1),
                "unstable_example_fraction": float(unstable),
                "expected_pairwise_label_flip_rate": float(pairwise_flip),
                "mean_positive_rate": float(s.mean()),
            }
        )
    return metrics


def collect_dataset(
    model,
    val: TokenBin,
    *,
    batches: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    if device.type != "cpu":
        raise RuntimeError("causal router audit is intentionally CPU-only")
    chunks = SEQ_LEN // model.cfg.chunk_size
    optional = model.cfg.n_stages - 1
    if optional != len(model.OPTIONAL_STAGE_RUN_RATES):
        raise RuntimeError("optional-stage schedule mismatch")

    current: dict[int, list[torch.Tensor]] = {i: [] for i in range(1, model.cfg.n_stages)}
    handles = []
    for stage_index, router in enumerate(model.stage_routers):
        if stage_index == model.FOUNDATION_STAGE:
            continue

        def make_hook(idx: int):
            def hook(_module, args):
                first_event, stream = args[:2]
                current[idx].append(
                    torch.cat((first_event.detach().float(), stream.detach().float()), dim=-1).cpu()
                )
            return hook

        handles.append(router.register_forward_pre_hook(make_hook(stage_index)))

    feature_rows: dict[int, list[torch.Tensor]] = {i: [] for i in current}
    label_rows: dict[int, list[torch.Tensor]] = {i: [] for i in current}
    probability_rows: dict[int, list[torch.Tensor]] = {i: [] for i in current}
    all_chunk_losses: list[torch.Tensor] = []
    g = torch.Generator(device="cpu").manual_seed(seed)

    try:
        for _ in range(batches):
            for values in current.values():
                values.clear()
            x, y = val.batch(batch_size, SEQ_LEN, g, device)
            with torch.no_grad():
                model.set_router_task_gradient_isolation(True)
                out = model(x, hard=False, route_mode="straight_through", update_memory=False)
            logits = out["logits"]
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("AERA output missing logits")
            chunk_losses = per_chunk_language_loss(logits, y)
            targets = model.chunk_difficulty_stage_targets(chunk_losses).view(batch_size, chunks, optional)
            probabilities = model.optional_stage_probabilities(out).detach().cpu().view(
                batch_size, chunks, optional
            )
            all_chunk_losses.append(chunk_losses.detach().cpu().reshape(-1))

            for stage_index in range(1, model.cfg.n_stages):
                captures = current[stage_index]
                if len(captures) != chunks:
                    raise RuntimeError(
                        f"stage {stage_index} captured {len(captures)} chunks, expected {chunks}"
                    )
                features = torch.stack(captures, dim=1)  # [B,chunks,2*d]
                feature_rows[stage_index].append(features)
                label_rows[stage_index].append(targets[:, :, stage_index - 1].cpu())
                probability_rows[stage_index].append(probabilities[:, :, stage_index - 1])
    finally:
        for handle in handles:
            handle.remove()

    return {
        "features": {k: torch.cat(v, dim=0) for k, v in feature_rows.items()},
        "labels": {k: torch.cat(v, dim=0) for k, v in label_rows.items()},
        "probabilities": {k: torch.cat(v, dim=0) for k, v in probability_rows.items()},
        "chunk_losses": torch.cat(all_chunk_losses, dim=0),
    }


def audit_checkpoint(
    *,
    checkpoint_path: str,
    data_dir: str,
    batches: int = DEFAULT_BATCHES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = AUDIT_SEED,
) -> dict[str, Any]:
    if batch_size != 8:
        raise ValueError("audit batch_size is frozen to 8 to match the training label quantization")
    validate_production_data(data_dir)
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"v14 checkpoint missing: {checkpoint}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_aera(torch.device("cpu"))
    model.load_state_dict(payload["model"])
    val = TokenBin(str(Path(data_dir) / "val.bin"))
    dataset = collect_dataset(model, val, batches=batches, batch_size=batch_size, seed=seed)

    n_sequences = next(iter(dataset["features"].values())).size(0)
    train_idx, test_idx = _split_sequences(n_sequences, seed=seed + 1)
    stage_results: list[dict[str, Any]] = []
    for stage_index in range(1, model.cfg.n_stages):
        features = dataset["features"][stage_index]
        labels = dataset["labels"][stage_index]
        probs = dataset["probabilities"][stage_index]

        train_x = features.index_select(0, train_idx).reshape(-1, features.size(-1))
        train_y = labels.index_select(0, train_idx).reshape(-1)
        test_x = features.index_select(0, test_idx).reshape(-1, features.size(-1))
        test_y = labels.index_select(0, test_idx).reshape(-1)
        test_prob = probs.index_select(0, test_idx).reshape(-1)

        probe = fit_linear_probe(
            train_x,
            train_y,
            test_x,
            test_y,
            seed=seed + 100 + stage_index,
        )
        trained = {
            "auc": binary_auc(test_prob, test_y),
            "balanced_accuracy": balanced_accuracy(test_prob, test_y),
            "hard_run_fraction": float((test_prob >= 0.5).float().mean()),
            "mean_probability": float(test_prob.mean()),
        }
        stage_results.append(
            {
                "optional_stage": stage_index,
                "target_run_rate": float(model.OPTIONAL_STAGE_RUN_RATES[stage_index - 1]),
                "heldout_target_positive_rate": float(test_y.float().mean()),
                "trained_router": trained,
                "fresh_linear_probe": probe,
                "probe_minus_router_auc": probe["auc"] - trained["auc"],
            }
        )

    probe_aucs = [row["fresh_linear_probe"]["auc"] for row in stage_results]
    probe_bacc = [row["fresh_linear_probe"]["balanced_accuracy"] for row in stage_results]
    router_aucs = [row["trained_router"]["auc"] for row in stage_results]
    mean_probe_auc = sum(probe_aucs) / len(probe_aucs)
    mean_probe_bacc = sum(probe_bacc) / len(probe_bacc)
    mean_router_auc = sum(router_aucs) / len(router_aucs)

    if all(a >= 0.65 for a in probe_aucs) and mean_probe_bacc >= 0.60:
        signal = "strong"
    elif mean_probe_auc < 0.58 or any(a < 0.55 for a in probe_aucs):
        signal = "weak"
    else:
        signal = "mixed"

    if signal == "weak":
        interpretation = "target_information_mismatch_likely"
    elif mean_probe_auc - mean_router_auc >= 0.08:
        interpretation = "optimization_or_router_training_mismatch_likely"
    else:
        interpretation = "causal_signal_present_and_router_tracks_target"

    rebatching = rebatching_flip_metrics(
        dataset["chunk_losses"],
        run_rates=tuple(float(x) for x in model.OPTIONAL_STAGE_RUN_RATES),
        batch_size=batch_size,
        repeats=32,
        seed=seed + 2,
    )

    return {
        "audit": "aera-v14-causal-router-predictability",
        "checkpoint": str(checkpoint),
        "seed": seed,
        "cpu_only": True,
        "batches": batches,
        "batch_size": batch_size,
        "sequences": n_sequences,
        "chunks_per_sequence": SEQ_LEN // model.cfg.chunk_size,
        "stage_results": stage_results,
        "mean_fresh_probe_auc": mean_probe_auc,
        "mean_fresh_probe_balanced_accuracy": mean_probe_bacc,
        "mean_trained_router_auc": mean_router_auc,
        "causal_target_signal": signal,
        "interpretation": interpretation,
        "rebatching_label_instability": rebatching,
        "claims": {
            "diagnostic_only": True,
            "no_aera_parameter_update": True,
            "gpu_used": False,
            "100m_authorized": False,
        },
    }


def save_audit(result: dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(result, indent=2))
