from __future__ import annotations

"""Authoritative #381 systems wrapper with the frozen v25↔v25.1 logit gate.

This module does not change the timed systems protocol. It runs the already-frozen
comparison, then performs one untimed full-FICEM equivalence call per implementation
for each registered batch on the exact same fixed random tokens. The additional
result is required for PASS at atol/rtol <= 1e-6, as preregistered by issue #381.
"""

from typing import Any

import torch

from . import aera_v25_1_systems as systems
from . import aera_v25_post8471_triage as triage


def _parameter_versions(model: torch.nn.Module) -> tuple[int, ...]:
    return tuple(int(parameter._version) for parameter in model.parameters())


def _fixed_system_tokens(batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size not in systems.SYSTEM_BATCH_SIZES:
        raise ValueError(f"unregistered issue381 batch size: {batch_size}")
    generator = torch.Generator(device="cpu").manual_seed(
        triage.DIAGNOSTIC_SEED + 10_000 + batch_size
    )
    return torch.randint(
        0,
        triage.VOCAB_SIZE,
        (batch_size, triage.SEQ_LEN),
        generator=generator,
    ).to(device)


def _logit_equivalence(
    original,
    candidate,
    tokens: torch.Tensor,
) -> dict[str, Any]:
    original_output = systems._model_call(original, tokens, update_memory=True)
    candidate_output = systems._model_call(candidate, tokens, update_memory=True)
    original_logits = original_output.get("logits")
    candidate_logits = candidate_output.get("logits")
    if not isinstance(original_logits, torch.Tensor) or not isinstance(
        candidate_logits, torch.Tensor
    ):
        raise RuntimeError("issue381 logit gate requires logits from both models")
    if original_logits.shape != candidate_logits.shape:
        return {
            "pass": False,
            "reason": "shape_mismatch",
            "original_shape": list(original_logits.shape),
            "candidate_shape": list(candidate_logits.shape),
            "atol": systems.CPU_EQUIVALENCE_ATOL,
            "rtol": systems.CPU_EQUIVALENCE_RTOL,
        }

    original_float = original_logits.float()
    candidate_float = candidate_logits.float()
    delta = (original_float - candidate_float).abs()
    maximum = float(delta.max()) if delta.numel() else 0.0
    mean = float(delta.mean()) if delta.numel() else 0.0
    equivalent = bool(
        torch.allclose(
            original_float,
            candidate_float,
            atol=systems.CPU_EQUIVALENCE_ATOL,
            rtol=systems.CPU_EQUIVALENCE_RTOL,
        )
    )
    return {
        "pass": equivalent,
        "atol": systems.CPU_EQUIVALENCE_ATOL,
        "rtol": systems.CPU_EQUIVALENCE_RTOL,
        "max_abs": maximum,
        "mean_abs": mean,
        "shape": list(original_logits.shape),
    }


def _apply_logit_gate(
    result: dict[str, Any],
    equivalence_rows: dict[str, dict[str, Any]],
    *,
    guard_parameter_versions_unchanged: bool,
    hashes_before_guard: dict[str, str],
    hashes_after_guard: dict[str, str],
) -> dict[str, Any]:
    rows = result.get("rows")
    per_batch_pass = result.get("per_batch_pass")
    if not isinstance(rows, dict) or not isinstance(per_batch_pass, dict):
        raise RuntimeError("issue381 base systems result missing rows/pass map")

    for batch_size in systems.SYSTEM_BATCH_SIZES:
        batch = str(batch_size)
        if batch not in rows or batch not in per_batch_pass or batch not in equivalence_rows:
            raise RuntimeError(f"issue381 base systems result missing batch {batch}")
        row = rows[batch]
        if not isinstance(row, dict):
            raise RuntimeError(f"issue381 batch {batch} row is not a mapping")
        equivalence = equivalence_rows[batch]
        row["logit_equivalence"] = equivalence
        per_batch_pass[batch] = bool(
            per_batch_pass[batch] and equivalence.get("pass") is True
        )

    base_hashes_before = result.get("checkpoint_hashes_before")
    checkpoint_hashes_unchanged = bool(
        result.get("checkpoint_hashes_unchanged") is True
        and isinstance(base_hashes_before, dict)
        and base_hashes_before == hashes_before_guard == hashes_after_guard
    )
    result["checkpoint_hashes_after"] = hashes_after_guard
    result["checkpoint_hashes_unchanged"] = checkpoint_hashes_unchanged
    result["logit_equivalence_required"] = True
    result["logit_guard_parameter_versions_unchanged"] = bool(
        guard_parameter_versions_unchanged
    )
    result["per_batch_pass"] = per_batch_pass

    overall_pass = bool(
        all(bool(value) for value in per_batch_pass.values())
        and result.get("parameter_versions_unchanged") is True
        and guard_parameter_versions_unchanged
        and checkpoint_hashes_unchanged
    )
    result["overall_pass"] = overall_pass
    result["decision"] = (
        "PASS_SYSTEMS_EQUIVALENT_CANDIDATE"
        if overall_pass
        else "FAIL_FROZEN_SYSTEMS_GATE"
    )
    return result


@torch.inference_mode()
def run_guarded_systems_comparison(*, run_dir: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("issue381 guarded systems comparison requires one NVIDIA L4")
    device = torch.device("cuda")

    hashes_before_guard = systems.checkpoint_hashes(run_dir)
    result = systems.run_systems_comparison(run_dir=run_dir)

    original, candidate, transformer = systems.load_models(
        run_dir=run_dir,
        device=device,
    )
    del transformer
    original_before = _parameter_versions(original)
    candidate_before = _parameter_versions(candidate)

    equivalence_rows: dict[str, dict[str, Any]] = {}
    for batch_size in systems.SYSTEM_BATCH_SIZES:
        tokens = _fixed_system_tokens(batch_size, device)
        equivalence_rows[str(batch_size)] = _logit_equivalence(
            original,
            candidate,
            tokens,
        )
        del tokens

    original_after = _parameter_versions(original)
    candidate_after = _parameter_versions(candidate)
    versions_unchanged = bool(
        original_before == original_after and candidate_before == candidate_after
    )
    del original, candidate
    torch.cuda.empty_cache()

    hashes_after_guard = systems.checkpoint_hashes(run_dir)
    return _apply_logit_gate(
        result,
        equivalence_rows,
        guard_parameter_versions_unchanged=versions_unchanged,
        hashes_before_guard=hashes_before_guard,
        hashes_after_guard=hashes_after_guard,
    )
