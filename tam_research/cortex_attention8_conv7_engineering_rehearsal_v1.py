"""Synthetic-only, zero-GPU rehearsal of Conv7 engineering screening.

Never accepts actual research records. A PASS means that mock decision logic
worked, not that training happened. No Modal, checkpoint, or seed side effects.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_PATH = ROOT / "research/reduced_attention/attention8_conv7_engineering_proposal_v1.json"
SYNTHETIC_TAG = "SYNTHETIC_ONLY_NO_TRAINING"


def classify_synthetic_screen(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    proposal = json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))
    if proposal["stage"] != "ZERO_GPU_PROPOSAL_NOT_PREREGISTERED":
        raise RuntimeError("proposal is no longer an unapproved draft")
    if not all(value is False for value in proposal["authority"].values()):
        raise RuntimeError("training authority drift in synthetic rehearsal")
    pilot = proposal["draft_engineering_pilot"]
    if pilot["engineering_seed"] is not None or pilot["strict_max_gpu_spend_usd"] is not None:
        raise RuntimeError("synthetic rehearsal must not run alongside an allocated seed/budget")
    if pilot["modal_account"] != "primary":
        raise RuntimeError("unexpected workspace in frozen proposal")
    expected = pilot["comparators"]
    gates = proposal["draft_screen_gates"]
    envelope = {
        "synthetic_only": True,
        "engineering_result_recorded": False,
        "scientific_evidence": False,
        "gpu_allocated": False,
        "writes_performed": False,
        "seed_consumed": False,
        "replication_authorized": False,
        "scale_250m_5b_authorized": False,
        "breakthrough_claim_allowed": False,
    }

    def stop(reason: str) -> dict[str, Any]:
        return {**envelope, "classification": "SYNTHETIC_SCREEN_REJECTED", "reason": reason}

    if not isinstance(records, Mapping) or set(records) != set(expected):
        return stop("missing_or_extra_comparator")
    for name in expected:
        record = records[name]
        if not isinstance(record, Mapping):
            return stop("not_a_record")
        if record.get("synthetic_tag") != SYNTHETIC_TAG:
            return stop("not_marked_synthetic")
        if record.get("model_name") != name:
            return stop("model_identity_mismatch")
        if record.get("status") != "COMPLETE":
            return stop("incomplete_result")
        if record.get("optimizer_steps") != pilot["per_model_optimizer_steps"]:
            return stop("optimizer_steps_mismatch")
        if record.get("token_exposures") != pilot["per_model_token_exposures"]:
            return stop("token_exposures_mismatch")
        if record.get("heldout_tokens") != pilot["final_evaluation_batches"] * pilot["evaluation_batch_size"] * pilot["sequence_length"]:
            return stop("evaluation_geometry_mismatch")
        if record.get("lr_horizon_steps") != pilot["full_horizon_optimizer_steps"]:
            return stop("optimizer_horizon_mismatch")
        if record.get("seed") != SYNTHETIC_TAG:
            return stop("seed_mismatch")
        if record.get("heldout_stream") != SYNTHETIC_TAG:
            return stop("evaluation_stream_mismatch")
        for field in ("nll", "training_tokens_per_second"):
            value = record.get(field)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                return stop("nonfinite_or_invalid_metric")

    transformer, parent, candidate = (records[name] for name in expected)
    nll_gap = candidate["nll"] - transformer["nll"]
    parent_improvement = parent["nll"] - candidate["nll"]
    throughput_ratio = (
        candidate["training_tokens_per_second"] /
        transformer["training_tokens_per_second"]
    )
    quality_pass = nll_gap <= gates["transformer_comparison_max_nll_delta"]
    parent_pass = parent_improvement >= gates["attention8_parent_min_quality_improvement_nll"]
    systems_pass = throughput_ratio >= gates["transformer_min_training_tokens_per_second_ratio"]
    all_pass = quality_pass and parent_pass and systems_pass
    return {
        **envelope,
        "classification": (
            "SYNTHETIC_ENGINEERING_SCREEN_PASS_NOT_SCIENTIFIC"
            if all_pass else "SYNTHETIC_ENGINEERING_SCREEN_FAIL"
        ),
        "quality_gate_pass": quality_pass,
        "parent_improvement_gate_pass": parent_pass,
        "systems_gate_pass": systems_pass,
        "nll_delta_candidate_minus_transformer": nll_gap,
        "nll_improvement_parent_minus_candidate": parent_improvement,
        "training_tps_candidate_over_transformer": throughput_ratio,
        "would_pass_draft_screen": all_pass,
    }
