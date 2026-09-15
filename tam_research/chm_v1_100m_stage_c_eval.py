from __future__ import annotations

"""Pre-result CHM-v1 ~100M Stage-C evaluator contract (#986).

CPU/CI evaluation engineering only.  This module freezes the aligned v4 probe
view, candidate-set scoring, exact flat final-token EIEM scoring, deterministic
stratified bootstrap, and the pure Stage-C PASS/STOP classifier.  It contains no
training launcher, paid-compute trigger, or scientific-seed execution authority.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from .chm_v1_100m_scale import (
    EXPECTED_EIEM_PARAMETERS,
    EXPECTED_LOCAL_PARAMETERS,
    FIRST_SCREEN_TOKEN_BUDGET,
    FUTURE_SCREEN_SEED,
    PARAMETER_MISMATCH_LIMIT,
)
from .chm_v1_long_memory_eval import EncodedProbe
from .chm_v1_long_memory_eval_v3 import generate_probe_suite as generate_v3_probe_suite
from .chm_v1_small_lm import EpisodicState, LOCAL_WINDOW, RETRIEVAL_HOPS, _hidden

ISSUE = 986
GENERATOR_VERSION = "chm-v1-100m-heldout-aligned-v4"
PROBE_SEED = 977_301
VALIDATION_SEED = 977_302
BOOTSTRAP_SEED = 977_303
RESERVED_SCIENTIFIC_SEED = FUTURE_SCREEN_SEED

LONG_RANGE_FAMILIES = ("rare_fact", "overwrite", "two_hop")
ALL_FAMILIES = LONG_RANGE_FAMILIES + ("local_negative",)
CASES_PER_FAMILY = 128
TOTAL_PROBES = CASES_PER_FAMILY * len(ALL_FAMILIES)
LONG_RANGE_PROBES = CASES_PER_FAMILY * len(LONG_RANGE_FAMILIES)
CANDIDATE_COUNT = 8
EXPECTED_PER_CANDIDATE_PER_FAMILY = CASES_PER_FAMILY // CANDIDATE_COUNT

VALIDATION_TOKENS = 1_048_576
VALIDATION_BATCHES = 128
VALIDATION_BATCH_SIZE = 8
VALIDATION_SESSION_LEN = 1_024
BOOTSTRAP_RESAMPLES = 10_000

MIN_AGGREGATE_ACCURACY_GAIN = 0.05
MIN_AGGREGATE_CANDIDATE_NLL_BENEFIT = 0.05
MIN_FAMILY_ACCURACY_GAIN = 0.03
MIN_LONG_RANGE_FAMILIES_AT_GAIN = 2
MIN_ANY_FAMILY_ACCURACY_GAIN = -0.01
MIN_LONG_RANGE_FAMILIES_POSITIVE_NLL = 2
MAX_LANGUAGE_NLL_REGRESSION = 0.03
MIN_LOCAL_NEGATIVE_ACCURACY_GAIN = -0.02
MAX_LOCAL_NEGATIVE_CANDIDATE_NLL_REGRESSION = 0.03

REQUIRED_INTEGRITY_FLAGS = (
    "backbone_initialization_identical",
    "byte_identical_training_stream",
    "matched_optimizer_schedule",
    "finite_losses",
    "finite_parameters",
    "no_cross_session_state_aliasing",
    "no_future_self_leakage",
)


def protocol_manifest() -> dict[str, Any]:
    return {
        "classification": "CHM_V1_100M_STAGE_C_EVALUATION_PREREGISTRATION_NO_EXECUTION_AUTHORITY",
        "issue": ISSUE,
        "generator_version": GENERATOR_VERSION,
        "probe_seed": PROBE_SEED,
        "validation_seed": VALIDATION_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cases_per_family": CASES_PER_FAMILY,
        "total_probes": TOTAL_PROBES,
        "long_range_probes": LONG_RANGE_PROBES,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "validation_tokens": VALIDATION_TOKENS,
        "validation_batches": VALIDATION_BATCHES,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "validation_session_len": VALIDATION_SESSION_LEN,
        "scientific_seed_reserved_not_authorized": RESERVED_SCIENTIFIC_SEED,
        "screen_training_tokens_per_model_not_authorized": FIRST_SCREEN_TOKEN_BUDGET,
        "gpu_authorized": False,
        "modal_authorized": False,
        "training_authorized": False,
        "scientific_execution_authorized": False,
    }


def validate_protocol_manifest() -> dict[str, Any]:
    if LOCAL_WINDOW != 512:
        raise RuntimeError("#986 requires the inherited 512-token local window")
    if CASES_PER_FAMILY != 128 or TOTAL_PROBES != 512 or LONG_RANGE_PROBES != 384:
        raise RuntimeError("#986 probe envelope drift")
    if EXPECTED_PER_CANDIDATE_PER_FAMILY != 16:
        raise RuntimeError("#986 candidate balance drift")
    if (VALIDATION_BATCHES * VALIDATION_BATCH_SIZE * VALIDATION_SESSION_LEN) != VALIDATION_TOKENS:
        raise RuntimeError("#986 validation-token accounting drift")
    if BOOTSTRAP_RESAMPLES != 10_000:
        raise RuntimeError("#986 bootstrap count drift")
    return protocol_manifest()


def generate_aligned_probe_suite(
    encode: Callable[[str], Sequence[int]],
    *,
    seed: int = PROBE_SEED,
    cases_per_family: int = CASES_PER_FAMILY,
) -> list[EncodedProbe]:
    """Reuse v3 probe content exactly while binding the v4 scoring contract."""
    if int(seed) != PROBE_SEED:
        raise RuntimeError(f"#986 accepts only probe seed {PROBE_SEED}")
    if int(cases_per_family) != CASES_PER_FAMILY:
        raise RuntimeError(f"#986 requires exactly {CASES_PER_FAMILY} cases/family")

    v3 = generate_v3_probe_suite(
        encode,
        seed=PROBE_SEED,
        cases_per_family=CASES_PER_FAMILY,
    )
    suite = [replace(probe, generator_version=GENERATOR_VERSION) for probe in v3]
    if len(suite) != TOTAL_PROBES:
        raise RuntimeError(f"#986 probe count drift: {len(suite)} != {TOTAL_PROBES}")

    by_family: dict[str, list[EncodedProbe]] = defaultdict(list)
    for probe in suite:
        probe.validate()
        if probe.used_for_training:
            raise RuntimeError("#986 scientific probes must remain held out")
        by_family[probe.family].append(probe)

    if set(by_family) != set(ALL_FAMILIES):
        raise RuntimeError(f"#986 family set drift: {sorted(by_family)}")
    for family in ALL_FAMILIES:
        rows = by_family[family]
        if len(rows) != CASES_PER_FAMILY:
            raise RuntimeError(f"#986 {family} count drift: {len(rows)}")
        candidate_ids = rows[0].candidate_token_ids
        if len(candidate_ids) != CANDIDATE_COUNT:
            raise RuntimeError(f"#986 {family} candidate count drift")
        counts = Counter(row.answer_token_id for row in rows)
        if set(counts) != set(candidate_ids) or any(
            count != EXPECTED_PER_CANDIDATE_PER_FAMILY for count in counts.values()
        ):
            raise RuntimeError(f"#986 {family} answer balance drift: {dict(counts)}")
    return suite


def aligned_local_token_ids(probe: EncodedProbe) -> tuple[int, ...]:
    """Return exactly the fixed 512-token partition containing the query."""
    if probe.query_token != len(probe.prompt_ids) - 1:
        raise ValueError("#986 scoring requires the query at the final prompt position")
    start = (probe.query_token // LOCAL_WINDOW) * LOCAL_WINDOW
    aligned = tuple(probe.prompt_ids[start : probe.query_token + 1])
    if not aligned or len(aligned) > LOCAL_WINDOW:
        raise RuntimeError("aligned local scoring slice escaped the inherited local window")
    return aligned


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def local_aligned_final_logits(model: torch.nn.Module, probe: EncodedProbe) -> torch.Tensor:
    ids = aligned_local_token_ids(probe)
    device = _model_device(model)
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(tokens)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(ids):
        raise RuntimeError("LOCAL evaluator returned unexpected logit shape")
    return logits[0, -1]


@torch.no_grad()
def eiem_flat_final_logits(model: torch.nn.Module, prompt_ids: Sequence[int]) -> torch.Tensor:
    """Exact final-token flat scorer with no indexed/clipped retrieval.

    Earlier chunks only need to write raw keys/hidden values: retrieval does not
    alter those writes in CHM-v1.  Therefore final-token-only scoring is exactly
    equivalent to the full flat evaluator at the scored query while avoiding
    unnecessary retrieval for unscored tokens.
    """
    ids = tuple(int(token) for token in prompt_ids)
    if not ids:
        raise ValueError("prompt_ids must be non-empty")
    device = _model_device(model)
    state = EpisodicState("stage-c-986-final-token")

    for start in range(0, len(ids), LOCAL_WINDOW):
        chunk_ids = ids[start : start + LOCAL_WINDOW]
        tokens = torch.tensor(chunk_ids, dtype=torch.long, device=device).unsqueeze(0)
        hidden = _hidden(model.backbone, tokens)
        is_final = start + len(chunk_ids) == len(ids)
        if not is_final:
            keys = model.key_for(hidden)
            state.write(keys[0], hidden[0])
            continue

        query_state = hidden[0, -1]
        if len(state):
            for _ in range(RETRIEVAL_HOPS):
                query = model.query_for(query_state)
                value, _, _, _, _, _ = state.retrieve(
                    query,
                    mode="flat",
                    verify_indexed_exactness=False,
                )
                query_state = model._integrate(query_state, value)
        return model.backbone.lm_head(query_state)

    raise RuntimeError("final EIEM chunk was not reached")


def candidate_score(
    logits: torch.Tensor,
    *,
    candidate_token_ids: Sequence[int],
    answer_token_id: int,
    stale_token_ids: Sequence[int] = (),
) -> dict[str, Any]:
    if logits.ndim != 1:
        raise ValueError("candidate scoring expects one full-vocabulary logit vector")
    candidates = tuple(int(token) for token in candidate_token_ids)
    if len(candidates) != CANDIDATE_COUNT or len(set(candidates)) != CANDIDATE_COUNT:
        raise ValueError("#986 requires exactly eight unique candidate IDs")
    answer = int(answer_token_id)
    if answer not in candidates:
        raise ValueError("answer token is absent from the frozen candidate set")

    index = torch.tensor(candidates, device=logits.device, dtype=torch.long)
    candidate_logits = logits.index_select(0, index).float()
    true_index = candidates.index(answer)
    log_probs = torch.log_softmax(candidate_logits, dim=0)
    predicted_index = int(torch.argmax(candidate_logits).item())
    predicted_token_id = candidates[predicted_index]
    return {
        "predicted_token_id": predicted_token_id,
        "correct": predicted_token_id == answer,
        "candidate_nll": float((-log_probs[true_index]).item()),
        "stale_choice": predicted_token_id in {int(x) for x in stale_token_ids},
    }


def paired_probe_row(
    *,
    probe: EncodedProbe,
    local_logits: torch.Tensor,
    eiem_logits: torch.Tensor,
) -> dict[str, Any]:
    local = candidate_score(
        local_logits,
        candidate_token_ids=probe.candidate_token_ids,
        answer_token_id=probe.answer_token_id,
        stale_token_ids=probe.stale_token_ids,
    )
    eiem = candidate_score(
        eiem_logits,
        candidate_token_ids=probe.candidate_token_ids,
        answer_token_id=probe.answer_token_id,
        stale_token_ids=probe.stale_token_ids,
    )
    return {
        "family": probe.family,
        "case_id": probe.case_id,
        "generator_version": probe.generator_version,
        "evidence_distance": probe.evidence_distance,
        "query_chunk_index": probe.query_token // LOCAL_WINDOW,
        "local_correct": bool(local["correct"]),
        "eiem_correct": bool(eiem["correct"]),
        "local_candidate_nll": float(local["candidate_nll"]),
        "eiem_candidate_nll": float(eiem["candidate_nll"]),
        "local_stale_choice": bool(local["stale_choice"]),
        "eiem_stale_choice": bool(eiem["stale_choice"]),
    }


def _validated_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    if len(rows) != TOTAL_PROBES:
        raise ValueError(f"#986 requires exactly {TOTAL_PROBES} scored probe rows")
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in rows:
        family = str(row.get("family"))
        if family not in ALL_FAMILIES:
            raise ValueError(f"unexpected #986 family {family!r}")
        if row.get("generator_version") != GENERATOR_VERSION:
            raise ValueError("#986 scored row has wrong evaluator version")
        case_id = int(row.get("case_id", -1))
        key = (family, case_id)
        if key in seen:
            raise ValueError(f"duplicate #986 scored case {key}")
        seen.add(key)
        for field in ("local_candidate_nll", "eiem_candidate_nll"):
            value = float(row[field])
            if not np.isfinite(value):
                raise ValueError(f"non-finite #986 score {field}")
        for field in ("local_correct", "eiem_correct", "local_stale_choice", "eiem_stale_choice"):
            if not isinstance(row.get(field), (bool, np.bool_)):
                raise ValueError(f"#986 {field} must be boolean")
        by_family[family].append(row)
    for family in ALL_FAMILIES:
        if len(by_family[family]) != CASES_PER_FAMILY:
            raise ValueError(f"#986 {family} scored-row count drift")
    return by_family


def _row_accuracy_gain(row: Mapping[str, Any]) -> float:
    return float(bool(row["eiem_correct"])) - float(bool(row["local_correct"]))


def _row_nll_benefit(row: Mapping[str, Any]) -> float:
    return float(row["local_candidate_nll"]) - float(row["eiem_candidate_nll"])


def stratified_paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, dict[str, float]]:
    if int(seed) != BOOTSTRAP_SEED:
        raise RuntimeError(f"#986 accepts only bootstrap seed {BOOTSTRAP_SEED}")
    if int(resamples) != BOOTSTRAP_RESAMPLES:
        raise RuntimeError(f"#986 requires exactly {BOOTSTRAP_RESAMPLES} bootstrap resamples")
    by_family = _validated_rows(rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    acc_means: list[np.ndarray] = []
    nll_means: list[np.ndarray] = []
    for family in LONG_RANGE_FAMILIES:
        family_rows = by_family[family]
        acc = np.asarray([_row_accuracy_gain(row) for row in family_rows], dtype=np.float64)
        nll = np.asarray([_row_nll_benefit(row) for row in family_rows], dtype=np.float64)
        indices = rng.integers(0, CASES_PER_FAMILY, size=(BOOTSTRAP_RESAMPLES, CASES_PER_FAMILY))
        acc_means.append(acc[indices].mean(axis=1))
        nll_means.append(nll[indices].mean(axis=1))
    aggregate_acc = np.stack(acc_means, axis=0).mean(axis=0)
    aggregate_nll = np.stack(nll_means, axis=0).mean(axis=0)

    def bounds(values: np.ndarray) -> dict[str, float]:
        low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
        return {"p2_5": float(low), "p50": float(median), "p97_5": float(high)}

    return {
        "accuracy_gain": bounds(aggregate_acc),
        "candidate_nll_benefit": bounds(aggregate_nll),
    }


def _integrity_stop_reasons(integrity: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    local_params = int(integrity.get("local_trainable_parameters", -1))
    eiem_params = int(integrity.get("eiem_trainable_parameters", -1))
    if local_params != EXPECTED_LOCAL_PARAMETERS:
        reasons.append("local_parameter_count_mismatch")
    if eiem_params != EXPECTED_EIEM_PARAMETERS:
        reasons.append("eiem_parameter_count_mismatch")
    if local_params > 0 and abs((eiem_params - local_params) / local_params) > PARAMETER_MISMATCH_LIMIT:
        reasons.append("parameter_mismatch_above_point_one_percent")
    if int(integrity.get("training_tokens_per_model", -1)) != FIRST_SCREEN_TOKEN_BUDGET:
        reasons.append("training_token_count_mismatch")
    for flag in REQUIRED_INTEGRITY_FLAGS:
        if integrity.get(flag) is not True:
            reasons.append(f"integrity_{flag}_failed")
    expected_metadata = {
        "generator_version": GENERATOR_VERSION,
        "probe_seed": PROBE_SEED,
        "validation_seed": VALIDATION_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cases_per_family": CASES_PER_FAMILY,
        "probe_count": TOTAL_PROBES,
        "validation_tokens": VALIDATION_TOKENS,
    }
    for key, expected in expected_metadata.items():
        if integrity.get(key) != expected:
            reasons.append(f"integrity_{key}_mismatch")
    return reasons


def classify_stage_c(
    *,
    integrity: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    local_language_nll: float,
    eiem_language_nll: float,
) -> dict[str, Any]:
    """Apply the frozen #986 single-seed scaling-screen gate mechanically."""
    reasons = _integrity_stop_reasons(integrity)
    by_family = _validated_rows(rows)

    family_metrics: dict[str, dict[str, float]] = {}
    for family in ALL_FAMILIES:
        family_rows = by_family[family]
        family_metrics[family] = {
            "accuracy_gain": float(np.mean([_row_accuracy_gain(row) for row in family_rows])),
            "candidate_nll_benefit": float(np.mean([_row_nll_benefit(row) for row in family_rows])),
        }

    long_rows = [row for family in LONG_RANGE_FAMILIES for row in by_family[family]]
    aggregate_accuracy_gain = float(np.mean([_row_accuracy_gain(row) for row in long_rows]))
    aggregate_nll_benefit = float(np.mean([_row_nll_benefit(row) for row in long_rows]))
    if aggregate_accuracy_gain < MIN_AGGREGATE_ACCURACY_GAIN:
        reasons.append("aggregate_long_range_accuracy_gain_below_gate")
    if aggregate_nll_benefit < MIN_AGGREGATE_CANDIDATE_NLL_BENEFIT:
        reasons.append("aggregate_long_range_candidate_nll_benefit_below_gate")
    family_acc_gains = [family_metrics[family]["accuracy_gain"] for family in LONG_RANGE_FAMILIES]
    if sum(gain >= MIN_FAMILY_ACCURACY_GAIN for gain in family_acc_gains) < MIN_LONG_RANGE_FAMILIES_AT_GAIN:
        reasons.append("fewer_than_two_long_range_families_meet_accuracy_gain")
    if any(gain < MIN_ANY_FAMILY_ACCURACY_GAIN for gain in family_acc_gains):
        reasons.append("long_range_family_accuracy_regression")
    family_nll = [family_metrics[family]["candidate_nll_benefit"] for family in LONG_RANGE_FAMILIES]
    if sum(benefit > 0.0 for benefit in family_nll) < MIN_LONG_RANGE_FAMILIES_POSITIVE_NLL:
        reasons.append("fewer_than_two_long_range_families_have_positive_nll_benefit")

    bootstrap = stratified_paired_bootstrap(rows)
    if bootstrap["accuracy_gain"]["p2_5"] <= 0.0:
        reasons.append("bootstrap_accuracy_lower_bound_not_positive")
    if bootstrap["candidate_nll_benefit"]["p2_5"] <= 0.0:
        reasons.append("bootstrap_nll_lower_bound_not_positive")

    local_language_nll = float(local_language_nll)
    eiem_language_nll = float(eiem_language_nll)
    if not np.isfinite(local_language_nll) or not np.isfinite(eiem_language_nll):
        reasons.append("nonfinite_ordinary_language_nll")
    language_delta = eiem_language_nll - local_language_nll
    if language_delta > MAX_LANGUAGE_NLL_REGRESSION:
        reasons.append("ordinary_language_nll_regression_above_gate")

    local_negative = family_metrics["local_negative"]
    if local_negative["accuracy_gain"] < MIN_LOCAL_NEGATIVE_ACCURACY_GAIN:
        reasons.append("local_negative_accuracy_regression_above_gate")
    local_negative_nll_regression = -local_negative["candidate_nll_benefit"]
    if local_negative_nll_regression > MAX_LOCAL_NEGATIVE_CANDIDATE_NLL_REGRESSION:
        reasons.append("local_negative_candidate_nll_regression_above_gate")

    overwrite = by_family["overwrite"]
    local_stale = float(np.mean([float(bool(row["local_stale_choice"])) for row in overwrite]))
    eiem_stale = float(np.mean([float(bool(row["eiem_stale_choice"])) for row in overwrite]))
    if local_stale < 0.01:
        stale_pass = eiem_stale <= local_stale + 0.01
        stale_rule = "eiem<=local+0.01_when_local<0.01"
    else:
        stale_pass = eiem_stale <= 0.80 * local_stale
        stale_rule = "eiem<=0.80*local"
    if not stale_pass:
        reasons.append("overwrite_stale_state_guard_failed")

    passed = not reasons
    return {
        "classification": (
            "CHM_V1_100M_STAGE_C_POSITIVE_SCALING_SCREEN"
            if passed
            else "CHM_V1_100M_STAGE_C_STOP_SCALE_BRANCH"
        ),
        "passed": passed,
        "stop_reasons": reasons,
        "metrics": {
            "aggregate_long_range_accuracy_gain": aggregate_accuracy_gain,
            "aggregate_long_range_candidate_nll_benefit": aggregate_nll_benefit,
            "family_metrics": family_metrics,
            "bootstrap": bootstrap,
            "ordinary_language_nll_delta": language_delta,
            "local_negative_candidate_nll_regression": local_negative_nll_regression,
            "local_overwrite_stale_choice_rate": local_stale,
            "eiem_overwrite_stale_choice_rate": eiem_stale,
            "overwrite_stale_rule": stale_rule,
        },
        "interpretation_ceiling": (
            "One-seed ~100M scaling-screen evidence only; not confirmation, novelty, SOTA, AGI, RSI, or breakthrough."
        ),
        "stage_d_automatically_authorized": False,
    }
