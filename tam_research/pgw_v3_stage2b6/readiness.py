"""PGW-v3 Stage-2B6: exact corpus-root and read-only fairness blocker audit.

No optimizer, training runner, scientific seed, GPU/Modal allocation, payment,
checkpoint, or authority to interpret a synthetic result as a breakthrough.

Source-bound contract: issue #1454; parent design #1437.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from tam_research.pgw_v3_stage2b4.preflight import gradient_activity_preflight
from tam_research.pgw_v3_stage2b5.corpus import (
    CandidateCorpusManifest, candidate_corpus_manifest,
)

# Immutable expected roots sealed independently BEFORE this source (issue
# #1429 comment 6101556127). Never regenerate and rewrite these from a result.
SEALED_ROOTS: tuple[tuple[str, str], ...] = (
    ("train", "0174103c2ef1cf14e5dec0df23498d994062dabd0e4151d02f7d2ba4f055d9a3"),
    ("validation", "c6b4fb7bf1943755510bb24fb98053a0ccf45cc83ecb4b51c821193845668611"),
    ("test", "3b3ab8e8b5d68337ad1608053541838f02369033b6c95e538a7266b330d8c7d3"),
    ("all", "36a40585fe3deb2c4fc268c32eb93092ebc46f7809b772cdbc223d0185d940ef"),
)
EXPECTED_SPLIT_SAMPLES = (("train", 2880), ("validation", 1440), ("test", 1440))
PRIMARY_ARMS = ("pgw_hybrid", "full_global")
DELAYS = (1, 2, 4)

# These are *required future science gates*, deliberately not dynamic switches
# that become true because one CPU derivative probe happened to succeed.
OPEN_SCIENCE_BLOCKERS = (
    "analytical_and_trajectory_active_parameter_parity_unproven",
    "auxiliary_prediction_information_objective_not_matched",
    "measured_whole_training_step_compute_and_time_not_equalized",
    "optimizer_learning_schedule_lambda_not_preregistered",
    "fresh_paired_scientific_seeds_and_durable_attempts_not_reserved",
    "explicit_source_bound_owner_execution_authorization_missing",
    "live_modal_balance_spend_cap_and_artifact_retention_unverified",
)


@dataclass(frozen=True)
class CorpusReadback:
    roots: tuple[tuple[str, str], ...]
    split_samples: tuple[tuple[str, int], ...]
    samples: int
    input_tokens: int
    unique_token_fingerprints: int
    unique_semantic_traces: int
    cells_per_split: tuple[int, ...]
    valid: bool


@dataclass(frozen=True)
class PairedDerivativeReadback:
    arm: str
    delay: int
    instantiated: int
    ce_nonzero: int
    aux_nonzero: int
    union_nonzero: int
    ce_predictor_nonzero: int
    aux_predictor_nonzero: int
    state_unchanged: bool


@dataclass(frozen=True)
class ReadonlyScientificGate:
    corpus: CorpusReadback
    paired_derivatives: tuple[PairedDerivativeReadback, ...]
    instantiated_parameter_delta: int
    science_blockers: tuple[str, ...]
    allowed_to_train: bool
    allowed_to_spend_modal: bool
    decision: str


def verify_sealed_corpus(
    manifest: CandidateCorpusManifest | None = None,
) -> CorpusReadback:
    """Fail closed on ANY mismatch from the independently frozen SHA roots."""
    if manifest is None:
        manifest = candidate_corpus_manifest()
    roots = tuple((x.name, x.root_sha256) for x in manifest.splits) + (
        ("all", manifest.root_sha256),
    )
    counts = tuple((x.name, x.sample_count) for x in manifest.splits)
    if roots != SEALED_ROOTS:
        raise ValueError("FROZEN PGW corpus root mismatch: stop, never train")
    if counts != EXPECTED_SPLIT_SAMPLES or manifest.total_samples != 5760:
        raise ValueError("FROZEN PGW corpus cardinality mismatch")
    if manifest.total_input_tokens != 522240:
        raise ValueError("FROZEN PGW corpus input token count mismatch")
    if any(
        x.cells != 180 or x.present != x.missing
        or x.sample_count != x.present + x.missing
        for x in manifest.splits
    ):
        raise ValueError("FROZEN PGW split factorial strata mismatch")
    records = manifest.records
    seen_tokens = len({r.token_sha256 for r in records})
    seen_traces = len({r.semantic_sha256 for r in records})
    if seen_tokens != 5760 or seen_traces != 5760:
        raise ValueError("PGW corpus token/semantic collision detected")
    if len(records) != 5760 or len({r.sample_id for r in records}) != 5760:
        raise ValueError("PGW corpus record identity collision")
    return CorpusReadback(
        roots=roots,
        split_samples=counts,
        samples=manifest.total_samples,
        input_tokens=manifest.total_input_tokens,
        unique_token_fingerprints=seen_tokens,
        unique_semantic_traces=seen_traces,
        cells_per_split=tuple(x.cells for x in manifest.splits),
        valid=True,
    )


def readonly_research_gate() -> ReadonlyScientificGate:
    """Validate sealed corpus and two control families' derivative paths.

    Deliberately *never* authorizes science execution, even if all structural
    checks pass. No model parameter is updated in the imported probe.
    """
    python_rng_before = random.getstate()
    torch_rng_before = torch.get_rng_state().clone()
    corpus = verify_sealed_corpus()
    measured = gradient_activity_preflight(
        arms=PRIMARY_ARMS, delays=DELAYS
    )
    if random.getstate() != python_rng_before:
        raise ValueError("global Python RNG state mutated in no-training audit")
    if not torch.equal(torch.get_rng_state(), torch_rng_before):
        raise ValueError("global torch RNG state mutated in no-training audit")
    if len(measured) != len(PRIMARY_ARMS) * len(DELAYS):
        raise ValueError("incomplete paired primary-arm gradient probe")

    out = []
    for row in measured:
        expected_parameters = 20354 if row.arm == "pgw_hybrid" else 20347
        if row.instantiated != expected_parameters:
            raise ValueError("frozen PGW-v3/reference parameter count drift")
        if (
            not row.state_unchanged
            or row.answer_ce.predictor_nonzero != 0
            or row.auxiliary.predictor_nonzero <= 0
            or row.auxiliary.nonzero_elements <= 0
            or row.answer_ce.nonzero_elements <= 0
            or row.union_nonzero > row.instantiated
        ):
            raise ValueError("PGW-v3 source derivative or state integrity failure")
        out.append(PairedDerivativeReadback(
            arm=row.arm,
            delay=row.delay,
            instantiated=row.instantiated,
            ce_nonzero=row.answer_ce.nonzero_elements,
            aux_nonzero=row.auxiliary.nonzero_elements,
            union_nonzero=row.union_nonzero,
            ce_predictor_nonzero=row.answer_ce.predictor_nonzero,
            aux_predictor_nonzero=row.auxiliary.predictor_nonzero,
            state_unchanged=row.state_unchanged,
        ))
    if {(r.arm, r.delay) for r in out} != {
        (arm, delay) for arm in PRIMARY_ARMS for delay in DELAYS
    }:
        raise ValueError("missing PGW/reference delay or duplicate row")
    return ReadonlyScientificGate(
        corpus=corpus,
        paired_derivatives=tuple(out),
        instantiated_parameter_delta=20354 - 20347,
        science_blockers=OPEN_SCIENCE_BLOCKERS,
        allowed_to_train=False,
        allowed_to_spend_modal=False,
        decision="BLOCKED_PENDING_APPROVED_FAIR_SCIENTIFIC_PROTOCOL",
    )
