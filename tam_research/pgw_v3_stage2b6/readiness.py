"""PGW-v3 Stage-2B6: exact corpus-root and read-only fairness blocker audit.

No optimizer, training runner, scientific seed, GPU/Modal allocation, payment,
checkpoint, or authority to interpret a synthetic result as a breakthrough.

Source-bound contract: issue #1454; parent design #1437.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Iterable

import torch

from tam_research.pgw_v3_stage2b4.preflight import gradient_activity_preflight
from tam_research.pgw_v3_stage2b5.corpus import (
    CandidateCorpusManifest, EvaluationScore, candidate_corpus_manifest,
    evaluate_complete_predictions, _ascii_row, _digest, _generate, _record_line,
    SPLIT_REPEATS, CELLS_PER_SPLIT,
)
from tam_research.pgw_v3_stage1b.oracle import (
    POSITIONS, OVERWRITES, DELAYS as ORACLE_DELAYS,
    oracle_latest_value_or_not_found,
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
    """Recompute every row and root; never trust supplied manifest metadata."""
    python_rng_before = random.getstate()
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
        x.cells != CELLS_PER_SPLIT or x.present != x.missing
        or x.sample_count != x.present + x.missing
        for x in manifest.splits
    ):
        raise ValueError("FROZEN PGW split factorial strata mismatch")
    records = manifest.records
    if len(records) != 5760:
        raise ValueError("FROZEN PGW corpus record cardinality mismatch")

    # Duplicate sample identities must be rejected as an identity collision
    # BEFORE checking canonical order. Both changes corrupt the manifest, but
    # a duplicate is a strictly stronger integrity defect; keep its original
    # diagnostic stable for the frozen negative regression.
    if len({row.sample_id for row in records}) != len(records):
        raise ValueError("PGW corpus sample identifier collision")

    # The immutable content digest sorts rows. Preserve its historical roots,
    # but independently fail closed on a permuted traversal manifest.
    expected_order = tuple(
        (split, index, position, overwrites, interference, delay, missing)
        for split, repeats in SPLIT_REPEATS
        for position in POSITIONS
        for overwrites in OVERWRITES
        for interference in (False, True)
        for delay in ORACLE_DELAYS
        for missing in (False, True)
        for index in range(repeats)
    )
    actual_order = tuple(
        (row.split, row.index, row.last_position, row.overwrite_count,
         row.interference, row.delay, row.missing)
        for row in records
    )
    if actual_order != expected_order:
        raise ValueError("FROZEN PGW canonical corpus traversal mismatch")

    split_repeats = dict(SPLIT_REPEATS)
    expected_cells = {
        (p, o, interference, delay, missing)
        for p in POSITIONS for o in OVERWRITES
        for interference in (False, True)
        for delay in ORACLE_DELAYS for missing in (False, True)
    }
    if len(expected_cells) != 180:
        raise ValueError("FROZEN PGW source factorial grammar drift")
    rows_by_split: dict[str, list] = {split: [] for split in split_repeats}
    seen_ids: set[str] = set()
    seen_tokens: set[str] = set()
    seen_semantics: set[str] = set()
    token_total = 0
    for row in records:
        if row.split not in split_repeats:
            raise ValueError("FROZEN PGW unknown split")
        if row.sample_id in seen_ids:
            raise ValueError("PGW corpus sample identifier collision")
        seen_ids.add(row.sample_id)
        if row.token_sha256 in seen_tokens or row.semantic_sha256 in seen_semantics:
            raise ValueError("PGW corpus token/semantic collision detected")
        seen_tokens.add(row.token_sha256)
        seen_semantics.add(row.semantic_sha256)
        metadata = (
            row.split, row.index, row.last_position, row.overwrite_count,
            int(row.interference), row.delay, int(row.missing), row.token_sha256,
        )
        expected_id = hashlib.sha256(
            _ascii_row(metadata).encode("ascii")
        ).hexdigest()
        if row.sample_id != expected_id:
            raise ValueError("PGW corpus sample identifier mismatch")
        sample = _generate(row)
        # Do not trust generator-reported SHA metadata: hash regenerated token
        # *bytes* independently, before accepting any corpus row or score.
        actual_token_sha = hashlib.sha256(bytes(sample.tokens)).hexdigest()
        if actual_token_sha != row.token_sha256 or sample.sha256 != actual_token_sha:
            raise ValueError("PGW corpus regenerated token-byte SHA mismatch")
        if oracle_latest_value_or_not_found(sample.tokens) != row.answer:
            raise ValueError("PGW corpus independent oracle disagreement")
        token_total += len(sample.tokens)
        rows_by_split[row.split].append(row)

    for root, (split, repeats) in zip(manifest.splits, SPLIT_REPEATS):
        rows = rows_by_split[split]
        if root.name != split or len(rows) != CELLS_PER_SPLIT * repeats:
            raise ValueError("FROZEN PGW split cardinality mismatch")
        by_cell: dict[tuple, set[int]] = {}
        for row in rows:
            if row.cell not in expected_cells:
                raise ValueError("FROZEN PGW split factorial strata mismatch")
            indexes = by_cell.setdefault(row.cell, set())
            if row.index in indexes:
                raise ValueError("FROZEN PGW duplicate factorial cell/index")
            indexes.add(row.index)
        if set(by_cell) != expected_cells or any(
            indexes != set(range(repeats)) for indexes in by_cell.values()
        ):
            raise ValueError("FROZEN PGW split factorial strata mismatch")
        present = sum(not row.missing for row in rows)
        missing = sum(row.missing for row in rows)
        if (
            root.per_cell != repeats or root.cells != CELLS_PER_SPLIT
            or root.sample_count != len(rows)
            or root.present != present or root.missing != missing
            or present != missing
        ):
            raise ValueError("FROZEN PGW split factorial strata mismatch")
        if _digest(_record_line(row) for row in rows) != root.root_sha256:
            raise ValueError("FROZEN PGW recomputed split root mismatch")

    if _digest(_record_line(row) for row in records) != manifest.root_sha256:
        raise ValueError("FROZEN PGW recomputed whole-corpus root mismatch")
    if token_total != manifest.total_input_tokens:
        raise ValueError("FROZEN PGW regenerated input token count mismatch")
    if random.getstate() != python_rng_before:
        raise ValueError("FROZEN PGW corpus regeneration changed Python RNG")
    return CorpusReadback(
        roots=roots,
        split_samples=counts,
        samples=manifest.total_samples,
        input_tokens=token_total,
        unique_token_fingerprints=len(seen_tokens),
        unique_semantic_traces=len(seen_semantics),
        cells_per_split=tuple(x.cells for x in manifest.splits),
        valid=True,
    )


def evaluate_verified_predictions(
    manifest: CandidateCorpusManifest, *, split: str,
    predictions: Iterable[tuple[str, int]],
) -> EvaluationScore:
    """Score only AFTER complete sealed source-bound corpus attestation.

    Stage-2B5's raw evaluator accepts caller-provided manifest metadata and
    therefore is not itself an authentication boundary. This wrapper checks
    every row, byte SHA, semantic/oracle trace, exact identities, complete
    factorial coverage, immutable original roots and canonical order first.
    It is CPU-only scoring of PROVIDED predictions; no model or optimizer.
    """
    verify_sealed_corpus(manifest)
    return evaluate_complete_predictions(
        manifest, split=split, predictions=predictions
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
