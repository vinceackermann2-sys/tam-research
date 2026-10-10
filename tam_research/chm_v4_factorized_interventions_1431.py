from __future__ import annotations

"""#1431: zero-GPU TRAIN-only explicit source/copy/gate counterfactual controls.

The hard-read interface sees ONLY SealedInput and an external physical source
proposal, never hidden Episode/role/gold/variant/split. No model is trained.
Deliberate diagnostic source/gate interventions are NOT learned predictions.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from .chm_v4_independent_binding_benchmark_1421 import (
    CODES, FAMILIES, POSITIVES, STALE_CODE, SPLIT_COUNTS,
    OLD_100M_STOP, OLD_SCIENTIFIC_SEED_CONSUMED, SealedInput,
    oracle_read, seal_input, train_counterfactual_group,
)
from .chm_v4_visible_hard_copy_train_1427 import (
    VisibleRecord, _parse_completed_memory, read_visible_text,
)

CLASSIFICATION = "CHM_V4_1431_TRAIN_ONLY_FACTORIZED_SOURCE_COPY_GATE"
GPU_AUTHORIZED = False
SCORED_HELDOUT_AUTHORIZED = False
SCIENTIFIC_EXECUTION_AUTHORIZED = False


@dataclass(frozen=True)
class PhysicalCopy:
    """The value actually fetched at a proposed physical memory source."""
    first_physical_source: int | None
    payload_physical_source: int | None
    copied_code: int | None
    route: Literal["direct", "relation", "absent"]


@dataclass(frozen=True)
class GatedAnswer:
    candidate_code: int | None
    abstained: bool


def copy_from_proposed_source(
    view: SealedInput, proposed_first_physical_source: int | None,
) -> PhysicalCopy:
    """Explicit HARD physical read. Source index is proposed, not inferred.

    Selection is strictly by [dddd] address visible in completed memory. For
    a relation line, follow its literal referenced document and select the
    newest matching code row by physical position. Wrong source proposals
    intentionally cause wrong copies; no query/gold authority repairs them.
    """
    if not isinstance(view, SealedInput):
        raise TypeError("only sealed text input is permitted")
    rows = _parse_completed_memory(view)
    if proposed_first_physical_source is None:
        return PhysicalCopy(None, None, None, "absent")
    if type(proposed_first_physical_source) is not int:
        raise TypeError("physical source must be integer or None")
    choices = [r for r in rows if r.position == proposed_first_physical_source]
    if len(choices) != 1:
        raise ValueError("proposed physical source missing/ambiguous")
    source = choices[0]
    if source.linked_document is not None:
        matches = [
            r for r in rows
            if r.subject == source.linked_document and r.code is not None
        ]
        if not matches:
            raise ValueError("proposed relation has no readable document payload")
        payload = max(matches, key=lambda r: r.position)
        route: Literal["direct", "relation", "absent"] = "relation"
    else:
        payload = source
        route = "direct"
    if payload.code not in CODES:
        raise ValueError("selected source has no valid candidate CODE payload")
    return PhysicalCopy(
        proposed_first_physical_source, payload.position, payload.code, route,
    )


def copy_value_at_explicit_payload(
    view: SealedInput, proposed_payload_physical_source: int,
) -> int:
    """Stage-two address->CODE only. Never consults chosen first address."""
    if not isinstance(view, SealedInput):
        raise TypeError("sealed input required")
    if type(proposed_payload_physical_source) is not int:
        raise TypeError("payload physical source must be integer")
    rows = _parse_completed_memory(view)
    matches = [r for r in rows if r.position == proposed_payload_physical_source]
    if len(matches) != 1 or matches[0].code not in CODES:
        raise ValueError("proposed payload is missing or not a candidate CODE")
    return matches[0].code


def gate_copied_value(
    copied_code: int | None,
    *,
    policy: Literal["copy_if_present", "force_abstain", "force_default"] =
    "copy_if_present",
) -> GatedAnswer:
    """Explicit answer/abstention gate, separated from address/copy.

    force_default and force_abstain are deliberate engineering ablations,
    NOT a deployable safety policy or model inference rule.
    """
    if copied_code is not None and copied_code not in CODES:
        raise ValueError("gate received noncandidate code")
    if policy == "force_abstain":
        return GatedAnswer(None, True)
    if policy == "force_default":
        return GatedAnswer(CODES[0] if copied_code is None else copied_code, False)
    if policy != "copy_if_present":
        raise ValueError("unknown gating policy")
    return GatedAnswer(copied_code, copied_code is None)


def train_only_factorized_audit() -> dict[str, object]:
    """Intervention diagnostics ONLY on preexisting V4 TRAIN groups.

    The visible baseline generates its own proposed first address. Evaluator
    oracle comparisons happen strictly AFTER all independent predictions.
    All alternate source/gate interventions are EXPLICIT counterfactuals.
    """
    normal_answers = Counter()
    normal_first = Counter()
    normal_payload = Counter()
    wrong_first_changes = Counter()
    wrong_payload_changes = Counter()
    forced_abstain_losses = Counter()
    older_relation_changes = Counter()
    stale_overwrite_rejections = 0
    forced_no_match_false_positives = 0
    no_match_correct_abstentions = 0
    cases = 0

    for family in FAMILIES:
        for entity in range(SPLIT_COUNTS["train"]):
            for ep in train_counterfactual_group(family, entity):
                if ep.split != "train":
                    raise RuntimeError("heldout episode entered TRAIN intervention")
                view = seal_input(ep)
                # Visible-only source proposal / hard physical copy / answer gate.
                proposal = read_visible_text(view)
                copied = copy_from_proposed_source(view, proposal.first_physical_source)
                gated = gate_copied_value(copied.copied_code)
                gold, relation_position, payload_position = oracle_read(ep)
                expected_first = (
                    relation_position if relation_position is not None
                    else payload_position
                )
                normal_answers[family] += gated.candidate_code == gold
                normal_first[family] += copied.first_physical_source == expected_first
                normal_payload[family] += copied.payload_physical_source == payload_position

                # Diagnostic perturbations chosen from VISIBLE text, not labels.
                rows = _parse_completed_memory(view)
                if family in POSITIVES:
                    # Every positive CODE appears exactly once; a different
                    # visible code row always contains a different answer.
                    decoys = sorted(
                        (r for r in rows
                         if r.code in CODES and r.code != copied.copied_code),
                        key=lambda r: r.position,
                    )
                    if len(decoys) != 7:
                        raise RuntimeError("positive value multiset no longer eight-way")
                    wrong_first = copy_from_proposed_source(view, decoys[0].position)
                    wrong_payload = copy_value_at_explicit_payload(
                        view, decoys[0].position,
                    )
                    wrong_first_changes[family] += (
                        gate_copied_value(wrong_first.copied_code).candidate_code != gold
                    )
                    wrong_payload_changes[family] += wrong_payload != gold
                    forced_abstain_losses[family] += (
                        gate_copied_value(
                            copied.copied_code, policy="force_abstain",
                        ).candidate_code != gold
                    )
                    if family == "two_hop":
                        linked = sorted(
                            (r for r in rows
                             if r.subject == ep.entity and
                             r.linked_document is not None),
                            key=lambda r: r.position,
                        )
                        if len(linked) != 2:
                            raise RuntimeError("expected old and new visible relation")
                        old_read = copy_from_proposed_source(
                            view, linked[0].position,
                        )
                        older_relation_changes[family] += old_read.copied_code != gold
                    if family == "overwrite":
                        old = [r for r in rows
                               if r.subject == ep.entity and r.code == STALE_CODE]
                        if len(old) != 1:
                            raise RuntimeError("missing old overwritten visible row")
                        try:
                            copy_from_proposed_source(view, old[0].position)
                        except ValueError as error:
                            if "no valid candidate" not in str(error):
                                raise
                            stale_overwrite_rejections += 1
                        else:
                            raise RuntimeError("noncandidate stale code was incorrectly copied")
                else:
                    if proposal.first_physical_source is not None or copied.copied_code is not None:
                        raise RuntimeError("negative no-match acquired a copy source")
                    no_match_correct_abstentions += gated.abstained
                    forced = gate_copied_value(None, policy="force_default")
                    forced_no_match_false_positives += not forced.abstained
                cases += 1

    counts = {name: SPLIT_COUNTS["train"] * 8 for name in FAMILIES}
    if cases != 2048 or any(
        normal_answers[family] != n or normal_first[family] != n
        or normal_payload[family] != n
        for family, n in counts.items()
    ):
        raise RuntimeError("factorized TRAIN source/copy/gate reference failed")
    if any(
        wrong_first_changes[f] != 512
        or wrong_payload_changes[f] != 512
        or forced_abstain_losses[f] != 512
        for f in POSITIVES
    ):
        raise RuntimeError("counterfactual wrong source/payload/gate did not fail")
    if (older_relation_changes["two_hop"] != 512
        or stale_overwrite_rejections != 512
        or no_match_correct_abstentions != 512
        or forced_no_match_false_positives != 512):
        raise RuntimeError("relation/overwrite/no-match interventions were ineffective")

    return {
        "classification": CLASSIFICATION,
        "train_independent_entities": SPLIT_COUNTS["train"],
        "training_cases_only": cases,
        "natural_answer_correct_by_family": dict(normal_answers),
        "natural_first_address_correct_by_family": dict(normal_first),
        "natural_payload_address_correct_by_family": dict(normal_payload),
        "wrong_first_source_changes_answer_by_positive_family": dict(wrong_first_changes),
        "wrong_payload_source_changes_answer_by_positive_family": dict(wrong_payload_changes),
        "forced_abstain_loses_answer_by_positive_family": dict(forced_abstain_losses),
        "outdated_two_hop_relation_changes_answer": older_relation_changes["two_hop"],
        "stale_overwrite_first_source_rejected": stale_overwrite_rejections,
        "no_match_natural_abstentions": no_match_correct_abstentions,
        "no_match_forced_default_false_positives": forced_no_match_false_positives,
        "development_test_examples_created": 0,
        "trained_model_forward_calls": 0,
        "optimizer_steps": 0,
        "gpu_used": False,
        "new_scientific_attempt": False,
        "historical_status": OLD_100M_STOP,
        "historical_seed_consumed": OLD_SCIENTIFIC_SEED_CONSUMED,
        "interpretation": (
            "Exact synthetic TRAIN-only intervention sensitivity of a hand-coded "
            "visible-text reader. It proves no neural learning, scientific win, "
            "heldout generalization or 100M upgrade."
        ),
    }
