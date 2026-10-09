from __future__ import annotations

from collections import Counter
from dataclasses import replace
import inspect

import pytest

import tam_research.chm_v3_counterfactual_multiset_suite_1373 as v3
from tam_research.chm_v3_counterfactual_memory_suite_1358 import (
    ANSWER_IDS, MEMORY_LENGTHS, SPLITS,
)
from tam_research.chm_v3_counterfactual_model_view_1365 import (
    STALE_CODE, oracle_read, seal_model_view, redact_memory_values,
)


@pytest.mark.parametrize("split", tuple(SPLITS))
@pytest.mark.parametrize("family", v3.FAMILIES)
def test_eight_way_pairing_preserves_multiset_query_and_addresses(split, family):
    for entity in range(SPLITS[split]):
        group = v3.paired_group(split, family, entity)
        assert len(group) == 8
        assert len({ep.query for ep in group}) == 1
        assert len({ep.entity for ep in group}) == 1
        assert len({tuple((f.position, f.kind, f.subject, f.document)
                          for f in ep.facts) for ep in group}) == 1
        views = [seal_model_view(ep) for ep in group]
        assert len({redact_memory_values(v) for v in views}) == 1
        assert len({v3.model_code_bag(v) for v in views}) == 1
        expected = Counter(f"CODE-{i}" for i in ANSWER_IDS)
        if family == "overwrite":
            expected[f"CODE-{STALE_CODE}"] += 1
        for view in views:
            assert Counter(v3.model_code_bag(view)) == expected
            assert not hasattr(view, "gold_answer")
            assert not hasattr(view, "variant")
            assert not hasattr(view, "facts")
        if family in v3.POSITIVES:
            assert {ep.gold_answer for ep in group} == set(ANSWER_IDS)
            assert len({v.memory_text for v in views}) == 8
            assert len({oracle_read(ep)[2] for ep in group}) == 1
        else:
            assert {ep.gold_answer for ep in group} == {None}
            assert len({v.memory_text for v in views}) == 1


@pytest.mark.parametrize("family", v3.FAMILIES)
@pytest.mark.parametrize("length", MEMORY_LENGTHS)
def test_oracle_reachable_and_bounded_at_every_memory_length(family, length):
    group = v3.paired_group("test", family, 1, memory_length=length)
    for ep in group:
        assert ep.memory_length == length
        assert all(0 <= fact.position < length for fact in ep.facts)
        answer, relation_pos, answer_pos = oracle_read(ep)
        assert answer == ep.gold_answer
        if family == "no_match":
            assert (answer, relation_pos, answer_pos) == (None, None, None)
        elif family == "two_hop":
            assert relation_pos is not None and answer_pos is not None
            r = next(f for f in ep.facts if f.position == relation_pos)
            val = next(f for f in ep.facts if f.position == answer_pos)
            assert r.kind == "relation" and val.kind == "record"
            assert r.document == val.subject
        elif family == "overwrite":
            assert relation_pos is None
            stored = sorted((x for x in ep.facts if x.subject == ep.entity),
                            key=lambda f: f.position)
            assert len(stored) == 2
            assert stored[0].code == STALE_CODE
            assert stored[-1].code == answer
            assert stored[-1].position == answer_pos
        else:
            assert relation_pos is None and answer_pos is not None


def test_each_nonbase_variant_swaps_exactly_target_and_one_unrelated_decoy():
    for family in v3.POSITIVES:
        group = v3.paired_group("test", family, 0)
        base = group[0]
        target = oracle_read(base)[2]
        for ep in group[1:]:
            differences = [(a, b) for a, b in zip(base.facts, ep.facts)
                           if a.code != b.code]
            assert len(differences) == 2
            changed_positions = {a.position for a, _ in differences}
            assert target in changed_positions
            assert len(changed_positions) == 2
            assert all((a.position, a.subject, a.kind, a.document)
                       == (b.position, b.subject, b.kind, b.document)
                       for a, b in differences)
            assert ep.gold_answer != base.gold_answer
            assert ep.query == base.query
            assert oracle_read(ep)[0] == ep.gold_answer


def test_three_memory_blind_controls_are_chance_positive_and_abstain_is_separate():
    for split in SPLITS:
        report = v3.control_report(split)
        assert report["trained_model_used"] is False
        assert report["gpu_allocated"] is False
        assert report["historical_scientific_classification"] == (
            "CHM_V3_100M_DAEC_STAGE_C_STOP"
        )
        assert report["historical_scientific_seed_consumed"] is True
        for family in v3.POSITIVES:
            x = report["families"][family]
            assert x["cases"] == 8 * SPLITS[split]
            assert x["query_only_accuracy"] == pytest.approx(0.125)
            assert x["layout_only_accuracy"] == pytest.approx(0.125)
            assert x["bag_of_codes_only_accuracy"] == pytest.approx(0.125)
            assert x["always_abstain_correct"] == 0
        neg = report["families"]["no_match"]
        assert neg["cases"] == 8 * SPLITS[split]
        assert neg["query_only_accuracy"] == 0.0
        assert neg["layout_only_accuracy"] == 0.0
        assert neg["bag_of_codes_only_accuracy"] == 0.0
        assert neg["always_abstain_correct"] == neg["cases"]


def test_multiset_and_layout_tamper_fail_closed():
    group = v3.paired_group("test", "two_hop", 0)
    target = oracle_read(group[1])[2]
    bad_bag = replace(group[1], facts=tuple(
        replace(f, code=ANSWER_IDS[0])
        if f.code in ANSWER_IDS and f.position != target
        else f for f in group[1].facts
    ))
    with pytest.raises(ValueError, match="multiset|frequency"):
        v3.validate_group((group[0], bad_bag, *group[2:]))
    bad_layout = replace(group[1], facts=tuple(
        replace(f, subject=f.subject + "-tampered")
        if f.position == group[1].facts[0].position else f
        for f in group[1].facts
    ))
    # An invalid relation subject may fail oracle reachability before the
    # pairwise-layout validator; either fail-closed check is valid.
    with pytest.raises(ValueError):
        v3.validate_group((group[0], bad_layout, *group[2:]))
    with pytest.raises(ValueError, match="rotation"):
        v3.validate_group((group[0], group[0], *group[2:]))


def test_negative_has_all_code_values_but_no_target_subject():
    group = v3.paired_group("test", "no_match", 2)
    for ep in group:
        assert oracle_read(ep) == (None, None, None)
        assert ep.gold_answer is None
        assert all(f.subject != ep.entity for f in ep.facts)
        assert Counter(v3.model_code_bag(seal_model_view(ep))) == Counter(
            f"CODE-{x}" for x in ANSWER_IDS
        )


def test_reproducible_and_no_old_probe_seed_or_authority():
    a = v3.generate_episode("test", "two_hop", 3, 5, memory_length=1024)
    b = v3.generate_episode("test", "two_hop", 3, 5, memory_length=1024)
    assert a == b
    assert v3.GPU_AUTHORIZED is False
    assert v3.SCIENTIFIC_EXECUTION_AUTHORIZED is False
    assert v3.ORIGINAL_SCIENTIFIC_SEED_CONSUMED == 2013161
    src = inspect.getsource(v3)
    for forbidden in (
        "modal.App(", "modal.run(", "torch.cuda", "optimizer.step(",
        "torch.load(", "checkpoint.load(", "generate_aligned_probe_suite(",
    ):
        assert forbidden not in src
