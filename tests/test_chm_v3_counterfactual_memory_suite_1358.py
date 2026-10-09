from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import inspect

import pytest

import tam_research.chm_v3_counterfactual_memory_suite_1358 as suite


@pytest.mark.parametrize("split,count", [("train", 256), ("development", 128), ("test", 128)])
def test_counterfactual_suite_is_balanced_per_entity_and_identity_disjoint(split, count):
    cases = suite.generate_suite(split)
    assert len(cases) == count
    assert len({(x.family, x.entity, x.counterfactual_index) for x in cases}) == count
    for family in suite.POSITIVE_FAMILIES:
        grouped = defaultdict(list)
        for x in cases:
            if x.family == family:
                grouped[x.entity].append(x)
        assert len(grouped) == suite.SPLITS[split]
        for records in grouped.values():
            assert len(records) == 8
            assert {r.correct_answer_id for r in records} == set(suite.ANSWER_IDS)
            assert len({r.query for r in records}) == 1
            assert len({r.counterfactual_index for r in records}) == 8
            assert all(str(v) not in records[0].query for v in suite.ANSWER_IDS)
            assert all(suite.oracle_read(r)[0] == r.correct_answer_id for r in records)
    for x in cases:
        suite.validate_episode(x)
        assert x.entity.startswith(split + "-")

    for other_split in suite.SPLITS:
        if other_split != split:
            assert {x.entity for x in cases}.isdisjoint(
                {y.entity for y in suite.generate_suite(other_split)}
            )


@pytest.mark.parametrize("memory_length", suite.MEMORY_LENGTHS)
def test_answer_oracle_reachable_across_memory_lengths_and_latest_overwrite(memory_length):
    for family in suite.FAMILIES:
        for counterfactual_index in range(8):
            ep = suite.generate_episode("development", family, 0,
                                        counterfactual_index, memory_length=memory_length)
            answer, first, second = suite.oracle_read(ep)
            assert answer == ep.correct_answer_id
            assert all(0 <= record.position < memory_length for record in ep.records)
            if family == "no_match":
                assert (answer, first, second) == (None, None, None)
            elif family == "two_hop":
                assert first is not None and second is not None and first != second
                rel = next(r for r in ep.records if r.position == first)
                val = next(r for r in ep.records if r.position == second)
                assert rel.role == "relation" and val.role == "record"
                assert rel.document == val.subject
            else:
                assert first is None and second is not None
            if family == "overwrite":
                target = [r for r in ep.records if r.subject == ep.entity]
                assert len(target) == 2
                assert target[0].position < target[1].position
                assert target[0].answer_id != target[1].answer_id
                assert answer == target[1].answer_id and second == target[1].position


def test_counterfactual_fact_rebinding_changes_answer_not_query():
    ep = suite.generate_episode("test", "direct", 1, 0)
    before = suite.oracle_read(ep)
    changed_answer = next(v for v in suite.ANSWER_IDS if v != ep.correct_answer_id)
    rebind = tuple(
        replace(record, answer_id=changed_answer) if record.subject == ep.entity
        else record for record in ep.records
    )
    changed = replace(ep, records=rebind, correct_answer_id=changed_answer)
    suite.validate_episode(changed)
    assert changed.query == ep.query and changed.entity == ep.entity
    assert suite.oracle_read(changed)[0] != before[0]


def test_two_hop_relation_link_permutation_changes_answer_and_preserves_query():
    ep = suite.generate_episode("test", "two_hop", 2, 1)
    relation = next(r for r in ep.records if r.role == "relation" and r.subject == ep.entity)
    decoy = next(r for r in ep.records if r.role == "record" and r.subject != relation.document)
    alternative = replace(ep, records=tuple(
        replace(r, document=decoy.subject) if r == relation else r for r in ep.records
    ), correct_answer_id=decoy.answer_id)
    suite.validate_episode(alternative)
    assert suite.oracle_read(alternative)[0] == decoy.answer_id
    assert decoy.answer_id != ep.correct_answer_id
    assert alternative.query == ep.query


def test_memory_record_listing_order_does_not_change_last_authoritative_value():
    ep = suite.generate_episode("test", "overwrite", 1, 4)
    reversed_order = replace(ep, records=tuple(reversed(ep.records)))
    suite.validate_episode(reversed_order)
    assert suite.oracle_read(reversed_order) == suite.oracle_read(ep)


def test_memory_free_policy_cannot_solve_changed_facts():
    for split in suite.SPLITS:
        report = suite.baseline_report(split)
        assert report["gpu_used"] is False
        assert report["trained_checkpoint_replayed"] is False
        assert report["historical_stage_c_stop_unchanged"] is True
        for family in suite.POSITIVE_FAMILIES:
            result = report["families"][family]
            assert result["query_only_accuracy"] == 0.125
            assert result["query_only_correct"] == suite.SPLITS[split]
            assert result["always_abstain_correct"] == 0
        neg = report["families"]["no_match"]
        assert neg["query_only_correct"] == 0
        assert neg["always_abstain_correct"] == neg["cases"]
        for family in suite.POSITIVE_FAMILIES:
            for index in range(suite.SPLITS[split]):
                episodes = [suite.generate_episode(split, family, index, i) for i in range(8)]
                assert len({suite.query_only_baseline(x) for x in episodes}) == 1
                assert sum(suite.query_only_baseline(x) == x.correct_answer_id for x in episodes) == 1


def test_adversarial_integrity_rejects_missing_fact_future_position_or_target_leak():
    ep = suite.generate_episode("test", "direct", 0, 0)
    answer_record = next(r for r in ep.records if r.subject == ep.entity)
    no_answer = replace(ep, records=tuple(r for r in ep.records if r != answer_record))
    with pytest.raises(ValueError, match="oracle value"):
        suite.validate_episode(no_answer)
    future = replace(ep, records=tuple(
        replace(r, position=ep.memory_length) if r == answer_record else r for r in ep.records))
    with pytest.raises(ValueError, match="out-of-bounds"):
        suite.validate_episode(future)
    leak = replace(ep, query=ep.query + " 18001")
    with pytest.raises(ValueError, match="query changed"):
        suite.validate_episode(leak)
    duplicate = replace(ep, records=(ep.records[0], ep.records[0]) + ep.records[1:])
    with pytest.raises(ValueError, match="duplicate"):
        suite.validate_episode(duplicate)
    with pytest.raises(ValueError, match="rotation"):
        suite.generate_episode("test", "direct", 0, 8)
    with pytest.raises(ValueError, match="memory length"):
        suite.generate_episode("test", "direct", 0, 0, memory_length=64)


def test_splits_reproducible_source_frozen_and_no_scientific_authority():
    ep1 = suite.generate_episode("test", "two_hop", 3, 6, memory_length=1024)
    ep2 = suite.generate_episode("test", "two_hop", 3, 6, memory_length=1024)
    assert ep1 == ep2
    assert suite.HISTORICAL_STAGE_C_SCIENTIFIC_SEED_CONSUMED == 2013161
    assert suite.MODAL_GPU_AUTHORIZED is False
    assert suite.STAGE_C_SCIENTIFIC_AUTHORITY is False
    source = inspect.getsource(suite)
    for forbidden in ("modal.App(", "torch.cuda", "optimizer.step(", "checkpoint.load(",
                      "torch.load(", "generate_aligned_probe_suite("):
        assert forbidden not in source
