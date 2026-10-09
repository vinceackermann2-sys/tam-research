from __future__ import annotations

from dataclasses import fields, replace
import inspect

import pytest

import tam_research.chm_v3_counterfactual_model_view_1365 as v2
from tam_research.chm_v3_counterfactual_memory_suite_1358 import ANSWER_IDS, SPLITS


@pytest.mark.parametrize("split", tuple(SPLITS))
@pytest.mark.parametrize("family", v2.FAMILIES)
def test_matched_counterfactual_groups_are_source_bound(split: str, family: str) -> None:
    for index in range(SPLITS[split]):
        group = v2.paired_group(split, family, index)
        assert len(group) == 8
        assert len({ep.query for ep in group}) == 1
        assert len({ep.entity for ep in group}) == 1
        assert len({ep.memory_length for ep in group}) == 1
        assert len({tuple((x.position, x.kind, x.subject, x.document)
                          for x in ep.facts) for ep in group}) == 1
        assert len({v2.redact_memory_values(v2.seal_model_view(ep)) for ep in group}) == 1
        if family != "no_match":
            assert {ep.gold_answer for ep in group} == set(ANSWER_IDS)
            assert len({v2.seal_model_view(ep).memory_text for ep in group}) == 8
            oracle_indices = {v2.oracle_read(ep)[2] for ep in group}
            assert len(oracle_indices) == 1
        else:
            assert {ep.gold_answer for ep in group} == {None}
            assert len({v2.seal_model_view(ep).memory_text for ep in group}) == 1


@pytest.mark.parametrize("length", (128, 256, 512, 1024))
@pytest.mark.parametrize("family", v2.FAMILIES)
def test_completed_memory_oracle_and_role_semantics(length: int, family: str) -> None:
    group = v2.paired_group("test", family, 1, memory_length=length)
    base_masked = v2.redact_memory_values(v2.seal_model_view(group[0]))
    for ep in group:
        assert ep.memory_length == length
        assert all(0 <= r.position < length for r in ep.facts)
        assert v2.redact_memory_values(v2.seal_model_view(ep)) == base_masked
        answer, first, target = v2.oracle_read(ep)
        assert answer == ep.gold_answer
        if family == "no_match":
            assert (first, target) == (None, None)
        elif family == "two_hop":
            assert first is not None and target is not None and first != target
            rel = next(x for x in ep.facts if x.position == first)
            val = next(x for x in ep.facts if x.position == target)
            assert rel.kind == "relation" and val.kind == "record"
            assert rel.document == val.subject
        elif family == "overwrite":
            assert first is None and target is not None
            values = sorted((r for r in ep.facts if r.subject == ep.entity),
                            key=lambda r: r.position)
            assert len(values) == 2
            assert values[0].code == v2.STALE_CODE
            assert values[1].code == answer
            assert values[0].code not in ANSWER_IDS
        else:
            assert first is None and target is not None


def test_only_gold_fact_value_changes_not_record_positions_or_document_identity() -> None:
    for family in v2.POSITIVE_FAMILIES:
        a, b = v2.paired_group("test", family, 0)[:2]
        assert a.query == b.query
        assert a.gold_answer != b.gold_answer
        different = [(x, y) for x, y in zip(a.facts, b.facts) if x != y]
        assert len(different) == 1
        lhs, rhs = different[0]
        assert lhs.position == rhs.position
        assert lhs.kind == rhs.kind and lhs.subject == rhs.subject
        assert lhs.document == rhs.document
        assert lhs.code != rhs.code
        assert v2.seal_model_view(a).answer_options == v2.seal_model_view(b).answer_options


def test_memory_free_and_layout_only_controls_are_exact_chance_per_positive_family() -> None:
    for split in SPLITS:
        result = v2.negative_control_report(split)
        assert result["model_trained"] is False
        assert result["no_gpu"] is True
        assert result["old_scientific_stop_unchanged"] is True
        for family in v2.POSITIVE_FAMILIES:
            family_report = result["families"][family]
            assert family_report["cases"] == 8 * SPLITS[split]
            assert family_report["query_only_accuracy"] == pytest.approx(0.125)
            assert family_report["layout_only_accuracy"] == pytest.approx(0.125)
            assert family_report["always_abstain_correct"] == 0
        negative = result["families"]["no_match"]
        assert negative["query_only_accuracy"] == 0.0
        assert negative["layout_only_accuracy"] == 0.0
        assert negative["always_abstain_correct"] == negative["cases"]


def test_model_view_is_actually_sealed_and_has_no_oracle_fields() -> None:
    assert [f.name for f in fields(v2.SealedModelView)] == [
        "query", "memory_text", "answer_options"
    ]
    for fam in v2.FAMILIES:
        ep = v2.generate_paired_episode("test", fam, 0, 1)
        view = v2.seal_model_view(ep)
        assert not hasattr(view, "gold_answer")
        assert not hasattr(view, "variant")
        assert not hasattr(view, "facts")
        assert not hasattr(view, "split")
        assert not hasattr(view, "family")
        assert not hasattr(view, "oracle_read")
        assert not hasattr(view, "target_position")
        assert all(str(answer_id) not in view.query for answer_id in ANSWER_IDS)
        assert all(str(answer_id) not in view.query for answer_id in (v2.STALE_CODE, v2.DECOY_CODE))
        assert all(f"counterfactual" not in view.memory_text for _ in range(1))
        assert all("variant" not in view.memory_text for _ in range(1))
        assert tuple(view.answer_options) == tuple(f"CODE-{x}" for x in ANSWER_IDS)


def test_adversarial_counterfactual_tamper_is_rejected() -> None:
    group = v2.paired_group("test", "two_hop", 0)
    shifted = replace(group[1], facts=tuple(
        replace(x, position=x.position + 1) if x.position == group[1].facts[0].position
        else x for x in group[1].facts
    ))
    with pytest.raises(ValueError):
        v2.validate_matched_group((group[0], shifted, *group[2:]))
    # Target text is unchanged but unrelated decoy information changes:
    bad_ep = replace(group[1], facts=tuple(
        replace(x, code=v2.STALE_CODE) if x.kind == "value" else x
        for x in group[1].facts
    ))
    with pytest.raises(ValueError, match="non-target|metadata"):
        v2.validate_matched_group((group[0], bad_ep, *group[2:]))

    other_family = v2.generate_paired_episode("test", "direct", 0, 1)
    # A mixed-family group can be rejected by either the balance precheck
    # or the later explicit identity guard; any ValueError is correct.
    with pytest.raises(ValueError):
        v2.validate_matched_group((group[0], other_family, *group[2:]))


def test_no_query_label_rewrite_no_future_memory_and_negative_reachability() -> None:
    ep = v2.generate_paired_episode("development", "direct", 0, 0)
    with pytest.raises(ValueError, match="query"):
        v2.validate_episode(replace(ep, query=ep.query + f" CODE-{ep.gold_answer}"))
    corrupted = replace(ep, facts=tuple(
        replace(x, position=ep.memory_length) if x.subject == ep.entity else x
        for x in ep.facts
    ))
    with pytest.raises(ValueError, match="outside"):
        v2.validate_episode(corrupted)
    assert v2.oracle_read(v2.generate_paired_episode("test", "no_match", 1, 3)) == (
        None, None, None
    )


def test_new_cpu_only_module_does_not_launch_science_or_reuse_old_probes() -> None:
    source = inspect.getsource(v2)
    assert v2.GPU_OR_PAID_TRAINING_AUTHORIZED is False
    assert v2.SCIENTIFIC_QUALITY_CLAIM_AUTHORIZED is False
    assert v2.ORIGINAL_SCIENTIFIC_SEED_CONSUMED == 2_013_161
    for bad in (
        "modal.App(", "modal.run(", "torch.cuda", "optimizer.step(",
        "torch.load(", "checkpoint.load(", "generate_aligned_probe_suite(",
    ):
        assert bad not in source
