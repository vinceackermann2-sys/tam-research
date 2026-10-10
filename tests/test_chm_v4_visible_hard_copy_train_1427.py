from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
import inspect

import pytest

import tam_research.chm_v4_visible_hard_copy_train_1427 as reader
import tam_research.chm_v4_independent_binding_benchmark_1421 as v4


@pytest.mark.parametrize("family", v4.FAMILIES)
@pytest.mark.parametrize("entity_idx", (0,1,2,13,39,63))
def test_reader_only_sealed_text_agrees_with_train_oracle_for_every_rotation(
    family, entity_idx,
):
    group = v4.train_counterfactual_group(family,entity_idx)
    assert len(group) == 8
    for ep in group:
        assert ep.split == "train"
        sealed = v4.seal_input(ep)
        # Reader has no reference to hidden ep fields or the evaluator oracle.
        result = reader.read_visible_text(sealed)
        actual_code, actual_relation, actual_value = v4.oracle_read(ep)
        actual_first = actual_relation if actual_relation is not None else actual_value
        assert result.answer_id == ep.gold_answer == actual_code
        assert result.first_physical_source == actual_first
        assert result.payload_physical_source == actual_value
        if family == "no_match":
            assert result.action == "abstain" and result.routing == "absent"
        else:
            assert result.action == "copy"
            assert result.routing == ("latest_link" if family=="two_hop"
                                      else "latest_direct")
    if family in v4.POSITIVES:
        assert len({reader.read_visible_text(v4.seal_input(x)).answer_id
                    for x in group}) == 8


def test_full_2048_train_case_rule_diagnostic_is_not_model_performance():
    report = reader.train_only_visible_reference_audit()
    assert report["classification"] == (
        "CHM_V4_1427_VISIBLE_HARD_COPY_TRAIN_ONLY_RULE"
    )
    assert report["train_cases"] == 2048
    assert report["train_entities_per_family"] == 64
    for field in ("train_oracle_answer_correct_by_family",
                  "train_first_source_correct_by_family",
                  "train_payload_source_correct_by_family"):
        assert report[field] == dict.fromkeys(v4.FAMILIES,512)
    assert report["visible_rule_routing_counts"] == {
        "latest_direct":1024, "latest_link":512, "absent":512,
    }
    assert report["development_test_cases_constructed"] is False
    assert report["trained_model_forward_calls"] == 0
    assert report["optimizer_steps"] == 0
    assert report["gpu_used"] is False
    assert report["new_scientific_attempt"] is False
    assert report["historical_scientific_stop"] == "CHM_V3_100M_DAEC_STAGE_C_STOP"
    assert report["historical_seed_consumed"] == 2013161
    assert "NOT learned" in report["interpretation"]


def test_train_only_audit_refuses_to_generate_dev_or_test(monkeypatch):
    original = v4.generate_episode
    seen: Counter[str] = Counter()

    def guard(split, family, index, variant):
        seen[split] += 1
        if split != "train":
            raise AssertionError("CI attempted a heldout episode")
        return original(split,family,index,variant)
    monkeypatch.setattr(v4, "generate_episode", guard)
    reader.train_only_visible_reference_audit()
    assert seen == Counter({"train":2048})


def test_last_overwrite_wins_and_rebound_document_is_followed():
    for idx in (0,1,7,21,63):
        overwrite = v4.train_counterfactual_group("overwrite",idx)[2]
        code, rel, value = v4.oracle_read(overwrite)
        result = reader.read_visible_text(v4.seal_input(overwrite))
        assert rel is None and result.first_physical_source == value
        assert result.answer_id == code
        twohop = v4.train_counterfactual_group("two_hop",idx)[4]
        link_positions = sorted(r.position for r in twohop.records
                                if r.role == "relation" and r.subject == twohop.entity)
        assert len(link_positions)==2
        gold, relation, target = v4.oracle_read(twohop)
        found = reader.read_visible_text(v4.seal_input(twohop))
        assert found.first_physical_source == relation == link_positions[-1]
        assert found.payload_physical_source == target
        assert found.answer_id == gold == twohop.gold_answer


def test_reject_missing_ambiguous_query_and_metadata_episode_input():
    ep = v4.train_counterfactual_group("direct",0)[0]
    view = v4.seal_input(ep)
    with pytest.raises(TypeError,match="sealed"):
        reader.read_visible_text(ep)
    with pytest.raises(ValueError,match="exactly one"):
        reader.read_visible_text(replace(view,query="Find current code."))
    with pytest.raises(ValueError,match="exactly one"):
        reader.read_visible_text(replace(view,query=view.query+" V4-train-E9999"))
    with pytest.raises(ValueError,match="options"):
        reader.read_visible_text(replace(view,answer_options=("CODE-18001",)))


def test_reject_duplicate_memory_sources_unrecognized_grammar_and_bag_tampering():
    ep = v4.train_counterfactual_group("direct",0)[0]
    view = v4.seal_input(ep)
    with pytest.raises(ValueError,match="duplicate|unsorted"):
        reader.read_visible_text(replace(
            view,memory_text=view.memory_text+"\n"+view.memory_text.splitlines()[0]
        ))
    with pytest.raises(ValueError,match="grammar|malformed"):
        reader.read_visible_text(replace(
            view,memory_text=view.memory_text.replace("CODE-","VAL-",1)
        ))
    with pytest.raises(ValueError,match="candidate"):
        reader.read_visible_text(replace(
            view,memory_text=view.memory_text.replace("CODE-18002","CODE-18001",1)
        ))
    with pytest.raises(ValueError,match="memory"):
        reader.read_visible_text(replace(view,memory_text=""))


def test_fail_closed_on_active_rebound_relation_to_nonexistent_document():
    ep = v4.train_counterfactual_group("two_hop",0)[0]
    value, pos, source = v4.oracle_read(ep)
    assert value in v4.CODES and pos != source
    view = v4.seal_input(ep)
    lines = view.memory_text.splitlines()
    original = next(i for i,line in enumerate(lines)
                    if line.startswith(f"[{pos:04d}]"))
    line=lines[original]
    assert "V4-train-DOC0000" in line
    lines[original] = line.replace("V4-train-DOC0000", "V4-train-NOTDOC0000")
    with pytest.raises(ValueError,match="no readable document payload"):
        reader.read_visible_text(replace(
            view,memory_text="\n".join(lines)
        ))


def test_static_ast_enforces_no_model_gpu_optimizers_or_heldout_calls():
    source = inspect.getsource(reader)
    tree = ast.parse(source)
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    forbidden = {
        "torch.cuda", "torch.load", "modal.App", "modal.run",
        "model.forward", "optimizer.step", "loss.backward",
        "generate_episode", "v4.generate_episode", "torch.save",
    }
    assert not forbidden.intersection(calls)
    assert "read_visible_text(visible)" in inspect.getsource(
        reader.train_only_visible_reference_audit
    )
    assert "oracle_read(ep)" in inspect.getsource(
        reader.train_only_visible_reference_audit
    )
    fn=inspect.getsource(reader.train_only_visible_reference_audit)
    assert fn.index("read_visible_text(visible)") < fn.index("oracle_read(ep)")
    assert "oracle_read(" not in inspect.getsource(reader.read_visible_text)
    assert "ep.gold_answer" not in inspect.getsource(reader.read_visible_text)
    assert v4.SPLIT_COUNTS == {"train":64,"development":16,"test":32}
    assert reader.HELDOUT_SCORING_AUTHORIZED is False
    assert reader.GPU_AUTHORIZED is False
    assert reader.SCIENTIFIC_RUN_AUTHORIZED is False
