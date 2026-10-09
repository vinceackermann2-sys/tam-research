from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch

import tam_research.chm_v3_daec_a_role_ablation_1357 as ablation


def test_preregistered_cpu_only_identity_and_split_isolation() -> None:
    assert ablation.PAID_GPU_AUTHORIZED is False
    assert ablation.HISTORICAL_SCIENTIFIC_SEED_CONSUMED == 2_013_161
    assert ablation.MEMORY_LENGTHS == (128, 256, 512, 1024)
    assert ablation.MODES == ("oracle", "learned", "soft")
    assert ablation.TRAIN_STEPS == 96
    assert ablation.TEST_CASES_PER_SPLIT_LENGTH == 48
    assert ablation.TRAIN_MEMORY_LENGTH == 128
    assert len(set(ablation.SPLIT_SEEDS.values())) == 3
    assert set(ablation.SPLIT_SEEDS.values()).isdisjoint({1350011, 1350101, 1350201, 2013161})
    src=inspect.getsource(ablation)
    for token in ("modal.App(", "modal.run(", "torch.cuda", "get_encoding(\"gpt2\")"):
        assert token not in src


@pytest.mark.parametrize("length", [128,256,512,1024])
def test_fresh_episode_labels_are_reachable_deterministic_and_disjoint(length: int) -> None:
    for idx in (0,1,2,5,9,13):
        ep=ablation.make_episode("development",idx,memory_length=length)
        ep2=ablation.make_episode("development",idx,memory_length=length)
        assert torch.equal(ep.base.key_views,ep2.base.key_views)
        assert torch.equal(ep.observed_role_cues,ep2.observed_role_cues)
        assert ep.base.query_view.shape==(ablation.DIM,)
        assert ep.observed_role_cues.shape==(length,2)
        assert ep.base.overwritten==(idx%4==1)
        if idx%5==0:
            assert ep.base.correct_answer_id is None
            assert ep.base.target_relation_pos is None
            assert ep.base.target_answer_pos is None
        else:
            rel,ans=ep.base.target_relation_pos,ep.base.target_answer_pos
            assert rel is not None and ans is not None and rel!=ans
            assert ep.base.role[rel].item()==ablation.ROLE_RELATION
            assert ep.base.role[ans].item()==ablation.ROLE_RECORD
            assert ep.base.correct_answer_id == ep.base.memory_answer_ids[ans].item()
            assert ep.base.memory_answer_ids[ans].item()>=8000


def test_train_dev_test_episodes_have_distinct_observations() -> None:
    t=ablation.make_episode("train",7,memory_length=128)
    d=ablation.make_episode("development",7,memory_length=128)
    h=ablation.make_episode("test",7,memory_length=128)
    assert not torch.equal(t.base.key_views,d.base.key_views)
    assert not torch.equal(d.base.key_views,h.base.key_views)
    assert not torch.equal(t.base.query_view,h.base.query_view)


def test_learned_and_soft_hard_read_do_not_consult_oracle_role_masks_or_labels() -> None:
    ep=ablation.make_episode("development",7,memory_length=128)
    flipped=torch.where(ep.base.role==ablation.ROLE_RELATION,
        torch.tensor(ablation.ROLE_RECORD),
        torch.where(ep.base.role==ablation.ROLE_RECORD,
                    torch.tensor(ablation.ROLE_RELATION),
                    ep.base.role))
    poisoned=replace(ep,base=replace(
        ep.base,role=flipped,target_relation_pos=None,
        target_answer_pos=None,correct_answer_id=None,
    ))
    for mode in ("learned","soft"):
        arm=ablation.AblationArm(mode).eval()
        a=ablation.hard_read(arm,ep)
        b=ablation.hard_read(arm,poisoned)
        assert (a["first"],a["second"],a["answer"]) == (b["first"],b["second"],b["answer"])
        assert torch.equal(arm.first_logits(ep),arm.first_logits(poisoned))
    oracle=ablation.AblationArm("oracle").eval()
    assert not torch.equal(oracle.first_logits(ep),oracle.first_logits(poisoned))


def test_three_arms_have_exact_same_address_initialization_and_finite_training_losses() -> None:
    arms={name:ablation.AblationArm(name) for name in ablation.MODES}
    reference=list(arms["oracle"].pointer.parameters())
    for name in ablation.MODES[1:]:
        other=list(arms[name].pointer.parameters())
        assert all(torch.equal(a,b) for a,b in zip(reference,other))
    for idx in (0,1,2):
        ep=ablation.make_episode("train",idx,memory_length=128)
        for name,arm in arms.items():
            arm.zero_grad(set_to_none=True)
            loss=ablation.loss_for_arm(arm,ep)
            assert bool(torch.isfinite(loss))
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in arm.pointer.parameters() if p.grad is not None)


def test_hard_eval_uses_selected_payload_and_reports_no_match_or_overwrite() -> None:
    for mode in ablation.MODES:
        arm=ablation.AblationArm(mode).eval()
        ep=ablation.make_episode("test",1,memory_length=128)
        result=ablation.hard_read(arm,ep)
        assert result["overwrite"] is True and result["negative"] is False
        assert result["first"] is None or 0<=result["first"]<128
        assert result["second"] is None or 0<=result["second"]<128
        epneg=ablation.make_episode("test",0,memory_length=128)
        neg=ablation.hard_read(arm,epneg)
        assert neg["negative"] is True
        assert neg["false_positive_on_negative"] == (neg["answer"] is not None)
        assert 0<=ablation.soft_target_probability(arm,ep)<=1.0


def test_panel_has_raw_counts_and_no_scientific_authority() -> None:
    for mode in ablation.MODES:
        arm=ablation.AblationArm(mode).eval()
        result=ablation.evaluate_panel(arm,"development",128,cases=8)
        assert result["mode"]==mode and result["cases"]==8
        assert result["positive_count"]+result["negative_count"]==8
        assert 0<=result["hard_answer_correct"]<=8
        assert 0<=result["first_pointer_correct"]<=8
        assert 0<=result["negative_false_positive_count"]<=result["negative_count"]
        assert 0<=result["mean_soft_target_probability"]<=1.0
        if mode=="learned":
            assert result["role_count"]==8*128
        else:
            assert result["role_count"] is None
    script=inspect.getsource(ablation)
    assert "torch.cuda" not in script
    assert "modal.run" not in script


def test_fails_closed_on_non_preregistered_episode_or_panel() -> None:
    with pytest.raises(ValueError):
        ablation.make_episode("train",96,memory_length=128)
    with pytest.raises(ValueError):
        ablation.make_episode("development",48,memory_length=128)
    with pytest.raises(ValueError):
        ablation.make_episode("train",0,memory_length=2048)
    with pytest.raises(ValueError):
        ablation.make_episode("other",0,memory_length=128)
    arm=ablation.AblationArm("oracle")
    with pytest.raises(ValueError):
        ablation.evaluate_panel(arm,"train",128)
    with pytest.raises(ValueError):
        ablation.evaluate_panel(arm,"test",128,cases=49)
