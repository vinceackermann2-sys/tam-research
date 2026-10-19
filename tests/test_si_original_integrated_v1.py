"""Functional fidelity checks for screenshot-inspired ORIGINAL SI system.

These are engineering smoke/causality tests, NOT learning-success scores.
No scientific seeds, hidden simulator labels, model training budget claims,
external services, GPU or sealed heldout tasks.
"""
import math

import pytest
import torch
from torch import nn

from tam_research.si_original_hybrid_v0 import (
    CausalAttention, StateSpaceMixer, SparseExperts,
)
from tam_research.si_original_integrated_v1 import (
    KINDS, PROVENANCE, SIOriginalIntegratedV1, VersionedVerifiedMemory,
)


def _model():
    torch.set_num_threads(2)
    torch.manual_seed(1042)  # ENGINEERING FIXTURE ONLY, no scientific attempt
    return SIOriginalIntegratedV1(width=16, layers=4, experts=4, max_steps=3)


def _steps(batch=1):
    return [
        ("text", torch.tensor([13]*batch,dtype=torch.long)),
        ("image", torch.randn(batch,3,8,8)),
        ("audio", torch.randn(batch,32)),
        ("video", torch.randn(batch,2,3,8,8)),
        ("motor", torch.randn(batch,4)),
    ]


def test_exact_original_attention_recurrence_and_sparse_expert_blocks_are_used():
    model=_model()
    assert len(model.blocks)==4
    assert [type(b.mixer) for b in model.blocks]==[
        CausalAttention, StateSpaceMixer, CausalAttention, StateSpaceMixer]
    assert all(type(b.ffn) is SparseExperts for b in model.blocks)
    assert all(len(b.ffn.experts)==4 for b in model.blocks)
    assert model.cfg.d_model==16
    assert model.cfg.window==12


def test_all_five_modalities_are_real_encoders_and_backpropagate_through_shared_backbone():
    model=_model()
    steps=_steps(batch=2)
    x=model.encode(steps)
    assert x.shape==(2,5,16)
    assert len(KINDS)==5
    out=model(steps, enable_search=False, enable_fast=False)
    assert out["logits"].shape==(2,4)
    assert out["action"].shape==(2,)
    assert len(out["expert_routes"])==4
    assert all(r.shape==(2,5) for r in out["expert_routes"])
    assert torch.isfinite(out["balance"])
    loss=out["logits"].square().mean()+.001*out["balance"]
    loss.backward()
    for name,module in (
        ("text",model.text_encoder),("image",model.image_encoder),
        ("audio",model.audio_encoder),("video",model.video_encoder),
        ("motor",model.motor_encoder),
    ):
        grads=[p.grad for p in module.parameters() if p.requires_grad]
        assert grads and any(g is not None and torch.isfinite(g).all()
                             and float(g.abs().sum())>0 for g in grads), name


@pytest.mark.parametrize("kind,value",[
    ("text",torch.zeros(1,3)),
    ("image",torch.zeros(1,3,7,8)),
    ("audio",torch.zeros(1,31)),
    ("video",torch.zeros(1,3,2,8,8)),
    ("motor",torch.zeros(1,5)),
])
def test_no_fake_multimodal_tag_substitution_or_invalid_sensor_shapes(kind,value):
    model=_model()
    with pytest.raises(ValueError):
        model.encode([(kind,value)])
    with pytest.raises(ValueError):
        model.encode([("unknown",torch.tensor([1]))])
    with pytest.raises(ValueError):
        model.encode([("text",torch.tensor([300]))])


def test_adaptive_latent_compute_physically_halts_early_and_uses_more_steps_when_forced():
    model=_model().eval()
    steps=_steps(batch=2)
    with torch.no_grad():
        model.halt.weight.zero_()
        model.halt.bias.zero_()
        model.halt_bias.fill_(12)
        fast=model(steps,enable_search=False)
        model.halt_bias.fill_(-12)
        slow=model(steps,enable_search=False)
    assert fast["latent_steps"].tolist()==[1,1]
    assert slow["latent_steps"].tolist()==[3,3]
    assert not torch.allclose(fast["state"],slow["state"])
    assert fast["search_expansions"]==slow["search_expansions"]==0


def test_actual_depth_two_imagined_action_tree_only_runs_when_verifier_requests_it():
    model=_model().eval()
    steps=_steps()
    with torch.no_grad():
        skipped=model(steps,uncertainty_threshold=1.0)
        searched=model(steps,uncertainty_threshold=0.0)
    assert skipped["searched"].tolist()==[False]
    assert skipped["search_expansions"]==0
    assert searched["searched"].tolist()==[True]
    assert searched["search_expansions"]==20
    assert searched["action"].item() in range(4)
    assert not hasattr(model,"oracle_outcome")
    assert not hasattr(model,"world_hidden_state")


def test_public_fact_writes_feed_forward_retrieval_with_world_boundary():
    model=_model().eval()
    before=_steps()
    after=before+[("text",torch.tensor([12]))]
    with torch.no_grad():
        no_record=model(before,world="A",enable_fast=False,enable_search=False)
        model.ingest_executed(before,1,1.0,after,world="A")
        remembered=model(before,world="A",enable_fast=False,enable_search=False)
        separate=model(before,world="B",enable_fast=False,enable_search=False)
    assert len(model.memory.rows)==1
    assert float(no_record["retrieval"].abs().max())==0
    assert float(remembered["retrieval"].abs().max())>0
    assert torch.allclose(separate["retrieval"],torch.zeros_like(separate["retrieval"]))
    assert not torch.allclose(remembered["state"],no_record["state"])


def test_inference_time_fast_weights_change_from_only_actual_reward():
    model=_model().eval()
    before=_steps()
    after=before+[("text",torch.tensor([2]))]
    with torch.no_grad():
        base=model(before,enable_memory=False,enable_search=False)
        model.ingest_executed(before,action=2,reward=1.0,after=after,world="A")
        updated=model(before,enable_memory=False,enable_search=False)
        no_fast=model(before,enable_memory=False,enable_fast=False,enable_search=False)
    assert model.fast_updates==1
    assert float(model.fast_v.abs().sum())>0
    assert not torch.allclose(updated["state"],no_fast["state"],atol=1e-8,rtol=1e-8)
    assert not torch.allclose(base["state"],updated["state"],atol=1e-8,rtol=1e-8)


def test_verified_only_experience_never_accepts_future_or_forged_reward():
    model=_model()
    before=_steps()
    after=before+[("text",torch.tensor([2]))]
    with pytest.raises(ValueError):
        model.ingest_executed(before,2,1.0,after,world="A",provenance="oracle")
    with pytest.raises(ValueError):
        model.ingest_executed(before,2,2.0,after,world="A")
    with pytest.raises(ValueError):
        model.ingest_executed(before,2,float("nan"),after,world="A")
    with pytest.raises(ValueError):
        model.ingest_executed(before,3,1.0,before,world="A")
    assert model.fast_updates==0 and not model.memory.rows and not model.replay


def test_verifier_and_world_transition_are_trained_from_executed_replay():
    model=_model()
    before=_steps()
    for i in range(3):
        after=before+[("text",torch.tensor([40+i]))]
        model.ingest_executed(before,i,1.0 if i==0 else -1.0,
                              after,world="A")
    assert len(model.replay)==3
    weights=model.transition[0].weight.detach().clone()
    verifier=model.verifier[-1].weight.detach().clone()
    optimizer=torch.optim.AdamW(
        list(model.transition.parameters())+
        list(model.reward_head.parameters())+
        list(model.verifier.parameters()),lr=.005)
    loss=model.consolidate_verified_replay(optimizer,count=3)
    assert math.isfinite(loss) and loss>=0
    assert not torch.equal(weights,model.transition[0].weight)
    assert not torch.equal(verifier,model.verifier[-1].weight)


def test_skill_adapter_is_separate_and_can_be_trained_without_overwriting_backbone():
    model=_model()
    model.add_skill("robotics")
    with pytest.raises(ValueError):
        model.add_skill("robotics")
    before=_steps()
    base=model(before,skill=None,enable_search=False)["state"]
    first=model(before,skill="robotics",enable_search=False)["state"]
    assert torch.allclose(first,base,atol=1e-7)
    frozen=model.blocks[0].norm1.weight.detach().clone()
    opt=torch.optim.SGD(model.skills["robotics"].parameters(),lr=.1)
    predicted=model(before,skill="robotics",enable_search=False)["logits"]
    loss=predicted[:,0].square().sum()
    opt.zero_grad();loss.backward();opt.step()
    modified=model(before,skill="robotics",enable_search=False)["state"]
    assert not torch.allclose(modified,base)
    assert torch.equal(frozen,model.blocks[0].norm1.weight)
    assert len(model.skills)==1


def test_memory_and_fast_weights_survive_explicit_serialization_and_rehydration():
    model=_model().eval()
    before=_steps()
    after=before+[("text",torch.tensor([17]))]
    model.ingest_executed(before,0,1.0,after,world="A")
    serialized=model.export_persistent_state()
    fresh=_model()
    fresh.import_persistent_state(serialized)
    assert fresh.memory.dump()==model.memory.dump()
    assert torch.equal(fresh.fast_u,model.fast_u)
    assert torch.equal(fresh.fast_v,model.fast_v)
    assert fresh.fast_updates==1
    with pytest.raises(ValueError):
        fresh.import_persistent_state({**serialized,"fast_updates":-1})
    with pytest.raises(ValueError):
        fresh.import_persistent_state({**serialized,"schema":"bad"})


def test_memory_rejects_unverified_arbitrary_entries_and_bad_serialized_source():
    memory=VersionedVerifiedMemory(4,capacity=2)
    with pytest.raises(ValueError):
        memory.write("A",torch.ones(4),torch.ones(4),provenance="model_prediction")
    memory.write("A",torch.ones(4),torch.ones(4),provenance=PROVENANCE)
    memory.write("B",torch.ones(4),torch.ones(4)*2,provenance=PROVENANCE)
    assert len(memory.rows)==2
    assert torch.equal(memory.read("B",torch.ones(4)),torch.ones(4)*2)
    assert torch.equal(memory.read("C",torch.ones(4)),torch.zeros(4))
    invalid=memory.dump()
    invalid[0]["provenance"]="untrusted"
    with pytest.raises(ValueError):
        memory.load(invalid)


def test_replay_capacity_is_bounded_and_no_claim_of_scientific_learning():
    model=SIOriginalIntegratedV1(width=16,layers=4,experts=4,
                                 replay_capacity=2,memory_capacity=2)
    before=_steps()
    for i in range(4):
        model.ingest_executed(
            before,i%4,1.0 if i%2 else 0.0,
            before+[("text",torch.tensor([10+i]))],world="A")
    assert len(model.memory.rows)==len(model.replay)==2
    assert model.fast_updates==4
    assert not hasattr(model,"scientific_breakthrough")
