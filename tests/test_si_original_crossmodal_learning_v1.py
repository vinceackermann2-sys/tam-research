"""Source-architecture training protocol: no label leak, exact backbone parity."""
import torch
import pytest
from torch import nn
from torch.nn import functional as F

from tam_research.si_original_hybrid_v0 import (
    SIHybridBlock, DenseBlock,
)
from tam_research.si_original_integrated_v1 import SIOriginalIntegratedV1
from tam_research.si_original_crossmodal_learning_v1 import (
    _public_batch, _shuffled_cue_batch, _configure_models,
    _policy_logits, UPDATES, BATCH, WIDTH, LAYERS, EXPERTS,
    TRAIN_ID_FIRST, VALIDATION_ID_FIRST,
    SIIntegratedTransformerControl,
)


def test_actual_user_integrated_source_and_dense_control_same_entire_periphery():
    hybrid, transformer=_configure_models()
    assert type(hybrid) is SIOriginalIntegratedV1
    assert type(transformer) is SIIntegratedTransformerControl
    assert all(type(b) is SIHybridBlock for b in hybrid.blocks)
    assert all(type(b) is DenseBlock for b in transformer.blocks)
    assert len(hybrid.blocks)==len(transformer.blocks)==LAYERS
    assert hybrid.width==transformer.width==WIDTH
    assert hybrid.cfg.experts==EXPERTS
    a,b=hybrid.state_dict(),transformer.state_dict()
    common=[k for k in a if k in b and a[k].shape==b[k].shape
            and not k.startswith("blocks.")]
    assert len(common)>20
    assert all(torch.equal(a[k],b[k]) for k in common)
    assert len(hybrid.skills)==len(transformer.skills)==0


def test_public_generator_no_target_in_query_and_all_five_sensor_paths():
    steps,target,query=_public_batch(11001,64)
    assert [kind for kind,_ in steps]==[
        "text","image","audio","video","motor"]
    assert target.shape==query.shape==(64,)
    assert set(target.tolist())=={0,1,2,3}
    assert set(query.tolist())=={0,1,2,3}
    assert steps[0][1].shape==(64,)
    assert steps[1][1].shape==(64,3,8,8)
    assert steps[2][1].shape==(64,32)
    assert steps[3][1].shape==(64,2,3,8,8)
    assert steps[4][1].shape==(64,4)
    assert torch.equal(steps[4][1],F.one_hot(query,4).float())
    # Each real query targets exactly one of the four public cues.
    decoded=torch.stack((
        steps[0][1]-20,
        ((steps[1][1][:,0,0,0]-.10)/.20).round().long(),
        ((steps[2][1][:,0]-.10)/.20).round().long(),
        ((steps[3][1][:,0,0,0,0]-.10)/.20).round().long(),
    ),dim=1)
    assert torch.equal(target,decoded.gather(1,query[:,None]).flatten())


def test_no_shuffled_hidden_target_or_same_train_validation_developer_id():
    assert UPDATES==96 and BATCH==16
    assert TRAIN_ID_FIRST+UPDATES-1<VALIDATION_ID_FIRST
    steps,target,query=_public_batch(TRAIN_ID_FIRST,32)
    noise=_shuffled_cue_batch(steps,seed=20001)
    assert torch.equal(steps[-1][1],noise[-1][1])
    assert all(a[0]==b[0] and a[1].shape==b[1].shape
               for a,b in zip(steps,noise))
    assert any(not torch.equal(a[1],b[1])
               for a,b in zip(steps[:-1],noise[:-1]))
    with pytest.raises(ValueError):
        _public_batch(1,0)
    with pytest.raises(ValueError):
        _public_batch(-1,1)


@pytest.mark.parametrize("kind",["hybrid","transformer"])
def test_real_one_step_cpu_trainability_of_each_source_backbone(kind):
    torch.set_num_threads(2)
    hybrid,transformer=_configure_models()
    model=hybrid if kind=="hybrid" else transformer
    model.train()
    steps,target,_=_public_batch(11234,6) # separate correctness-only dev ID
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    before=model.text_encoder.weight.detach().clone()
    direct=model(steps,enable_memory=False,enable_fast=False,
                 enable_search=False,reason_budget=3)
    assert direct["logits"].shape==(6,4)
    loss=F.cross_entropy(direct["logits"],target)+.001*direct["balance"]
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True);loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum())>0
               for p in model.blocks[0].parameters())
    assert float(model.text_encoder.weight.grad.abs().sum())>0
    opt.step()
    assert not torch.equal(model.text_encoder.weight,before)
    model.eval()
    with torch.no_grad():
        after=_policy_logits(model,steps)
    assert after.shape==(6,4) and torch.isfinite(after).all()


def test_explicitly_disabled_untrained_search_and_caller_memory_for_pilot():
    hybrid,_=_configure_models()
    steps,_,_=_public_batch(11234,2)
    out=hybrid(steps,enable_memory=False,enable_fast=False,
               enable_search=False,reason_budget=3)
    assert not bool(out["searched"].any())
    assert out["search_expansions"]==0
    assert not hybrid.memory.rows and not hybrid.replay
    assert hybrid.fast_updates==0
    assert all(float(x.abs().sum())==0 for x in out["retrieval"])


def test_public_validation_not_claimed_independent_sealed_heldout():
    import tam_research.si_original_crossmodal_learning_v1 as module
    assert not hasattr(module,"scientific_go")
    assert not hasattr(module,"heldout_truth")
    assert not hasattr(module,"oracle_reward")
    assert module.VALIDATION_BATCHES==8
    assert module.VALIDATION_BATCH==32
