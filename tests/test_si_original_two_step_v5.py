import torch
from scripts.si_original_planning_transfer_v3 import world, advance
from scripts.si_original_two_step_v5 import true_two_step, ridge_two_step, learned_two_step, score_plan
from scripts.si_original_fast_memory_v4 import FastEpisodicMemory
from tam_research.si_original_hybrid_v0 import OriginalSIHybrid, TransformerControl, SIConfig


def test_two_step_oracle_matches_replayed_actions():
    for fam in ('direct','delayed','switch','nonlinear'):
        ex=world(5,20,97900,fam)
        truth=true_two_step(ex,fam)
        # Replay action pair 1,2 directly from the same private model.
        a1=torch.eye(4)[1:2].expand(5,-1)
        a2=torch.eye(4)[2:3].expand(5,-1)
        first=advance(ex[0][:,-1],a1,ex[-1],ex[3],ex[4],fam)
        second=advance(first,a2,a1,ex[3],ex[4],fam)
        assert torch.allclose(truth[:,6],second,atol=1e-6)


def test_two_step_candidate_values_and_no_oracle_leakage():
    data=world(2,20,97901,'delayed')
    # Hidden matrices and query targets can be changed without altering model predictions.
    torch.manual_seed(97901)
    for ctor in (OriginalSIHybrid,TransformerControl):
        model=FastEpisodicMemory(ctor(SIConfig())).eval()
        predicted=learned_two_step(model,data)
        changed=list(data)
        changed[2]=data[2]+234
        changed[3]=data[3]+23
        changed[4]=data[4]-45
        changed[5]=data[5]+13
        assert predicted.shape==(2,16,3)
        assert torch.allclose(predicted,learned_two_step(model,changed),atol=1e-6)
        assert torch.isfinite(predicted).all()
        assert torch.isfinite(ridge_two_step(data)).all()
        assert torch.allclose(ridge_two_step(data),ridge_two_step(changed),atol=1e-6)


def test_oracle_plan_zero_regret():
    ex=world(8,20,97902,'nonlinear')
    states=true_two_step(ex,'nonlinear')
    goal=states[:,11]+.04
    val=score_plan(states,states,goal)
    assert val[0]==1
    assert val[1].abs()<1e-7
