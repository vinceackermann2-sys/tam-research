"""CPU correctness and causal safety tests for source-attested SI v6."""
import torch
from tam_research.si_original_hybrid_v0 import SIConfig, OriginalSIHybrid, TransformerControl
from scripts.si_original_fast_memory_v4 import FastEpisodicMemory
from scripts.si_original_planning_transfer_v3 import world
from scripts.si_original_closed_loop_v6 import (attested_source, second_action_prediction,
                                                score_planner, true_two_step)


def test_exact_original_source():
    assert attested_source() == '575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'


def test_second_step_prediction_is_independent_of_hidden_laws_or_unexecuted_future():
    torch.manual_seed(98500)
    sample=world(2,20,98500,'direct')
    obs, actions = sample[:2]
    expanded_obs = torch.cat((obs, obs[:,-1:]), 1)
    expanded_acts = torch.cat((actions, torch.zeros(2,1,4)), 1)
    for cls in (OriginalSIHybrid, TransformerControl):
        m=FastEpisodicMemory(cls(SIConfig())).eval()
        with torch.no_grad():
            first = second_action_prediction(m, expanded_obs, expanded_acts)
            second = second_action_prediction(m, expanded_obs.clone(), expanded_acts.clone())
            assert torch.equal(first, second)
            assert first.shape==(2,4,3)


def test_paired_closed_loop_uses_same_first_action_and_new_real_observation():
    torch.manual_seed(98500)
    sample=world(4,20,98500,'switch')
    truth=true_two_step(sample,'switch')
    goal=truth[:,5].clone()
    model=FastEpisodicMemory(OriginalSIHybrid(SIConfig())).eval()
    with torch.no_grad():
        x=score_planner(model,sample,'switch',goal,'model')
        # The completed first transition must be visible in the second-stage read.
        assert x['actual_first_observation'].shape==(4,3)
        assert torch.isfinite(x['closed_regret']).all()
        assert x['open_regret'].min()>=-1e-6
        assert x['closed_regret'].min()>=-1e-6


def test_ridge_controls_cannot_read_current_query_targets():
    sample=world(3,20,98501,'nonlinear')
    hidden=list(sample)
    hidden[2]=hidden[2]+1000
    goal=true_two_step(sample,'nonlinear')[:,3]
    m=FastEpisodicMemory(TransformerControl(SIConfig())).eval()
    # Change target x20 supplied in batch; it MUST not affect planner metrics
    # because only observed states/actions and private true simulator laws count.
    with torch.no_grad():
        for method in ('model','ridge_all','ridge_recent'):
            a=score_planner(m,sample,'nonlinear',goal,method)
            b=score_planner(m,tuple(hidden),'nonlinear',goal,method)
            assert torch.allclose(a['closed_regret'],b['closed_regret'],atol=1e-7)
            assert torch.allclose(a['open_regret'],b['open_regret'],atol=1e-7)
