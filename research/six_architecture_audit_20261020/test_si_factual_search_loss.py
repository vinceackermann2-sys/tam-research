"""Synthetic derivative checks, not a research run or real action feedback."""
import pytest
import torch
from torch import nn
from si_factual_search_loss import ExecutedPlanningTrajectory,discounted_factual_returns,factual_search_training_loss

class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__();self.width=8
        self.transition=nn.Linear(12,8);self.reward_head=nn.Linear(8,1)
        self.value_head=nn.Linear(8,1);self.verifier=nn.Linear(8,1)

def test_returns_use_only_observed_completed_rewards():
    r=discounted_factual_returns(torch.tensor([.5,.75]))
    torch.testing.assert_close(r,torch.tensor([1.175,.75,0.]))

def test_all_search_heads_receive_factual_gradient():
    torch.manual_seed(310987)
    model=FixtureModel()
    t=ExecutedPlanningTrajectory(torch.randn(4,8),torch.tensor([0,2,1]),torch.tensor([0.,1.,-.5]),True,'executed_public_feedback')
    loss,_=factual_search_training_loss(model,t);loss.backward()
    for head in ('transition','reward_head','value_head','verifier'):
        assert getattr(model,head).weight.grad.norm()>0

@pytest.mark.parametrize('terminal,provenance',[(False,'executed_public_feedback'),(True,'imagined')])
def test_unobserved_or_incomplete_rollouts_rejected(terminal,provenance):
    t=ExecutedPlanningTrajectory(torch.zeros(2,8),torch.tensor([0]),torch.zeros(1),terminal,provenance)
    with pytest.raises(ValueError):factual_search_training_loss(FixtureModel(),t)
