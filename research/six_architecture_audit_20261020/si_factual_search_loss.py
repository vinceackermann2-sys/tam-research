"""Proposed SI training repair, not an efficacy claim or a trained successor.

The native search consumes value_head, but the original policy/replay objectives
never train it. This loss trains search heads in the SAME logged planning-state
coordinate system using only completed, actually executed public trajectories.
Caller provenance is an assertion, not cryptographic authentication.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F

@dataclass(frozen=True)
class ExecutedPlanningTrajectory:
    # Capture forward()['state'] at each executed decision, then terminal state.
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminal: bool
    provenance: str

def discounted_factual_returns(rewards, discount=.9):
    if rewards.ndim!=1 or not torch.isfinite(rewards).all():raise ValueError('finite rewards required')
    if not 0<=discount<=1:raise ValueError('invalid discount')
    ret=rewards.new_zeros(len(rewards)+1)
    for t in range(len(rewards)-1,-1,-1):ret[t]=rewards[t]+discount*ret[t+1]
    return ret

def factual_search_training_loss(model,trajectory:ExecutedPlanningTrajectory,discount=.9):
    if trajectory.provenance!='executed_public_feedback' or trajectory.terminal is not True:
        raise ValueError('completed executed public trajectory required; no imagined rewards')
    states,actions,rewards=trajectory.states.detach(),trajectory.actions,trajectory.rewards.detach()
    if (states.ndim!=2 or actions.ndim!=1 or rewards.ndim!=1 or len(actions)<1
        or len(states)!=len(actions)+1 or len(rewards)!=len(actions)
        or states.shape[1]!=model.width or actions.dtype!=torch.long
        or bool(((actions<0)|(actions>=4)).any()) or not torch.isfinite(states).all()
        or not torch.isfinite(rewards).all() or bool((rewards.abs()>1).any())):
        raise ValueError('invalid executed state/action/reward sequence')
    if sum(p.numel() for p in model.parameters())>100_000_000:raise ValueError('parameter ceiling')
    before,observed_next=states[:-1],states[1:]
    onehot=F.one_hot(actions,4).to(before)
    predicted_next=before+model.transition(torch.cat((before,onehot),-1))
    errors=(predicted_next-observed_next).square().mean(-1)
    transition_loss=errors.mean()
    reward_loss=F.mse_loss(model.reward_head(predicted_next).flatten(),rewards)
    returns=discounted_factual_returns(rewards,discount)
    value_loss=F.mse_loss(model.value_head(states).flatten(),returns)
    # Inference queries verifier(current planning state), so train at that state.
    verifier_loss=F.mse_loss(torch.sigmoid(model.verifier(before)).flatten(),errors.detach().clamp(0,1))
    return transition_loss+reward_loss+value_loss+.1*verifier_loss,{
        'transition':transition_loss.detach(),'reward':reward_loss.detach(),
        'value':value_loss.detach(),'verifier':verifier_loss.detach()}
