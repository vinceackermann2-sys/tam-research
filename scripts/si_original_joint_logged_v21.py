"""SI original v21: two-stage factual feedback with costs factored out.

Uses an unchanged, source-attested v19 original hybrid/Transformer/GRU actor;
replaces the second-stage critics with disjoint logged-outcome success heads
and learns first-action continuation reward using only factual trajectories
matching the frozen deployed second policy. No Bayesian purchase labels.
"""
from __future__ import annotations
import argparse, copy, json, time
import torch
from torch import nn
import torch.nn.functional as F
from scripts import si_original_crossfit_value_v19 as v19
from scripts import si_original_logged_value_v18 as v18
from scripts import si_original_direct_first_value_v20 as v20
from scripts.si_original_active_experiment_v17 import subset, score_policy
from scripts import si_original_linear_context_v21 as linear

FAMILIES = v19.EXTENDED_FAMILIES

class FactoredPolicy(nn.Module):
    def __init__(self, frozen):
        super().__init__()
        self.core = copy.deepcopy(frozen)
        for p in self.core.parameters(): p.requires_grad_(False)
        # Cost is a KNOWN public consequence of purchasing: never predict it.
        # Q1 heads predict P(correct | selected arm, observed history), not net reward.
        self.success_a = nn.Linear(3, 2)
        self.success_b = nn.Linear(3, 2)
        with torch.no_grad():
            self.success_a.weight.copy_(self.core.actor.q1_head.weight)
            self.success_a.bias.copy_(self.core.actor.q1_head.bias)
            self.success_b.weight.copy_(self.core.q1_b.weight)
            self.success_b.bias.copy_(self.core.q1_b.bias)
        self.first_head = copy.deepcopy(self.core.actor.q0_head)

    def features(self, e, first):
        return self.core.features(e, first)

    def second_success_logits(self, e, first):
        f = self.features(e, first)
        return self.success_a(f), self.success_b(f)

    def q1(self, e, first):
        qa, qb = self.second_success_logits(e, first)
        success = .5*(qa.sigmoid()+qb.sigmoid())
        other_cost = torch.where(first==1, e['cost_high'],e['cost_low'])
        return success - torch.stack((torch.zeros_like(other_cost), other_cost),dim=1)

    def decision(self, e, first, second=None):
        return self.core.decision(e,first,second)

    def q0(self,e):
        raw=self.first_head(self.core.features(e))
        # First-stage known expenditure, deducted at decision time.
        costs = torch.stack((torch.zeros_like(e['cost_low']), e['cost_low'],e['cost_high']),dim=-1)
        return raw-costs


def fit_second(seed,model,steps=240,batch=96):
    """Randomized logged buy/skip, frozen actor, disjoint observed arm BCE."""
    for p in model.parameters():p.requires_grad_(False)
    for p in list(model.success_a.parameters())+list(model.success_b.parameters()): p.requires_grad_(True)
    params = list(model.success_a.parameters())+list(model.success_b.parameters())
    opt=torch.optim.AdamW(params,lr=.003,weight_decay=0.)
    t0=time.perf_counter(); counts=[0,0,0,0]
    for i in range(steps):
        e=v19.evaluation_world(batch//2,seed*1000000+700000+i,'base')
        n=len(e['mu_low'])
        first=torch.where(v18.random_boolean(n,seed*1000000+700000+i+991111),1,2).long()
        second=v18.random_boolean(n,seed*1000000+700000+i+551111)
        remaining=3-first
        with torch.no_grad():
            action=torch.zeros(n,dtype=torch.bool)
            if (~second).any():action[~second]=model.decision(subset(e,~second),first[~second])>=0
            if second.any():action[second]=model.decision(subset(e,second),first[second],remaining[second])>=0
            observed_correct=v18.logged_feedback(e,action).float()
        qa,qb=model.second_success_logits(e,first)
        left=torch.arange(n)%2==0; right=~left
        loss=qa.sum()*0 + qb.sum()*0
        for head,mask in ((qa,left),(qb,right)):
            if mask.any():
                chosen=head[mask].gather(1,second[mask,None].long()).squeeze(1)
                loss=loss+F.binary_cross_entropy_with_logits(chosen,observed_correct[mask])
        for mask in (left,right):
            for arm in (False,True):counts[(0 if mask is left else 2)+int(arm)] +=int((mask&(second==arm)).sum())
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.)
        opt.step()
    for p in model.parameters():p.requires_grad_(False)
    return {'seconds':time.perf_counter()-t0,'observed_counts':counts}


def fit_first(seed,model,steps=160,batch=96):
    """Q0 is continuation reward excluding known first cost; factual full return."""
    for p in model.parameters():p.requires_grad_(False)
    for p in model.first_head.parameters():p.requires_grad_(True)
    opt=torch.optim.AdamW(model.first_head.parameters(),lr=.003,weight_decay=0.)
    t0=time.perf_counter();counts=[0,0,0]; requested=[0,0,0]
    for i in range(steps):
        e=v19.evaluation_world(batch//2,seed*1000000+800000+i,'base')
        n=len(e['mu_low'])
        first=v18.random_first(n,seed*1000000+800000+i+991111)
        second=v18.random_boolean(n,seed*1000000+800000+i+551111)
        stopped=v18.random_boolean(n,seed*1000000+800000+i+441111)
        with torch.no_grad():
            target, accepted=v20.logged_first_trajectories(e,model,first,second,stopped)
            first_cost=torch.where(first==1,e['cost_low'],torch.where(first==2,e['cost_high'],0.))
            # Target continuation value, no first cost; deterministic first fee
            # deducted by q0() only when the policy compares the three arms.
            target=target+first_cost
        for k in range(3):
            requested[k]+=int((first==k).sum());counts[k]+=int(((first==k)&accepted).sum())
        pred=model.first_head(model.core.features(e)).gather(1,first[:,None]).squeeze(1)
        loss=F.mse_loss(pred[accepted],target[accepted])
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.first_head.parameters(),1.)
        opt.step()
    for p in model.parameters():p.requires_grad_(False)
    return {'seconds':time.perf_counter()-t0,'counts':counts,'requested':requested}


@torch.no_grad()
def second_calibration(model,e,seed):
    n=len(e['mu_low']);first=torch.where(v18.random_boolean(n,seed+100),1,2).long()
    second=v18.random_boolean(n,seed+200);remaining=3-first
    action=torch.zeros(n,dtype=torch.bool)
    if (~second).any():action[~second]=model.decision(subset(e,~second),first[~second])>=0
    if second.any():action[second]=model.decision(subset(e,second),first[second],remaining[second])>=0
    correct=v18.logged_feedback(e,action).float()
    if isinstance(model,FactoredPolicy):
        qa,qb=model.second_success_logits(e,first)
        pred=(qa.sigmoid()+qb.sigmoid())*.5
    else:
        q=model.q1(e,first)
        costs=torch.where(first==1,e['cost_high'],e['cost_low'])
        pred=q+torch.stack((torch.zeros_like(costs),costs),dim=1)
    chosen=pred.gather(1,second.long()[:,None]).squeeze(1)
    return float((chosen-correct).square().mean())


def run(seed,actor_steps=160,critic_steps=240,second_steps=240,first_steps=160,batch=96,n_pairs=512):
    torch.set_num_threads(2)
    attested=v18.verify_source()
    worlds={f:v19.evaluation_world(n_pairs,seed*1000000+81000+131*j,f) for j,f in enumerate(FAMILIES)}
    out={'seed':seed,'canonical_source':attested,'settings':dict(actor_steps=actor_steps,critic_steps=critic_steps,second_steps=second_steps,first_steps=first_steps,batch=batch,n_pairs=n_pairs),'controls':{},'models':{},'linear_contextual':{}}
    for fam,e in worlds.items():out['controls'][fam]={p:score_policy(e,p) for p in ('never','always_low','always_high','always_both','greedy','bayes')}
    simple,simple_t=linear.train(seed,n_train=6000)
    out['linear_contextual']={'training':simple_t,'families':{f:linear.score(simple,e) for f,e in worlds.items()},'prior_note':'Terminal action uses known Gaussian likelihood ratio; purchase values use factual logged ridge regression.'}
    for arch in ('hybrid','transformer','gru'):
        start=time.perf_counter()
        base,base_times=v19.train_one(seed,arch,actor_steps,critic_steps,batch)
        direct,refit=v20.refit_q0(base,seed,'direct',first_steps,batch)
        combined=FactoredPolicy(base)
        st=fit_second(seed,combined,second_steps,batch)
        ft=fit_first(seed,combined,first_steps,batch)
        out['models'][arch]={'base_params':sum(p.numel() for p in base.parameters()),'factored_params':sum(p.numel() for p in combined.parameters()),'times':{'base':base_times,'direct_q0':refit,'factored_q1':st,'factored_q0':ft,'all_cpu':time.perf_counter()-start},'policies':{}}
        for name,model in (('v19',base),('v20_direct_q0',direct),('v21_factored_both',combined)):
            out['models'][arch]['policies'][name]={f:{**v18.score_logged(e,model), 'observed_q1_correct_mse':second_calibration(model,e,seed+191+j*137)} for j,(f,e) in enumerate(worlds.items())}
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for name,val in (('seed',113000),('actor_steps',160),('critic_steps',240),('second_steps',240),('first_steps',160),('batch',96),('n_pairs',512)):
        ap.add_argument('--'+name.replace('_','-'),type=int,default=val)
    v=ap.parse_args()
    print(json.dumps(run(v.seed,v.actor_steps,v.critic_steps,v.second_steps,v.first_steps,v.batch,v.n_pairs),indent=2))
