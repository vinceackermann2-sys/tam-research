"""Original SI v20: direct, logged, complete-trajectory first-action value.

Controlled CPU-only ablation of v19. Train a full v19 model ONCE per seed,
then clone identical frozen actor and Q1 critics. Compare two Q0-only fitting
methods under equal extra optimizer steps and fresh identical randomized logs:
(1) observed complete returns for the chosen first arm, accepting purchased
first trajectories only when a separately randomized second action matches the
frozen Q1-deployed second decision; (2) the original double-Q bootstrap.
The filter is independent of hidden outcomes: no unpurchased probe, private
labels, or Bayesian decision/purchase labels enter the learner.
"""
from __future__ import annotations
import argparse, copy, json, time
import torch
import torch.nn.functional as F
from scripts import si_original_crossfit_value_v19 as v19
from scripts import si_original_logged_value_v18 as v18
from scripts.si_original_active_experiment_v17 import subset, score_policy

FAMILIES=v19.EXTENDED_FAMILIES


def logged_first_trajectories(e, model, first, second, stop_random):
    """Executed factual rollouts and Q1 policy-agreement, no counterfactual labels.

    first is randomized {0,1,2}, second is independently randomized 0/1.
    The acceptance event has propensity .5 conditional on bought-first state.
    We do not train Q0 on unmatched trajectories: their returns correspond
    to a different follow-up behavior than the deployed Q1-greedy policy.
    """
    with torch.no_grad():
        n=len(first)
        observed=torch.zeros(n)
        accepted=first==0
        if accepted.any():
            # Stop action is actually executed, correctness is observed.
            ep=subset(e,accepted)
            stopped=stop_random[accepted]
            observed[accepted]=v18.logged_feedback(ep,stopped).float()
        bought=first!=0
        if bought.any():
            ep=subset(e,bought); f=first[bought]; other=3-f
            greedy=model.q1(ep,f).argmax(1).bool()
            logged_buy=second[bought]
            agree=logged_buy==greedy
            accepted[bought]=agree
            chosen=torch.zeros(len(f),dtype=torch.bool)
            if (~logged_buy).any():
                chosen[~logged_buy]=model.decision(subset(ep,~logged_buy),f[~logged_buy])>=0
            if logged_buy.any():
                chosen[logged_buy]=model.decision(subset(ep,logged_buy),f[logged_buy],other[logged_buy])>=0
            correct=v18.logged_feedback(ep,chosen).float()
            cost_first=torch.where(f==1,ep['cost_low'],ep['cost_high'])
            cost_second=torch.where(f==1,ep['cost_high'],ep['cost_low'])
            result=correct-cost_first-logged_buy.float()*cost_second
            observed[bought]=result
        return observed,accepted


def first_logged_q_mse(model,e,seed):
    """Heldout chosen-arm MSE based on factual complete returns and agreement."""
    n=len(e['mu_low'])
    first=v18.random_first(n,seed+101)
    second=v18.random_boolean(n,seed+202)
    stop=v18.random_boolean(n,seed+303)
    target,accepted=logged_first_trajectories(e,model,first,second,stop)
    q=model.q0(e).gather(1,first[:,None]).squeeze(1)
    return dict(mse=float((q[accepted]-target[accepted]).square().mean()),
                accepted_fraction=float(accepted.float().mean()),
                arm_coverage={str(k):int(((first==k)&accepted).sum()) for k in range(3)})


def clone_freeze(model):
    cloned=copy.deepcopy(model)
    for p in cloned.parameters(): p.requires_grad_(False)
    for p in cloned.actor.q0_head.parameters():p.requires_grad_(True)
    return cloned


def refit_q0(base,seed,mode='direct',steps=160,batch=96):
    """Matched extra Q0-only steps; same logged worlds/actions across arms."""
    assert mode in ('direct','bootstrap')
    model=clone_freeze(base)
    q0params=list(model.actor.q0_head.parameters())
    opt=torch.optim.AdamW(q0params,lr=.003,weight_decay=0.)
    wall=time.perf_counter()
    count=[0,0,0];requested=[0,0,0]
    for i in range(steps):
        e=v19.evaluation_world(batch//2,seed*1000000+600000+i,'base')
        n=len(e['mu_low'])
        first=v18.random_first(n,seed*1000000+600000+i+991111)
        second=v18.random_boolean(n,seed*1000000+600000+i+551111)
        stop=v18.random_boolean(n,seed*1000000+600000+i+441111)
        with torch.no_grad():
            target,accepted=logged_first_trajectories(e,model,first,second,stop)
            # Fair same-update control: original double-Q Q0 target computed
            # from actual observed first reading and detached frozen Q1 heads.
            if mode=='bootstrap':
                active=first!=0
                if active.any():
                    selected=subset(e,active);f=first[active]
                    qa,qb=model.critic_pair(selected,f)
                    cost=torch.where(f==1,selected['cost_low'],selected['cost_high'])
                    target[active]=v19.crossfit_bootstrap(qa,qb,cost)
                accepted=torch.ones_like(accepted)
        for k in range(3):
            requested[k]+=int((first==k).sum())
            count[k]+=int(((first==k)&accepted).sum())
        q0=model.q0(e).gather(1,first[:,None]).squeeze(1)
        if not accepted.any():raise RuntimeError('No eligible observed-arm trajectories')
        loss=F.mse_loss(q0[accepted],target[accepted])
        if not torch.isfinite(loss):raise RuntimeError('nonfinite Q0 loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(q0params,1.0);opt.step()
    for p in model.parameters():p.requires_grad_(False)
    return model,{'q0_refit_seconds':time.perf_counter()-wall,
                  'accepted_arm_counts':count,'randomized_arm_counts':requested}


def run(seed,actor_steps=160,critic_steps=240,refit_steps=160,batch=96,n_pairs=512):
    torch.set_num_threads(2)
    blob=v18.verify_source()
    worlds={f:v19.evaluation_world(n_pairs,seed*1000000+81000+131*j,f)
            for j,f in enumerate(FAMILIES)}
    out={'seed':seed,'canonical_blob':blob,'actor_steps':actor_steps,'critic_steps':critic_steps,
         'refit_steps':refit_steps,'batch':batch,'n_pairs':n_pairs,'architectures':{},
         'controls':{f:{p:score_policy(e,p) for p in ('never','always_low','always_high','always_both','greedy','bayes')}
                     for f,e in worlds.items()}}
    for arch in ('hybrid','transformer','gru'):
        # The complete v19 double-Q model is trained exactly ONCE per seed.
        shared,timing=v19.train_one(seed,arch,actor_steps,critic_steps,batch)
        direct,td=refit_q0(shared,seed,'direct',refit_steps,batch)
        boot,tb=refit_q0(shared,seed,'bootstrap',refit_steps,batch)
        nets={'frozen_v19':(shared,{}),'extra_bootstrap':(boot,tb),'direct_logged_return':(direct,td)}
        out['architectures'][arch]={}
        for method,(net,extra) in nets.items():
            out['architectures'][arch][method]={'parameters':sum(p.numel() for p in net.parameters()),
               'base_train_timing':timing,'extra_timing':extra,
               'families':{f:{**v18.score_logged(e,net),
                      'first_logged_calibration':first_logged_q_mse(net,e,seed+191+j*137)}
                      for j,(f,e) in enumerate(worlds.items())}}
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=112000)
    p.add_argument('--actor-steps',type=int,default=160)
    p.add_argument('--critic-steps',type=int,default=240)
    p.add_argument('--refit-steps',type=int,default=160)
    p.add_argument('--batch',type=int,default=96)
    p.add_argument('--n-pairs',type=int,default=512)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.actor_steps,a.critic_steps,a.refit_steps,a.batch,a.n_pairs),indent=2))
