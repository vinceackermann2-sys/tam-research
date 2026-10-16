"""Original SI v13: causal model-family uncertainty with conservative raw fallback.

Builds explicit known-vs-broad-pattern likelihood from *observable* calibration
pairs only. Reuses unmodified v12 raw and belief trainer for original hybrid,
matched Transformer and small GRU. Evaluates safe fallbacks versus all-raw,
all-belief, shuffled-gate and expanded-Bayes references. CPU only.

Important: this is a hand-designed *expanded* world-model hypothesis class.
It is NOT unsupervised discovery of arbitrary new environmental laws.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import time
import torch
from scripts.si_original_belief_filter_v12 import (episode, Controller, train_one,
    belief, evaluate, verify_original, FAMILIES)
from scripts.si_original_contextual_value_v10 import feedback, rand_buy
from scripts.si_original_paired_drift_v11 import policy_given_polarity_posterior
from scripts.si_original_belief_transfer_v8 import assess_baseline

BASE_FAMILIES = ('base','one_flip','early_flip','double_flip','weak_calibration','reference_drift')
# The last calibration time is constrained to have the final sensor polarity.
# Each pattern is a sign sequence relative to the *current/query* polarity.
ALL_PATTERNS = tuple(tuple(signs)+(1,) for signs in itertools.product((-1,1),repeat=5))
KNOWN_PATTERNS = ((1,1,1,1,1,1),(-1,-1,1,1,1,1),(-1,-1,-1,1,1,1),(-1,-1,-1,-1,1,1))
UNKNOWN_PATTERNS = tuple(p for p in ALL_PATTERNS if p not in KNOWN_PATTERNS)
assert len(UNKNOWN_PATTERNS)==28


def reference_drift(n, seed):
    # Previously unmodeled corruption: *trusted reference* flips its sign
    # during the final three calibration observations (not target sensor).
    e=episode(n,seed,'base')
    e['y_cal']=e['y_cal'].clone()
    e['y_cal'][:,3:] *= -1
    e['private_reference_drift']=torch.ones(n,dtype=torch.bool)
    return e


def make_world(n,seed,family):
    return reference_drift(n,seed) if family=='reference_drift' else episode(n,seed,family)


def evidence_by_pattern(e):
    """log-likelihood for every (z_final, calibration pattern).

    This function NEVER reads latent z, theta, unpurchased probe, true
    changed family or private switchpoint. Known observation noise variance.
    """
    x,y=e['x_cal'],e['y_cal']
    mu=e['cal_mu'][:,None,None,None]
    ref=e['mu_ref'][:,None,None,None]
    s=torch.as_tensor(ALL_PATTERNS,dtype=x.dtype,device=x.device)[None,None,:,:]
    z=torch.as_tensor([1.,-1.],dtype=x.dtype,device=x.device)[None,:,None,None]
    mean=z*s*mu
    x4=x[:,None,None,:];y4=y[:,None,None,:]
    lp=-.5*((x4-mean).square()+(y4-ref).square())
    lm=-.5*((x4+mean).square()+(y4+ref).square())
    return torch.logaddexp(lp,lm).sum(-1)-math.log(2.)


def unknown_model_belief(e):
    """Posterior outside known 4 patterns, plus expanded polarity belief.

    0.5 prior on known pattern group, 0.5 on the 28 other patterns;
    known prior 0.5 stationary/0.5 evenly over 2,3,4 as in v12.
    Returns unknown posterior and expanded P(z_current=+).
    """
    L=evidence_by_pattern(e)
    weights=[]
    for p in ALL_PATTERNS:
        if p==KNOWN_PATTERNS[0]: weights.append(.5*.5)
        elif p in KNOWN_PATTERNS: weights.append(.5*.5/3.)
        else: weights.append(.5/len(UNKNOWN_PATTERNS))
    prior=torch.tensor(weights,dtype=L.dtype,device=L.device).log()
    joint=L+prior[None,None,:]-math.log(2.)
    normalizer=torch.logsumexp(joint.flatten(1),-1)
    prob=(joint-normalizer[:,None,None]).exp()
    outmask=torch.tensor([p not in KNOWN_PATTERNS for p in ALL_PATTERNS],device=L.device)
    unknown=prob[:,:,outmask].sum((1,2))
    pz=prob[:,0].sum(1)
    return unknown,pz


@torch.no_grad()
def model_actions(model,e):
    q=model.values(e).sigmoid()
    purchase=q[:,1]-q[:,0]>e['c']
    final=model.decision_logits(e,purchase)>=0
    success,_,net=feedback(e,final,purchase)
    return {'purchase':purchase,'final':final,'reward':net,'success':success,'q':q}


def gate_policy(raw,believed,e,gate,logseed=0):
    # Gate is permitted to depend ONLY on observed sensor calibration values.
    # Each underlying branch chooses its own purchased probe, then final action.
    a=model_actions(raw,e); b=model_actions(believed,e)
    choose_raw=gate.bool()
    buy=torch.where(choose_raw,a['purchase'],b['purchase'])
    action=torch.where(choose_raw,a['final'],b['final'])
    success,_,net=feedback(e,action,buy)
    logbuy=rand_buy(len(e['z']),logseed)
    a_success,_,_=feedback(e,raw.decision_logits(e,logbuy)>=0,logbuy)
    b_success,_,_=feedback(e,believed.decision_logits(e,logbuy)>=0,logbuy)
    q=torch.where(choose_raw[:,None],a['q'],b['q'])
    actual_success=torch.where(choose_raw,a_success,b_success)
    selected_q=q.gather(1,logbuy.long()[:,None]).squeeze(1)
    return dict(reward=float(net.mean()),accuracy=float(success.mean()),
                probe_fraction=float(buy.float().mean()),fallback_fraction=float(choose_raw.float().mean()),
                logged_arm_brier=float((selected_q-actual_success).square().mean()))


@torch.no_grad()
def evaluate_pair(raw,believed,e,logseed,threshold=.5):
    unknown,pz=unknown_model_belief(e)
    gate=unknown>threshold
    # Negative control: exactly the same fallback count as detection gate,
    # but with assignment shuffled in a deterministic seeded permutation.
    g=torch.Generator().manual_seed(logseed+99999)
    random_gate=gate[torch.randperm(len(gate),generator=g)]
    result={
        'raw':gate_policy(raw,believed,e,torch.ones_like(gate),logseed),
        'belief':gate_policy(raw,believed,e,torch.zeros_like(gate),logseed),
        'fallback':gate_policy(raw,believed,e,gate,logseed),
        'shuffled_gate':gate_policy(raw,believed,e,random_gate,logseed),
        'unknown_posterior_mean':float(unknown.mean()),
        'unknown_gt_threshold':float(gate.float().mean()),
        'expanded_pz_brier':float(((pz-(e['z']>0).float())**2).mean()),
    }
    return result


@torch.no_grad()
def bayes_references(e):
    punknown,pz=unknown_model_belief(e)
    buy,choice,_=policy_given_polarity_posterior(e,pz)
    correct=(choice==e['truth'].gt(0)).float()
    extended={'reward':float((correct-buy.float()*e['c']).mean()),
              'accuracy':float(correct.mean()),'probe_fraction':float(buy.float().mean())}
    return {'expanded_pattern_bayes':extended,
            'stationary_bayes':assess_baseline(e,'bayes'),
            'stationary_always':assess_baseline(e,'always_probe'),
            'stationary_never':assess_baseline(e,'never_probe')}


def run(seed, actor_steps=180, critic_steps=200, batch=96, n_eval=2048, threshold=.5):
    torch.set_num_threads(2)
    source=verify_original()
    worlds={f:make_world(n_eval,seed*1000000+83000+131*i,f)
            for i,f in enumerate(BASE_FAMILIES)}
    report={'seed':seed,'source_blob':source,'actor_steps':actor_steps,
            'critic_steps':critic_steps,'batch':batch,'n_eval':n_eval,
            'threshold':threshold,'models':{},'references':{}}
    for name,e in worlds.items(): report['references'][name]=bayes_references(e)
    for arch in ('hybrid','transformer','gru'):
        raw,t_raw=train_one(seed,arch,False,actor_steps,critic_steps,batch)
        bel,t_bel=train_one(seed,arch,True,actor_steps,critic_steps,batch)
        report['models'][arch]={
            'params_per_policy':sum(p.numel() for p in raw.parameters()),
            'paired_params':sum(p.numel() for p in raw.parameters())+sum(p.numel() for p in bel.parameters()),
            'train_seconds_raw':sum(t_raw.values()),'train_seconds_belief':sum(t_bel.values()),
            'families':{f:evaluate_pair(raw,bel,e,seed+90335,threshold) for f,e in worlds.items()}
        }
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=105000)
    p.add_argument('--actor-steps',type=int,default=180)
    p.add_argument('--critic-steps',type=int,default=200)
    p.add_argument('--batch',type=int,default=96)
    p.add_argument('--n-eval',type=int,default=2048)
    p.add_argument('--threshold',type=float,default=.5)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.actor_steps,a.critic_steps,a.batch,a.n_eval,a.threshold),indent=2))
