"""SI original v14: causal three-sensor consistency and fault-isolation ablation.

All learned neural weights/trainers imported UNCHANGED from the canonical
original SI v12/v13 stack. The independent third calibration sensor is a
new *observed* measurement: the environment simulates it from private
calibration truth, but the neural policy only receives it through an explicitly
priced sensor-consistency / substitution rule. This is a hand-designed
redundancy mechanism, not generalized world-model discovery.
"""
from __future__ import annotations
import argparse
import json
import time
import torch
from scripts.si_original_unknown_fallback_v13 import make_world, unknown_model_belief, gate_policy
from scripts.si_original_belief_filter_v12 import train_one, verify_original
from scripts.si_original_contextual_value_v10 import feedback,rand_buy

FAMILIES = ('base','one_flip','double_flip','reference_drift',
            'reference_drift_early','reference_dropout','anchor_drift','weak_calibration')
ANCHOR_MU=1.8
ANCHOR_FEE=.025
REF_THRESHOLD=.5
UNKNOWN_THRESHOLD=.5


def world_with_anchor(n,seed,family):
    base_family = 'base' if family in ('reference_drift_early','reference_dropout','anchor_drift') else family
    e=make_world(n,seed,base_family)
    # Simulated physical independent sensor: the hidden calibration modes are
    # used ONLY to generate the sensor reading, never in any inference code.
    g=torch.Generator().manual_seed(seed+843427)
    a=e['cal_truth']*ANCHOR_MU+torch.randn(n,6,generator=g)
    e['anchor_cal']=a
    e['anchor_mu']=torch.full((n,),ANCHOR_MU)
    if family=='reference_drift_early':
        e['y_cal']=e['y_cal'].clone();e['y_cal'][:,:3]*=-1
    elif family=='reference_dropout':
        e['y_cal']=e['y_cal'].clone();e['y_cal'][:,3:]=5*torch.randn(n,3,generator=g)
    elif family=='anchor_drift':
        e['anchor_cal']=e['anchor_cal'].clone();e['anchor_cal'][:,3:]*=-1
    return e


def consistency(e):
    """Normalized covariance of nominal reference with independent anchor.

    Uses only physically observed readings and stated strengths. A low
    time-average is evidence the TWO sensors disagree, NOT proof the nominal
    reference rather than the independent anchor is the faulty one.
    """
    return (e['y_cal']*e['anchor_cal'] /
            (e['mu_ref'][:,None]*e['anchor_mu'][:,None])).mean(1)


def repaired_observation(e):
    """Replace suspect nominal reference with independently observed sensor.

    Express anchor reading on nominal reference strength scale. The new
    observation has a different noise law; call this model mismatch explicitly.
    """
    new=dict(e)
    new['y_cal']=e['anchor_cal']*(e['mu_ref']/e['anchor_mu'])[:,None]
    return new


@torch.no_grad()
def score_policy(raw,bel,e,mode,shuffler_seed):
    n=len(e['z'])
    disagreement=consistency(e)<REF_THRESHOLD
    unknown,_=unknown_model_belief(e)
    raw_gate=unknown>UNKNOWN_THRESHOLD
    rand=torch.Generator().manual_seed(shuffler_seed+99197)
    shuffle=disagreement[torch.randperm(n,generator=rand)]
    if mode=='raw':
        return _execute(raw,e,False,torch.zeros(n,dtype=torch.bool))
    if mode=='belief':
        return _execute(bel,e,False,torch.zeros(n,dtype=torch.bool))
    if mode=='v13_fallback':
        return _execute_mix(raw,bel,e,raw_gate,False,torch.zeros(n,dtype=torch.bool))
    if mode=='always_anchor_belief':
        return _execute(bel,repaired_observation(e),True,torch.ones(n,dtype=torch.bool))
    if mode in ('consistency_repair','shuffled_repair','repair_plus_v13_fallback'):
        repair=disagreement if mode!='shuffled_repair' else shuffle
        # Two separately evaluated worlds, chosen PER episode before actions.
        e_fix=repaired_observation(e)
        if mode=='repair_plus_v13_fallback':
            # Coexisting original v13 OOD rule. Selector sees only real x/y,
            # never hidden simulator labels or future measurements.
            a=_actions(raw,e);b=_actions(bel,e)
            a_fix=_actions(raw,e_fix);b_fix=_actions(bel,e_fix)
            base=_select_actions(a,b,raw_gate)
            fixed=_select_actions(a_fix,b_fix,raw_gate)
        else:
            base=_actions(bel,e);fixed=_actions(bel,e_fix)
        selected=_select_actions(fixed,base,repair)
        return _summarize(selected,e,True,repair,raw_gate if mode=='repair_plus_v13_fallback' else torch.zeros_like(repair))
    raise ValueError(mode)


def _actions(net,e):
    q=net.values(e).sigmoid()
    purchased=q[:,1]-q[:,0]>e['c']
    chosen=net.decision_logits(e,purchased)>=0
    return purchased,chosen,q


def _select_actions(a,b,g):
    return tuple(torch.where(g[:,None],x,y) if x.ndim==2 else torch.where(g,x,y)
                 for x,y in zip(a,b))


def _summarize(actions,truth_env,priced,repair,raw_gate):
    buy,action,q=actions
    success,_,reward=feedback(truth_env,action,buy)
    if priced:reward=reward-ANCHOR_FEE
    return dict(reward=float(reward.mean()),accuracy=float(success.mean()),
                probe_rate=float(buy.float().mean()),
                paid_anchor=bool(priced),anchor_fee=ANCHOR_FEE if priced else 0.,
                repaired=float(repair.float().mean()),raw_fallback=float(raw_gate.float().mean()))


def _execute(net,e,priced,repair):
    return _summarize(_actions(net,e),e,priced,repair,torch.zeros_like(repair))


def _execute_mix(raw,bel,e,g,priced,repair):
    return _summarize(_select_actions(_actions(raw,e),_actions(bel,e),g),e,priced,repair,g)


@torch.no_grad()
def evaluate_cached(raw,bel,e,logseed):
    """Equivalent to score_policy variants, but only four model-action calls.

    Cache is per-world, per-architecture. No hidden metadata controls routing.
    Random negative-control permutation remains seeded and family-matched.
    """
    fault=consistency(e)<REF_THRESHOLD
    unknown,_=unknown_model_belief(e)
    raw_gate=unknown>UNKNOWN_THRESHOLD
    g=torch.Generator().manual_seed(logseed+99197)
    shuffled=fault[torch.randperm(len(fault),generator=g)]
    repaired=repaired_observation(e)
    a=_actions(raw,e)
    b=_actions(bel,e)
    afix=_actions(raw,repaired)
    bfix=_actions(bel,repaired)
    zeros=torch.zeros_like(fault)
    ones=torch.ones_like(fault)
    base_fallback=_select_actions(a,b,raw_gate)
    repaired_fallback=_select_actions(afix,bfix,raw_gate)
    result={
        'raw':_summarize(a,e,False,zeros,zeros),
        'belief':_summarize(b,e,False,zeros,zeros),
        'v13_fallback':_summarize(base_fallback,e,False,zeros,raw_gate),
        'always_anchor_belief':_summarize(bfix,e,True,ones,zeros),
        'consistency_repair':_summarize(_select_actions(bfix,b,fault),e,True,fault,zeros),
        'shuffled_repair':_summarize(_select_actions(bfix,b,shuffled),e,True,shuffled,zeros),
        'repair_plus_v13_fallback':_summarize(
            _select_actions(repaired_fallback,base_fallback,fault),e,True,fault,raw_gate),
    }
    return result


@torch.no_grad()
def detector_metrics(e):
    c=consistency(e)
    v13,_=unknown_model_belief(e)
    return dict(consistency_mean=float(c.mean()),
                fault_flag_fraction=float((c<REF_THRESHOLD).float().mean()),
                v13_unknown_fraction=float((v13>UNKNOWN_THRESHOLD).float().mean()))


def run(seed,actor_steps=180,critic_steps=200,batch=96,n_eval=2048):
    torch.set_num_threads(2)
    source=verify_original()
    worlds={fam:world_with_anchor(n_eval,seed*1000000+83000+131*j,fam)
            for j,fam in enumerate(FAMILIES)}
    t=time.perf_counter()
    panel={'seed':seed,'source_blob':source,'params':{},'cpu_train_seconds':{},
           'actor_steps':actor_steps,'critic_steps':critic_steps,'n_eval':n_eval,
           'anchor_fee':ANCHOR_FEE,'anchor_mu':ANCHOR_MU,
           'detector':{k:detector_metrics(v) for k,v in worlds.items()},'models':{}}
    modes=('raw','belief','v13_fallback','always_anchor_belief',
           'consistency_repair','shuffled_repair','repair_plus_v13_fallback')
    for arch in ('hybrid','transformer','gru'):
        raw,tr=train_one(seed,arch,False,actor_steps,critic_steps,batch)
        bel,tb=train_one(seed,arch,True,actor_steps,critic_steps,batch)
        panel['params'][arch]=sum(p.numel() for p in raw.parameters())
        panel['cpu_train_seconds'][arch]=sum(tr.values())+sum(tb.values())
        panel['models'][arch]={fam:evaluate_cached(raw,bel,e,
                          seed*1000000+83000+131*j+419)
                         for j,(fam,e) in enumerate(worlds.items())}
    panel['cpu_total_seconds']=time.perf_counter()-t
    return panel

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=106000)
    p.add_argument('--actor-steps',type=int,default=180)
    p.add_argument('--critic-steps',type=int,default=200)
    p.add_argument('--batch',type=int,default=96)
    p.add_argument('--n-eval',type=int,default=2048)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.actor_steps,a.critic_steps,a.batch,a.n_eval),indent=2))
