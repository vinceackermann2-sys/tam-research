"""SI-original v11: paired v9/v10 methods and causal hidden sensor-polarity drift.

CPU-only mechanism audit. Reuses UNMODIFIED canonical original hybrid,
Transformer, GRU and v9/v10 trainers; v11 only defines the shared evaluator
and a previously unseen within-episode polarity-switch world. No oracle
labels, unseen probe, or hidden switch point enter trained neural policies.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import torch
from scripts import si_original_reward_belief_v9 as v9
from scripts import si_original_contextual_value_v10 as v10
from scripts.si_original_belief_transfer_v8 import make_episode, bayes_policy, posterior_z

FAMILIES = ('base', 'polarity_flip', 'weak_calibration')
SWITCH_CANDIDATES = (2, 3, 4)


def drift_episode(n: int, seed: int):
    """True polarity at query = z; calibration sensor polarity flips at hidden c.

    Calibration i<c uses -z, i>=c uses +z. Reference calibration and true
    query remain unchanged. c is sampled from {2,3,4} and not given to models.
    """
    d = make_episode(n, seed, 'base')
    g = torch.Generator().manual_seed(seed + 51007)
    c = torch.randint(2, 5, (n,), generator=g)
    old_phase = torch.arange(6)[None] < c[:, None]
    x = torch.where(old_phase, -d['x_cal'], d['x_cal'])
    return {**d, 'x_cal': x, 'private_flip_at': c}


def switch_aware_posterior(d):
    """Exact p(z_final=+|x_0:5,y_0:5) marginalizing all hidden switch times.

    Assumes known generative family with uniform switch prior but does not
    receive the realized switch position or query latent. Evaluator ONLY.
    """
    x, y = d['x_cal'], d['y_cal']
    mu, ref = d['cal_mu'][:, None], d['mu_ref'][:, None]
    candidates = []
    for z in (1., -1.):
        times=[]
        for switch in SWITCH_CANDIDATES:
            signal_sign=torch.where(torch.arange(6)[None] < switch, -z, z)
            signed_mu=signal_sign*mu
            p_plus=-.5*((x-signed_mu)**2+(y-ref)**2)
            p_minus=-.5*((x+signed_mu)**2+(y+ref)**2)
            times.append(torch.logaddexp(p_plus,p_minus).sum(1))
        candidates.append(torch.logsumexp(torch.stack(times, dim=1), dim=1) - math.log(3.))
    return torch.sigmoid(candidates[0] - candidates[1])


def policy_given_polarity_posterior(d, pz, force_buy=None):
    """Bayes VOI given calibrated p(z); no unpurchased probe used to decide buy."""
    x, mu = d['x_query'], d['mu0']
    pz = pz.clamp(1e-9, 1-1e-9)
    logs=torch.stack([pz.log(), torch.log1p(-pz)], dim=-1)
    plus=torch.logsumexp(logs + torch.stack([-.5*(x-mu)**2,-.5*(x+mu)**2],dim=-1),-1)
    minus=torch.logsumexp(logs + torch.stack([-.5*(x+mu)**2,-.5*(x-mu)**2],dim=-1),-1)
    odds=plus-minus
    q=torch.sigmoid(odds)
    threshold=-odds/(2*d['mu_p'])
    cdf=lambda a: .5*(1+torch.erf(a/math.sqrt(2.)))
    after=q*cdf(d['mu_p']-threshold)+(1-q)*cdf(d['mu_p']+threshold)
    gain=after-torch.maximum(q,1-q)
    buy=gain>d['c']
    if force_buy is not None:
        buy=torch.full_like(buy, bool(force_buy))
    updated=odds+buy.float()*2*d['mu_p']*d['y_probe']
    chosen=updated>=0
    return buy,chosen,gain


def known_drift_reference(d, method):
    """Only evaluator knows drift FAMILY. Switch location remains hidden."""
    pz=switch_aware_posterior(d)
    buy,chosen,gain=policy_given_polarity_posterior(d,pz)
    if method in ('never_probe', 'always_probe'):
        # Force observed arm directly. Zero cost does NOT guarantee a buy,
        # because an essentially certain posterior can make VOI round to 0.
        buy,chosen,_=policy_given_polarity_posterior(
            d,pz,force_buy=(method=='always_probe'))
    if method=='misspecified_stationary':
        buy=bayes_policy(d)['buy']
        chosen=bayes_policy(d)['choice']>0
    success=(chosen==d['truth'].gt(0)).float()
    return {'reward':float((success-buy.float()*d['c']).mean()),
            'accuracy':float(success.mean()),'probe_rate':float(buy.float().mean()),
            'latent_brier':float(((pz-(d['z']>0).float())**2).mean())}


def build_evaluations(n_eval,seed):
    return {
        'base':make_episode(n_eval,seed*1000000+88000,'base'),
        'polarity_flip':drift_episode(n_eval,seed*1000000+88000+127),
        'weak_calibration':make_episode(n_eval,seed*1000000+88000+254,'weak_calibration'),
    }


def all_references(e, family):
    if family=='polarity_flip':
        return {k:known_drift_reference(e,k) for k in ('correct_drift_bayes','always_probe','never_probe','misspecified_stationary')}
    from scripts.si_original_belief_transfer_v8 import assess_baseline
    return {k:assess_baseline(e,k) for k in ('bayes','always_probe','never_probe','no_cal_bayes')}


def score_model(model, method, e, seed):
    if method.startswith('v9_'):
        return v9.evaluate_model(model,e)
    metrics=v10.evaluate(model,e)
    metrics['randomized_critic_brier']=v10.randomized_value_calibration(model,e,seed+90335)
    return metrics


def run(seed,steps_v9=240,steps_equal=500,steps_actor=240,steps_critic=260,batch=128,n_eval=3072):
    torch.set_num_threads(2)
    provenance=v10.verify_original()
    worlds=build_evaluations(n_eval,seed)
    results={'seed':seed,'source_blob':provenance,'n_eval_per_family':n_eval,
             'steps_v9':steps_v9,'steps_equal':steps_equal,
             'steps_actor':steps_actor,'steps_critic':steps_critic,
             'baseline':{f:all_references(e,f) for f,e in worlds.items()},
             'methods':{}}
    for kind in ('hybrid','transformer','gru'):
        for method in ('v9_original','v9_500steps','v10_contextual'):
            if method=='v9_original':
                model,seconds,telemetry=v9.train_one(seed,kind,steps_v9,batch)
                timing={'train_seconds':seconds}
            elif method=='v9_500steps':
                model,seconds,telemetry=v9.train_one(seed,kind,steps_equal,batch)
                timing={'train_seconds':seconds}
            else:
                model,timing=v10.train_one(seed,kind,steps_actor,steps_critic,batch)
                telemetry=[]
            results['methods'][f'{kind}_{method}']={
                'parameters':sum(p.numel() for p in model.parameters()),
                'timing':timing,'telemetry':telemetry,
                'families':{f:score_model(model,method,e,seed) for f,e in worlds.items()},
            }
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=103000)
    p.add_argument('--steps-v9',type=int,default=240)
    p.add_argument('--steps-equal',type=int,default=500)
    p.add_argument('--steps-actor',type=int,default=240)
    p.add_argument('--steps-critic',type=int,default=260)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--n-eval',type=int,default=3072)
    x=p.parse_args()
    print(json.dumps(run(x.seed,x.steps_v9,x.steps_equal,x.steps_actor,x.steps_critic,x.batch,x.n_eval),indent=2))
