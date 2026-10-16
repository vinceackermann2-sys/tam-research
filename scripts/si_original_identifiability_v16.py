"""Original SI v16: observationally identical sensor worlds with opposite answers.

Purely CPU synthetic decision-theoretic test. Two hypotheses have *identical*
observed calibrations, strengths/costs and unprobed target queries but
opposite query labels. This is an impossibility certificate for fault identity
from these observations alone, NOT a new SI architecture capability.

A separately trusted *optional* query probe breaks the symmetry at a price.
OriginalSIHybrid, TransformerControl and GRU use unchanged canonical v12
training on stationary/single-flip worlds; no private fault labels are inputs.
"""
from __future__ import annotations
import argparse, json, math, time
import torch
from scripts.si_original_belief_transfer_v8 import make_episode, BASE_BLOB
from scripts.si_original_sensor_fault_v15 import fault_posterior, majority_pz, decision_from_pz
from scripts.si_original_belief_filter_v12 import Controller, train_one, verify_original
from scripts.si_original_contextual_value_v10 import feedback

FEE_ANCHOR=.025


def identical_opposite_worlds(n: int, seed: int):
    """Twin A target reversed and refs healthy, twin B target normal and refs reversed.

    Every observation available BEFORE optional query probe is bitwise equal.
    The calibration latent modes and query outcome labels are opposite.
    The trusted probe itself has a different signed mean; it is hidden unless paid.
    """
    base=make_episode(n,seed,'base')
    g=torch.Generator().manual_seed(seed+791197)
    rand=lambda *s: torch.rand(*s,generator=g)
    normal=lambda *s: torch.randn(*s,generator=g)
    t=base['cal_truth']; answer=base['truth']; mu=base['cal_mu']; ref=base['mu_ref']
    q_mu=.35+1.55*rand(n)
    price=.03+.5*rand(n)
    x=t*mu[:,None]+normal(n,6)
    y=-t*ref[:,None]+normal(n,6)
    anchor_mu=torch.full((n,),1.8)
    a=-t*anchor_mu[:,None]+normal(n,6)
    xq=answer*base['mu0']+normal(n)
    eps_probe=normal(n)
    fields=dict(base)
    fields.update(cal_mu=mu,mu_p=q_mu,c=price,x_cal=x,y_cal=y,
                  anchor_cal=a,anchor_mu=anchor_mu,x_query=xq)
    # B: target healthy, both reference-like sensors reverse sign.
    B={**fields,'z':torch.ones(n), 'cal_truth':t.clone(), 'truth':answer.clone(),
       'y_probe':answer*q_mu+eps_probe,
       'private_world_kind':torch.ones(n,dtype=torch.int64)}
    # A: target reversed, both reference-like sensors healthy. Re-labeling
    # t'= -t and answer'= -answer produces identical x,y,anchor,x_query.
    A={**fields,'z':-torch.ones(n), 'cal_truth':-t,
       'truth':-answer, 'y_probe':-answer*q_mu+eps_probe,
       'private_world_kind':torch.zeros(n,dtype=torch.int64)}
    return A,B


def concatenate_twins(A,B):
    return {k:torch.cat((A[k],B[k]),0) for k in A}


def assert_twin_certificate(A,B):
    private={'z','cal_truth','truth','y_probe','private_world_kind'}
    for k in A:
        if k not in private and not torch.equal(A[k],B[k]):
            raise AssertionError(f'Not observationally identical before purchase: {k}')
    assert torch.equal(A['truth'],-B['truth'])
    assert torch.equal(A['cal_truth'],-B['cal_truth'])


def normal_cdf(x):
    return .5*(1+torch.erf(x/math.sqrt(2.)))


@torch.no_grad()
def trusted_probe_baselines(e):
    """Exact Bayes under equal, indistinguishable A/B prior for query theta.

    No-probe success is exactly .5. With a purchased independent Gaussian
    probe (mean +/- mu_p, variance1), MAP choice sign(y_probe) is optimal.
    Buy iff Phi(mu_p)-.5 exceeds the actual observable price.
    """
    gain=normal_cdf(e['mu_p'])-.5
    methods={
      'never_probe':torch.zeros_like(gain,dtype=torch.bool),
      'always_probe':torch.ones_like(gain,dtype=torch.bool),
      'bayes_selective':gain>e['c'],
    }
    scores={}
    for name,buy in methods.items():
        act=torch.where(buy,torch.where(e['y_probe']>=0,1.,-1.),torch.ones_like(e['truth']))
        correct=(act==e['truth']).float()
        reward=correct-buy.float()*e['c']
        scores[name]=dict(reward=float(reward.mean()),accuracy=float(correct.mean()),
                          probe_fraction=float(buy.float().mean()))
    scores['expected_optimal_bayes_reward']=float((.5+(gain-e['c']).clamp_min(0)).mean())
    scores['expected_always_probe_reward']=float((normal_cdf(e['mu_p'])-e['c']).mean())
    return scores


@torch.no_grad()
def neural_score(model,e):
    q=model.values(e).sigmoid()
    buy=(q[:,1]-q[:,0])>e['c']
    decision=model.decision_logits(e,buy)>=0
    correctness,_,net=feedback(e,decision,buy)
    no_probe=model.decision_logits(e,torch.zeros_like(buy))>=0
    forced_success,_,_=feedback(e,no_probe,torch.zeros_like(buy))
    return {'reward':float(net.mean()),'accuracy':float(correctness.mean()),
            'probe_fraction':float(buy.float().mean()),
            'forced_no_probe_accuracy':float(forced_success.mean()),
            'q_skip_mean':float(q[:,0].mean()),'q_buy_mean':float(q[:,1].mean())}


@torch.no_grad()
def classical_scores(e):
    majority=decision_from_pz(e,majority_pz(e),True)
    time_start=time.perf_counter()
    pz,pr,pa,h=fault_posterior(e)
    bayes=decision_from_pz(e,pz,True)
    return {'majority':majority,'handcoded_fault_bayes':bayes,
            'mean_p_ref_fault':float(pr.mean()),'mean_p_anchor_fault':float(pa.mean()),
            'posterior_entropy_mean':float(h.mean()),
            'fault_model_wall_seconds':time.perf_counter()-time_start}


def run(seed,actor_steps=180,critic_steps=200,batch=96,n_pairs=512):
    torch.set_num_threads(2)
    attested=verify_original()
    A,B=identical_opposite_worlds(n_pairs,seed*1000000+94000)
    assert_twin_certificate(A,B)
    e=concatenate_twins(A,B)
    result=dict(seed=seed,source_blob=attested,n_pairs=n_pairs,
                trained_actor_steps=actor_steps,trained_critic_steps=critic_steps,batch=batch,
                exact_observation_twin_certificate=True,
                analytic=trusted_probe_baselines(e),classical=classical_scores(e),models={})
    for arch in ('hybrid','transformer','gru'):
        result['models'][arch]={}
        for features in (False,True):
            model,times=train_one(seed,arch,features,actor_steps,critic_steps,batch)
            name='belief' if features else 'raw'
            result['models'][arch][name]={
               'train_cpu_seconds':sum(times.values()),
               'parameters':sum(p.numel() for p in model.parameters()),
               'score':neural_score(model,e)}
            # Crucial: any no-probe policy must make EXACTLY same decision
            # on the two observationally identical but opposite-label worlds.
            with torch.no_grad():
                no=torch.zeros(n_pairs,dtype=torch.bool)
                pre_a=model.decision_logits(A,no)>=0
                pre_b=model.decision_logits(B,no)>=0
                if not torch.equal(pre_a,pre_b):
                    raise AssertionError('Causal neural policy used hidden information')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=108000)
    p.add_argument('--actor-steps',type=int,default=180)
    p.add_argument('--critic-steps',type=int,default=200)
    p.add_argument('--batch',type=int,default=96)
    p.add_argument('--n-pairs',type=int,default=512)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.actor_steps,a.critic_steps,a.batch,a.n_pairs),indent=2))
