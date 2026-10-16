"""Original SI v22: second-probe counterfactual value *audit only*.

Frozen original v21 learner is trained without counterfactual outcomes. On strictly
heldout worlds, simulate independent alternative second-probe observations to
estimate second-purchase value for that SAME frozen terminal-action classifier.
Counterfactual knowledge is never used for model training or selection.
"""
from __future__ import annotations
import argparse, json, time
import torch
from scripts import si_original_crossfit_value_v19 as v19
from scripts import si_original_logged_value_v18 as v18
from scripts import si_original_joint_logged_v21 as v21
from scripts.si_original_active_experiment_v17 import subset

FAMILIES=('base','high_cost_ood','weak_sensors_ood','strong_discount_ood')
BINS=(-100.,-.20,-.05,0.,.05,.20,100.)

@torch.no_grad()
def audit_first(model,e,first_id,seed,replications=12):
    """Compare predicted net uplift to heldout MC *frozen-policy* uplift.

    Counterfactual samples generated solely for evaluation. For each context the
    already observed first probe is held fixed; replicate only the unpurchased
    independent second probe. Costs and hidden answer are unchanged.
    """
    first=torch.full((len(e['mu_low']),),first_id,dtype=torch.long)
    q=model.q1(e,first)
    predicted=(q[:,1]-q[:,0]).detach()
    skipped=(model.decision(e,first)>=0)
    base_correct=(skipped==(e['latent_answer']>0)).float()
    mu_other=e['mu_high'] if first_id==1 else e['mu_low']
    cost=e['cost_high'] if first_id==1 else e['cost_low']
    # In a genuine decision this noise is unobserved and unavailable; this
    # controlled evaluation deliberately samples it with separate RNG.
    gen=torch.Generator().manual_seed(seed)
    n=len(first)
    new_signal=e['latent_answer'][:,None]*mu_other[:,None]+torch.randn(n,replications,generator=gen)
    expanded={k:x.repeat_interleave(replications,dim=0) for k,x in e.items()}
    if first_id==1:expanded['probe_high']=new_signal.reshape(-1)
    else: expanded['probe_low']=new_signal.reshape(-1)
    first_exp=first.repeat_interleave(replications)
    other_exp=3-first_exp
    purchased=(model.decision(expanded,first_exp,other_exp)>=0)
    bought_accuracy=(purchased.reshape(n,replications)==(e['latent_answer'][:,None]>0)).float().mean(1)
    realized_uplift=bought_accuracy-base_correct-cost
    act=predicted>0
    oracle=realized_uplift>0
    missed=~act & oracle
    unnecessary=act & ~oracle
    foregone=(realized_uplift.clamp_min(0)-torch.where(act,realized_uplift,0.)).mean()
    bins=[]
    for lo,hi in zip(BINS[:-1],BINS[1:]):
        mask=(predicted>=lo)&(predicted<hi)
        if mask.any():bins.append(dict(lo=lo,hi=hi,n=int(mask.sum()),predicted=float(predicted[mask].mean()),empirical=float(realized_uplift[mask].mean()),buy_rate=float(act[mask].float().mean())))
        else:bins.append(dict(lo=lo,hi=hi,n=0,predicted=None,empirical=None,buy_rate=None))
    # Binwise oracle optimizes ONE decision per publicly predicted-delta bin,
    # rather than illegitimately choosing using individual hidden truth.
    bin_oracle=sum(max(0.,b['empirical'])*b['n']/n for b in bins if b['n'])
    return dict(n=n,replications=replications,first=first_id,
        predicted_mean=float(predicted.mean()),empirical_mean=float(realized_uplift.mean()),
        predicted_positive=float(act.float().mean()),empirical_positive=float(oracle.float().mean()),
        missed_beneficial=float(missed.float().mean()),unnecessary_purchase=float(unnecessary.float().mean()),
        forecast_error=float((predicted-realized_uplift).mean()),
        forecast_mse=float((predicted-realized_uplift).square().mean()),
        oracle_given_frozen_actor_gain=float(realized_uplift.clamp_min(0).mean()),
        enacted_gain=float(torch.where(act,realized_uplift,0.).mean()),
        clairvoyant_per_world_regret_upper_bound=float(foregone),
        bin_oracle_gain=float(bin_oracle),
        bin_policy_gap=float(bin_oracle-float(torch.where(act,realized_uplift,0.).mean())),
        expected_buy_reward=float(bought_accuracy.mean()-cost.mean()),
        expected_skip_reward=float(base_correct.mean()),
        bins=bins)

@torch.no_grad()
def privacy_check(model,e):
    f=torch.ones(len(e['mu_low']),dtype=torch.long)
    modified={**e,'latent_answer':-e['latent_answer'],'probe_high':e['probe_high']+123.}
    return bool(torch.equal(model.q1(e,f),model.q1(modified,f)) and
                torch.equal(model.decision(e,f),model.decision(modified,f)))

def run(seed,actor_steps=160,critic_steps=240,second_steps=160,first_steps=160,batch=96,n_pairs=256,replications=12):
    torch.set_num_threads(2)
    attested=v18.verify_source()
    worlds={f:v19.evaluation_world(n_pairs,seed*1000000+84000+131*i,f)
            for i,f in enumerate(FAMILIES)}
    out={'seed':seed,'canonical_model_blob':attested,'params':dict(actor_steps=actor_steps,critic_steps=critic_steps,second_steps=second_steps,first_steps=first_steps,batch=batch,n_pairs=n_pairs,replications=replications),'architectures':{}}
    for ai,arch in enumerate(('hybrid','transformer','gru')):
        t0=time.perf_counter()
        base,basetime=v19.train_one(seed,arch,actor_steps,critic_steps,batch)
        model=v21.FactoredPolicy(base)
        secondtime=v21.fit_second(seed,model,second_steps,batch)
        firsttime=v21.fit_first(seed,model,first_steps,batch)
        archdata={'total_train_sec':time.perf_counter()-t0,'base_time':basetime,'second_time':secondtime,'first_time':firsttime,'families':{}}
        for i,(fam,e) in enumerate(worlds.items()):
            assert privacy_check(model,e),'Private/unpurchased info leak'
            archdata['families'][fam]={
                'executed_policy':v18.score_logged(e,model),
                'forced_first':{str(f):audit_first(model,e,f,seed*1000000+490000+ai*2000+i*17+f,replications) for f in (1,2)}
            }
        out['architectures'][arch]=archdata
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k,v in [('seed',114000),('actor_steps',160),('critic_steps',240),('second_steps',160),('first_steps',160),('batch',96),('n_pairs',256),('replications',12)]:p.add_argument('--'+k.replace('_','-'),type=int,default=v)
    a=p.parse_args();print(json.dumps(run(a.seed,a.actor_steps,a.critic_steps,a.second_steps,a.first_steps,a.batch,a.n_pairs,a.replications),indent=2))
