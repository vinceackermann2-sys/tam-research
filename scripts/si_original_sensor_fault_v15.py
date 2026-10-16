"""Original SI v15: three-channel Bayesian fault localization vs neural controls.

A deliberately specified finite sensor-law hypothesis model, NOT self-discovery.
Observe three noisy calibration sensors x/y/anchor. Infer which may have changed
without reading hidden simulator truth, failure labels, or future probe samples.
Train unchanged canonical v12 raw/belief neural controllers; score a conservative
reference-substitution rule, classical Bayes and simple redundant-sensor vote.
"""
from __future__ import annotations
import argparse, json, math, time
import torch
from scripts.si_original_sensor_consistency_v14 import (world_with_anchor, ANCHOR_FEE,
    _actions, _select_actions, _summarize, repaired_observation, evaluate_cached)
from scripts.si_original_belief_filter_v12 import train_one, verify_original
from scripts.si_original_paired_drift_v11 import policy_given_polarity_posterior

FAMILIES=('base','one_flip','double_flip','reference_drift','reference_drift_early',
          'anchor_drift','reference_dropout','both_reference_anchor_drift','target_anchor_drift')
# Inference hypotheses are deliberately incomplete. Fault timing is known in the
# candidate family (first/last three), but fault identity and actual state aren't.
XPATTERNS=((1,1,1,1,1,1),(-1,-1,1,1,1,1),(-1,-1,-1,1,1,1),
           (-1,-1,-1,-1,1,1),(1,1,-1,-1,-1,1))
XPRIOR=(.65,.08,.08,.08,.11)
YFAULT=('ok','late','early','noisy_late')
YPRIOR=(.76,.12,.08,.04)
AFAULT=('ok','late','early')
APRIOR=(.83,.12,.05)
SENSOR_MU=1.8


def sensor_world(n,seed,family):
    if family=='both_reference_anchor_drift':
        e=world_with_anchor(n,seed,'reference_drift')
        e['anchor_cal']=e['anchor_cal'].clone();e['anchor_cal'][:,3:]*=-1
        return e
    if family=='target_anchor_drift':
        e=world_with_anchor(n,seed,'double_flip')
        e['anchor_cal']=e['anchor_cal'].clone();e['anchor_cal'][:,3:]*=-1
        return e
    return world_with_anchor(n,seed,family)


def _sensor_sign(which):
    if which=='ok':return [1]*6
    if which=='late':return [1,1,1,-1,-1,-1]
    if which=='early':return [-1,-1,-1,1,1,1]
    if which=='noisy_late':return [1]*6
    raise ValueError(which)


def fault_posterior(e):
    """Exact finite-class posterior by summing over unknown per-time theta_i.

    Observed x/y/a and known nominal strengths only. No hidden theta, z,
    family, change labels or y_probe. Returns P(z_final=+), P(y_fault),
    P(anchor_fault), posterior fault-state entropy. priors are specified.
    """
    x=e['x_cal']; y=e['y_cal']; a=e['anchor_cal']
    mu=e['cal_mu']; my=e['mu_ref']; ma=e['anchor_mu']
    n=x.shape[0]; terms=[]; metadata=[]
    for zi,z in enumerate((1.,-1.)):
        for xi,xpat in enumerate(XPATTERNS):
            for yi,ytype in enumerate(YFAULT):
                for ai,atype in enumerate(AFAULT):
                    sx=torch.as_tensor(xpat,dtype=x.dtype)
                    sy=torch.as_tensor(_sensor_sign(ytype),dtype=x.dtype)
                    sa=torch.as_tensor(_sensor_sign(atype),dtype=x.dtype)
                    mx=z*mu[:,None]*sx[None,:]
                    yy=my[:,None]*sy[None,:]
                    aa=ma[:,None]*sa[None,:]
                    # y may be large-variance late noise, not a systematic sign flip.
                    sigmay=torch.ones_like(yy)
                    if ytype=='noisy_late':sigmay[:,3:]=5.
                    def nlogpdf(obs,loc,std):return -.5*((obs-loc)/std).square()-std.log()
                    lp=nlogpdf(x,mx,torch.ones_like(x))+nlogpdf(y,yy,sigmay)+nlogpdf(a,aa,torch.ones_like(a))
                    lm=nlogpdf(x,-mx,torch.ones_like(x))+nlogpdf(y,-yy,sigmay)+nlogpdf(a,-aa,torch.ones_like(a))
                    loglike=torch.logaddexp(lp,lm).sum(1)-6*math.log(2.)
                    prior=math.log(.5*XPRIOR[xi]*YPRIOR[yi]*APRIOR[ai])
                    terms.append(loglike+prior)
                    metadata.append((zi,yi,ai))
    logits=torch.stack(terms,dim=1)
    prob=torch.softmax(logits,dim=1)
    iz=torch.tensor([m[0]==0 for m in metadata])
    iy=torch.tensor([m[1]!=0 for m in metadata])
    ia=torch.tensor([m[2]!=0 for m in metadata])
    pz=prob[:,iz].sum(1)
    py=prob[:,iy].sum(1)
    pa=prob[:,ia].sum(1)
    entropy=-(prob*prob.clamp_min(1e-30).log()).sum(1)
    return pz,py,pa,entropy


def majority_pz(e):
    """Cheap nonparametric control: late 3 sensor-pair observations.

    Use y and anchor as two redundant noisy references. Sign of their sum
    estimates calibration truth; if they conflict, vote is weak/near random.
    """
    signal=e['y_cal']/e['mu_ref'][:,None]+e['anchor_cal']/e['anchor_mu'][:,None]
    vote=torch.tanh(signal)
    correlation=(e['x_cal'][:,-3:]/e['cal_mu'][:,None])*vote[:,-3:]
    return torch.sigmoid(1.5*correlation.sum(1))


@torch.no_grad()
def decision_from_pz(e,pz,priced):
    buy,positive,_=policy_given_polarity_posterior(e,pz.clamp(1e-6,1-1e-6))
    correct=(positive==e['truth'].gt(0)).float()
    reward=correct-buy.float()*e['c']-(ANCHOR_FEE if priced else 0.)
    return dict(reward=float(reward.mean()),accuracy=float(correct.mean()),
                probe_fraction=float(buy.float().mean()),fee=float(ANCHOR_FEE if priced else 0.))


def _metrics_selected(a,b,e,mask):
    chosen=_select_actions(a,b,mask)
    return _summarize(chosen,e,True,mask,torch.zeros_like(mask))


@torch.no_grad()
def neural_panel(raw,bel,e,py,pa):
    """Guarded repair selects reference replacement ONLY if evidence says y fault.

    If anchor likely faulty or ambiguous, retain original y; never transfer
    hidden fault types. Cost of observing anchor is charged regardless.
    """
    orig=_actions(bel,e)
    alt=_actions(bel,repaired_observation(e))
    # Frozen conservative threshold .55 and margin .20 above anchor posterior.
    choose=(py>.55)&((py-pa)>.20)
    out=_metrics_selected(alt,orig,e,choose)
    out['repair_fraction']=float(choose.float().mean())
    return out


def run(seed,actor_steps=180,critic_steps=200,batch=96,n_eval=512):
    torch.set_num_threads(2)
    source=verify_original()
    envs={f:sensor_world(n_eval,seed*1000000+83000+131*j,f) for j,f in enumerate(FAMILIES)}
    out=dict(seed=seed,canonical_source=source,params={},train_cpu={},families={},
             models={},settings=dict(actor_steps=actor_steps,critic_steps=critic_steps,batch=batch,n_eval=n_eval))
    pz={};py={};pa={}
    for name,e in envs.items():
        t0=time.perf_counter();pz[name],py[name],pa[name],entropy=fault_posterior(e)
        out['families'][name]={
            'prob_reference_fault':float(py[name].mean()),
            'prob_anchor_fault':float(pa[name].mean()),
            'majority':decision_from_pz(e,majority_pz(e),True),
            'joint_fault_bayes':decision_from_pz(e,pz[name],True),
            'fault_model_inference_sec':time.perf_counter()-t0,
        }
    for arch in ('hybrid','transformer','gru'):
        raw,tr=train_one(seed,arch,False,actor_steps,critic_steps,batch)
        bel,tb=train_one(seed,arch,True,actor_steps,critic_steps,batch)
        out['params'][arch]=sum(p.numel() for p in raw.parameters())
        out['train_cpu'][arch]=sum(tr.values())+sum(tb.values())
        out['models'][arch]={}
        for k,e in envs.items():
            old=evaluate_cached(raw,bel,e,seed*1000000+40000)
            guarded=neural_panel(raw,bel,e,py[k],pa[k])
            out['models'][arch][k]={
                'raw':old['raw'],'belief':old['belief'],'always_anchor':old['always_anchor_belief'],
                'v14_consistency':old['consistency_repair'],'v15_guarded':guarded}
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=107000)
    p.add_argument('--actor-steps',type=int,default=180)
    p.add_argument('--critic-steps',type=int,default=200)
    p.add_argument('--batch',type=int,default=96)
    p.add_argument('--n-eval',type=int,default=512)
    args=p.parse_args()
    print(json.dumps(run(args.seed,args.actor_steps,args.critic_steps,args.batch,args.n_eval),indent=2))
