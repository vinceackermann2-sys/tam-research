"""SI original v23: prospective logged, frozen-policy second-probe calibration.

Train tiny post-hoc success-value heads ONLY from randomized purchased second-arm
outcomes and executed correctness feedback. No analytic Bayes labels, hidden
answers, unpurchased probe values, or per-example counterfactual outcomes enter
optimizer. Scores frozen original SI/Transformer/GRU classifiers and compares
unchanged v21, pooled offset, and contextual ridge for second-stage decisions.
"""
from __future__ import annotations
import argparse, json, time
import torch
from scripts import si_original_joint_logged_v21 as v21
from scripts import si_original_crossfit_value_v19 as v19
from scripts import si_original_logged_value_v18 as v18
from scripts.si_original_active_experiment_v17 import subset, score_policy

FAMILIES=('base','high_cost_ood','weak_sensors_ood','strong_discount_ood')

@torch.no_grad()
def public_features(e,first,base_model):
    """Only information observed before buying the remaining probe."""
    first_mu=torch.where(first==1,e['mu_low'],e['mu_high'])
    second_mu=torch.where(first==1,e['mu_high'],e['mu_low'])
    first_read=torch.where(first==1,e['probe_low'],e['probe_high'])
    confidence=(first_mu*first_read).abs().clamp(max=5)
    high_cost=torch.where(first==1,e['cost_high'],e['cost_low'])
    q=base_model.q1(e,first)
    # v21 q already has second probe cost subtracted; undo it to get its
    # observed-policy correctness estimates as calibration features.
    pskip=q[:,0].clamp(0,1)
    pbuy=(q[:,1]+high_cost).clamp(0,1)
    low=(first==1).float()
    one=torch.ones_like(low)
    return torch.stack((one,low,confidence,confidence.square(),first_mu,
                        second_mu,second_mu.square(),high_cost,
                        pskip,pbuy,pbuy-pskip,low*confidence,
                        low*second_mu,confidence*second_mu),dim=1)


def ridge(x,y,lam=20.):
    x=x.double();y=y.double()
    reg=torch.eye(x.shape[1],dtype=torch.float64)*lam
    reg[0,0]=1e-6
    return torch.linalg.solve(x.T@x+reg,x.T@y).float()


@torch.no_grad()
def logged_batch(model,e,seed):
    n=len(e['mu_low'])
    first=torch.where(v18.random_boolean(n,seed+11),1,2).long()
    purchase=v18.random_boolean(n,seed+29)
    rem=3-first
    outcome=torch.zeros(n,dtype=torch.bool)
    for arm in (False,True):
        mask=purchase==arm
        if mask.any():
            s=subset(e,mask)
            if arm: outcome[mask]=(model.decision(s,first[mask],rem[mask])>=0)
            else: outcome[mask]=(model.decision(s,first[mask])>=0)
    # Ground truth is used ONLY by environment-facing feedback after the
    # logged action, never in the policy, regression features, or optimizer.
    correct=v18.logged_feedback(e,outcome).float()
    return public_features(e,first,model),purchase,correct,first


@torch.no_grad()
def train_calibration(model,seed,n_train=12000):
    """Frozen core; separate arm regression and a group-only offset baseline.

    The two arm datasets share the same RANDOMIZED base family distribution,
    but no synthetic counterfactual outcomes are used. Public cost is factored
    exactly at decision time; features are restricted to pre-second state.
    """
    t=time.perf_counter()
    allx=[]; allarm=[]; ally=[]; allfirst=[]
    for i in range(0,n_train,1024):
        n=min(1024,n_train-i)
        # n must be even because twin examples are exactly coupled.
        assert n%2==0
        e=v19.evaluation_world(n//2,seed*1000000+900000+i,'base')
        x,a,y,f=logged_batch(model,e,seed*1000000+570000+i)
        allx.append(x);allarm.append(a);ally.append(y);allfirst.append(f)
    X=torch.cat(allx);arms=torch.cat(allarm);y=torch.cat(ally);first=torch.cat(allfirst)
    weights=[ridge(X[arms==arm],y[arms==arm],lam=20.) for arm in (False,True)]
    # Coarse bias for existing predicted net uplift, learned only from actual
    # logged success within forced-first groups; never from test set.
    shifts={}
    for f in (1,2):
        g=first==f
        # Randomization is independent, so separately observed successes
        # estimate the two arm means without per-world counterfactual labels.
        predicted_v21=(X[g,9]-X[g,8]).mean()
        observed_delta=y[g&arms].mean()-y[g&~arms].mean()
        shifts[f]=float(observed_delta-predicted_v21)
    return {'weights':weights,'shifts':shifts,'n':int(len(X)),
            'counts':{f'{f}_{arm}':int(((first==f)&(arms==arm)).sum()) for f in (1,2) for arm in (False,True)},
            'train_seconds':time.perf_counter()-t}


class CorrectedPolicy:
    """Frozen original v21 q0 + frozen original terminal classifier + Q1 correction."""
    def __init__(self,model,cal,method):
        self.model=model;self.cal=cal;self.method=method
        assert method in ('offset','ridge')
    def eval(self):
        self.model.eval()
        return self
    def q0(self,e):return self.model.q0(e)
    def decision(self,e,first,second=None):return self.model.decision(e,first,second)
    @torch.no_grad()
    def q1(self,e,first):
        q=self.model.q1(e,first)
        cost=torch.where(first==1,e['cost_high'],e['cost_low'])
        if self.method=='offset':
            shift=torch.where(first==1,torch.full_like(q[:,0],self.cal['shifts'][1]),torch.full_like(q[:,0],self.cal['shifts'][2]))
            return torch.stack([q[:,0],q[:,1]+shift],dim=1)
        x=public_features(e,first,self.model)
        probs=torch.stack([x@w for w in self.cal['weights']],dim=1).clamp(0,1)
        return probs-torch.stack([torch.zeros_like(cost),cost],dim=1)


@torch.no_grad()
def observed_arm_mse(model,e,seed):
    x,second,correct,first=logged_batch(model.model if isinstance(model,CorrectedPolicy) else model,e,seed)
    q=model.q1(e,first)
    cost=torch.where(first==1,e['cost_high'],e['cost_low'])
    p=q+torch.stack([torch.zeros_like(cost),cost],dim=1)
    selected=p.gather(1,second.long()[:,None]).squeeze(1)
    return float((selected-correct).square().mean())


@torch.no_grad()
def forced_first_gain(model,e,first):
    # Policy-selected second decision after first purchase, actual net reward.
    from scripts.si_original_second_value_audit_v22 import audit_first
    # This is an evaluation-only held-out MC reference, not a training label.
    # Separate from the deployed policy's REAL reward metric.
    return float((model.q1(e,torch.full((len(e['mu_low']),),first,dtype=torch.long))[:,1]-
                  model.q1(e,torch.full((len(e['mu_low']),),first,dtype=torch.long))[:,0]).mean())


def run(seed,actor_steps=160,critic_steps=240,second_steps=160,first_steps=160,batch=96,n_pairs=512,n_train=12000):
    torch.set_num_threads(2)
    original=v18.verify_source()
    worlds={f:v19.evaluation_world(n_pairs,seed*1000000+81000+131*j,f) for j,f in enumerate(FAMILIES)}
    out={'seed':seed,'original_source':original,'settings':dict(actor_steps=actor_steps,critic_steps=critic_steps,second_steps=second_steps,first_steps=first_steps,batch=batch,n_pairs=n_pairs,n_train=n_train),'models':{},'controls':{f:{p:score_policy(e,p) for p in ('never','always_low','always_high','always_both','greedy','bayes')} for f,e in worlds.items()}}
    for arch in ('hybrid','transformer','gru'):
        start=time.perf_counter()
        base,t0=v19.train_one(seed,arch,actor_steps,critic_steps,batch)
        frozen=v21.FactoredPolicy(base)
        t1=v21.fit_second(seed,frozen,second_steps,batch)
        t2=v21.fit_first(seed,frozen,first_steps,batch)
        cal=train_calibration(frozen,seed+100003,n_train=n_train)
        choices={'v21_unchanged':frozen,'offset':CorrectedPolicy(frozen,cal,'offset'),
                 'contextual_ridge':CorrectedPolicy(frozen,cal,'ridge')}
        out['models'][arch]={'train_seconds':time.perf_counter()-start,'base_timing':t0,'second_timing':t1,'first_timing':t2,'calibration':{k:v for k,v in cal.items() if k!='weights'},'policies':{}}
        for name,policy in choices.items():
            out['models'][arch]['policies'][name]={f:{**v18.score_logged(e,policy),
                     'factual_q1_mse':observed_arm_mse(policy,e,seed*1000000+1200+j*137),
                     'cheap_first_predicted_delta':forced_first_gain(policy,e,1)} for j,(f,e) in enumerate(worlds.items())}
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name,value in [('seed',115000),('actor_steps',160),('critic_steps',240),('second_steps',160),('first_steps',160),('batch',96),('n_pairs',512),('n_train',12000)]:p.add_argument('--'+name.replace('_','-'),type=int,default=value)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.actor_steps,a.critic_steps,a.second_steps,a.first_steps,a.batch,a.n_pairs,a.n_train),indent=2))
