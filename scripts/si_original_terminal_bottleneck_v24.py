"""V24: freeze SI/Transformer/GRU purchase policies, ablate terminal decision only.

Factual logged action+correctness labels train simple terminal ridge models. Gaussian
known-law sufficient-statistic terminal is an EXPLICIT privileged reference.
No hidden query label, unpurchased sensor or purchase teacher reaches inference.
"""
from __future__ import annotations
import argparse, json, time
import torch
from scripts import si_original_crossfit_value_v19 as v19
from scripts import si_original_logged_value_v18 as v18
from scripts import si_original_joint_logged_v21 as v21
from scripts import si_original_uplift_calibration_v23 as v23
from scripts.si_original_active_experiment_v17 import score_policy

FAMILIES=('base','high_cost_ood','weak_sensors_ood','strong_discount_ood')


def _masks(first,second=None):
    low=(first==1)
    high=(first==2)
    if second is not None:
        low=low | (second==1)
        high=high | (second==2)
    return low,high


def terminal_features(e,first,second=None,feature='raw'):
    """Mask ALL unpurchased readings; never include hidden answer or y_probe."""
    low,high=_masks(first,second)
    a=e['probe_low']*low.float(); b=e['probe_high']*high.float()
    one=torch.ones_like(a)
    x=[one,a,b,e['mu_low'],e['mu_high'],low.float(),high.float()]
    if feature=='likelihood':
        # Engineered Gaussian-product sufficient statistics; privileged form.
        x += [e['mu_low']*a,e['mu_high']*b]
    elif feature!='raw':raise ValueError(feature)
    return torch.stack(x,dim=1)


def fit_terminal(seed,n_train=4000,kind='raw',penalty=25.):
    """Randomized ACTUALLY purchased log, executed action+correctness supervision."""
    xx=[];yy=[];start=time.perf_counter()
    for offset in range(0,n_train,1000):
        n=min(1000,n_train-offset)
        assert n%2==0
        e=v19.evaluation_world(n//2,seed*1000000+1000000+offset,'base')
        g=torch.Generator().manual_seed(seed*1000000+1300000+offset)
        logged=torch.randint(0,4,(n,),generator=g)
        low=(logged==1)|(logged==3)
        high=(logged==2)|(logged==3)
        # The terminal classifier sees only readings actually purchased.
        first=torch.where(low,1,torch.where(high,2,0)).long()
        second=torch.where(logged==3,2,0).long()
        X=terminal_features(e,first,second,kind)
        executed=torch.rand(n,generator=g)<.5
        correctness=v18.logged_feedback(e,executed)
        recovered=torch.where(correctness,executed,~executed).float()*2-1
        xx.append(X);yy.append(recovered)
    X=torch.cat(xx).double();Y=torch.cat(yy).double()
    reg=torch.eye(X.shape[1],dtype=torch.float64)*penalty
    reg[0,0]=1e-5
    w=torch.linalg.solve(X.T@X+reg,X.T@Y).float()
    return w,{'n_train':n_train,'seconds':time.perf_counter()-start,'features':kind}


class TerminalPolicy:
    def __init__(self,purchase,method,weights=None):
        self.purchase=purchase;self.method=method;self.weights=weights
    def eval(self):
        self.purchase.eval();return self
    def q0(self,e):return self.purchase.q0(e)
    def q1(self,e,first):return self.purchase.q1(e,first)
    def decision(self,e,first,second=None):
        if self.method=='original':return self.purchase.decision(e,first,second)
        if self.method=='bayes_terminal':
            low,high=_masks(first,second)
            # Exact two-independent-Gaussian likelihood for purchased probes.
            val=low.float()*e['mu_low']*e['probe_low']+high.float()*e['mu_high']*e['probe_high']
            return val
        return terminal_features(e,first,second,self.method)@self.weights


@torch.no_grad()
@torch.no_grad()
def score_terminal_accuracy(e,policy):
    """Evaluate *identical* policy acquisitions and explicit decision effectiveness."""
    first=policy.q0(e).argmax(1)
    active=first!=0
    bought=torch.zeros(len(first),dtype=torch.bool)
    guess=torch.ones(len(first),dtype=torch.bool)
    if active.any():
        sub={k:v[active] for k,v in e.items()};f=first[active]
        take=policy.q1(sub,f)[:,1]>policy.q1(sub,f)[:,0]
        bought[active]=take
        local=torch.ones(len(f),dtype=torch.bool)
        for buy in (False,True):
            mask=take==buy
            if mask.any():
                piece={k:v[mask] for k,v in sub.items()}
                local[mask]=policy.decision(piece,f[mask],3-f[mask] if buy else None)>=0
        guess[active]=local
    correct=(guess==(e['latent_answer']>0)).float()
    first_price=torch.where(first==1,e['cost_low'],torch.where(first==2,e['cost_high'],0.))
    second_price=torch.where(first==1,e['cost_high'],e['cost_low'])*bought.float()
    return {'reward':float((correct-first_price-second_price).mean()),'correct':float(correct.mean()),
            'mean_cost':float((first_price+second_price).mean()),
            'first_low':float((first==1).float().mean()),'first_high':float((first==2).float().mean()),
            'second_buy':float(bought.float().mean()),'no_probe':float((~active).float().mean()),
            'accuracy_purchased':float(correct[active].mean()) if active.any() else None}


@torch.no_grad()
def score_cached_terminal(e,policies):
    """Exact same purchases reused across final-action ablations; no Q refit.

    Mathematically equivalent to calling score_terminal_accuracy separately;
    eliminates redundant original-core Q0/Q1 passes.
    """
    purchase=next(iter(policies.values())).purchase
    first=purchase.q0(e).argmax(1)
    active=first!=0
    bought=torch.zeros(len(first),dtype=torch.bool)
    sub=None;f=None;choose=None
    if active.any():
        sub={k:v[active] for k,v in e.items()};f=first[active]
        q=purchase.q1(sub,f)
        choose=q[:,1]>q[:,0]
        bought[active]=choose
    first_price=torch.where(first==1,e['cost_low'],torch.where(first==2,e['cost_high'],0.))
    second_price=torch.where(first==1,e['cost_high'],e['cost_low'])*bought.float()
    prices=first_price+second_price
    results={}
    for name,policy in policies.items():
        guesses=torch.ones(len(first),dtype=torch.bool)
        if active.any():
            local=torch.ones(len(f),dtype=torch.bool)
            for buying in (False,True):
                mask=choose==buying
                if mask.any():
                    piece={k:v[mask] for k,v in sub.items()}
                    local[mask]=policy.decision(piece,f[mask],3-f[mask] if buying else None)>=0
            guesses[active]=local
        correct=(guesses==(e['latent_answer']>0)).float()
        results[name]={'reward':float((correct-prices).mean()),'correct':float(correct.mean()),
            'mean_cost':float(prices.mean()),'first_low':float((first==1).float().mean()),
            'first_high':float((first==2).float().mean()),'second_buy':float(bought.float().mean()),
            'no_probe':float((~active).float().mean()),
            'accuracy_purchased':float(correct[active].mean()) if active.any() else None}
    return results


@torch.no_grad()
def score_forced_probe(e,policy,which):
    """Diagnostic only: force identical low, high or both purchases, then score final decision."""
    n=len(e['mu_low'])
    first=torch.ones(n,dtype=torch.long)
    second=None
    if which=='strong':first.fill_(2)
    elif which=='both':second=torch.full_like(first,2)
    elif which!='cheap':raise ValueError(which)
    correct=(policy.decision(e,first,second)>=0)==(e['latent_answer']>0)
    cost=e['cost_low'] if which=='cheap' else e['cost_high'] if which=='strong' else e['cost_low']+e['cost_high']
    return dict(accuracy=float(correct.float().mean()),reward=float((correct.float()-cost).mean()),mean_cost=float(cost.mean()))


def run(seed,actor_steps=80,critic_steps=120,second_steps=80,first_steps=80,batch=64,n_pairs=256,n_cal=4000,n_terminal=4000):
    torch.set_num_threads(2)
    model_hash=v18.verify_source()
    worlds={fam:v19.evaluation_world(n_pairs,seed*1000000+81000+j*131,fam) for j,fam in enumerate(FAMILIES)}
    refs={f:{p:score_policy(e,p) for p in ('never','always_low','always_high','always_both','greedy','bayes')} for f,e in worlds.items()}
    rw,rt=fit_terminal(seed+200031,n_terminal,'raw')
    lw,lt=fit_terminal(seed+200031,n_terminal,'likelihood')
    out={'seed':seed,'canonical_blob':model_hash,'settings':dict(actor_steps=actor_steps,critic_steps=critic_steps,second_steps=second_steps,first_steps=first_steps,batch=batch,n_pairs=n_pairs,n_cal=n_cal,n_terminal=n_terminal),
         'terminal_training':{'raw':rt,'likelihood':lt,'raw_weights':rw.tolist(),'likelihood_weights':lw.tolist()},'controls':refs,'architectures':{}}
    for arch in ('hybrid','transformer','gru'):
        t0=time.perf_counter()
        base,btime=v19.train_one(seed,arch,actor_steps,critic_steps,batch)
        frozen=v21.FactoredPolicy(base)
        q1time=v21.fit_second(seed,frozen,second_steps,batch)
        q0time=v21.fit_first(seed,frozen,first_steps,batch)
        cal=v23.train_calibration(frozen,seed+100003,n_train=n_cal)
        buy=v23.CorrectedPolicy(frozen,cal,'ridge')
        # ALL four final classifiers evaluated with the same exact frozen
        # Q0/Q1 acquisition policy; no refitting Q values to new terminal model.
        policies={name:TerminalPolicy(buy,name,weight) for name,weight in
                  [('original',None),('raw',rw),('likelihood',lw),('bayes_terminal',None)]}
        fams={fam:score_cached_terminal(e,policies) for fam,e in worlds.items()}
        out['architectures'][arch]={'wall_seconds':time.perf_counter()-t0,'model_parameters':sum(p.numel() for p in frozen.parameters()),
            'calibration_counts':cal['counts'],'training_time':dict(v19=btime,q1=q1time,q0=q0time,calibration=cal['train_seconds']),
            'families':fams,'forced_probe_diagnostic':{kind:{name:score_forced_probe(worlds['base'],policy,kind) for name,policy in policies.items()} for kind in ('cheap','strong','both')}}
        for fam,data in fams.items():
            for key in ('first_low','first_high','second_buy','no_probe','mean_cost'):
                if max(v[key] for v in data.values())-min(v[key] for v in data.values())>1e-7:
                    raise AssertionError(f'Policy changed its purchases in {arch}/{fam}: {key}')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name,value in [('seed',116000),('actor_steps',80),('critic_steps',120),('second_steps',80),('first_steps',80),('batch',64),('n_pairs',256),('n_cal',4000),('n_terminal',4000)]:
        p.add_argument('--'+name.replace('_','-'),type=int,default=value)
    a=p.parse_args();print(json.dumps(run(a.seed,a.actor_steps,a.critic_steps,a.second_steps,a.first_steps,a.batch,a.n_pairs,a.n_cal,a.n_terminal),indent=2))
