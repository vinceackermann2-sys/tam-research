"""V25 original SI: first-probe acquisition fitted from factual randomized full returns.

Hold the original full hybrid/Transformer/GRU, final classifier and logged-trained
second-purchase Q1 fixed. Randomly assign FIRST arm stop/cheap/strong and execute
that chosen arm plus its frozen Q1 continuation. Regress each arm's ACTUALLY
observed final correctness less incurred SECOND cost against ONLY public
pre-first features; subtract the first fee algebraically at deployment.

No Bayesian choice labels, unpurchased measurements in a decision, latent truth
as model training input, or unobserved arm/counterfactual outcome supervision.
"""
from __future__ import annotations
import argparse, json, time
import torch
from scripts import si_original_crossfit_value_v19 as v19
from scripts import si_original_logged_value_v18 as v18
from scripts import si_original_joint_logged_v21 as v21
from scripts import si_original_uplift_calibration_v23 as v23
from scripts.si_original_active_experiment_v17 import subset, score_policy

FAMILIES=('base','high_cost_ood','weak_sensors_ood','strong_discount_ood')


def before_first(e):
    """Only public prices/qualities, NEVER latent answer or either probe reading."""
    a,b=e['mu_low'],e['mu_high'];ca,cb=e['cost_low'],e['cost_high']
    return torch.stack([torch.ones_like(a),a,b,a*a,b*b,a*b,ca,cb,ca*cb,cb-ca,
                        a*ca,b*cb],dim=1)


@torch.no_grad()
def logged_first_return(policy,e,first):
    """Execute observed first action and Q1 continuation; return factual reward.

    Training loss uses ONLY the logged chosen first arm. Private truth is read
    via environment-facing correctness feedback AFTER executing final action.
    """
    n=len(first)
    active=first!=0
    second=torch.zeros(n,dtype=torch.bool)
    executed=torch.ones(n,dtype=torch.bool)
    if active.any():
        sub=subset(e,active);f=first[active]
        buy=policy.q1(sub,f)[:,1]>policy.q1(sub,f)[:,0]
        second[active]=buy
        action=torch.ones(len(f),dtype=torch.bool)
        for val in (False,True):
            mask=buy==val
            if mask.any():
                subsub=subset(sub,mask)
                action[mask]=policy.decision(subsub,f[mask],3-f[mask] if val else None)>=0
        executed[active]=action
    correct=v18.logged_feedback(e,executed).float()
    first_price=torch.where(first==1,e['cost_low'],torch.where(first==2,e['cost_high'],0.))
    remaining=torch.where(first==1,e['cost_high'],e['cost_low'])
    second_price=second.float()*remaining
    # Regression predicts continuation value before first fee, which is public
    # and subtracted at decision time in the model wrapper.
    return correct-second_price, correct-first_price-second_price,second


def ridge(X,y,lam=12.):
    X=X.double();y=y.double()
    reg=torch.eye(X.shape[1],dtype=torch.float64)*lam
    reg[0,0]=1e-6
    return torch.linalg.solve(X.T@X+reg,X.T@y).float()


@torch.no_grad()
def train_acquisition(policy,seed,n_train=3600):
    start=time.perf_counter()
    xs=[];actions=[];targets=[]
    for off in range(0,n_train,600):
        n=min(600,n_train-off)
        if n%2: raise ValueError('Logging batch must be twin paired')
        e=v19.evaluation_world(n//2,seed*1000000+1300000+off,'base')
        first=v18.random_first(n,seed*1000000+1400000+off)
        continued,_,_=logged_first_return(policy,e,first)
        xs.append(before_first(e));actions.append(first);targets.append(continued)
    X=torch.cat(xs);A=torch.cat(actions);Y=torch.cat(targets)
    weights=[ridge(X[A==arm],Y[A==arm]) for arm in range(3)]
    per_arm=[int((A==arm).sum()) for arm in range(3)]
    means=[float(Y[A==arm].mean()) for arm in range(3)]
    return {'weights':weights,'counts':per_arm,'logged_continuation_means':means,
            'seconds':time.perf_counter()-start,'n_train':len(A)}


class NewFirstPolicy:
    """Factual Q0 contextual ridge, SAME frozen Q1/terminal decision as v23."""
    def __init__(self,source,learned,mode='ridge'):
        self.source=source;self.learned=learned;self.mode=mode
        if mode not in ('ridge','constant','fixed_high','fixed_cheap','fixed_high_only'):
            raise ValueError(mode)
    def eval(self):self.source.eval();return self
    @torch.no_grad()
    def q0(self,e):
        n=len(e['mu_low'])
        if self.mode.startswith('fixed_'):
            arm=1 if self.mode=='fixed_cheap' else 2
            q=torch.zeros((n,3))
            q[:,arm]=1.
            return q
        if self.mode=='constant':
            raw=torch.stack([torch.full((n,),self.learned['logged_continuation_means'][i]) for i in range(3)],dim=1)
        else:
            X=before_first(e)
            raw=torch.stack([X@w for w in self.learned['weights']],dim=1)
        return raw-torch.stack((torch.zeros(n),e['cost_low'],e['cost_high']),dim=1)
    def q1(self,e,first):
        if self.mode=='fixed_high_only':
            # Diagnostic fixed strong+skip second: use actual neural terminal.
            return torch.stack((torch.ones(len(first)),torch.zeros(len(first))),dim=1)
        return self.source.q1(e,first)
    def decision(self,e,first,second=None):return self.source.decision(e,first,second)


@torch.no_grad()
def factual_first_arm_mse(policy,e,seed):
    a=v18.random_first(len(e['mu_low']),seed+330)
    target,_,_=logged_first_return(policy,e,a)
    pred=policy.q0(e).gather(1,a[:,None]).squeeze(1)
    first_price=torch.where(a==1,e['cost_low'],torch.where(a==2,e['cost_high'],0.))
    return {'mse':float(((pred+first_price)-target).square().mean()),
            'coverage':{str(i):int((a==i).sum()) for i in range(3)}}


def run(seed,actor_steps=80,critic_steps=120,second_steps=80,first_steps=80,
        batch=64,n_pairs=256,n_q1_logs=4000,n_q0_logs=3600):
    torch.set_num_threads(2)
    original=v18.verify_source()
    worlds={fam:v19.evaluation_world(n_pairs,seed*1000000+81000+j*131,fam)
            for j,fam in enumerate(FAMILIES)}
    out={'seed':seed,'canonical_original_blob':original,'settings':dict(actor_steps=actor_steps,
         critic_steps=critic_steps,second_steps=second_steps,first_steps=first_steps,batch=batch,
         n_pairs=n_pairs,n_q1_logs=n_q1_logs,n_q0_logs=n_q0_logs),
         'controls':{f:{p:score_policy(e,p) for p in ('never','always_low','always_high','always_both','greedy','bayes')}
                   for f,e in worlds.items()},'models':{}}
    for arch in ('hybrid','transformer','gru'):
        start=time.perf_counter()
        base,t0=v19.train_one(seed,arch,actor_steps,critic_steps,batch)
        frozen=v21.FactoredPolicy(base)
        t1=v21.fit_second(seed,frozen,second_steps,batch)
        t2=v21.fit_first(seed,frozen,first_steps,batch)
        calibrated=v23.train_calibration(frozen,seed+100003,n_train=n_q1_logs)
        old=v23.CorrectedPolicy(frozen,calibrated,'ridge')
        new=train_acquisition(old,seed+330101,n_train=n_q0_logs)
        policies={'original_q0':old,'logged_constant':NewFirstPolicy(old,new,'constant'),
                  'logged_contextual':NewFirstPolicy(old,new,'ridge'),
                  'fixed_cheap_neural':NewFirstPolicy(old,new,'fixed_cheap'),
                  'fixed_high_neural':NewFirstPolicy(old,new,'fixed_high'),
                  'fixed_high_only_neural':NewFirstPolicy(old,new,'fixed_high_only')}
        results={}
        for fam,e in worlds.items():
            results[fam]={}
            for name,p in policies.items():
                metrics=v18.score_logged(e,p)
                if name in ('original_q0','logged_constant','logged_contextual'):
                    metrics['logged_first_mse']=factual_first_arm_mse(p,e,seed*1000000+1200+FAMILIES.index(fam)*137)
                results[fam][name]=metrics
        out['models'][arch]={'parameters':sum(p.numel() for p in frozen.parameters()),
            'seconds':time.perf_counter()-start,
            'timing':{'actor_v19':t0,'second_q1':t1,'first_q0':t2,
                      'q1_calibration_sec':calibrated['train_seconds'],'new_first_log_sec':new['seconds']},
            'q1_arm_counts':calibrated['counts'],
            'first_arm_counts':new['counts'],'first_arm_logged_means':new['logged_continuation_means'],
            'families':results}
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg,default in [('seed',117000),('actor_steps',80),('critic_steps',120),('second_steps',80),
                        ('first_steps',80),('batch',64),('n_pairs',256),('n_q1_logs',4000),('n_q0_logs',3600)]:
        p.add_argument('--'+arg.replace('_','-'),type=int,default=default)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.actor_steps,a.critic_steps,a.second_steps,a.first_steps,
                         a.batch,a.n_pairs,a.n_q1_logs,a.n_q0_logs),indent=2))
