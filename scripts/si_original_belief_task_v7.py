"""CPU-only *benchmark validation* for a partial-observation SI successor.

No claim of original SI/Transformer superiority: these compact policies merely
establish test learnability and the exact Bayes reference before SI integration.
A binary hidden world rule must be inferred from a noisy initial observation.
The agent may buy a further noisy measurement, then act toward the rule.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass(frozen=True)
class Config:
    steps: int = 240
    batch: int = 256
    lr: float = 0.002
    flip_probability: float = 0.20
    n_eval: int = 8192


def sample_world(batch: int, seed: int, *, flipped: bool=False, cost_shift: bool=False):
    g=torch.Generator(device='cpu').manual_seed(int(seed))
    theta=(torch.randint(0,2,(batch,),generator=g)*2-1).float()
    # Sensor quality and information price are observable; hidden theta is not.
    mu0=.12+1.38*torch.rand(batch,generator=g)
    mup=.8+1.3*torch.rand(batch,generator=g)
    cost=(.04+(.27 if not cost_shift else .45)*torch.rand(batch,generator=g))
    cue0=theta*mu0+torch.randn(batch,generator=g)
    if flipped:
        corruption=torch.rand(batch,generator=g)<.20
        cue0=torch.where(corruption,-theta*mu0+torch.randn(batch,generator=g),cue0)
    probe=theta*mup+torch.randn(batch,generator=g)
    return dict(theta=theta,mu0=mu0,mup=mup,cost=cost,cue0=cue0,probe=probe)


def initial_log_odds(d, corruption=0.):
    x,mu=d['cue0'],d['mu0']
    if corruption == 0.:
        return 2*mu*x
    # Correct mixture likelihood for a sign-corrupted initial sensor.
    a=-.5*(x-mu).square()
    b=-.5*(x+mu).square()
    p=math.log(1-corruption)
    q=math.log(corruption)
    return torch.logaddexp(a+p,b+q)-torch.logaddexp(b+p,a+q)


def bayesian_decisions(d, corruption=0.):
    l=initial_log_odds(d,corruption)
    prob=torch.sigmoid(l)
    signal=d['mup']
    # Probe posterior threshold xp>threshold for theta=+1.
    threshold=-l/(2*signal)
    std_cdf=lambda x: .5*(1+torch.erf(x/(2**.5)))
    after=prob*std_cdf(signal-threshold)+(1-prob)*std_cdf(signal+threshold)
    before=torch.maximum(prob,1-prob)
    should_probe=(after-before)>d['cost']
    new_l=l+2*signal*d['probe']*should_probe.float()
    choice=torch.where(new_l>=0.,torch.ones_like(l),-torch.ones_like(l))
    return should_probe, choice, after-before, prob


def compact_features(d, probe_mask):
    # Stage0 never contains actual probe signal, including in the second token's causal prefix.
    base=torch.stack([d['cue0'],d['mu0'],d['mup'],d['cost']],-1)
    observed=d['probe']*probe_mask.float()
    after=torch.cat((base,observed[:,None],probe_mask.float()[:,None]),-1)
    return base,after


class DenseControl(nn.Module):
    def __init__(self,width=24):
        super().__init__()
        self.probe_head=nn.Sequential(nn.Linear(4,width),nn.Tanh(),nn.Linear(width,1))
        self.action_head=nn.Sequential(nn.Linear(6,width),nn.Tanh(),nn.Linear(width,1))
    def forward(self,d,probe_mask):
        before,after=compact_features(d,probe_mask)
        return self.probe_head(before).flatten(),self.action_head(after).flatten()


class RecurrentControl(nn.Module):
    def __init__(self,width=20):
        super().__init__()
        self.encoder=nn.Linear(7,width)
        self.gru=nn.GRU(width,width,batch_first=True)
        self.probe_head=nn.Linear(width,1)
        self.action_head=nn.Linear(width,1)
    def forward(self,d,probe_mask):
        before,after=compact_features(d,probe_mask)
        initial=torch.cat([before,torch.zeros(len(before),2),torch.zeros(len(before),1)],-1)
        follow=torch.cat([after,torch.ones(len(after),1)],-1)
        seq=torch.stack([initial,follow],1)
        output,_=self.gru(torch.tanh(self.encoder(seq)))
        return self.probe_head(output[:,0]).flatten(),self.action_head(output[:,1]).flatten()


def realized_reward(choice,probe_mask,d):
    return (choice==d['theta']).float()-probe_mask.float()*d['cost']


@torch.no_grad()
def score_controller(controller,d):
    controller.eval()
    no_probe=torch.zeros_like(d['theta'],dtype=torch.bool)
    step0,_=controller(d,no_probe)
    take_probe=step0>0
    _,final_logits=controller(d,take_probe)
    chosen=torch.where(final_logits>=0,1.,-1.)
    return dict(reward=realized_reward(chosen,take_probe,d).mean().item(),
                accuracy=(chosen==d['theta']).float().mean().item(),
                probe_rate=take_probe.float().mean().item(),
                parameters=sum(p.numel() for p in controller.parameters()))


def classical_scores(d,corruption):
    decisions={}
    for label,allow,correct in [('bayes_correct',None,corruption),
                                ('bayes_gaussian_misspecified',None,0.),
                                ('never_probe',False,corruption),
                                ('always_probe',True,corruption),
                                ('random_action',False,corruption)]:
        probe,choice,_,_=bayesian_decisions(d,correct)
        if allow is not None:
            probe=torch.full_like(probe,allow)
            log_odds=initial_log_odds(d,correct)+probe.float()*2*d['mup']*d['probe']
            choice=torch.where(log_odds>=0,1.,-1.)
        if label=='random_action':
            g=torch.Generator().manual_seed(7777+int(len(d['theta'])))
            choice=(torch.randint(0,2,(len(choice),),generator=g)*2-1).float()
        decisions[label]=dict(reward=float(realized_reward(choice,probe,d).mean()),
                              accuracy=float((choice==d['theta']).float().mean()),
                              probe_rate=float(probe.float().mean()))
    return decisions


def train_control(seed,cls,c:Config):
    torch.manual_seed(seed)
    model=cls()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c.lr,weight_decay=.01)
    start=time.perf_counter()
    last_loss=None
    for i in range(c.steps):
        d=sample_world(c.batch,seed*1000000+i)
        target_probe=bayesian_decisions(d,0.)[0].float()
        # Observe the secondary cue in 50% of supervised training examples.
        # The stage0 probe prediction is never given its value.
        train_mask=torch.rand(c.batch)<.5
        probe_logit,action_logit=model(d,train_mask)
        target_action=(d['theta']>0).float()
        loss=F.binary_cross_entropy_with_logits(probe_logit,target_probe)+F.binary_cross_entropy_with_logits(action_logit,target_action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        last_loss=float(loss.detach())
    seconds=time.perf_counter()-start
    return model,seconds,last_loss


def run(seed,c=Config()):
    torch.set_num_threads(2)
    results=dict(seed=seed,config=c.__dict__.copy(),policies={})
    for key,cls in [('dense',DenseControl),('recurrent',RecurrentControl)]:
        model,seconds,train_loss=train_control(seed,cls,c)
        results['policies'][key]=dict(parameters=sum(p.numel() for p in model.parameters()),
                                     cpu_train_seconds=round(seconds,4),training_last_loss=train_loss,
                                     evaluations={})
        for kind,flipped,cost_shift in [('gaussian',False,False),('sensor_flip_ood',True,False),('high_cost_ood',False,True)]:
            d=sample_world(c.n_eval,seed*1000000+99000,flipped=flipped,cost_shift=cost_shift)
            results['policies'][key]['evaluations'][kind]=score_controller(model,d)
    results['baselines']={}
    for kind,flipped,cost_shift in [('gaussian',False,False),('sensor_flip_ood',True,False),('high_cost_ood',False,True)]:
        d=sample_world(c.n_eval,seed*1000000+99000,flipped=flipped,cost_shift=cost_shift)
        results['baselines'][kind]=classical_scores(d,.2 if flipped else 0.)
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=99400)
    parser.add_argument('--steps',type=int,default=240)
    parser.add_argument('--n-eval',type=int,default=8192)
    a=parser.parse_args()
    print(json.dumps(run(a.seed,Config(steps=a.steps,n_eval=a.n_eval)),indent=2))
