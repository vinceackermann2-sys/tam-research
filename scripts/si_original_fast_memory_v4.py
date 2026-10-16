"""Original SI track — learned, causal fast-weight memory in the SAME backbone.

Adds a learned feature-keyed online delta-rule memory to the original hybrid
and parameter-matched Transformer. Only past *completed* transitions write.
This is an explicit inductive bias and NOT an SI breakthrough.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tam_research.si_original_hybrid_v0 import SIConfig, OriginalSIHybrid, TransformerControl, prediction_loss
from scripts.si_original_planning_transfer_v3 import world, evaluate


class FastEpisodicMemory(nn.Module):
    """Differentiable sample-local fast matrix; resets between episodes.

    At t, prediction sees memory written from i<t only, and never
    state[t+1] or hidden transition labels. The fast update is a learned
    feature-normalized online error correction (not a fitted oracle ridge).
    """
    def __init__(self, backbone, rank=16):
        super().__init__()
        self.backbone=backbone
        self.key_encoder=nn.Sequential(nn.Linear(11,32),nn.Tanh(),nn.Linear(32,rank))
        self.eta_logit=nn.Parameter(torch.tensor(math.log(.65/.35)))
        self.register_buffer('forget',torch.tensor(.985))
        self.rank=rank

    def forward(self,obs,actions,**kwargs):
        b,t,_=obs.shape
        out=self.backbone(obs,actions,**kwargs)
        base=out['prediction']
        prior_actions=torch.cat((torch.zeros_like(actions[:,:1]),actions[:,:-1]),dim=1)
        keys=F.normalize(self.key_encoder(torch.cat((obs,actions,prior_actions),dim=-1)),dim=-1)
        fast=base.new_zeros(b,self.rank,3)
        predictions=[]
        eta=torch.sigmoid(self.eta_logit)
        for j in range(t):
            residual=(keys[:,j:j+1] @ fast).squeeze(1)
            updated_prediction=base[:,j]+residual
            predictions.append(updated_prediction)
            # x[j+1] is visible at step j+1, so may be written only AFTER
            # producing prediction for step j. No y[j] target is ever needed.
            if j<t-1:
                innovation=obs[:,j+1]-updated_prediction
                fast=self.forget*fast+eta*keys[:,j,:,None]*innovation[:,None,:]
        result=dict(out)
        result['prediction']=torch.stack(predictions,dim=1)
        return result


def run(seed,steps=450,batch_size=16):
    torch.set_num_threads(2)
    config=SIConfig()
    candidates=(('original_hybrid',OriginalSIHybrid,False),
                ('transformer',TransformerControl,False),
                ('hybrid_fast_memory',OriginalSIHybrid,True),
                ('transformer_fast_memory',TransformerControl,True))
    results={'seed':seed,'steps':steps,'batch_size':batch_size,'models':{}}
    for name,backbone,with_memory in candidates:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        base=backbone(config)
        model=FastEpisodicMemory(base) if with_memory else base
        optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        start=time.perf_counter()
        for step in range(steps):
            family=('direct','delayed','switch')[step%3]
            data=world(batch_size,20,seed*1000000+step,family)
            model.train();optim.zero_grad(set_to_none=True)
            output=model(data[0],data[1]);loss,_=prediction_loss(output,data[2])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
        result={'params':sum(p.numel() for p in model.parameters()),
                'cpu_train_seconds':time.perf_counter()-start,
                'heldout':evaluate(model,seed*1000000+75000,steps)}
        if with_memory:result['learned_eta']=float(torch.sigmoid(model.eta_logit).detach())
        results['models'][name]=result
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=97300)
    p.add_argument('--steps',type=int,default=450);p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args();print(json.dumps(run(a.seed,a.steps,a.batch_size),indent=2))
