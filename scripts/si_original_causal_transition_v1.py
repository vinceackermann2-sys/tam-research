"""New causal rule-identification benchmark: source-frozen before execution.

A new hidden +/- sign is drawn per episode. Each action's effect on the
next state depends on it. Sign is *not* provided to the model; only a causal
history of state/action outcomes can reveal it. In switch mode sign flips
once mid-episode. Persistence is an explicit no-learning control.
"""
from __future__ import annotations
import argparse
import json
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from tam_research.si_original_hybrid_v0 import (
    OriginalSIHybrid, TransformerControl, SIConfig,
    prediction_loss, count_parameters,
)


def batch_worlds(batch, length, seed, mode="stable"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand = lambda *dims: torch.randn(*dims, generator=g)
    # Action effects are shared across worlds and held stable for training/eval.
    # The hidden episode-specific binary sign creates genuine memory dependence.
    effect = torch.tensor([[1., 0., -.7], [0., 1., .4], [-.8, -.6, 0.], [.5, -1., .6]])
    signs = torch.where(rand(batch, 1) >= 0, 1., -1.)
    state = rand(batch, 3) * .15
    xs, acts, ys = [], [], []
    for i in range(length):
        act_id = torch.randint(0, 4, (batch,), generator=g)
        action = F.one_hot(act_id, num_classes=4).float()
        sign_t = -signs if mode == "switch" and i >= length // 2 else signs
        effect_t = (action @ effect) * sign_t
        pred = .24 * state + .67 * torch.tanh(effect_t + .1 * state.roll(1, -1))
        if mode == "nonlinear":
            pred += .16 * state.roll(1, -1) * state.roll(2, -1)
        xs.append(state)
        acts.append(action)
        ys.append(pred)
        state = pred
    return torch.stack(xs, 1), torch.stack(acts, 1), torch.stack(ys, 1)


def eval_model(model, seed, mode):
    out = []
    baseline = []
    model.eval()
    with torch.no_grad():
        for batch_i in range(3):
            x,a,y=batch_worlds(32,16,seed+batch_i,mode)
            estimate=model(x,a)["prediction"]
            # Report *last four* transitions: memory use and post-change adaptation.
            out.append((estimate[:,-4:]-y[:,-4:]).square().mean().item())
            baseline.append((x[:,-4:]-y[:,-4:]).square().mean().item())
    return float(np.mean(out)),float(np.mean(baseline))


def run(seed=94011,steps=180,batch_size=16):
    torch.set_num_threads(2)
    config=SIConfig()
    result={"seed":seed,"steps":steps,"batch":batch_size,"models":{}}
    for name,make in [
        ("hybrid",lambda: OriginalSIHybrid(config)),
        ("transformer",lambda: TransformerControl(config)),
    ]:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=make();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        t0=time.monotonic()
        for step in range(steps):
            x,a,y=batch_worlds(batch_size,16,seed*10000+step)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out=model(x,a)
            loss,_=prediction_loss(out,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        duration=time.monotonic()-t0
        metrics={}
        for mode in ("stable","switch","nonlinear"):
            mse,persistence=eval_model(model,seed*10000+85000,mode)
            metrics[mode]={"mse":mse,"persistence_mse":persistence}
        result["models"][name]={"params":count_parameters(model),"wall_seconds":duration,"modes":metrics}
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=94011)
    p.add_argument("--steps",type=int,default=180)
    p.add_argument("--batch",type=int,default=16)
    a=p.parse_args()
    print(json.dumps(run(a.seed,a.steps,a.batch),indent=2))
