from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10

DT = 0.08


def nls_step(u, v, alpha, g, dt=DT):
    r2 = u*u + v*v
    u2 = u + dt * (-alpha * p10.lap(v) + g * r2 * v)
    v2 = v + dt * (+alpha * p10.lap(u) - g * r2 * u)
    return u2.astype(np.float32), v2.astype(np.float32)


def gen_nls(num, seq_len, grid, alpha_range, g_range, seed, max_mode=3, dt=DT):
    rng = np.random.default_rng(seed)
    out = np.empty((num, seq_len, 2, grid, grid), np.float32)
    alphas = rng.uniform(*alpha_range, size=num).astype(np.float32)
    gs = rng.uniform(*g_range, size=num).astype(np.float32)
    for i in range(num):
        u = p10.smooth_field(rng, grid, max_mode) * float(rng.uniform(.25, .55))
        v = p10.smooth_field(rng, grid, max_mode) * float(rng.uniform(.25, .55))
        for t in range(seq_len):
            out[i,t,0] = u; out[i,t,1] = v
            u, v = nls_step(u, v, float(alphas[i]), float(gs[i]), dt)
            if not (np.isfinite(u).all() and np.isfinite(v).all()):
                raise FloatingPointError('non-finite NLS trajectory')
    return out, alphas, gs


@dataclass
class Seeds:
    wave: int
    gray: int
    sc1: int
    sc2: int
    ic1: int
    ic2: int
    nls_id: int
    nls_spectral: int
    nls_parameter: int
    nls_combined: int


REGISTERED = [
    Seeds(781240001,781240002,781240003,781240004,781240005,781240006,781240011,781240012,781240013,781240014),
    Seeds(781240101,781240102,781240103,781240104,781240105,781240106,781240111,781240112,781240113,781240114),
    Seeds(781240201,781240202,781240203,781240204,781240205,781240206,781240211,781240212,781240213,781240214),
]


def fit_frozen_representation(seeds, grid=16, train_n=64, code_sizes=(2048,128), iters=20, max_points=60000):
    k1,k2 = code_sizes
    w,_ = p10.gen_wave(train_n,8,grid,(.6,1.),seeds.wave,3)
    g = p10.gen_gray(train_n,8,grid,(.025,.045),(.055,.065),seeds.gray)
    joint = np.concatenate([w,g],0)
    sm,ss = p10.stats(joint)
    sc1,sc2 = p10.fit_rvq(p10.flat2(p10.norm5(joint,sm,ss)),k1,k2,seeds.sc1,seeds.sc2,iters,max_points)
    state = (sm,ss,sc1,sc2)
    rawdiff = joint[:,1:] - joint[:,:-1]
    dm,ds = p10.stats(rawdiff)
    ic1,ic2 = p10.fit_rvq(p10.flat2(p10.norm5(rawdiff,dm,ds)),k1,k2,seeds.ic1,seeds.ic2,iters,max_points)
    plain = (dm,ds,ic1,ic2)
    return state, plain, dm, ds


def analytic_map(alpha, g, dt=DT):
    W = np.zeros((2,16), np.float64)
    W[0,1] = 1.0
    W[0,5] = 4.0*dt*alpha
    W[0,6] = -dt*alpha
    W[0,13] = dt*g
    W[0,15] = dt*g
    W[1,5] = 1.0
    W[1,1] = -4.0*dt*alpha
    W[1,2] = dt*alpha
    W[1,12] = -dt*g
    W[1,14] = -dt*g
    return W


def representability_error(seed=42424242, grid=8):
    rng=np.random.default_rng(seed)
    u=p10.smooth_field(rng,grid,4)*.37; v=p10.smooth_field(rng,grid,4)*.41
    alpha=.071; g=.19
    u2,v2=nls_step(u,v,alpha,g)
    s=np.stack([u,v],0)[None]
    pred=np.einsum('bhwf,cf->bchw',p10.features(s),analytic_map(alpha,g),optimize=True)[0]
    target=np.stack([u2,v2],0)
    return float(np.mean((pred-target)**2))


def run_rep(seeds, grid=16, train_n=64, eval_n=48, code_sizes=(2048,128), iters=20, max_points=60000):
    t0=time.time()
    state,plain,dm,ds = fit_frozen_representation(seeds,grid,train_n,code_sizes,iters,max_points)
    nls_id,_,_=gen_nls(eval_n,16,grid,(.04,.08),(.10,.25),seeds.nls_id,3)
    nls_spec,_,_=gen_nls(eval_n,8,grid,(.04,.08),(.10,.25),seeds.nls_spectral,6)
    nls_param,_,_=gen_nls(eval_n,8,grid,(.10,.14),(.30,.45),seeds.nls_parameter,3)
    nls_comb,_,_=gen_nls(eval_n,8,grid,(.10,.14),(.30,.45),seeds.nls_combined,6)
    splits={
      'nls_id_h8': p10.eval_split(nls_id,state,plain,dm,ds,8),
      'nls_spectral_ood_h3': p10.eval_split(nls_spec,state,plain,dm,ds,3),
      'nls_parameter_ood_h3': p10.eval_split(nls_param,state,plain,dm,ds,3),
      'nls_combined_ood_h3': p10.eval_split(nls_comb,state,plain,dm,ds,3),
    }
    return {
      'seeds': asdict(seeds),
      'representation_training_families': ['wave','gray_scott'],
      'nls_used_in_representation_training': False,
      'codec': {
        'state_mean': state[0].tolist(), 'state_std': state[1].tolist(),
        'innovation_mean': dm.tolist(), 'innovation_std': ds.tolist(),
        'state_coarse': code_sizes[0], 'state_residual': code_sizes[1],
        'companded_levels_per_channel': p10.LEVELS, 'companded_zmax': p10.ZMAX, 'nominal_bits': 18,
      },
      'splits': splits,
      'primary_companded_ratios': {k:float(v['companded_ratio']) for k,v in splits.items()},
      'representability_mse': representability_error(),
      'runtime_seconds': time.time()-t0,
    }


def validate_invariants():
    assert p10.LEVELS == 512 and p10.ZMAX == 16.0
    e=representability_error()
    assert e < 1e-12, e
    a,aa,gg=gen_nls(3,6,8,(.04,.08),(.10,.25),931337001,3)
    b,aa2,gg2=gen_nls(3,6,8,(.04,.08),(.10,.25),931337001,3)
    assert np.array_equal(a,b) and np.array_equal(aa,aa2) and np.array_equal(gg,gg2)
    assert np.isfinite(a).all()
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    return {'phase10_levels':p10.LEVELS,'phase10_zmax':p10.ZMAX,'representability_mse':e,'nls_deterministic':True,'registered_seed_count':len(vals)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    inv=validate_invariants()
    if a.smoke:
        s=Seeds(991240001,991240002,991240003,991240004,991240005,991240006,991240011,991240012,991240013,991240014)
        r=run_rep(s,grid=8,train_n=12,eval_n=5,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE12_UNSEEN_NLS_REGISTERED_TRANSFER'
    r['invariants']=inv
    Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))

if __name__=='__main__': main()
