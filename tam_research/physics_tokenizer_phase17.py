from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase16 as p16

CONTEXT = 8
RIDGE = 0.001


def stencil_features(s):
    """Unnamed periodic 3x3 neighborhood values for each channel + center polynomials."""
    cols = [np.ones_like(s[:, 0])]
    for ch in range(2):
        z = s[:, ch]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                # Positive offsets mean sample the positive coordinate direction.
                cols.append(np.roll(np.roll(z, -dy, -2), -dx, -1))
    u, v = s[:, 0], s[:, 1]
    cols += [u*u, u*v, v*v, u*u*u, u*u*v, u*v*v, v*v*v]
    return np.stack(cols, -1).astype(np.float32)


def fit_maps(ctx, feature_fn=stencil_features, dim=26, ridge=RIDGE):
    B = ctx.shape[0]
    maps = np.empty((B, 2, dim), np.float32)
    for b in range(B):
        X, Y = [], []
        for t in range(ctx.shape[1] - 1):
            X.append(feature_fn(ctx[b:b+1, t])[0].reshape(-1, dim))
            Y.append(ctx[b, t+1].transpose(1, 2, 0).reshape(-1, 2))
        X = np.concatenate(X).astype(np.float64)
        Y = np.concatenate(Y).astype(np.float64)
        sc = np.sqrt(np.mean(X*X, axis=0) + 1e-12)
        sc[0] = 1.
        Xs = X / sc
        W = np.linalg.solve(Xs.T @ Xs + ridge*np.eye(dim), Xs.T @ Y).T
        maps[b] = (W / sc[None, :]).astype(np.float32)
    return maps


def predict(s, maps, feature_fn=stencil_features):
    return np.einsum('bhwf,bcf->bchw', feature_fn(s), maps, optimize=True).astype(np.float32)


def eval_companded(raw, state, dm, ds, horizon):
    qC, ctxdiag = p10.encode_comp_seq(raw[:, :CONTEXT], state, dm, ds, True)
    maps = fit_maps(qC)
    cur = qC[:, -1].copy(); clips = []
    for _ in range(horizon):
        y = predict(cur, maps)
        q, d = p10.q_comp_delta(y-cur, dm, ds, True)
        cur = (cur + q).astype(np.float32)
        clips.append(d['clip_fraction'])
    target = raw[:, CONTEXT+horizon-1]
    persist = raw[:, CONTEXT-1]
    pm = p10.mse(persist, target)

    raw26 = fit_maps(raw[:, :CONTEXT])
    r26 = raw[:, CONTEXT-1].copy()
    for _ in range(horizon):
        r26 = predict(r26, raw26)

    raw18 = p16.fit_maps(raw[:, :CONTEXT])
    r18 = raw[:, CONTEXT-1].copy()
    for _ in range(horizon):
        r18 = p16.predict(r18, raw18)

    return {
        'persistence_mse': pm,
        'companded_mse': p10.mse(cur, target),
        'companded_ratio': p10.mse(cur, target)/pm,
        'raw_stencil26_rollout_ratio': p10.mse(r26, target)/pm if np.isfinite(r26).all() else None,
        'raw_named18_rollout_ratio': p10.mse(r18, target)/pm if np.isfinite(r18).all() else None,
        'context_mse': p10.mse(qC, raw[:, :CONTEXT]),
        'context_clip_fraction': ctxdiag['clip_fraction'],
        'rollout_clip_fraction': float(np.mean(clips)) if clips else 0.0,
    }


@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    wave_id:int; wave_spectral:int; wave_combined:int
    gray_id:int; gray_sharp:int; gray_parameter:int; gray_combined:int
    nls_id:int; nls_spectral:int; nls_parameter:int; nls_combined:int
    adv_id:int; adv_spectral:int; adv_speed:int; adv_combined:int


REGISTERED = [
    Seeds(817170001,817170002,817170003,817170004,817170005,817170006,817170011,817170012,817170013,817170021,817170022,817170023,817170024,817170031,817170032,817170033,817170034,817170041,817170042,817170043,817170044),
    Seeds(817170101,817170102,817170103,817170104,817170105,817170106,817170111,817170112,817170113,817170121,817170122,817170123,817170124,817170131,817170132,817170133,817170134,817170141,817170142,817170143,817170144),
    Seeds(817170201,817170202,817170203,817170204,817170205,817170206,817170211,817170212,817170213,817170221,817170222,817170223,817170224,817170231,817170232,817170233,817170234,817170241,817170242,817170243,817170244),
]


def build_codecs(s, grid=16, train_n=64, code_sizes=(2048,128), iters=20, max_points=60000):
    # Keep representation training identical to Phase 16.
    return p16.build_codecs(s, grid, train_n, code_sizes, iters, max_points)


def run_rep(s, grid=16, train_n=64, eval_n=48, code_sizes=(2048,128), iters=20, max_points=60000):
    t0 = time.time()
    state, dm, ds = build_codecs(s, grid, train_n, code_sizes, iters, max_points)
    raw = {
        'wave_id_h8': p10.gen_wave(eval_n,16,grid,(.6,1.0),s.wave_id,3)[0],
        'wave_spectral_ood_h3': p10.gen_wave(eval_n,11,grid,(.6,1.0),s.wave_spectral,6)[0],
        'wave_combined_ood_h3': p10.gen_wave(eval_n,11,grid,(1.15,1.35),s.wave_combined,6)[0],
        'gray_id_h8': p16.gen_gray(eval_n,16,grid,(.025,.045),(.055,.065),s.gray_id,False)[0],
        'gray_sharp_ood_h3': p16.gen_gray(eval_n,11,grid,(.025,.045),(.055,.065),s.gray_sharp,True)[0],
        'gray_parameter_ood_h3': p16.gen_gray(eval_n,11,grid,(.050,.060),(.045,.055),s.gray_parameter,False)[0],
        'gray_combined_ood_h3': p16.gen_gray(eval_n,11,grid,(.050,.060),(.045,.055),s.gray_combined,True)[0],
        'nls_id_h8': p16.gen_nls(eval_n,16,grid,(.04,.08),(.10,.25),s.nls_id,3)[0],
        'nls_spectral_ood_h3': p16.gen_nls(eval_n,11,grid,(.04,.08),(.10,.25),s.nls_spectral,6)[0],
        'nls_parameter_ood_h3': p16.gen_nls(eval_n,11,grid,(.10,.14),(.30,.45),s.nls_parameter,3)[0],
        'nls_combined_ood_h3': p16.gen_nls(eval_n,11,grid,(.10,.14),(.30,.45),s.nls_combined,6)[0],
        'adv_id_h8': p16.gen_adv(eval_n,16,grid,(.18,.30),(-.20,.20),s.adv_id,3)[0],
        'adv_spectral_ood_h3': p16.gen_adv(eval_n,11,grid,(.18,.30),(-.20,.20),s.adv_spectral,6)[0],
        'adv_speed_ood_h3': p16.gen_adv(eval_n,11,grid,(.45,.60),(-.20,.20),s.adv_speed,3)[0],
        'adv_combined_ood_h3': p16.gen_adv(eval_n,11,grid,(.45,.60),(1.20,1.50),s.adv_combined,6)[0],
    }
    splits = {name: eval_companded(x,state,dm,ds,8 if name.endswith('h8') else 3) for name,x in raw.items()}
    primary = {k:float(v['companded_ratio']) for k,v in splits.items()}
    return {
        'seeds':asdict(s), 'feature_library':'raw_3x3_stencil_plus_center_polynomial_v1', 'feature_dim':26,
        'context_frames':CONTEXT, 'ridge':RIDGE, 'representation_training_families':['wave','gray_scott'],
        'nls_used_in_representation_training':False, 'advection_used_in_representation_training':False,
        'primary_companded_ratios':primary, 'all_fifteen_below_persistence':bool(all(v<1 for v in primary.values())),
        'splits':splits, 'runtime_seconds':time.time()-t0,
    }


def validate_invariants():
    z=np.random.default_rng(1).normal(size=(2,2,8,8)).astype(np.float32)
    f=stencil_features(z)
    assert f.shape[-1]==26
    for ch in range(2):
        off=1+ch*9; zz=z[:,ch]
        left,right,up,down,center=f[...,off+3],f[...,off+5],f[...,off+1],f[...,off+7],f[...,off+4]
        np.testing.assert_allclose(.5*(right-left),p16.centered_dx(zz),rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(.5*(down-up),p16.centered_dy(zz),rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(left+right+up+down-4*center,p10.lap(zz),rtol=1e-6,atol=1e-6)
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    return {'feature_dim':26,'context_frames':CONTEXT,'ridge':RIDGE,'registered_seed_count':len(vals),'named_operators_span_exactly':True}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    inv=validate_invariants()
    if a.smoke:
        s=Seeds(997174001,997174002,997174003,997174004,997174005,997174006,997174011,997174012,997174013,997174021,997174022,997174023,997174024,997174031,997174032,997174033,997174034,997174041,997174042,997174043,997174044)
        r=run_rep(s,grid=8,train_n=12,eval_n=5,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE17_LOCAL_STENCIL_REGISTERED_CONFIRMATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))

if __name__=='__main__': main()
