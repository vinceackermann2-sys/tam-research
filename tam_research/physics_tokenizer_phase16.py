from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10

CONTEXT = 8
RIDGE = 0.001


def centered_dx(z):
    return .5 * (np.roll(z, -1, -1) - np.roll(z, 1, -1))


def centered_dy(z):
    return .5 * (np.roll(z, -1, -2) - np.roll(z, 1, -2))


def generic_features(s):
    cols = [np.ones_like(s[:, 0])]
    for ch in range(2):
        z = s[:, ch]
        l = p10.lap(z)
        cols += [z, centered_dx(z), centered_dy(z), l, p10.lap(l)]
    u, v = s[:, 0], s[:, 1]
    cols += [u*u, u*v, v*v, u*u*u, u*u*v, u*v*v, v*v*v]
    return np.stack(cols, -1).astype(np.float32)


def fit_maps(ctx, feature_fn=generic_features, dim=18, ridge=RIDGE):
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


def predict(s, maps, feature_fn=generic_features):
    return np.einsum('bhwf,bcf->bchw', feature_fn(s), maps, optimize=True).astype(np.float32)


def eval_companded(raw, state, dm, ds, horizon):
    qC, ctxdiag = p10.encode_comp_seq(raw[:, :CONTEXT], state, dm, ds, True)
    maps = fit_maps(qC)
    cur = qC[:, -1].copy(); clips = []
    for _ in range(horizon):
        y = predict(cur, maps)
        q, d = p10.q_comp_delta(y-cur, dm, ds, True)
        cur = (cur + q).astype(np.float32); clips.append(d['clip_fraction'])
    target = raw[:, CONTEXT+horizon-1]
    persist = raw[:, CONTEXT-1]
    pm = p10.mse(persist, target)

    raw_maps18 = fit_maps(raw[:, :CONTEXT])
    r18 = raw[:, CONTEXT-1].copy()
    for _ in range(horizon): r18 = predict(r18, raw_maps18)

    raw_maps16 = fit_maps(raw[:, :CONTEXT], p10.features, 16, RIDGE)
    r16 = raw[:, CONTEXT-1].copy()
    for _ in range(horizon): r16 = predict(r16, raw_maps16, p10.features)

    return {
        'persistence_mse': pm,
        'companded_mse': p10.mse(cur, target),
        'companded_ratio': p10.mse(cur, target)/pm,
        'raw_generic18_rollout_ratio': p10.mse(r18, target)/pm if np.isfinite(r18).all() else None,
        'raw_frozen16_rollout_ratio': p10.mse(r16, target)/pm if np.isfinite(r16).all() else None,
        'context_mse': p10.mse(qC, raw[:, :CONTEXT]),
        'context_clip_fraction': ctxdiag['clip_fraction'],
        'rollout_clip_fraction': float(np.mean(clips)) if clips else 0.0,
    }


def gen_gray(num, seq_len, grid, F_range, k_range, seed, sharp=False, substeps=4):
    rng = np.random.default_rng(seed)
    out = np.empty((num, seq_len, 2, grid, grid), np.float32)
    Fs = rng.uniform(*F_range, size=num).astype(np.float32)
    ks = rng.uniform(*k_range, size=num).astype(np.float32)
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
    for i in range(num):
        u = np.ones((grid, grid), np.float32); v = np.zeros((grid, grid), np.float32)
        for _ in range(int(rng.integers(1, 4))):
            cx, cy = float(rng.uniform(0, grid)), float(rng.uniform(0, grid))
            r = float(rng.uniform(.8, 1.8) if sharp else rng.uniform(1.5, 3.0))
            dx = np.minimum(np.abs(xx-cx), grid-np.abs(xx-cx)); dy = np.minimum(np.abs(yy-cy), grid-np.abs(yy-cy))
            blob = np.exp(-(dx*dx+dy*dy)/(2*r*r)).astype(np.float32)
            u -= float(rng.uniform(.25, .5))*blob; v += float(rng.uniform(.18, .35))*blob
        u += rng.normal(0, .02 if sharp else .01, (grid, grid)).astype(np.float32)
        v += rng.normal(0, .01 if sharp else .005, (grid, grid)).astype(np.float32)
        for t in range(seq_len):
            out[i,t,0] = u; out[i,t,1] = v
            for _ in range(substeps): u, v = p10.gray_step(u, v, Fs[i], ks[i])
    return out, Fs, ks


NLS_DT = 0.08

def nls_step(u, v, alpha, g, dt=NLS_DT):
    r2 = u*u + v*v
    u2 = u + dt * (-alpha*p10.lap(v) + g*r2*v)
    v2 = v + dt * (+alpha*p10.lap(u) - g*r2*u)
    return u2.astype(np.float32), v2.astype(np.float32)


def gen_nls(num, seq_len, grid, alpha_range, g_range, seed, max_mode=3):
    rng = np.random.default_rng(seed)
    out = np.empty((num, seq_len, 2, grid, grid), np.float32)
    alphas = rng.uniform(*alpha_range, size=num).astype(np.float32)
    gs = rng.uniform(*g_range, size=num).astype(np.float32)
    for i in range(num):
        u = p10.smooth_field(rng, grid, max_mode) * float(rng.uniform(.25, .55))
        v = p10.smooth_field(rng, grid, max_mode) * float(rng.uniform(.25, .55))
        for t in range(seq_len):
            out[i,t,0] = u; out[i,t,1] = v
            u, v = nls_step(u, v, float(alphas[i]), float(gs[i]))
    return out, alphas, gs


ADV_DT = .18; ADV_NU = .01

def adv_step(x, vx, vy):
    return (x - ADV_DT*(vx*centered_dx(x) + vy*centered_dy(x)) + ADV_DT*ADV_NU*p10.lap(x)).astype(np.float32)


def gen_adv(num, seq_len, grid, speed_range, angle_range, seed, max_mode=3):
    rng = np.random.default_rng(seed)
    out = np.empty((num, seq_len, 2, grid, grid), np.float32)
    speeds = rng.uniform(*speed_range, size=num).astype(np.float32)
    angles = rng.uniform(*angle_range, size=num).astype(np.float32)
    for i in range(num):
        u = p10.smooth_field(rng, grid, max_mode)*float(rng.uniform(.35, .75))
        v = p10.smooth_field(rng, grid, max_mode)*float(rng.uniform(.25, .65))
        vx = float(speeds[i]*np.cos(angles[i])); vy = float(speeds[i]*np.sin(angles[i]))
        for t in range(seq_len):
            out[i,t,0] = u; out[i,t,1] = v
            u = adv_step(u, vx, vy); v = adv_step(v, vx, vy)
    return out, speeds, angles


@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    wave_id:int; wave_spectral:int; wave_combined:int
    gray_id:int; gray_sharp:int; gray_parameter:int; gray_combined:int
    nls_id:int; nls_spectral:int; nls_parameter:int; nls_combined:int
    adv_id:int; adv_spectral:int; adv_speed:int; adv_combined:int


REGISTERED = [
    Seeds(816160001,816160002,816160003,816160004,816160005,816160006,816160011,816160012,816160013,816160021,816160022,816160023,816160024,816160031,816160032,816160033,816160034,816160041,816160042,816160043,816160044),
    Seeds(816160101,816160102,816160103,816160104,816160105,816160106,816160111,816160112,816160113,816160121,816160122,816160123,816160124,816160131,816160132,816160133,816160134,816160141,816160142,816160143,816160144),
    Seeds(816160201,816160202,816160203,816160204,816160205,816160206,816160211,816160212,816160213,816160221,816160222,816160223,816160224,816160231,816160232,816160233,816160234,816160241,816160242,816160243,816160244),
]


def build_codecs(s, grid=16, train_n=64, code_sizes=(2048,128), iters=20, max_points=60000):
    k1, k2 = code_sizes
    w, _ = p10.gen_wave(train_n, 8, grid, (.6,1.0), s.wave_train, 3)
    g, _, _ = gen_gray(train_n, 8, grid, (.025,.045), (.055,.065), s.gray_train, False)
    joint = np.concatenate([w,g], 0)
    sm, ss = p10.stats(joint)
    sc1, sc2 = p10.fit_rvq(p10.flat2(p10.norm5(joint,sm,ss)), k1,k2,s.sc1,s.sc2,iters,max_points)
    state = (sm,ss,sc1,sc2)
    rawdiff = joint[:,1:] - joint[:,:-1]
    dm, ds = p10.stats(rawdiff)
    # Fit the unused plain innovation RVQ too so representation training remains recipe-identical.
    p10.fit_rvq(p10.flat2(p10.norm5(rawdiff,dm,ds)), k1,k2,s.ic1,s.ic2,iters,max_points)
    return state, dm, ds


def run_rep(s, grid=16, train_n=64, eval_n=48, code_sizes=(2048,128), iters=20, max_points=60000):
    t0 = time.time(); state, dm, ds = build_codecs(s,grid,train_n,code_sizes,iters,max_points)
    raw = {
        'wave_id_h8': p10.gen_wave(eval_n,16,grid,(.6,1.0),s.wave_id,3)[0],
        'wave_spectral_ood_h3': p10.gen_wave(eval_n,11,grid,(.6,1.0),s.wave_spectral,6)[0],
        'wave_combined_ood_h3': p10.gen_wave(eval_n,11,grid,(1.15,1.35),s.wave_combined,6)[0],
        'gray_id_h8': gen_gray(eval_n,16,grid,(.025,.045),(.055,.065),s.gray_id,False)[0],
        'gray_sharp_ood_h3': gen_gray(eval_n,11,grid,(.025,.045),(.055,.065),s.gray_sharp,True)[0],
        'gray_parameter_ood_h3': gen_gray(eval_n,11,grid,(.050,.060),(.045,.055),s.gray_parameter,False)[0],
        'gray_combined_ood_h3': gen_gray(eval_n,11,grid,(.050,.060),(.045,.055),s.gray_combined,True)[0],
        'nls_id_h8': gen_nls(eval_n,16,grid,(.04,.08),(.10,.25),s.nls_id,3)[0],
        'nls_spectral_ood_h3': gen_nls(eval_n,11,grid,(.04,.08),(.10,.25),s.nls_spectral,6)[0],
        'nls_parameter_ood_h3': gen_nls(eval_n,11,grid,(.10,.14),(.30,.45),s.nls_parameter,3)[0],
        'nls_combined_ood_h3': gen_nls(eval_n,11,grid,(.10,.14),(.30,.45),s.nls_combined,6)[0],
        'adv_id_h8': gen_adv(eval_n,16,grid,(.18,.30),(-.20,.20),s.adv_id,3)[0],
        'adv_spectral_ood_h3': gen_adv(eval_n,11,grid,(.18,.30),(-.20,.20),s.adv_spectral,6)[0],
        'adv_speed_ood_h3': gen_adv(eval_n,11,grid,(.45,.60),(-.20,.20),s.adv_speed,3)[0],
        'adv_combined_ood_h3': gen_adv(eval_n,11,grid,(.45,.60),(1.20,1.50),s.adv_combined,6)[0],
    }
    splits = {}
    for name, x in raw.items():
        splits[name] = eval_companded(x,state,dm,ds,8 if name.endswith('h8') else 3)
    primary = {k:float(v['companded_ratio']) for k,v in splits.items()}
    return {
        'seeds':asdict(s),'feature_library':'generic_parity_complete_differential_v1','feature_dim':18,
        'context_frames':CONTEXT,'ridge':RIDGE,'representation_training_families':['wave','gray_scott'],
        'nls_used_in_representation_training':False,'advection_used_in_representation_training':False,
        'primary_companded_ratios':primary,'all_fifteen_below_persistence':bool(all(v<1 for v in primary.values())),
        'splits':splits,'runtime_seconds':time.time()-t0,
    }


def validate_invariants():
    z = np.random.default_rng(1).normal(size=(2,2,8,8)).astype(np.float32)
    f = generic_features(z)
    assert f.shape[-1] == 18
    # Odd derivatives flip sign under axis reversal; Laplacian does not.
    zr = z[..., ::-1]
    assert np.allclose(centered_dx(zr), centered_dx(z)[..., ::-1] * -1)
    assert np.allclose(p10.lap(zr), p10.lap(z)[..., ::-1])
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    return {'feature_dim':18,'context_frames':CONTEXT,'ridge':RIDGE,'registered_seed_count':len(vals),'parity_checks':True}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    inv=validate_invariants()
    if a.smoke:
        s=Seeds(996160001,996160002,996160003,996160004,996160005,996160006,996160011,996160012,996160013,996160021,996160022,996160023,996160024,996160031,996160032,996160033,996160034,996160041,996160042,996160043,996160044)
        r=run_rep(s,grid=8,train_n=12,eval_n=5,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE16_GENERIC_OPERATORS_REGISTERED_CONFIRMATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))

if __name__=='__main__': main()
