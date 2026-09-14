from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase12 as p12
from tam_research import physics_tokenizer_phase13 as p13

DT = 0.18
NU = 0.01


def centered_dx(z):
    return .5 * (np.roll(z, -1, -1) - np.roll(z, 1, -1))


def centered_dy(z):
    return .5 * (np.roll(z, -1, -2) - np.roll(z, 1, -2))


def adv_step(x, vx, vy, dt=DT, nu=NU):
    return (x - dt * (vx * centered_dx(x) + vy * centered_dy(x)) + dt * nu * p10.lap(x)).astype(np.float32)


def gen_adv(num, seq_len, grid, speed_range, angle_range, seed, max_mode=3):
    rng = np.random.default_rng(seed)
    out = np.empty((num, seq_len, 2, grid, grid), np.float32)
    speeds = rng.uniform(*speed_range, size=num).astype(np.float32)
    angles = rng.uniform(*angle_range, size=num).astype(np.float32)
    for i in range(num):
        u = p10.smooth_field(rng, grid, max_mode) * float(rng.uniform(.35, .75))
        v = p10.smooth_field(rng, grid, max_mode) * float(rng.uniform(.25, .65))
        vx = float(speeds[i] * np.cos(angles[i]))
        vy = float(speeds[i] * np.sin(angles[i]))
        for t in range(seq_len):
            out[i, t, 0] = u
            out[i, t, 1] = v
            u = adv_step(u, vx, vy)
            v = adv_step(v, vx, vy)
    return out, speeds, angles


def features_directional(s):
    base = p10.features(s)
    extras = []
    for ch in range(2):
        z = s[:, ch]
        extras += [centered_dx(z), centered_dy(z)]
    return np.concatenate([base, np.stack(extras, -1).astype(np.float32)], -1)


def fit_maps_generic(ctx, feature_fn, dim, ridge=.001):
    B = ctx.shape[0]
    maps = np.empty((B, 2, dim), np.float32)
    for b in range(B):
        X, Y = [], []
        for t in range(ctx.shape[1] - 1):
            X.append(feature_fn(ctx[b:b+1, t])[0].reshape(-1, dim))
            Y.append(ctx[b, t+1].transpose(1, 2, 0).reshape(-1, 2))
        X = np.concatenate(X).astype(np.float64)
        Y = np.concatenate(Y).astype(np.float64)
        sc = np.sqrt(np.mean(X * X, axis=0) + 1e-12)
        sc[0] = 1.
        Xs = X / sc
        W = np.linalg.solve(Xs.T @ Xs + ridge * np.eye(dim), Xs.T @ Y).T
        maps[b] = (W / sc[None, :]).astype(np.float32)
    return maps


def predict_generic(s, maps, feature_fn):
    return np.einsum('bhwf,bcf->bchw', feature_fn(s), maps, optimize=True).astype(np.float32)


def raw_control(raw, horizon, feature_fn, dim):
    ctx = raw[:, :p13.CONTEXT]
    maps = fit_maps_generic(ctx, feature_fn, dim, p13.RIDGE)
    cur = ctx[:, -1].copy()
    for _ in range(horizon):
        cur = predict_generic(cur, maps, feature_fn)
    target = raw[:, p13.CONTEXT + horizon - 1]
    persist = raw[:, p13.CONTEXT - 1]
    pm = p10.mse(persist, target)
    ratio = p10.mse(cur, target) / pm
    return float(ratio) if np.isfinite(ratio) else None


def context_one_step_ratio(raw, feature_fn, dim):
    ctx = raw[:, :p13.CONTEXT]
    maps = fit_maps_generic(ctx, feature_fn, dim, p13.RIDGE)
    preds, targets, persists = [], [], []
    for t in range(p13.CONTEXT - 1):
        preds.append(predict_generic(ctx[:, t], maps, feature_fn))
        targets.append(ctx[:, t+1])
        persists.append(ctx[:, t])
    pred = np.stack(preds, 1)
    target = np.stack(targets, 1)
    persist = np.stack(persists, 1)
    return float(p10.mse(pred, target) / p10.mse(persist, target))


@dataclass
class Seeds:
    wave: int
    gray: int
    sc1: int
    sc2: int
    ic1: int
    ic2: int
    adv_id: int
    adv_spectral: int
    adv_speed: int
    adv_combined: int


REGISTERED = [
    Seeds(815150001,815150002,815150003,815150004,815150005,815150006,815150011,815150012,815150013,815150014),
    Seeds(815150101,815150102,815150103,815150104,815150105,815150106,815150111,815150112,815150113,815150114),
    Seeds(815150201,815150202,815150203,815150204,815150205,815150206,815150211,815150212,815150213,815150214),
]


def to_phase12_seeds(s):
    return p12.Seeds(s.wave, s.gray, s.sc1, s.sc2, s.ic1, s.ic2,
                     s.adv_id, s.adv_spectral, s.adv_speed, s.adv_combined)


def run_rep(seeds, grid=16, train_n=64, eval_n=48, code_sizes=(2048,128), iters=20, max_points=60000):
    t0 = time.time()
    state, _plain, dm, ds = p12.fit_frozen_representation(
        to_phase12_seeds(seeds), grid, train_n, code_sizes, iters, max_points)
    raw = {
        'adv_id_h8': gen_adv(eval_n, 16, grid, (.18,.30), (-.20,.20), seeds.adv_id, 3)[0],
        'adv_spectral_ood_h3': gen_adv(eval_n, 11, grid, (.18,.30), (-.20,.20), seeds.adv_spectral, 6)[0],
        'adv_speed_ood_h3': gen_adv(eval_n, 11, grid, (.45,.60), (-.20,.20), seeds.adv_speed, 3)[0],
        'adv_combined_ood_h3': gen_adv(eval_n, 11, grid, (.45,.60), (1.20,1.50), seeds.adv_combined, 6)[0],
    }
    splits, controls = {}, {}
    for name, x in raw.items():
        h = 8 if name.endswith('h8') else 3
        splits[name] = p13.eval_companded_context8(x, state, dm, ds, h)
        controls[name] = {
            'raw_frozen16_rollout_ratio': raw_control(x, h, p10.features, 16),
            'raw_directional20_rollout_ratio': raw_control(x, h, features_directional, 20),
            'raw_frozen16_context_one_step_ratio': context_one_step_ratio(x, p10.features, 16),
            'raw_directional20_context_one_step_ratio': context_one_step_ratio(x, features_directional, 20),
        }
    primary = {k: float(v['companded_ratio']) for k, v in splits.items()}
    return {
        'seeds': asdict(seeds),
        'context_frames': p13.CONTEXT,
        'observed_transitions_used': p13.CONTEXT - 1,
        'representation_training_families': ['wave','gray_scott'],
        'advection_used_in_representation_training': False,
        'frozen_feature_dim': 16,
        'directional_control_feature_dim': 20,
        'primary_companded_ratios': primary,
        'all_four_below_persistence': bool(all(v < 1.0 for v in primary.values())),
        'splits': splits,
        'controls': controls,
        'runtime_seconds': time.time() - t0,
    }


def validate_invariants():
    assert p10.LEVELS == 512 and p10.ZMAX == 16.0
    assert p13.CONTEXT == 8 and p13.RIDGE == .001
    z = np.zeros((1,2,8,8), np.float32)
    assert p10.features(z).shape[-1] == 16
    assert features_directional(z).shape[-1] == 20
    a, sp, an = gen_adv(3, 6, 8, (.18,.30), (-.20,.20), 915150001, 3)
    b, sp2, an2 = gen_adv(3, 6, 8, (.18,.30), (-.20,.20), 915150001, 3)
    assert np.array_equal(a,b) and np.array_equal(sp,sp2) and np.array_equal(an,an2)
    assert np.isfinite(a).all()
    vals = []
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals) == len(set(vals))
    # Directional features should expose information absent from the frozen symmetric library.
    probe, _, _ = gen_adv(4, 8, 8, (.30,.35), (.70,.80), 915150002, 4)
    r16 = context_one_step_ratio(probe, p10.features, 16)
    r20 = context_one_step_ratio(probe, features_directional, 20)
    assert r20 < r16 * 1e-2, (r16, r20)
    return {'phase10_levels':p10.LEVELS,'phase10_zmax':p10.ZMAX,'context_frames':p13.CONTEXT,
            'ridge':p13.RIDGE,'frozen_feature_dim':16,'directional_feature_dim':20,
            'advection_deterministic':True,'registered_seed_count':len(vals),
            'probe_frozen16_one_step_ratio':r16,'probe_directional20_one_step_ratio':r20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--rep', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    inv = validate_invariants()
    if a.smoke:
        s = Seeds(995152001,995152002,995152003,995152004,995152005,995152006,995152011,995152012,995152013,995152014)
        r = run_rep(s, grid=8, train_n=12, eval_n=5, code_sizes=(32,8), iters=3, max_points=3000)
        r['status'] = 'SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r = run_rep(REGISTERED[a.rep-1])
        r['replicate'] = a.rep
        r['status'] = 'PHASE15_OUTOFCLASS_ADVECTION_REGISTERED_FALSIFICATION'
    r['invariants'] = inv
    Path(a.out).write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))

if __name__ == '__main__': main()
