from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase16 as p16
from tam_research import physics_tokenizer_phase25 as p25

CONTEXT = 8
RIDGE = 0.001

@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    wave_id:int; wave_spectral:int; wave_combined:int
    gray_id:int; gray_sharp:int; gray_parameter:int; gray_combined:int
    nls_id:int; nls_spectral:int; nls_parameter:int; nls_combined:int
    adv_id:int; adv_spectral:int; adv_speed:int; adv_combined:int
    burg_id:int; burg_spectral:int; burg_parameter:int; burg_combined:int

REGISTERED = [
    Seeds(827270001,827270002,827270003,827270004,827270005,827270006,
          827270011,827270012,827270013,
          827270021,827270022,827270023,827270024,
          827270031,827270032,827270033,827270034,
          827270041,827270042,827270043,827270044,
          827270051,827270052,827270053,827270054),
    Seeds(827270101,827270102,827270103,827270104,827270105,827270106,
          827270111,827270112,827270113,
          827270121,827270122,827270123,827270124,
          827270131,827270132,827270133,827270134,
          827270141,827270142,827270143,827270144,
          827270151,827270152,827270153,827270154),
    Seeds(827270201,827270202,827270203,827270204,827270205,827270206,
          827270211,827270212,827270213,
          827270221,827270222,827270223,827270224,
          827270231,827270232,827270233,827270234,
          827270241,827270242,827270243,827270244,
          827270251,827270252,827270253,827270254),
]


def build_all(s, grid=16, train_n=64, code_sizes=(2048,128), iters=20, max_points=60000):
    return p25.build_all(s, grid, train_n, code_sizes, iters, max_points)


def build_eval(s, grid=16, eval_n=48):
    return {
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
      'burgers_id_h8': p25.gen_burgers(eval_n,16,grid,(.01,.03),(.15,.35),s.burg_id,3)[0],
      'burgers_spectral_ood_h3': p25.gen_burgers(eval_n,11,grid,(.01,.03),(.15,.35),s.burg_spectral,6)[0],
      'burgers_parameter_ood_h3': p25.gen_burgers(eval_n,11,grid,(.04,.06),(.15,.35),s.burg_parameter,3)[0],
      'burgers_combined_ood_h3': p25.gen_burgers(eval_n,11,grid,(.04,.06),(.35,.55),s.burg_combined,6)[0],
    }


def eval_split(raw, state, dm, ds, basis, horizon):
    qC, ctxdiag = p10.encode_comp_seq(raw[:,:CONTEXT], state, dm, ds, True)
    target = raw[:,CONTEXT+horizon-1]
    persist = raw[:,CONTEXT-1]
    pm = p10.mse(persist, target)

    m62 = p25.fit_maps(qC, p25.interaction62_features, 62, basis, RIDGE)
    y62, clip62 = p25.token_rollout(qC[:,-1], m62, horizon, p25.interaction62_features, basis, dm, ds)

    m26 = p25.fit_maps(qC, p25.base26_features, 26, basis, RIDGE)
    y26, clip26 = p25.token_rollout(qC[:,-1], m26, horizon, p25.base26_features, basis, dm, ds)

    return {
      'persistence_mse': pm,
      'primary_token62_mse': p10.mse(y62,target),
      'primary_token62_ratio': p10.mse(y62,target)/pm,
      'secondary_token26_mse': p10.mse(y26,target),
      'secondary_token26_ratio': p10.mse(y26,target)/pm,
      'context_mse': p10.mse(qC, raw[:,:CONTEXT]),
      'context_clip_fraction': ctxdiag['clip_fraction'],
      'primary_rollout_clip_fraction': clip62,
      'secondary_rollout_clip_fraction': clip26,
    }


def family_name(split):
    if split.startswith('wave_'): return 'wave'
    if split.startswith('gray_'): return 'gray_scott'
    if split.startswith('nls_'): return 'nls'
    if split.startswith('adv_'): return 'advection'
    if split.startswith('burgers_'): return 'burgers'
    raise KeyError(split)


def run_rep(s, grid=16, train_n=64, eval_n=48, code_sizes=(2048,128), iters=20, max_points=60000):
    t0=time.time()
    state,dm,ds,basis = build_all(s,grid,train_n,code_sizes,iters,max_points)
    raw = build_eval(s,grid,eval_n)
    splits = {name:eval_split(x,state,dm,ds,basis,8 if name.endswith('h8') else 3) for name,x in raw.items()}
    primary = {k:float(v['primary_token62_ratio']) for k,v in splits.items()}
    control = {k:float(v['secondary_token26_ratio']) for k,v in splits.items()}
    fam = {}
    for f in ('wave','gray_scott','nls','advection','burgers'):
        vals=[v for k,v in primary.items() if family_name(k)==f]
        fam[f]={'passing_cells':int(sum(x<1 for x in vals)),'total_cells':len(vals),'all_below_persistence':bool(all(x<1 for x in vals)),'worst_ratio':float(max(vals))}
    return {
      'seeds':asdict(s),
      'primary_feature_library':'plus_center_times_spatial_62',
      'secondary_control_library':'full_fourier_linear_plus_center_polynomial_26',
      'feature_dim':62,'context_frames':CONTEXT,'ridge':RIDGE,
      'representation_training_families':['wave','gray_scott'],
      'nls_used_in_representation_training':False,
      'advection_used_in_representation_training':False,
      'burgers_used_in_representation_training':False,
      'primary_ratios':primary,
      'secondary_control26_ratios':control,
      'primary_passing_cells':int(sum(v<1 for v in primary.values())),
      'primary_total_cells':len(primary),
      'primary_all_nineteen_below_persistence':bool(all(v<1 for v in primary.values())),
      'per_family':fam,
      'splits':splits,
      'runtime_seconds':time.time()-t0,
    }


def validate_invariants():
    z=np.random.default_rng(27).normal(size=(2,2,8,8)).astype(np.float32)
    b=p25.p22.fourier_basis()
    assert p25.interaction62_features(z,b).shape[-1]==62
    assert p25.base26_features(z,b).shape[-1]==26
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    assert len(vals)==75
    names=build_all.__code__.co_names
    assert 'gen_nls' not in names and 'gen_adv' not in names and 'gen_burgers' not in names
    return {'primary_feature_dim':62,'secondary_feature_dim':26,'context_frames':CONTEXT,'ridge':RIDGE,'registered_seed_count':len(vals),'representation_training_excludes_nls_advection_burgers':True}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); inv=validate_invariants()
    if a.smoke:
        s=Seeds(999270001,999270002,999270003,999270004,999270005,999270006,
                999270011,999270012,999270013,
                999270021,999270022,999270023,999270024,
                999270031,999270032,999270033,999270034,
                999270041,999270042,999270043,999270044,
                999270051,999270052,999270053,999270054)
        r=run_rep(s,grid=8,train_n=12,eval_n=5,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE27_UNIFIED_INTERACTION_REGISTERED_CONFIRMATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))

if __name__=='__main__': main()
