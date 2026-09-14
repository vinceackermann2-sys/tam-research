from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase12 as p12

CONTEXT = 8
RIDGE = 0.001


def fit_maps_context8(ctx, ridge=RIDGE):
    if ctx.shape[1] != CONTEXT:
        raise ValueError(f'expected exactly {CONTEXT} context frames, got {ctx.shape[1]}')
    B=ctx.shape[0]; maps=np.empty((B,2,16),np.float32)
    for b in range(B):
        X=[]; Y=[]
        for t in range(CONTEXT-1):
            X.append(p10.features(ctx[b:b+1,t])[0].reshape(-1,16))
            Y.append(ctx[b,t+1].transpose(1,2,0).reshape(-1,2))
        X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y).astype(np.float64)
        sc=np.sqrt(np.mean(X*X,axis=0)+1e-12); sc[0]=1.; Xs=X/sc
        W=np.linalg.solve(Xs.T@Xs+ridge*np.eye(16),Xs.T@Y).T
        maps[b]=(W/sc[None,:]).astype(np.float32)
    return maps


def eval_companded_context8(raw, state, dm, ds, horizon, ridge=RIDGE):
    if raw.shape[1] < CONTEXT+horizon:
        raise ValueError('sequence too short for context+horizon')
    qC,ctxdiag=p10.encode_comp_seq(raw[:,:CONTEXT],state,dm,ds,True)
    maps=fit_maps_context8(qC,ridge)
    cur=qC[:,-1].copy(); rollout_clips=[]
    for _ in range(horizon):
        y=p10.predict(cur,maps)
        q,d=p10.q_comp_delta(y-cur,dm,ds,True)
        cur=(cur+q).astype(np.float32); rollout_clips.append(d['clip_fraction'])
    target=raw[:,CONTEXT+horizon-1]; persist=raw[:,CONTEXT-1]; pm=p10.mse(persist,target)
    # Diagnostic: fit the same context8 map from raw observations, then project through the same frozen codec.
    raw_maps=fit_maps_context8(raw[:,:CONTEXT],ridge); rc=qC[:,-1].copy()
    for _ in range(horizon):
        y=p10.predict(rc,raw_maps); q,_=p10.q_comp_delta(y-rc,dm,ds,True); rc=(rc+q).astype(np.float32)
    true_d=raw[:,1:CONTEXT]-raw[:,:CONTEXT-1]; q_d=qC[:,1:]-qC[:,:-1]
    signal=float(np.mean(true_d**2)); noise=p10.mse(q_d,true_d)
    return {
        'persistence_mse':pm,
        'companded_mse':p10.mse(cur,target),
        'companded_ratio':p10.mse(cur,target)/pm,
        'raw_context_fit_companded_projection_ratio':p10.mse(rc,target)/pm,
        'context_mse':p10.mse(qC,raw[:,:CONTEXT]),
        'context_delta_mse':noise,
        'true_transition_signal_mse':signal,
        'transition_noise_over_signal':noise/signal,
        'context_clip_fraction':ctxdiag['clip_fraction'],
        'rollout_clip_fraction':float(np.mean(rollout_clips)) if rollout_clips else 0.0,
    }


@dataclass
class Seeds:
    wave:int; gray:int; sc1:int; sc2:int; ic1:int; ic2:int
    nls_id:int; nls_spectral:int; nls_parameter:int; nls_combined:int

REGISTERED=[
    Seeds(641390001,641390002,641390003,641390004,641390005,641390006,641390011,641390012,641390013,641390014),
    Seeds(641390101,641390102,641390103,641390104,641390105,641390106,641390111,641390112,641390113,641390114),
    Seeds(641390201,641390202,641390203,641390204,641390205,641390206,641390211,641390212,641390213,641390214),
]


def to_phase12_seeds(s):
    return p12.Seeds(s.wave,s.gray,s.sc1,s.sc2,s.ic1,s.ic2,s.nls_id,s.nls_spectral,s.nls_parameter,s.nls_combined)


def run_rep(seeds, grid=16, train_n=64, eval_n=48, code_sizes=(2048,128), iters=20, max_points=60000):
    t0=time.time(); p12s=to_phase12_seeds(seeds)
    state,plain,dm,ds=p12.fit_frozen_representation(p12s,grid,train_n,code_sizes,iters,max_points)
    nid,_,_=p12.gen_nls(eval_n,16,grid,(.04,.08),(.10,.25),seeds.nls_id,3)
    ns,_,_=p12.gen_nls(eval_n,11,grid,(.04,.08),(.10,.25),seeds.nls_spectral,6)
    np_,_,_=p12.gen_nls(eval_n,11,grid,(.10,.14),(.30,.45),seeds.nls_parameter,3)
    nc,_,_=p12.gen_nls(eval_n,11,grid,(.10,.14),(.30,.45),seeds.nls_combined,6)
    splits={
      'nls_id_h8':eval_companded_context8(nid,state,dm,ds,8),
      'nls_spectral_ood_h3':eval_companded_context8(ns,state,dm,ds,3),
      'nls_parameter_ood_h3':eval_companded_context8(np_,state,dm,ds,3),
      'nls_combined_ood_h3':eval_companded_context8(nc,state,dm,ds,3),
    }
    return {
      'seeds':asdict(seeds),
      'context_frames':CONTEXT,
      'observed_transitions_used':CONTEXT-1,
      'nls_used_in_representation_training':False,
      'primary_companded_ratios':{k:float(v['companded_ratio']) for k,v in splits.items()},
      'splits':splits,
      'runtime_seconds':time.time()-t0,
    }


def validate_invariants():
    assert p10.LEVELS==512 and p10.ZMAX==16.0
    assert p10.features(np.zeros((1,2,4,4),np.float32)).shape[-1]==16
    assert CONTEXT==8 and RIDGE==0.001
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    raw,_,_=p12.gen_nls(2,11,8,(.04,.08),(.10,.25),939130001,3)
    assert np.isfinite(raw).all()
    # synthetic check that all seven transitions affect the estimator
    c=raw[:,:8]
    m1=fit_maps_context8(c)
    c2=c.copy(); c2[:,7]+=0.01
    m2=fit_maps_context8(c2)
    assert not np.array_equal(m1,m2)
    return {'phase10_levels':p10.LEVELS,'phase10_zmax':p10.ZMAX,'feature_dim':16,'context_frames':CONTEXT,'transitions_used':CONTEXT-1,'ridge':RIDGE,'registered_seed_count':len(vals)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    inv=validate_invariants()
    if a.smoke:
        s=Seeds(939131001,939131002,939131003,939131004,939131005,939131006,939131011,939131012,939131013,939131014)
        r=run_rep(s,grid=8,train_n=12,eval_n=5,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE13_CONTEXT8_NLS_REGISTERED_CONFIRMATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))

if __name__=='__main__': main()
