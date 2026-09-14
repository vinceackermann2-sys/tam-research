from __future__ import annotations
import argparse, itertools, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase16 as p16
from tam_research import physics_tokenizer_phase22 as p22

CONTEXT=8
RIDGE=0.001
SINE_POOL=('sin_x','sin_y','sin_diag_plus','sin_diag_minus')
MODE_INDEX={name:i for i,name in enumerate(p22.MODE_NAMES)}
OMISSION_SUBSETS=tuple(combo for r in range(5) for combo in itertools.combinations(SINE_POOL,r))


def variant_label(omitted):
    return 'none' if not omitted else 'omit__'+'__'.join(omitted)


def selected_features(s,basis,omitted=()):
    omit_idx={MODE_INDEX[x] for x in omitted}
    keep=[j for j in range(9) if j not in omit_idx]
    cols=[np.ones_like(s[:,0])]
    for ch in range(2):
        P=p22.patch9(s,ch)
        Y=np.einsum('bhwp,kp->bhwk',P,basis[keep],optimize=True)
        cols.extend([Y[...,j] for j in range(len(keep))])
    u,v=s[:,0],s[:,1]
    cols += [u*u,u*v,v*v,u*u*u,u*u*v,u*v*v,v*v*v]
    return np.stack(cols,-1).astype(np.float32)


def fit_maps(ctx,basis,omitted=(),ridge=RIDGE):
    k=9-len(omitted); dim=8+2*k; B=ctx.shape[0]
    maps=np.empty((B,2,dim),np.float32)
    for b in range(B):
        X=[];Y=[]
        for t in range(ctx.shape[1]-1):
            X.append(selected_features(ctx[b:b+1,t],basis,omitted)[0].reshape(-1,dim))
            Y.append(ctx[b,t+1].transpose(1,2,0).reshape(-1,2))
        X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y).astype(np.float64)
        sc=np.sqrt(np.mean(X*X,axis=0)+1e-12); sc[0]=1.; Xs=X/sc
        W=np.linalg.solve(Xs.T@Xs+ridge*np.eye(dim),Xs.T@Y).T
        maps[b]=(W/sc[None,:]).astype(np.float32)
    return maps


def predict(s,maps,basis,omitted=()):
    return np.einsum('bhwf,bcf->bchw',selected_features(s,basis,omitted),maps,optimize=True).astype(np.float32)


def eval_prepared(prepared,basis,omitted,dm,ds):
    splits={}
    for name,(qC,target,pm,diag) in prepared.items():
        h=8 if name.endswith('h8') else 3
        maps=fit_maps(qC,basis,omitted); cur=qC[:,-1].copy(); clips=[]
        for _ in range(h):
            y=predict(cur,maps,basis,omitted); q,d=p10.q_comp_delta(y-cur,dm,ds,True)
            cur=(cur+q).astype(np.float32); clips.append(d['clip_fraction'])
        splits[name]={
            'persistence_mse':pm,
            'companded_mse':p10.mse(cur,target),
            'companded_ratio':p10.mse(cur,target)/pm,
            'context_clip_fraction':diag['clip_fraction'],
            'rollout_clip_fraction':float(np.mean(clips)) if clips else 0.0,
        }
    primary={k:float(v['companded_ratio']) for k,v in splits.items()}
    return {
        'omitted_modes':list(omitted),
        'retained_spatial_rank':9-len(omitted),
        'feature_dim':8+2*(9-len(omitted)),
        'primary_companded_ratios':primary,
        'all_fifteen_below_persistence':bool(all(v<1 for v in primary.values())),
        'passing_cells':int(sum(v<1 for v in primary.values())),
        'splits':splits,
    }


@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    wave_id:int; wave_spectral:int; wave_combined:int
    gray_id:int; gray_sharp:int; gray_parameter:int; gray_combined:int
    nls_id:int; nls_spectral:int; nls_parameter:int; nls_combined:int
    adv_id:int; adv_spectral:int; adv_speed:int; adv_combined:int

REGISTERED=[
Seeds(824240001,824240002,824240003,824240004,824240005,824240006,824240011,824240012,824240013,824240021,824240022,824240023,824240024,824240031,824240032,824240033,824240034,824240041,824240042,824240043,824240044),
Seeds(824240101,824240102,824240103,824240104,824240105,824240106,824240111,824240112,824240113,824240121,824240122,824240123,824240124,824240131,824240132,824240133,824240134,824240141,824240142,824240143,824240144),
Seeds(824240201,824240202,824240203,824240204,824240205,824240206,824240211,824240212,824240213,824240221,824240222,824240223,824240224,824240231,824240232,824240233,824240234,824240241,824240242,824240243,824240244)]


def build_all(s,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    state,dm,ds=p16.build_codecs(s,grid,train_n,code_sizes,iters,max_points)
    return state,dm,ds,p22.fourier_basis()


def build_eval(s,grid=16,eval_n=48):
    return p22.build_eval(s,grid,eval_n)


def run_rep(s,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time(); state,dm,ds,basis=build_all(s,grid,train_n,code_sizes,iters,max_points); raw=build_eval(s,grid,eval_n)
    prepared={name:p22.prepare_split(x,state,dm,ds,8 if name.endswith('h8') else 3) for name,x in raw.items()}
    variants={variant_label(o):eval_prepared(prepared,basis,o,dm,ds) for o in OMISSION_SUBSETS}
    return {
        'seeds':asdict(s),
        'basis':'fixed_real_3x3_dft_sine_subset_lattice_v1',
        'sine_pool':list(SINE_POOL),
        'variant_count':len(variants),
        'context_frames':CONTEXT,'ridge':RIDGE,
        'representation_training_families':['wave','gray_scott'],
        'nls_used_in_representation_training':False,
        'advection_used_in_representation_training':False,
        'variants':variants,
        'runtime_seconds':time.time()-t0,
    }


def validate_invariants():
    assert len(OMISSION_SUBSETS)==16
    counts={r:sum(len(o)==r for o in OMISSION_SUBSETS) for r in range(5)}
    assert counts=={0:1,1:4,2:6,3:4,4:1}
    b=p22.fourier_basis(); np.testing.assert_allclose(b@b.T,np.eye(9),atol=1e-6)
    z=np.zeros((2,2,8,8),np.float32)
    for o in OMISSION_SUBSETS:
        assert selected_features(z,b,o).shape[-1]==8+2*(9-len(o))
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    return {'variant_count':16,'omission_counts':counts,'ranks_tested':[9,8,7,6,5],
            'context_frames':CONTEXT,'ridge':RIDGE,'basis_data_independent':True,'registered_seed_count':len(vals)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); inv=validate_invariants()
    if a.smoke:
        s=Seeds(999240001,999240002,999240003,999240004,999240005,999240006,999240011,999240012,999240013,999240021,999240022,999240023,999240024,999240031,999240032,999240033,999240034,999240041,999240042,999240043,999240044)
        r=run_rep(s,grid=8,train_n=12,eval_n=5,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE24_FOURIER_SUBSET_LATTICE_REGISTERED_CONFIRMATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
