from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase16 as p16
from tam_research import physics_tokenizer_phase22 as p22

CONTEXT=8
RIDGE=0.001
BURGERS_DT=0.05


def smooth_field(rng,n,max_mode=3):
    yy,xx=np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
    f=np.zeros((n,n),np.float32)
    for _ in range(int(rng.integers(2,6))):
        kx=int(rng.integers(0,max_mode+1)); ky=int(rng.integers(0,max_mode+1))
        if kx==0 and ky==0: kx=1
        amp=float(rng.normal())/(1+kx*kx+ky*ky)
        ph=float(rng.uniform(0,2*np.pi))
        f += amp*np.cos(2*np.pi*(kx*xx+ky*yy)/n+ph)
    f-=f.mean(); f/=f.std()+1e-6
    return f.astype(np.float32)


def centered_dx(z):
    return .5*(np.roll(z,-1,-1)-np.roll(z,1,-1))


def centered_dy(z):
    return .5*(np.roll(z,-1,-2)-np.roll(z,1,-2))


def burgers_step(u,v,nu,dt=BURGERS_DT):
    du = -u*centered_dx(u) - v*centered_dy(u) + nu*p10.lap(u)
    dv = -u*centered_dx(v) - v*centered_dy(v) + nu*p10.lap(v)
    return (u+dt*du).astype(np.float32),(v+dt*dv).astype(np.float32)


def gen_burgers(num,seq_len,grid,nu_range,amp_range,seed,max_mode=3):
    rng=np.random.default_rng(seed)
    out=np.empty((num,seq_len,2,grid,grid),np.float32)
    nus=rng.uniform(*nu_range,size=num).astype(np.float32)
    amps=np.empty((num,2),np.float32)
    for i in range(num):
        a0=float(rng.uniform(*amp_range)); a1=float(rng.uniform(*amp_range)); amps[i]=[a0,a1]
        u=smooth_field(rng,grid,max_mode)*a0
        v=smooth_field(rng,grid,max_mode)*a1
        for t in range(seq_len):
            out[i,t,0]=u; out[i,t,1]=v
            u,v=burgers_step(u,v,float(nus[i]))
    return out,nus,amps


def base26_features(s,basis):
    return p22.selected_features(s,basis,None)


def interaction62_features(s,basis):
    base=base26_features(s,basis)
    spatial=base[...,1:19]
    u=s[:,0][...,None]; v=s[:,1][...,None]
    return np.concatenate([base,u*spatial,v*spatial],axis=-1).astype(np.float32)


def fit_maps(ctx,feature_fn,dim,basis,ridge=RIDGE):
    B=ctx.shape[0]; maps=np.empty((B,2,dim),np.float32)
    for b in range(B):
        X=[];Y=[]
        for t in range(ctx.shape[1]-1):
            X.append(feature_fn(ctx[b:b+1,t],basis)[0].reshape(-1,dim))
            Y.append(ctx[b,t+1].transpose(1,2,0).reshape(-1,2))
        X=np.concatenate(X).astype(np.float64);Y=np.concatenate(Y).astype(np.float64)
        sc=np.sqrt(np.mean(X*X,axis=0)+1e-12);sc[0]=1.;Xs=X/sc
        W=np.linalg.solve(Xs.T@Xs+ridge*np.eye(dim),Xs.T@Y).T
        maps[b]=(W/sc[None,:]).astype(np.float32)
    return maps


def predict(s,maps,feature_fn,basis):
    return np.einsum('bhwf,bcf->bchw',feature_fn(s,basis),maps,optimize=True).astype(np.float32)


def continuous_rollout(start,maps,h,feature_fn,basis):
    cur=start.copy()
    for _ in range(h): cur=predict(cur,maps,feature_fn,basis)
    return cur


def token_rollout(start,maps,h,feature_fn,basis,dm,ds):
    cur=start.copy(); clips=[]
    for _ in range(h):
        y=predict(cur,maps,feature_fn,basis); q,d=p10.q_comp_delta(y-cur,dm,ds,True)
        cur=(cur+q).astype(np.float32); clips.append(d['clip_fraction'])
    return cur,float(np.mean(clips)) if clips else 0.0


def eval_split(raw,state,dm,ds,basis,h):
    qC,diag=p10.encode_comp_seq(raw[:,:CONTEXT],state,dm,ds,True)
    target=raw[:,CONTEXT+h-1]; persist=raw[:,CONTEXT-1]; pm=p10.mse(persist,target)

    m26_tok=fit_maps(qC,base26_features,26,basis)
    y26_tok,clip26=token_rollout(qC[:,-1],m26_tok,h,base26_features,basis,dm,ds)

    m26_raw=fit_maps(raw[:,:CONTEXT],base26_features,26,basis)
    y26_raw=continuous_rollout(raw[:,CONTEXT-1],m26_raw,h,base26_features,basis)

    m62_raw=fit_maps(raw[:,:CONTEXT],interaction62_features,62,basis)
    y62_raw=continuous_rollout(raw[:,CONTEXT-1],m62_raw,h,interaction62_features,basis)

    m62_tok=fit_maps(qC,interaction62_features,62,basis)
    y62_tok,clip62=token_rollout(qC[:,-1],m62_tok,h,interaction62_features,basis,dm,ds)

    # One-step raw diagnostics use the same context-fitted maps.
    one_target=raw[:,CONTEXT]
    one26=predict(raw[:,CONTEXT-1],m26_raw,base26_features,basis)
    one62=predict(raw[:,CONTEXT-1],m62_raw,interaction62_features,basis)
    one_pm=p10.mse(raw[:,CONTEXT-1],one_target)

    return {
      'persistence_mse':pm,
      'primary_token26_mse':p10.mse(y26_tok,target),
      'primary_token26_ratio':p10.mse(y26_tok,target)/pm,
      'raw26_rollout_ratio':p10.mse(y26_raw,target)/pm if np.isfinite(y26_raw).all() else None,
      'raw62_rollout_ratio':p10.mse(y62_raw,target)/pm if np.isfinite(y62_raw).all() else None,
      'token62_rollout_ratio':p10.mse(y62_tok,target)/pm,
      'raw26_one_step_ratio':p10.mse(one26,one_target)/one_pm,
      'raw62_one_step_ratio':p10.mse(one62,one_target)/one_pm,
      'context_mse':p10.mse(qC,raw[:,:CONTEXT]),
      'context_clip_fraction':diag['clip_fraction'],
      'primary_rollout_clip_fraction':clip26,
      'token62_rollout_clip_fraction':clip62,
    }


@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    burg_id:int; burg_spectral:int; burg_parameter:int; burg_combined:int

REGISTERED=[
Seeds(825250001,825250002,825250003,825250004,825250005,825250006,825250011,825250012,825250013,825250014),
Seeds(825250101,825250102,825250103,825250104,825250105,825250106,825250111,825250112,825250113,825250114),
Seeds(825250201,825250202,825250203,825250204,825250205,825250206,825250211,825250212,825250213,825250214)]


def build_all(s,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    state,dm,ds=p16.build_codecs(s,grid,train_n,code_sizes,iters,max_points)
    return state,dm,ds,p22.fourier_basis()


def build_eval(s,grid=16,eval_n=48):
    return {
      'burgers_id_h8':gen_burgers(eval_n,16,grid,(.01,.03),(.15,.35),s.burg_id,3)[0],
      'burgers_spectral_ood_h3':gen_burgers(eval_n,11,grid,(.01,.03),(.15,.35),s.burg_spectral,6)[0],
      'burgers_parameter_ood_h3':gen_burgers(eval_n,11,grid,(.04,.06),(.15,.35),s.burg_parameter,3)[0],
      'burgers_combined_ood_h3':gen_burgers(eval_n,11,grid,(.04,.06),(.35,.55),s.burg_combined,6)[0],
    }


def run_rep(s,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time();state,dm,ds,basis=build_all(s,grid,train_n,code_sizes,iters,max_points);raw=build_eval(s,grid,eval_n)
    splits={name:eval_split(x,state,dm,ds,basis,8 if name.endswith('h8') else 3) for name,x in raw.items()}
    primary={k:float(v['primary_token26_ratio']) for k,v in splits.items()}
    return {
      'seeds':asdict(s),'family':'two_component_viscous_burgers_2d','dt':BURGERS_DT,
      'primary_feature_library':'full_fourier_linear_plus_center_polynomial_26','diagnostic_feature_library':'plus_center_times_spatial_62',
      'context_frames':CONTEXT,'ridge':RIDGE,'representation_training_families':['wave','gray_scott'],
      'burgers_used_in_representation_training':False,
      'primary_ratios':primary,'primary_all_four_below_persistence':bool(all(v<1 for v in primary.values())),
      'splits':splits,'runtime_seconds':time.time()-t0,
    }


def validate_invariants():
    rng=np.random.default_rng(25);z=rng.normal(size=(2,2,8,8)).astype(np.float32);b=p22.fourier_basis()
    assert base26_features(z,b).shape[-1]==26
    assert interaction62_features(z,b).shape[-1]==62
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    x,nu,amp=gen_burgers(3,6,8,(.01,.03),(.15,.35),999250001,3)
    assert np.isfinite(x).all() and x.shape==(3,6,2,8,8)
    return {'base_feature_dim':26,'interaction_feature_dim':62,'context_frames':CONTEXT,'ridge':RIDGE,
            'burgers_dt':BURGERS_DT,'registered_seed_count':len(vals),'simulator_finite_smoke':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--rep',type=int,default=0);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();inv=validate_invariants()
    if a.smoke:
      s=Seeds(999250101,999250102,999250103,999250104,999250105,999250106,999250111,999250112,999250113,999250114)
      r=run_rep(s,grid=8,train_n=12,eval_n=5,code_sizes=(32,8),iters=3,max_points=3000);r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
      if a.rep not in (1,2,3):raise SystemExit('--rep must be 1,2,3')
      r=run_rep(REGISTERED[a.rep-1]);r['replicate']=a.rep;r['status']='PHASE25_BURGERS_TRANSFER_REGISTERED_CONFIRMATION'
    r['invariants']=inv;Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
