from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase16 as p16
from tam_research import physics_tokenizer_phase17 as p17

CONTEXT=8
RIDGE=0.001
RANKS=(5,6,7,8,9)


def patch9(s,ch):
    z=s[:,ch]
    cols=[]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            cols.append(np.roll(np.roll(z,-dy,-2),-dx,-1))
    return np.stack(cols,-1).astype(np.float32)


def full_random_basis(seed0,seed1):
    rows=[]
    for seed in (seed0,seed1):
        rng=np.random.default_rng(seed)
        A=rng.normal(size=(9,9))
        Q,_=np.linalg.qr(A)
        W=Q.T.copy()
        for j in range(9):
            idx=int(np.argmax(np.abs(W[j])))
            if W[j,idx] < 0: W[j]*=-1
        rows.append(W.astype(np.float32))
    return np.stack(rows)


def projected_features(s,basis,k):
    cols=[np.ones_like(s[:,0])]
    for ch in range(2):
        P=patch9(s,ch)
        Y=np.einsum('bhwp,kp->bhwk',P,basis[ch,:k],optimize=True)
        cols.extend([Y[...,j] for j in range(k)])
    u,v=s[:,0],s[:,1]
    cols += [u*u,u*v,v*v,u*u*u,u*u*v,u*v*v,v*v*v]
    return np.stack(cols,-1).astype(np.float32)


def fit_maps(ctx,basis,k,ridge=RIDGE):
    dim=8+2*k; B=ctx.shape[0]; maps=np.empty((B,2,dim),np.float32)
    for b in range(B):
        X=[];Y=[]
        for t in range(ctx.shape[1]-1):
            X.append(projected_features(ctx[b:b+1,t],basis,k)[0].reshape(-1,dim))
            Y.append(ctx[b,t+1].transpose(1,2,0).reshape(-1,2))
        X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y).astype(np.float64)
        sc=np.sqrt(np.mean(X*X,axis=0)+1e-12); sc[0]=1.; Xs=X/sc
        W=np.linalg.solve(Xs.T@Xs+ridge*np.eye(dim),Xs.T@Y).T
        maps[b]=(W/sc[None,:]).astype(np.float32)
    return maps


def predict(s,maps,basis,k):
    return np.einsum('bhwf,bcf->bchw',projected_features(s,basis,k),maps,optimize=True).astype(np.float32)


def eval_rank(raw,state,dm,ds,basis,k,h):
    qC,diag=p10.encode_comp_seq(raw[:,:CONTEXT],state,dm,ds,True)
    maps=fit_maps(qC,basis,k); cur=qC[:,-1].copy(); clips=[]
    for _ in range(h):
        y=predict(cur,maps,basis,k); q,d=p10.q_comp_delta(y-cur,dm,ds,True)
        cur=(cur+q).astype(np.float32); clips.append(d['clip_fraction'])
    target=raw[:,CONTEXT+h-1]; persist=raw[:,CONTEXT-1]; pm=p10.mse(persist,target)
    raw_maps=fit_maps(raw[:,:CONTEXT],basis,k); r=raw[:,CONTEXT-1].copy()
    for _ in range(h): r=predict(r,raw_maps,basis,k)
    raw26=p17.fit_maps(raw[:,:CONTEXT]); r26=raw[:,CONTEXT-1].copy()
    for _ in range(h): r26=p17.predict(r26,raw26)
    return {
      'persistence_mse':pm,'companded_mse':p10.mse(cur,target),'companded_ratio':p10.mse(cur,target)/pm,
      'raw_projected_rollout_ratio':p10.mse(r,target)/pm if np.isfinite(r).all() else None,
      'raw_full_stencil26_rollout_ratio':p10.mse(r26,target)/pm if np.isfinite(r26).all() else None,
      'context_mse':p10.mse(qC,raw[:,:CONTEXT]),'context_clip_fraction':diag['clip_fraction'],
      'rollout_clip_fraction':float(np.mean(clips)) if clips else 0.}

@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int; basis0:int; basis1:int
    wave_id:int; wave_spectral:int; wave_combined:int
    gray_id:int; gray_sharp:int; gray_parameter:int; gray_combined:int
    nls_id:int; nls_spectral:int; nls_parameter:int; nls_combined:int
    adv_id:int; adv_spectral:int; adv_speed:int; adv_combined:int

REGISTERED=[
Seeds(821210001,821210002,821210003,821210004,821210005,821210006,821210007,821210008,821210011,821210012,821210013,821210021,821210022,821210023,821210024,821210031,821210032,821210033,821210034,821210041,821210042,821210043,821210044),
Seeds(821210101,821210102,821210103,821210104,821210105,821210106,821210107,821210108,821210111,821210112,821210113,821210121,821210122,821210123,821210124,821210131,821210132,821210133,821210134,821210141,821210142,821210143,821210144),
Seeds(821210201,821210202,821210203,821210204,821210205,821210206,821210207,821210208,821210211,821210212,821210213,821210221,821210222,821210223,821210224,821210231,821210232,821210233,821210234,821210241,821210242,821210243,821210244)]


def build_all(s,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    state,dm,ds=p16.build_codecs(s,grid,train_n,code_sizes,iters,max_points)
    return state,dm,ds,full_random_basis(s.basis0,s.basis1)


def build_eval(s,grid=16,eval_n=48):
    return {
      'wave_id_h8':p10.gen_wave(eval_n,16,grid,(.6,1.),s.wave_id,3)[0],
      'wave_spectral_ood_h3':p10.gen_wave(eval_n,11,grid,(.6,1.),s.wave_spectral,6)[0],
      'wave_combined_ood_h3':p10.gen_wave(eval_n,11,grid,(1.15,1.35),s.wave_combined,6)[0],
      'gray_id_h8':p16.gen_gray(eval_n,16,grid,(.025,.045),(.055,.065),s.gray_id,False)[0],
      'gray_sharp_ood_h3':p16.gen_gray(eval_n,11,grid,(.025,.045),(.055,.065),s.gray_sharp,True)[0],
      'gray_parameter_ood_h3':p16.gen_gray(eval_n,11,grid,(.050,.060),(.045,.055),s.gray_parameter,False)[0],
      'gray_combined_ood_h3':p16.gen_gray(eval_n,11,grid,(.050,.060),(.045,.055),s.gray_combined,True)[0],
      'nls_id_h8':p16.gen_nls(eval_n,16,grid,(.04,.08),(.10,.25),s.nls_id,3)[0],
      'nls_spectral_ood_h3':p16.gen_nls(eval_n,11,grid,(.04,.08),(.10,.25),s.nls_spectral,6)[0],
      'nls_parameter_ood_h3':p16.gen_nls(eval_n,11,grid,(.10,.14),(.30,.45),s.nls_parameter,3)[0],
      'nls_combined_ood_h3':p16.gen_nls(eval_n,11,grid,(.10,.14),(.30,.45),s.nls_combined,6)[0],
      'adv_id_h8':p16.gen_adv(eval_n,16,grid,(.18,.30),(-.20,.20),s.adv_id,3)[0],
      'adv_spectral_ood_h3':p16.gen_adv(eval_n,11,grid,(.18,.30),(-.20,.20),s.adv_spectral,6)[0],
      'adv_speed_ood_h3':p16.gen_adv(eval_n,11,grid,(.45,.60),(-.20,.20),s.adv_speed,3)[0],
      'adv_combined_ood_h3':p16.gen_adv(eval_n,11,grid,(.45,.60),(1.20,1.50),s.adv_combined,6)[0]}


def run_rep(s,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time(); state,dm,ds,basis=build_all(s,grid,train_n,code_sizes,iters,max_points); raw=build_eval(s,grid,eval_n)
    ranks={}
    for k in RANKS:
        splits={name:eval_rank(x,state,dm,ds,basis,k,8 if name.endswith('h8') else 3) for name,x in raw.items()}
        primary={name:float(v['companded_ratio']) for name,v in splits.items()}
        ranks[str(k)]={'feature_dim':8+2*k,'primary_companded_ratios':primary,'all_fifteen_below_persistence':bool(all(v<1 for v in primary.values())),'passing_cells':int(sum(v<1 for v in primary.values())),'splits':splits}
    return {'seeds':asdict(s),'basis_method':'nested_data_independent_random_orthogonal_9d_v1','ranks_tested':list(RANKS),'context_frames':CONTEXT,'ridge':RIDGE,'representation_training_families':['wave','gray_scott'],'nls_used_in_representation_training':False,'advection_used_in_representation_training':False,'ranks':ranks,'runtime_seconds':time.time()-t0}


def validate_invariants():
    b=full_random_basis(21001,21002); assert b.shape==(2,9,9)
    for ch in range(2): np.testing.assert_allclose(b[ch]@b[ch].T,np.eye(9),atol=1e-6)
    np.testing.assert_array_equal(b,full_random_basis(21001,21002))
    z=np.zeros((2,2,8,8),np.float32)
    for k in RANKS: assert projected_features(z,b,k).shape[-1]==8+2*k
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    return {'ranks':list(RANKS),'feature_dims':{str(k):8+2*k for k in RANKS},'context_frames':CONTEXT,'ridge':RIDGE,'basis_data_independent':True,'basis_orthonormal':True,'nested_prefixes':True,'registered_seed_count':len(vals)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); inv=validate_invariants()
    if a.smoke:
        s=Seeds(999210001,999210002,999210003,999210004,999210005,999210006,999210007,999210008,999210011,999210012,999210013,999210021,999210022,999210023,999210024,999210031,999210032,999210033,999210034,999210041,999210042,999210043,999210044)
        r=run_rep(s,grid=8,train_n=12,eval_n=5,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE21_PAIRED_RANDOM_RANK_SWEEP_REGISTERED_CONFIRMATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
