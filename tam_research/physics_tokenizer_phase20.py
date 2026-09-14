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
K=5


def patch9(s,ch):
    z=s[:,ch]
    cols=[]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            cols.append(np.roll(np.roll(z,-dy,-2),-dx,-1))
    return np.stack(cols,-1).astype(np.float32)


def random_basis(seed0, seed1, k=K):
    rows=[]
    for seed in (seed0,seed1):
        rng=np.random.default_rng(seed)
        A=rng.normal(size=(9,k))
        Q,_=np.linalg.qr(A,mode='reduced')
        W=Q.T.copy()
        for j in range(k):
            idx=int(np.argmax(np.abs(W[j])))
            if W[j,idx] < 0: W[j]*=-1
        rows.append(W.astype(np.float32))
    return np.stack(rows)


def projected_features(s,basis):
    cols=[np.ones_like(s[:,0])]
    for ch in range(2):
        P=patch9(s,ch)
        Y=np.einsum('bhwp,kp->bhwk',P,basis[ch],optimize=True)
        cols.extend([Y[...,j] for j in range(K)])
    u,v=s[:,0],s[:,1]
    cols += [u*u,u*v,v*v,u*u*u,u*u*v,u*v*v,v*v*v]
    return np.stack(cols,-1).astype(np.float32)


def fit_maps(ctx,basis,ridge=RIDGE):
    dim=18; B=ctx.shape[0]; maps=np.empty((B,2,dim),np.float32)
    for b in range(B):
        X=[];Y=[]
        for t in range(ctx.shape[1]-1):
            X.append(projected_features(ctx[b:b+1,t],basis)[0].reshape(-1,dim))
            Y.append(ctx[b,t+1].transpose(1,2,0).reshape(-1,2))
        X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y).astype(np.float64)
        sc=np.sqrt(np.mean(X*X,axis=0)+1e-12); sc[0]=1.; Xs=X/sc
        W=np.linalg.solve(Xs.T@Xs+ridge*np.eye(dim),Xs.T@Y).T
        maps[b]=(W/sc[None,:]).astype(np.float32)
    return maps


def predict(s,maps,basis):
    return np.einsum('bhwf,bcf->bchw',projected_features(s,basis),maps,optimize=True).astype(np.float32)


def eval_companded(raw,state,dm,ds,basis,h):
    qC,diag=p10.encode_comp_seq(raw[:,:CONTEXT],state,dm,ds,True)
    maps=fit_maps(qC,basis); cur=qC[:,-1].copy(); clips=[]
    for _ in range(h):
        y=predict(cur,maps,basis); q,d=p10.q_comp_delta(y-cur,dm,ds,True)
        cur=(cur+q).astype(np.float32); clips.append(d['clip_fraction'])
    target=raw[:,CONTEXT+h-1]; persist=raw[:,CONTEXT-1]; pm=p10.mse(persist,target)
    raw18=fit_maps(raw[:,:CONTEXT],basis); r18=raw[:,CONTEXT-1].copy()
    for _ in range(h): r18=predict(r18,raw18,basis)
    raw26=p17.fit_maps(raw[:,:CONTEXT]); r26=raw[:,CONTEXT-1].copy()
    for _ in range(h): r26=p17.predict(r26,raw26)
    return {
      'persistence_mse':pm,'companded_mse':p10.mse(cur,target),'companded_ratio':p10.mse(cur,target)/pm,
      'raw_random18_rollout_ratio':p10.mse(r18,target)/pm if np.isfinite(r18).all() else None,
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
Seeds(820200001,820200002,820200003,820200004,820200005,820200006,820200007,820200008,820200011,820200012,820200013,820200021,820200022,820200023,820200024,820200031,820200032,820200033,820200034,820200041,820200042,820200043,820200044),
Seeds(820200101,820200102,820200103,820200104,820200105,820200106,820200107,820200108,820200111,820200112,820200113,820200121,820200122,820200123,820200124,820200131,820200132,820200133,820200134,820200141,820200142,820200143,820200144),
Seeds(820200201,820200202,820200203,820200204,820200205,820200206,820200207,820200208,820200211,820200212,820200213,820200221,820200222,820200223,820200224,820200231,820200232,820200233,820200234,820200241,820200242,820200243,820200244)]


def build_all(s,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    state,dm,ds=p16.build_codecs(s,grid,train_n,code_sizes,iters,max_points)
    return state,dm,ds,random_basis(s.basis0,s.basis1,K)


def run_rep(s,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time(); state,dm,ds,basis=build_all(s,grid,train_n,code_sizes,iters,max_points)
    raw={
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
    splits={k:eval_companded(x,state,dm,ds,basis,8 if k.endswith('h8') else 3) for k,x in raw.items()}
    primary={k:float(v['companded_ratio']) for k,v in splits.items()}
    return {'seeds':asdict(s),'feature_library':'random_orthogonal_5x2_patch_projection_plus_center_polynomial_v1','feature_dim':18,'basis_components_per_channel':K,'basis_data_independent':True,'context_frames':CONTEXT,'ridge':RIDGE,'representation_training_families':['wave','gray_scott'],'nls_used_in_representation_training':False,'advection_used_in_representation_training':False,'primary_companded_ratios':primary,'all_fifteen_below_persistence':bool(all(v<1 for v in primary.values())),'splits':splits,'runtime_seconds':time.time()-t0}


def validate_invariants():
    b=random_basis(12345,67890)
    assert b.shape==(2,K,9)
    for ch in range(2): np.testing.assert_allclose(b[ch]@b[ch].T,np.eye(K),atol=1e-6)
    np.testing.assert_array_equal(b,random_basis(12345,67890))
    z=np.zeros((2,2,8,8),np.float32); assert projected_features(z,b).shape[-1]==18
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    return {'feature_dim':18,'components_per_channel':K,'context_frames':CONTEXT,'ridge':RIDGE,'basis_data_independent':True,'basis_orthonormal':True,'registered_seed_count':len(vals)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); inv=validate_invariants()
    if a.smoke:
        s=Seeds(999200001,999200002,999200003,999200004,999200005,999200006,999200007,999200008,999200011,999200012,999200013,999200021,999200022,999200023,999200024,999200031,999200032,999200033,999200034,999200041,999200042,999200043,999200044)
        r=run_rep(s,grid=8,train_n=12,eval_n=5,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE20_RANDOM_STENCIL_PROJECTION_REGISTERED_CONFIRMATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
