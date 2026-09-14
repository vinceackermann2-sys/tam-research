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


def geometry_features(s, rank=4):
    if rank not in (4,5):
        raise ValueError('rank must be 4 or 5')
    cols=[np.ones_like(s[:,0])]
    for ch in range(2):
        z=s[:,ch]
        n=np.roll(z,1,-2); south=np.roll(z,-1,-2)
        w=np.roll(z,1,-1); east=np.roll(z,-1,-1)
        cols += [z, n+south+east+w, east-w, south-n]
        if rank==5:
            ne=np.roll(np.roll(z,1,-2),-1,-1)
            nw=np.roll(np.roll(z,1,-2),1,-1)
            se=np.roll(np.roll(z,-1,-2),-1,-1)
            sw=np.roll(np.roll(z,-1,-2),1,-1)
            cols.append(ne+nw+se+sw)
    u,v=s[:,0],s[:,1]
    cols += [u*u,u*v,v*v,u*u*u,u*u*v,u*v*v,v*v*v]
    return np.stack(cols,-1).astype(np.float32)


def fit_maps(ctx, rank=4, ridge=RIDGE):
    dim=8+2*rank; B=ctx.shape[0]
    maps=np.empty((B,2,dim),np.float32)
    for b in range(B):
        X=[];Y=[]
        for t in range(ctx.shape[1]-1):
            X.append(geometry_features(ctx[b:b+1,t],rank)[0].reshape(-1,dim))
            Y.append(ctx[b,t+1].transpose(1,2,0).reshape(-1,2))
        X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y).astype(np.float64)
        sc=np.sqrt(np.mean(X*X,axis=0)+1e-12); sc[0]=1.; Xs=X/sc
        W=np.linalg.solve(Xs.T@Xs+ridge*np.eye(dim),Xs.T@Y).T
        maps[b]=(W/sc[None,:]).astype(np.float32)
    return maps


def predict(s,maps,rank=4):
    return np.einsum('bhwf,bcf->bchw',geometry_features(s,rank),maps,optimize=True).astype(np.float32)


def prepare_split(raw,state,dm,ds,h):
    qC,diag=p10.encode_comp_seq(raw[:,:CONTEXT],state,dm,ds,True)
    target=raw[:,CONTEXT+h-1]; persist=raw[:,CONTEXT-1]; pm=p10.mse(persist,target)
    return qC,target,pm,diag,raw


def eval_variant(prepared,rank,dm,ds):
    out={}
    for name,(qC,target,pm,diag,raw) in prepared.items():
        h=8 if name.endswith('h8') else 3
        maps=fit_maps(qC,rank); cur=qC[:,-1].copy(); clips=[]
        for _ in range(h):
            y=predict(cur,maps,rank); q,d=p10.q_comp_delta(y-cur,dm,ds,True)
            cur=(cur+q).astype(np.float32); clips.append(d['clip_fraction'])
        ratio=p10.mse(cur,target)/pm
        out[name]={
            'persistence_mse':pm,
            'companded_mse':p10.mse(cur,target),
            'companded_ratio':ratio,
            'context_clip_fraction':diag['clip_fraction'],
            'rollout_clip_fraction':float(np.mean(clips)) if clips else 0.0,
        }
    primary={k:float(v['companded_ratio']) for k,v in out.items()}
    return {
        'spatial_rank_per_channel':rank,
        'feature_dim':8+2*rank,
        'primary_companded_ratios':primary,
        'all_fifteen_below_persistence':bool(all(v<1 for v in primary.values())),
        'passing_cells':int(sum(v<1 for v in primary.values())),
        'splits':out,
    }


@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    wave_id:int; wave_spectral:int; wave_combined:int
    gray_id:int; gray_sharp:int; gray_parameter:int; gray_combined:int
    nls_id:int; nls_spectral:int; nls_parameter:int; nls_combined:int
    adv_id:int; adv_spectral:int; adv_speed:int; adv_combined:int

REGISTERED=[
Seeds(823230001,823230002,823230003,823230004,823230005,823230006,823230011,823230012,823230013,823230021,823230022,823230023,823230024,823230031,823230032,823230033,823230034,823230041,823230042,823230043,823230044),
Seeds(823230101,823230102,823230103,823230104,823230105,823230106,823230111,823230112,823230113,823230121,823230122,823230123,823230124,823230131,823230132,823230133,823230134,823230141,823230142,823230143,823230144),
Seeds(823230201,823230202,823230203,823230204,823230205,823230206,823230211,823230212,823230213,823230221,823230222,823230223,823230224,823230231,823230232,823230233,823230234,823230241,823230242,823230243,823230244)]


def build_all(s,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    return p16.build_codecs(s,grid,train_n,code_sizes,iters,max_points)


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
    t0=time.time(); state,dm,ds=build_all(s,grid,train_n,code_sizes,iters,max_points); raw=build_eval(s,grid,eval_n)
    prepared={name:prepare_split(x,state,dm,ds,8 if name.endswith('h8') else 3) for name,x in raw.items()}
    rank4=eval_variant(prepared,4,dm,ds); rank5=eval_variant(prepared,5,dm,ds)
    return {
        'seeds':asdict(s),
        'basis':'fixed_geometry_symmetry_v1',
        'rank4_terms':['center','axial_even','x_odd','y_odd'],
        'rank5_extra_term':'diagonal_even',
        'context_frames':CONTEXT,'ridge':RIDGE,
        'representation_training_families':['wave','gray_scott'],
        'nls_used_in_representation_training':False,
        'advection_used_in_representation_training':False,
        'rank4':rank4,'rank5':rank5,
        'runtime_seconds':time.time()-t0,
    }


def validate_invariants():
    z=np.random.default_rng(2301).normal(size=(2,2,8,8)).astype(np.float32)
    assert geometry_features(z,4).shape[-1]==16
    assert geometry_features(z,5).shape[-1]==18
    f=geometry_features(z,4)
    for ch in range(2):
        off=1+ch*4; zz=z[:,ch]
        axial=np.roll(zz,1,-2)+np.roll(zz,-1,-2)+np.roll(zz,-1,-1)+np.roll(zz,1,-1)
        xodd=np.roll(zz,-1,-1)-np.roll(zz,1,-1)
        yodd=np.roll(zz,-1,-2)-np.roll(zz,1,-2)
        np.testing.assert_allclose(f[...,off+1],axial)
        np.testing.assert_allclose(f[...,off+2],xodd)
        np.testing.assert_allclose(f[...,off+3],yodd)
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    return {'rank4_feature_dim':16,'rank5_feature_dim':18,'context_frames':CONTEXT,'ridge':RIDGE,'basis_data_independent':True,'registered_seed_count':len(vals)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--rep',type=int,default=0);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();inv=validate_invariants()
    if a.smoke:
        s=Seeds(999230001,999230002,999230003,999230004,999230005,999230006,999230011,999230012,999230013,999230021,999230022,999230023,999230024,999230031,999230032,999230033,999230034,999230041,999230042,999230043,999230044)
        r=run_rep(s,grid=8,train_n=12,eval_n=5,code_sizes=(32,8),iters=3,max_points=3000);r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]);r['replicate']=a.rep;r['status']='PHASE23_GEOMETRY_BASIS_REGISTERED_CONFIRMATION'
    r['invariants']=inv;Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
