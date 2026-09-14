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
MODE_NAMES=(
    'dc','cos_x','sin_x','cos_y','sin_y',
    'cos_diag_plus','sin_diag_plus','cos_diag_minus','sin_diag_minus')


def patch9(s,ch):
    z=s[:,ch]
    cols=[]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            cols.append(np.roll(np.roll(z,-dy,-2),-dx,-1))
    return np.stack(cols,-1).astype(np.float32)


def fourier_basis():
    coords=[(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)]
    rows=[np.ones(9,dtype=np.float64)/3.0]
    for kx,ky in ((1,0),(0,1),(1,1),(1,-1)):
        ph=np.array([2*np.pi*(kx*dx+ky*dy)/3.0 for dy,dx in coords],dtype=np.float64)
        rows.append(np.sqrt(2.0/9.0)*np.cos(ph))
        rows.append(np.sqrt(2.0/9.0)*np.sin(ph))
    return np.stack(rows).astype(np.float32)


def selected_features(s,basis,omit=None):
    keep=list(range(9)) if omit is None else [j for j in range(9) if j!=omit]
    cols=[np.ones_like(s[:,0])]
    for ch in range(2):
        P=patch9(s,ch)
        Y=np.einsum('bhwp,kp->bhwk',P,basis[keep],optimize=True)
        cols.extend([Y[...,j] for j in range(len(keep))])
    u,v=s[:,0],s[:,1]
    cols += [u*u,u*v,v*v,u*u*u,u*u*v,u*v*v,v*v*v]
    return np.stack(cols,-1).astype(np.float32)


def fit_maps(ctx,basis,omit=None,ridge=RIDGE):
    k=9 if omit is None else 8
    dim=8+2*k; B=ctx.shape[0]; maps=np.empty((B,2,dim),np.float32)
    for b in range(B):
        X=[];Y=[]
        for t in range(ctx.shape[1]-1):
            X.append(selected_features(ctx[b:b+1,t],basis,omit)[0].reshape(-1,dim))
            Y.append(ctx[b,t+1].transpose(1,2,0).reshape(-1,2))
        X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y).astype(np.float64)
        sc=np.sqrt(np.mean(X*X,axis=0)+1e-12); sc[0]=1.; Xs=X/sc
        W=np.linalg.solve(Xs.T@Xs+ridge*np.eye(dim),Xs.T@Y).T
        maps[b]=(W/sc[None,:]).astype(np.float32)
    return maps


def predict(s,maps,basis,omit=None):
    return np.einsum('bhwf,bcf->bchw',selected_features(s,basis,omit),maps,optimize=True).astype(np.float32)


def prepare_split(raw,state,dm,ds,h):
    qC,diag=p10.encode_comp_seq(raw[:,:CONTEXT],state,dm,ds,True)
    target=raw[:,CONTEXT+h-1]; persist=raw[:,CONTEXT-1]; pm=p10.mse(persist,target)
    return qC,target,pm,diag


@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    wave_id:int; wave_spectral:int; wave_combined:int
    gray_id:int; gray_sharp:int; gray_parameter:int; gray_combined:int
    nls_id:int; nls_spectral:int; nls_parameter:int; nls_combined:int
    adv_id:int; adv_spectral:int; adv_speed:int; adv_combined:int

REGISTERED=[
Seeds(822220001,822220002,822220003,822220004,822220005,822220006,822220011,822220012,822220013,822220021,822220022,822220023,822220024,822220031,822220032,822220033,822220034,822220041,822220042,822220043,822220044),
Seeds(822220101,822220102,822220103,822220104,822220105,822220106,822220111,822220112,822220113,822220121,822220122,822220123,822220124,822220131,822220132,822220133,822220134,822220141,822220142,822220143,822220144),
Seeds(822220201,822220202,822220203,822220204,822220205,822220206,822220211,822220212,822220213,822220221,822220222,822220223,822220224,822220231,822220232,822220233,822220234,822220241,822220242,822220243,822220244)]


def build_all(s,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    state,dm,ds=p16.build_codecs(s,grid,train_n,code_sizes,iters,max_points)
    return state,dm,ds,fourier_basis()


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


def eval_prepared_with_codec(prepared,basis,omit,h,dm,ds):
    qC,target,pm,diag=prepared
    maps=fit_maps(qC,basis,omit); cur=qC[:,-1].copy(); clips=[]
    for _ in range(h):
        y=predict(cur,maps,basis,omit); q,d=p10.q_comp_delta(y-cur,dm,ds,True)
        cur=(cur+q).astype(np.float32); clips.append(d['clip_fraction'])
    return {
      'persistence_mse':pm,'companded_mse':p10.mse(cur,target),'companded_ratio':p10.mse(cur,target)/pm,
      'context_clip_fraction':diag['clip_fraction'],
      'rollout_clip_fraction':float(np.mean(clips)) if clips else 0.}


def summarize_variant(prepared,basis,omit,dm,ds):
    splits={name:eval_prepared_with_codec(prep,basis,omit,8 if name.endswith('h8') else 3,dm,ds) for name,prep in prepared.items()}
    primary={name:float(v['companded_ratio']) for name,v in splits.items()}
    return {'feature_dim':26 if omit is None else 24,'primary_companded_ratios':primary,
            'all_fifteen_below_persistence':bool(all(v<1 for v in primary.values())),
            'passing_cells':int(sum(v<1 for v in primary.values())),'splits':splits}


def run_rep(s,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time(); state,dm,ds,basis=build_all(s,grid,train_n,code_sizes,iters,max_points); raw=build_eval(s,grid,eval_n)
    prepared={name:prepare_split(x,state,dm,ds,8 if name.endswith('h8') else 3) for name,x in raw.items()}
    full=summarize_variant(prepared,basis,None,dm,ds)
    lodo={name:summarize_variant(prepared,basis,i,dm,ds) for i,name in enumerate(MODE_NAMES)}
    return {'seeds':asdict(s),'basis':'fixed_real_3x3_dft_v1','mode_names':list(MODE_NAMES),
            'context_frames':CONTEXT,'ridge':RIDGE,'representation_training_families':['wave','gray_scott'],
            'nls_used_in_representation_training':False,'advection_used_in_representation_training':False,
            'full_control':full,'leave_one_out':lodo,'runtime_seconds':time.time()-t0}


def validate_invariants():
    b=fourier_basis(); np.testing.assert_allclose(b@b.T,np.eye(9),atol=1e-6)
    z=np.random.default_rng(22).normal(size=(3,2,8,8)).astype(np.float32)
    for ch in range(2):
        P=patch9(z,ch); Y=np.einsum('bhwp,kp->bhwk',P,b,optimize=True); rec=np.einsum('bhwk,kp->bhwp',Y,b,optimize=True)
        np.testing.assert_allclose(P,rec,rtol=1e-5,atol=1e-5)
    assert selected_features(z,b,None).shape[-1]==26
    for i in range(9): assert selected_features(z,b,i).shape[-1]==24
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    return {'basis_orthonormal':True,'basis_data_independent':True,'full_feature_dim':26,'lodo_feature_dim':24,
            'mode_names':list(MODE_NAMES),'context_frames':CONTEXT,'ridge':RIDGE,'registered_seed_count':len(vals)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); inv=validate_invariants()
    if a.smoke:
        s=Seeds(999220001,999220002,999220003,999220004,999220005,999220006,999220011,999220012,999220013,999220021,999220022,999220023,999220024,999220031,999220032,999220033,999220034,999220041,999220042,999220043,999220044)
        r=run_rep(s,grid=8,train_n=12,eval_n=5,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE22_FOURIER_LODO_REGISTERED_CONFIRMATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
