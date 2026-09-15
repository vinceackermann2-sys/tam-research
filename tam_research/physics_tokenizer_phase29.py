from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase22 as p22
from tam_research import physics_tokenizer_phase25 as p25
from tam_research import physics_tokenizer_phase28 as p28

CONTEXT=8
RIDGE=0.001
HJ_DT=0.08
HJ_SUBSTEPS=4
ODD_MODES=(2,4,6,8)

@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    hj_id:int; hj_spectral:int; hj_parameter:int; hj_combined:int

REGISTERED=[
    Seeds(829290001,829290002,829290003,829290004,829290005,829290006,829290011,829290012,829290013,829290014),
    Seeds(829290101,829290102,829290103,829290104,829290105,829290106,829290111,829290112,829290113,829290114),
    Seeds(829290201,829290202,829290203,829290204,829290205,829290206,829290211,829290212,829290213,829290214),
]

def centered_dx(z):
    return .5*(np.roll(z,-1,-1)-np.roll(z,1,-1))

def centered_dy(z):
    return .5*(np.roll(z,-1,-2)-np.roll(z,1,-2))

def hj_step(z,nu,lam,dt):
    gx=centered_dx(z); gy=centered_dy(z)
    return (z + dt*(nu*p10.lap(z) - .5*lam*(gx*gx+gy*gy))).astype(np.float32)

def gen_hj(num,seq_len,grid,nu_range,lam_range,amp_range,seed,max_mode=3):
    rng=np.random.default_rng(seed)
    out=np.empty((num,seq_len,2,grid,grid),np.float32)
    nus=rng.uniform(*nu_range,size=(num,2)).astype(np.float32)
    lams=rng.uniform(*lam_range,size=(num,2)).astype(np.float32)
    amps=rng.uniform(*amp_range,size=(num,2)).astype(np.float32)
    subdt=HJ_DT/HJ_SUBSTEPS
    for i in range(num):
        zs=[p10.smooth_field(rng,grid,max_mode)*float(amps[i,ch]) for ch in range(2)]
        for t in range(seq_len):
            out[i,t,0]=zs[0]; out[i,t,1]=zs[1]
            for _ in range(HJ_SUBSTEPS):
                zs=[hj_step(zs[ch],float(nus[i,ch]),float(lams[i,ch]),subdt) for ch in range(2)]
    return out,nus,lams,amps

def odd_spatial_quadratic82(s,basis):
    base=p25.interaction62_features(s,basis)
    extra=[]
    for ch in range(2):
        P=p22.patch9(s,ch)
        Y=np.einsum('bhwp,kp->bhwk',P,basis,optimize=True)
        odd=Y[...,ODD_MODES]
        for i in range(4):
            for j in range(i,4):
                extra.append(odd[...,i]*odd[...,j])
    return np.concatenate([base,np.stack(extra,-1)],axis=-1).astype(np.float32)

def build_all(s,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    return p25.build_all(s,grid,train_n,code_sizes,iters,max_points)

def build_eval(s,grid=16,eval_n=48):
    return {
      'hj_id_h8':gen_hj(eval_n,16,grid,(.015,.035),(.20,.40),(.25,.50),s.hj_id,3)[0],
      'hj_spectral_ood_h3':gen_hj(eval_n,11,grid,(.015,.035),(.20,.40),(.25,.50),s.hj_spectral,6)[0],
      'hj_parameter_ood_h3':gen_hj(eval_n,11,grid,(.05,.08),(.50,.80),(.25,.50),s.hj_parameter,3)[0],
      'hj_combined_ood_h3':gen_hj(eval_n,11,grid,(.05,.08),(.50,.80),(.45,.70),s.hj_combined,6)[0],
    }

def eval_split(raw,state,dm,ds,basis,h):
    qc,diag=p10.encode_comp_seq(raw[:,:CONTEXT],state,dm,ds,True)
    target=raw[:,CONTEXT+h-1]; persist=raw[:,CONTEXT-1]; pm=p10.mse(persist,target)

    choose62,mse26,mse62,s26,s62=p28.select_models(qc,basis)
    m26=p25.fit_maps(qc,p25.base26_features,26,basis,RIDGE)
    m62=p25.fit_maps(qc,p25.interaction62_features,62,basis,RIDGE)
    ys,selclip=p28.selected_rollout(qc[:,-1],m26,m62,choose62,h,basis,dm,ds)
    y26,c26=p25.token_rollout(qc[:,-1],m26,h,p25.base26_features,basis,dm,ds)
    y62,c62=p25.token_rollout(qc[:,-1],m62,h,p25.interaction62_features,basis,dm,ds)

    m82tok=p25.fit_maps(qc,odd_spatial_quadratic82,82,basis,RIDGE)
    y82tok,c82=p25.token_rollout(qc[:,-1],m82tok,h,odd_spatial_quadratic82,basis,dm,ds)

    m82raw=p25.fit_maps(raw[:,:CONTEXT],odd_spatial_quadratic82,82,basis,RIDGE)
    one_target=raw[:,CONTEXT]; one_pred=p25.predict(raw[:,CONTEXT-1],m82raw,odd_spatial_quadratic82,basis)
    one_pm=p10.mse(raw[:,CONTEXT-1],one_target)
    yr82=p25.continuous_rollout(raw[:,CONTEXT-1],m82raw,h,odd_spatial_quadratic82,basis)
    raw_roll=p10.mse(yr82,target)/pm if np.isfinite(yr82).all() else None

    return {
      'persistence_mse':pm,
      'selected_mse':p10.mse(ys,target),'selected_ratio':p10.mse(ys,target)/pm,
      'simple26_ratio':p10.mse(y26,target)/pm,'interaction62_ratio':p10.mse(y62,target)/pm,
      'selected_interaction_fraction':float(np.mean(choose62)),
      'diagnostic_token82_ratio':p10.mse(y82tok,target)/pm,
      'diagnostic_raw82_one_step_ratio':p10.mse(one_pred,one_target)/one_pm,
      'diagnostic_raw82_rollout_ratio':raw_roll,
      'validation_mse26_mean':float(np.mean(mse26)),'validation_mse62_mean':float(np.mean(mse62)),
      'validation_score26_mean':float(np.mean(s26)),'validation_score62_mean':float(np.mean(s62)),
      'context_mse':p10.mse(qc,raw[:,:CONTEXT]),'context_clip_fraction':diag['clip_fraction'],
      'selected_rollout_clip_fraction':selclip,'simple_rollout_clip_fraction':c26,
      'interaction_rollout_clip_fraction':c62,'diagnostic_token82_clip_fraction':c82,
    }

def run_rep(s,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time(); state,dm,ds,basis=build_all(s,grid,train_n,code_sizes,iters,max_points); raw=build_eval(s,grid,eval_n)
    splits={k:eval_split(x,state,dm,ds,basis,8 if k.endswith('h8') else 3) for k,x in raw.items()}
    primary={k:float(v['selected_ratio']) for k,v in splits.items()}
    tok82={k:float(v['diagnostic_token82_ratio']) for k,v in splits.items()}
    return {
      'seeds':asdict(s),'family':'two_component_viscous_hamilton_jacobi_2d','frame_dt':HJ_DT,'substeps':HJ_SUBSTEPS,
      'primary_model':'unchanged_phase28_heldout_context_bic_26_vs_62','diagnostic_library':'odd_spatial_quadratic82',
      'context_frames':CONTEXT,'ridge':RIDGE,'representation_training_families':['wave','gray_scott'],
      'hamilton_jacobi_used_in_representation_training':False,
      'primary_ratios':primary,'primary_all_four_below_persistence':bool(all(v<1 for v in primary.values())),
      'diagnostic_token82_ratios':tok82,'diagnostic_token82_all_four_below_persistence':bool(all(v<1 for v in tok82.values())),
      'splits':splits,'runtime_seconds':time.time()-t0,
    }

def validate_invariants():
    rng=np.random.default_rng(29); z=rng.normal(size=(2,2,8,8)).astype(np.float32); b=p22.fourier_basis()
    assert odd_spatial_quadratic82(z,b).shape[-1]==82
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==30 and len(vals)==len(set(vals))
    x,*_=gen_hj(3,7,8,(.015,.035),(.20,.40),(.25,.50),999290001,3)
    assert x.shape==(3,7,2,8,8) and np.isfinite(x).all()
    assert p28.FIT_FRAMES==6 and p28.VALIDATION_PAIRS==((5,6),(6,7))
    return {'diagnostic_feature_dim':82,'registered_seed_count':len(vals),'context_frames':CONTEXT,'ridge':RIDGE,
            'phase28_selector_unchanged':True,'simulator_finite_smoke':True,'odd_modes':list(ODD_MODES)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); inv=validate_invariants()
    if a.smoke:
        s=Seeds(999290101,999290102,999290103,999290104,999290105,999290106,999290111,999290112,999290113,999290114)
        r=run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE29_HAMILTON_JACOBI_REGISTERED_BOUNDARY'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))

if __name__=='__main__': main()
