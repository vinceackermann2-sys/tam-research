from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase25 as p25
from tam_research import physics_tokenizer_phase27 as p27

CONTEXT=8
RIDGE=0.001
FIT_FRAMES=6
VALIDATION_PAIRS=((5,6),(6,7))

@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    wave_id:int; wave_spectral:int; wave_combined:int
    gray_id:int; gray_sharp:int; gray_parameter:int; gray_combined:int
    nls_id:int; nls_spectral:int; nls_parameter:int; nls_combined:int
    adv_id:int; adv_spectral:int; adv_speed:int; adv_combined:int
    burg_id:int; burg_spectral:int; burg_parameter:int; burg_combined:int

REGISTERED=[
    Seeds(828280001,828280002,828280003,828280004,828280005,828280006,828280011,828280012,828280013,828280021,828280022,828280023,828280024,828280031,828280032,828280033,828280034,828280041,828280042,828280043,828280044,828280051,828280052,828280053,828280054),
    Seeds(828280101,828280102,828280103,828280104,828280105,828280106,828280111,828280112,828280113,828280121,828280122,828280123,828280124,828280131,828280132,828280133,828280134,828280141,828280142,828280143,828280144,828280151,828280152,828280153,828280154),
    Seeds(828280201,828280202,828280203,828280204,828280205,828280206,828280211,828280212,828280213,828280221,828280222,828280223,828280224,828280231,828280232,828280233,828280234,828280241,828280242,828280243,828280244,828280251,828280252,828280253,828280254),
]

def build_all(s,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    return p25.build_all(s,grid,train_n,code_sizes,iters,max_points)

def build_eval(s,grid=16,eval_n=48):
    return p27.build_eval(s,grid,eval_n)

def validation_score(qc, maps, feature_fn, dim, basis):
    B=qc.shape[0]; errs=np.zeros(B,np.float64)
    for a,b in VALIDATION_PAIRS:
        pred=p25.predict(qc[:,a],maps,feature_fn,basis)
        d=(pred-qc[:,b]).astype(np.float64)
        errs += np.mean(d*d,axis=(1,2,3))
    mse=errs/len(VALIDATION_PAIRS)
    n=len(VALIDATION_PAIRS)*2*qc.shape[-2]*qc.shape[-1]
    k=2*dim
    score=n*np.log(mse+1e-12)+k*np.log(n)
    return mse,score

def select_models(qc,basis):
    prefix=qc[:,:FIT_FRAMES]
    m26_prefix=p25.fit_maps(prefix,p25.base26_features,26,basis,RIDGE)
    m62_prefix=p25.fit_maps(prefix,p25.interaction62_features,62,basis,RIDGE)
    mse26,s26=validation_score(qc,m26_prefix,p25.base26_features,26,basis)
    mse62,s62=validation_score(qc,m62_prefix,p25.interaction62_features,62,basis)
    choose62=s62 < s26
    return choose62,mse26,mse62,s26,s62

def selected_rollout(start,m26,m62,choose62,h,basis,dm,ds):
    cur=start.copy(); clips=[]
    mask=choose62[:,None,None,None]
    for _ in range(h):
        y26=p25.predict(cur,m26,p25.base26_features,basis)
        y62=p25.predict(cur,m62,p25.interaction62_features,basis)
        y=np.where(mask,y62,y26).astype(np.float32)
        q,d=p10.q_comp_delta(y-cur,dm,ds,True)
        cur=(cur+q).astype(np.float32); clips.append(d['clip_fraction'])
    return cur,float(np.mean(clips)) if clips else 0.0

def eval_split(raw,state,dm,ds,basis,h):
    qc,diag=p10.encode_comp_seq(raw[:,:CONTEXT],state,dm,ds,True)
    target=raw[:,CONTEXT+h-1]; persist=raw[:,CONTEXT-1]; pm=p10.mse(persist,target)
    choose62,mse26,mse62,s26,s62=select_models(qc,basis)
    m26=p25.fit_maps(qc,p25.base26_features,26,basis,RIDGE)
    m62=p25.fit_maps(qc,p25.interaction62_features,62,basis,RIDGE)
    ys,clips=selected_rollout(qc[:,-1],m26,m62,choose62,h,basis,dm,ds)
    y26,c26=p25.token_rollout(qc[:,-1],m26,h,p25.base26_features,basis,dm,ds)
    y62,c62=p25.token_rollout(qc[:,-1],m62,h,p25.interaction62_features,basis,dm,ds)
    return {
        'persistence_mse':pm,
        'selected_mse':p10.mse(ys,target),'selected_ratio':p10.mse(ys,target)/pm,
        'simple26_ratio':p10.mse(y26,target)/pm,'interaction62_ratio':p10.mse(y62,target)/pm,
        'selected_interaction_fraction':float(np.mean(choose62)),
        'validation_mse26_mean':float(np.mean(mse26)),'validation_mse62_mean':float(np.mean(mse62)),
        'validation_score26_mean':float(np.mean(s26)),'validation_score62_mean':float(np.mean(s62)),
        'context_mse':p10.mse(qc,raw[:,:CONTEXT]),'context_clip_fraction':diag['clip_fraction'],
        'selected_rollout_clip_fraction':clips,'simple_rollout_clip_fraction':c26,'interaction_rollout_clip_fraction':c62,
    }

def family_name(k): return p27.family_name(k)

def run_rep(s,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time(); state,dm,ds,basis=build_all(s,grid,train_n,code_sizes,iters,max_points); raw=build_eval(s,grid,eval_n)
    splits={k:eval_split(x,state,dm,ds,basis,8 if k.endswith('h8') else 3) for k,x in raw.items()}
    primary={k:float(v['selected_ratio']) for k,v in splits.items()}
    fam={}
    for f in ('wave','gray_scott','nls','advection','burgers'):
        ks=[k for k in primary if family_name(k)==f]; vals=[primary[k] for k in ks]
        fam[f]={'passing_cells':int(sum(v<1 for v in vals)),'total_cells':len(vals),'all_below_persistence':bool(all(v<1 for v in vals)),
                'worst_ratio':float(max(vals)),'mean_interaction_selection_fraction':float(np.mean([splits[k]['selected_interaction_fraction'] for k in ks]))}
    return {'seeds':asdict(s),'selector':'heldout_context_bic_nested_26_vs_62','context_frames':CONTEXT,'fit_frames':FIT_FRAMES,
            'validation_pairs':[list(x) for x in VALIDATION_PAIRS],'ridge':RIDGE,'representation_training_families':['wave','gray_scott'],
            'primary_ratios':primary,'primary_passing_cells':int(sum(v<1 for v in primary.values())),'primary_total_cells':len(primary),
            'primary_all_nineteen_below_persistence':bool(all(v<1 for v in primary.values())),'per_family':fam,'splits':splits,'runtime_seconds':time.time()-t0}

def validate_invariants():
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==75 and len(vals)==len(set(vals))
    assert FIT_FRAMES==6 and VALIDATION_PAIRS==((5,6),(6,7))
    n=2*2*8*8; assert 2*62>2*26 and n>0
    z=np.zeros((2,8,2,8,8),np.float32); b=p25.p22.fourier_basis(); c,*_=select_models(z,b); assert c.shape==(2,)
    return {'registered_seed_count':len(vals),'context_frames':CONTEXT,'prefix_fit_frames':FIT_FRAMES,'validation_pairs':[list(x) for x in VALIDATION_PAIRS],
            'simple_params':52,'interaction_params':124,'tie_break':'simple','family_label_used':False}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); inv=validate_invariants()
    if a.smoke:
        s=Seeds(999280001,999280002,999280003,999280004,999280005,999280006,999280011,999280012,999280013,999280021,999280022,999280023,999280024,999280031,999280032,999280033,999280034,999280041,999280042,999280043,999280044,999280051,999280052,999280053,999280054)
        r=run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE28_CONTEXT_MDL_SELECTOR_REGISTERED_CONFIRMATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
