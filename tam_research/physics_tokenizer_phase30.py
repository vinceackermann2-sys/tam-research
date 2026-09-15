from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase25 as p25
from tam_research import physics_tokenizer_phase27 as p27
from tam_research import physics_tokenizer_phase28 as p28
from tam_research import physics_tokenizer_phase29 as p29

CONTEXT=8
RIDGE=0.001
MEDIAN_GAIN_MIN=0.10
CLEAN_PREFIX_TRANSITIONS=(1,2,3,4)
CLEAN_ALL_TRANSITIONS=(1,2,3,4,5,6)
HELDOUT_TRANSITIONS=(5,6)

@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    wave_id:int; wave_spectral:int; wave_combined:int
    gray_id:int; gray_sharp:int; gray_parameter:int; gray_combined:int
    nls_id:int; nls_spectral:int; nls_parameter:int; nls_combined:int
    adv_id:int; adv_spectral:int; adv_speed:int; adv_combined:int
    burg_id:int; burg_spectral:int; burg_parameter:int; burg_combined:int
    hj_id:int; hj_spectral:int; hj_parameter:int; hj_combined:int

def _seedset(base:int)->Seeds:
    return Seeds(base+1,base+2,base+3,base+4,base+5,base+6,
                 base+11,base+12,base+13,
                 base+21,base+22,base+23,base+24,
                 base+31,base+32,base+33,base+34,
                 base+41,base+42,base+43,base+44,
                 base+51,base+52,base+53,base+54,
                 base+61,base+62,base+63,base+64)
REGISTERED=[_seedset(830300000),_seedset(830300100),_seedset(830300200)]

def build_all(s,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    return p25.build_all(s,grid,train_n,code_sizes,iters,max_points)

def build_eval(s,grid=16,eval_n=48):
    d=p27.build_eval(s,grid,eval_n)
    d.update({
      'hj_id_h8':p29.gen_hj(eval_n,16,grid,(.015,.035),(.20,.40),(.25,.50),s.hj_id,3)[0],
      'hj_spectral_ood_h3':p29.gen_hj(eval_n,11,grid,(.015,.035),(.20,.40),(.25,.50),s.hj_spectral,6)[0],
      'hj_parameter_ood_h3':p29.gen_hj(eval_n,11,grid,(.05,.08),(.50,.80),(.25,.50),s.hj_parameter,3)[0],
      'hj_combined_ood_h3':p29.gen_hj(eval_n,11,grid,(.05,.08),(.50,.80),(.45,.70),s.hj_combined,6)[0],
    })
    return d

def fit_transition_subset(ctx,feature_fn,dim,basis,transitions,ridge=RIDGE):
    B=ctx.shape[0]; maps=np.empty((B,2,dim),np.float32)
    for b in range(B):
        X=[];Y=[]
        for t in transitions:
            X.append(feature_fn(ctx[b:b+1,t],basis)[0].reshape(-1,dim))
            Y.append(ctx[b,t+1].transpose(1,2,0).reshape(-1,2))
        X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y).astype(np.float64)
        sc=np.sqrt(np.mean(X*X,axis=0)+1e-12); sc[0]=1.; Xs=X/sc
        W=np.linalg.solve(Xs.T@Xs+ridge*np.eye(dim),Xs.T@Y).T
        maps[b]=(W/sc[None,:]).astype(np.float32)
    return maps

def shared_support_gain(qc,basis):
    m62=fit_transition_subset(qc,p25.interaction62_features,62,basis,CLEAN_PREFIX_TRANSITIONS)
    m82=fit_transition_subset(qc,p29.odd_spatial_quadratic82,82,basis,CLEAN_PREFIX_TRANSITIONS)
    e62=np.zeros(len(qc),np.float64); e82=np.zeros(len(qc),np.float64)
    for t in HELDOUT_TRANSITIONS:
        tgt=qc[:,t+1]
        a=p25.predict(qc[:,t],m62,p25.interaction62_features,basis)
        b=p25.predict(qc[:,t],m82,p29.odd_spatial_quadratic82,basis)
        e62 += np.mean((a-tgt).astype(np.float64)**2,axis=(1,2,3))
        e82 += np.mean((b-tgt).astype(np.float64)**2,axis=(1,2,3))
    gain=1.0-e82/(e62+1e-30)
    return float(np.median(gain)),gain

def rollout82(start,maps,h,basis,dm,ds):
    cur=start.copy(); clips=[]
    for _ in range(h):
        y=p25.predict(cur,maps,p29.odd_spatial_quadratic82,basis)
        q,d=p10.q_comp_delta(y-cur,dm,ds,True); cur=(cur+q).astype(np.float32); clips.append(d['clip_fraction'])
    return cur,float(np.mean(clips)) if clips else 0.0

def eval_split(raw,state,dm,ds,basis,h):
    qc,diag=p10.encode_comp_seq(raw[:,:CONTEXT],state,dm,ds,True)
    target=raw[:,CONTEXT+h-1]; persist=raw[:,CONTEXT-1]; pm=p10.mse(persist,target)
    med,gains=shared_support_gain(qc,basis); use82=bool(med>MEDIAN_GAIN_MIN)
    if use82:
        m82=fit_transition_subset(qc,p29.odd_spatial_quadratic82,82,basis,CLEAN_ALL_TRANSITIONS)
        pred,clips=rollout82(qc[:,-1],m82,h,basis,dm,ds)
        selected_fraction=1.0
    else:
        choose62,*_=p28.select_models(qc,basis)
        m26=p25.fit_maps(qc,p25.base26_features,26,basis,RIDGE)
        m62=p25.fit_maps(qc,p25.interaction62_features,62,basis,RIDGE)
        pred,clips=p28.selected_rollout(qc[:,-1],m26,m62,choose62,h,basis,dm,ds)
        selected_fraction=float(np.mean(choose62))
    ratio=p10.mse(pred,target)/pm
    return {'persistence_mse':pm,'selected_mse':p10.mse(pred,target),'selected_ratio':ratio,
            'use82_group':use82,'median_heldout_gain82_vs62':med,'gain_positive_fraction':float(np.mean(gains>0)),
            'parent_interaction_selection_fraction':selected_fraction if not use82 else None,
            'context_mse':p10.mse(qc,raw[:,:CONTEXT]),'context_clip_fraction':diag['clip_fraction'],
            'rollout_clip_fraction':clips}

def family_name(k):
    if k.startswith('hj_'): return 'hamilton_jacobi'
    return p27.family_name(k)

def run_rep(s,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time(); state,dm,ds,basis=build_all(s,grid,train_n,code_sizes,iters,max_points); raw=build_eval(s,grid,eval_n)
    splits={k:eval_split(v,state,dm,ds,basis,8 if k.endswith('h8') else 3) for k,v in raw.items()}
    primary={k:float(v['selected_ratio']) for k,v in splits.items()}
    fam={}
    for f in ('wave','gray_scott','nls','advection','burgers','hamilton_jacobi'):
        ks=[k for k in primary if family_name(k)==f]; vals=[primary[k] for k in ks]
        fam[f]={'passing_cells':int(sum(x<1 for x in vals)),'total_cells':len(vals),'all_below_persistence':bool(all(x<1 for x in vals)),
                'worst_ratio':float(max(vals)),'use82_splits':int(sum(bool(splits[k]['use82_group']) for k in ks))}
    return {'seeds':asdict(s),'selector':'phase28_parent_plus_shared_support_82','shared_support_median_gain_min':MEDIAN_GAIN_MIN,
            'clean_prefix_transitions':list(CLEAN_PREFIX_TRANSITIONS),'heldout_transitions':list(HELDOUT_TRANSITIONS),'clean_all_transitions':list(CLEAN_ALL_TRANSITIONS),
            'context_frames':CONTEXT,'ridge':RIDGE,'representation_training_families':['wave','gray_scott'],
            'nls_advection_burgers_hamilton_jacobi_excluded_from_representation_training':True,
            'primary_ratios':primary,'primary_passing_cells':int(sum(x<1 for x in primary.values())),'primary_total_cells':len(primary),
            'primary_all_23_below_persistence':bool(all(x<1 for x in primary.values())),'per_family':fam,'splits':splits,'runtime_seconds':time.time()-t0}

def validate_invariants():
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==87 and len(vals)==len(set(vals))
    assert MEDIAN_GAIN_MIN==.10 and CLEAN_PREFIX_TRANSITIONS==(1,2,3,4) and CLEAN_ALL_TRANSITIONS==(1,2,3,4,5,6) and HELDOUT_TRANSITIONS==(5,6)
    z=np.zeros((2,2,8,8),np.float32); b=p25.p22.fourier_basis(); assert p29.odd_spatial_quadratic82(z,b).shape[-1]==82
    assert p28.FIT_FRAMES==6 and p28.VALIDATION_PAIRS==((5,6),(6,7))
    return {'registered_seed_count':len(vals),'primary_cells_per_replicate':23,'context_frames':CONTEXT,'ridge':RIDGE,'median_gain_min':MEDIAN_GAIN_MIN,
            'parent_phase28_unchanged':True,'reset_transition_excluded_only_for_82':True,'family_label_used':False}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); inv=validate_invariants()
    if a.smoke:
        s=_seedset(999300000); r=run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE30_SHARED_SUPPORT_REGISTERED_CONFIRMATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
