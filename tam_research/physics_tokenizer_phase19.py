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
        for dx in (-1,0,1): cols.append(np.roll(np.roll(z,-dy,-2),-dx,-1))
    return np.stack(cols,-1).astype(np.float32)


def fit_basis(joint,k=K):
    means=[]; comps=[]; scores=[]
    ntraj,nt=joint.shape[:2]
    deltas=joint[:,1:]-joint[:,:-1]
    for ch in range(2):
        Ps=[]; Ts=[]
        for t in range(nt-1):
            P=patch9(joint[:,t],ch).reshape(-1,9).astype(np.float64)
            d=deltas[:,t]
            T=np.concatenate([patch9(d,0).reshape(-1,9),patch9(d,1).reshape(-1,9)],axis=1).astype(np.float64)
            Ps.append(P); Ts.append(T)
        P=np.concatenate(Ps,0); T=np.concatenate(Ts,0)
        mu=P.mean(0); X=P-mu; Y=T-T.mean(0)
        n=max(1,len(X)-1); Cpp=(X.T@X)/n; Cpt=(X.T@Y)/n
        reg=1e-6*float(np.trace(Cpp))/9.0 + 1e-12
        vals,U=np.linalg.eigh(Cpp + reg*np.eye(9)); vals=np.maximum(vals,1e-18)
        invsqrt=(U*(1.0/np.sqrt(vals))[None,:])@U.T
        A=invsqrt@(Cpt@Cpt.T)@invsqrt
        ev,Q=np.linalg.eigh((A+A.T)*0.5); order=np.argsort(ev)[::-1]; ev=ev[order]; Q=Q[:,order]
        V=invsqrt@Q[:,:k]
        W=V.T.copy()
        Creg=Cpp+reg*np.eye(9)
        for j in range(k):
            den=float(np.sqrt(max(W[j]@Creg@W[j],1e-30))); W[j]/=den
            idx=int(np.argmax(np.abs(W[j])))
            if W[j,idx] < 0: W[j]*=-1
        means.append(mu.astype(np.float32)); comps.append(W.astype(np.float32)); scores.append([float(x) for x in ev[:k]])
    return {'means':np.stack(means),'components':np.stack(comps),'predictive_scores':scores}


def learned_features(s,basis):
    cols=[np.ones_like(s[:,0])]
    for ch in range(2):
        P=patch9(s,ch)
        Z=P-basis['means'][ch][None,None,None,:]
        Y=np.einsum('bhwp,kp->bhwk',Z,basis['components'][ch],optimize=True)
        cols.extend([Y[...,j] for j in range(Y.shape[-1])])
    u,v=s[:,0],s[:,1]; cols += [u*u,u*v,v*v,u*u*u,u*u*v,u*v*v,v*v*v]
    return np.stack(cols,-1).astype(np.float32)


def fit_maps(ctx,basis,ridge=RIDGE):
    dim=18; B=ctx.shape[0]; maps=np.empty((B,2,dim),np.float32)
    for b in range(B):
        X=[];Y=[]
        for t in range(ctx.shape[1]-1):
            X.append(learned_features(ctx[b:b+1,t],basis)[0].reshape(-1,dim)); Y.append(ctx[b,t+1].transpose(1,2,0).reshape(-1,2))
        X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y).astype(np.float64); sc=np.sqrt(np.mean(X*X,axis=0)+1e-12); sc[0]=1.; Xs=X/sc; W=np.linalg.solve(Xs.T@Xs+ridge*np.eye(dim),Xs.T@Y).T; maps[b]=(W/sc[None,:]).astype(np.float32)
    return maps

def predict(s,maps,basis): return np.einsum('bhwf,bcf->bchw',learned_features(s,basis),maps,optimize=True).astype(np.float32)

def eval_companded(raw,state,dm,ds,basis,h):
    qC,diag=p10.encode_comp_seq(raw[:,:CONTEXT],state,dm,ds,True); maps=fit_maps(qC,basis); cur=qC[:,-1].copy(); clips=[]
    for _ in range(h):
        y=predict(cur,maps,basis); q,d=p10.q_comp_delta(y-cur,dm,ds,True); cur=(cur+q).astype(np.float32); clips.append(d['clip_fraction'])
    target=raw[:,CONTEXT+h-1]; persist=raw[:,CONTEXT-1]; pm=p10.mse(persist,target)
    raw18=fit_maps(raw[:,:CONTEXT],basis); r18=raw[:,CONTEXT-1].copy()
    for _ in range(h): r18=predict(r18,raw18,basis)
    raw26=p17.fit_maps(raw[:,:CONTEXT]); r26=raw[:,CONTEXT-1].copy()
    for _ in range(h): r26=p17.predict(r26,raw26)
    return {'persistence_mse':pm,'companded_mse':p10.mse(cur,target),'companded_ratio':p10.mse(cur,target)/pm,'raw_learned18_rollout_ratio':p10.mse(r18,target)/pm if np.isfinite(r18).all() else None,'raw_full_stencil26_rollout_ratio':p10.mse(r26,target)/pm if np.isfinite(r26).all() else None,'context_mse':p10.mse(qC,raw[:,:CONTEXT]),'context_clip_fraction':diag['clip_fraction'],'rollout_clip_fraction':float(np.mean(clips)) if clips else 0.}

@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    wave_id:int; wave_spectral:int; wave_combined:int
    gray_id:int; gray_sharp:int; gray_parameter:int; gray_combined:int
    nls_id:int; nls_spectral:int; nls_parameter:int; nls_combined:int
    adv_id:int; adv_spectral:int; adv_speed:int; adv_combined:int
REGISTERED=[
Seeds(819190001,819190002,819190003,819190004,819190005,819190006,819190011,819190012,819190013,819190021,819190022,819190023,819190024,819190031,819190032,819190033,819190034,819190041,819190042,819190043,819190044),
Seeds(819190101,819190102,819190103,819190104,819190105,819190106,819190111,819190112,819190113,819190121,819190122,819190123,819190124,819190131,819190132,819190133,819190134,819190141,819190142,819190143,819190144),
Seeds(819190201,819190202,819190203,819190204,819190205,819190206,819190211,819190212,819190213,819190221,819190222,819190223,819190224,819190231,819190232,819190233,819190234,819190241,819190242,819190243,819190244)]

def build_all(s,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    state,dm,ds=p16.build_codecs(s,grid,train_n,code_sizes,iters,max_points)
    w=p10.gen_wave(train_n,8,grid,(.6,1.),s.wave_train,3)[0]; g=p16.gen_gray(train_n,8,grid,(.025,.045),(.055,.065),s.gray_train,False)[0]; joint=np.concatenate([w,g],0); basis=fit_basis(joint,K)
    return state,dm,ds,basis

def run_rep(s,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time();state,dm,ds,basis=build_all(s,grid,train_n,code_sizes,iters,max_points)
    raw={'wave_id_h8':p10.gen_wave(eval_n,16,grid,(.6,1.),s.wave_id,3)[0],'wave_spectral_ood_h3':p10.gen_wave(eval_n,11,grid,(.6,1.),s.wave_spectral,6)[0],'wave_combined_ood_h3':p10.gen_wave(eval_n,11,grid,(1.15,1.35),s.wave_combined,6)[0],'gray_id_h8':p16.gen_gray(eval_n,16,grid,(.025,.045),(.055,.065),s.gray_id,False)[0],'gray_sharp_ood_h3':p16.gen_gray(eval_n,11,grid,(.025,.045),(.055,.065),s.gray_sharp,True)[0],'gray_parameter_ood_h3':p16.gen_gray(eval_n,11,grid,(.050,.060),(.045,.055),s.gray_parameter,False)[0],'gray_combined_ood_h3':p16.gen_gray(eval_n,11,grid,(.050,.060),(.045,.055),s.gray_combined,True)[0],'nls_id_h8':p16.gen_nls(eval_n,16,grid,(.04,.08),(.10,.25),s.nls_id,3)[0],'nls_spectral_ood_h3':p16.gen_nls(eval_n,11,grid,(.04,.08),(.10,.25),s.nls_spectral,6)[0],'nls_parameter_ood_h3':p16.gen_nls(eval_n,11,grid,(.10,.14),(.30,.45),s.nls_parameter,3)[0],'nls_combined_ood_h3':p16.gen_nls(eval_n,11,grid,(.10,.14),(.30,.45),s.nls_combined,6)[0],'adv_id_h8':p16.gen_adv(eval_n,16,grid,(.18,.30),(-.20,.20),s.adv_id,3)[0],'adv_spectral_ood_h3':p16.gen_adv(eval_n,11,grid,(.18,.30),(-.20,.20),s.adv_spectral,6)[0],'adv_speed_ood_h3':p16.gen_adv(eval_n,11,grid,(.45,.60),(-.20,.20),s.adv_speed,3)[0],'adv_combined_ood_h3':p16.gen_adv(eval_n,11,grid,(.45,.60),(1.20,1.50),s.adv_combined,6)[0]}
    splits={k:eval_companded(x,state,dm,ds,basis,8 if k.endswith('h8') else 3) for k,x in raw.items()}; primary={k:float(v['companded_ratio']) for k,v in splits.items()}
    return {'seeds':asdict(s),'feature_library':'predictive_5x2_patch_components_plus_center_polynomial_v1','feature_dim':18,'basis_components_per_channel':K,'basis_predictive_scores':basis['predictive_scores'],'context_frames':CONTEXT,'ridge':RIDGE,'representation_training_families':['wave','gray_scott'],'basis_training_families':['wave','gray_scott'],'nls_used_in_representation_or_basis_training':False,'advection_used_in_representation_or_basis_training':False,'primary_companded_ratios':primary,'all_fifteen_below_persistence':bool(all(v<1 for v in primary.values())),'splits':splits,'runtime_seconds':time.time()-t0}

def validate_invariants():
    rng=np.random.default_rng(818189001); joint=np.empty((4,3,2,8,8),np.float32); joint[:,:,0]=rng.normal(size=(4,3,8,8)); joint[:,:,1]=rng.normal(size=(4,3,8,8)); b1=fit_basis(joint); b2=fit_basis(joint); np.testing.assert_array_equal(b1['means'],b2['means']); np.testing.assert_array_equal(b1['components'],b2['components']); assert learned_features(joint[:,0],b1).shape[-1]==18; vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals)); return {'feature_dim':18,'components_per_channel':K,'context_frames':CONTEXT,'ridge':RIDGE,'basis_deterministic':True,'registered_seed_count':len(vals)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--rep',type=int,default=0);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();inv=validate_invariants()
    if a.smoke:
        s=Seeds(999190001,999190002,999190003,999190004,999190005,999190006,999190011,999190012,999190013,999190021,999190022,999190023,999190024,999190031,999190032,999190033,999190034,999190041,999190042,999190043,999190044);r=run_rep(s,grid=8,train_n=12,eval_n=5,code_sizes=(32,8),iters=3,max_points=3000);r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]);r['replicate']=a.rep;r['status']='PHASE19_PREDICTIVE_STENCIL_BASIS_REGISTERED_CONFIRMATION'
    r['invariants']=inv;Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
