from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

# Synthetic systems and Phase-9 matched controls.
def lap(x):
    return np.roll(x,1,-1)+np.roll(x,-1,-1)+np.roll(x,1,-2)+np.roll(x,-1,-2)-4*x

def wave_step(u,v,c,dt=.12):
    a=(c*c)*lap(u); vh=v+.5*dt*a; u2=u+dt*vh; a2=(c*c)*lap(u2); v2=vh+.5*dt*a2
    return u2.astype(np.float32),v2.astype(np.float32)

def smooth_field(rng,n,max_mode=3):
    yy,xx=np.meshgrid(np.arange(n),np.arange(n),indexing='ij'); f=np.zeros((n,n),np.float32)
    for _ in range(int(rng.integers(2,6))):
        kx=int(rng.integers(0,max_mode+1)); ky=int(rng.integers(0,max_mode+1))
        if kx==0 and ky==0: kx=1
        amp=float(rng.normal())/(1+kx*kx+ky*ky); ph=float(rng.uniform(0,2*np.pi))
        f += amp*np.cos(2*np.pi*(kx*xx+ky*yy)/n+ph)
    f-=f.mean(); f/=f.std()+1e-6
    return f.astype(np.float32)

def gen_wave(num,seq_len,grid,c_range,seed,max_mode=3,dt=.12):
    rng=np.random.default_rng(seed); xs=np.empty((num,seq_len,2,grid,grid),np.float32); cs=np.empty(num,np.float32)
    for i in range(num):
        c=float(rng.uniform(*c_range)); cs[i]=c
        u=smooth_field(rng,grid,max_mode)*float(rng.uniform(.5,1.)); v=smooth_field(rng,grid,max_mode)*float(rng.uniform(.05,.25))
        for t in range(seq_len): xs[i,t,0]=u; xs[i,t,1]=v; u,v=wave_step(u,v,c,dt)
    return xs,cs

def gray_step(u,v,F,k,du=.16,dv=.08,dt=1.):
    uvv=u*v*v
    return (u+(du*lap(u)-uvv+F*(1-u))*dt).astype(np.float32),(v+(dv*lap(v)+uvv-(F+k)*v)*dt).astype(np.float32)

def gen_gray(num,seq_len,grid,F_range,k_range,seed,substeps=4):
    rng=np.random.default_rng(seed); out=np.empty((num,seq_len,2,grid,grid),np.float32)
    Fs=rng.uniform(*F_range,size=num).astype(np.float32); ks=rng.uniform(*k_range,size=num).astype(np.float32)
    yy,xx=np.meshgrid(np.arange(grid),np.arange(grid),indexing='ij')
    for i in range(num):
        u=np.ones((grid,grid),np.float32); v=np.zeros((grid,grid),np.float32)
        for _ in range(int(rng.integers(1,4))):
            cx,cy=float(rng.uniform(0,grid)),float(rng.uniform(0,grid)); r=float(rng.uniform(1.5,3.0))
            dx=np.minimum(np.abs(xx-cx),grid-np.abs(xx-cx)); dy=np.minimum(np.abs(yy-cy),grid-np.abs(yy-cy)); blob=np.exp(-(dx*dx+dy*dy)/(2*r*r)).astype(np.float32)
            u-=float(rng.uniform(.25,.5))*blob; v+=float(rng.uniform(.18,.35))*blob
        u+=rng.normal(0,.01,(grid,grid)).astype(np.float32); v+=rng.normal(0,.005,(grid,grid)).astype(np.float32)
        for t in range(seq_len):
            out[i,t,0]=u; out[i,t,1]=v
            for _ in range(substeps): u,v=gray_step(u,v,Fs[i],ks[i])
    return out

def stats(x):
    axes=tuple(i for i in range(x.ndim) if i != x.ndim-3)
    return x.mean(axis=axes).astype(np.float32),(x.std(axis=axes)+1e-6).astype(np.float32)

def norm5(x,m,s):
    shape=(1,)*(x.ndim-3)+(2,1,1); return ((x-m.reshape(shape))/s.reshape(shape)).astype(np.float32)

def flat2(x):
    return x.transpose(*range(x.ndim-3),x.ndim-2,x.ndim-1,x.ndim-3).reshape(-1,2).astype(np.float32)

def fit_kmeans(x_np,k,seed,iters=20,max_points=60000,chunk=4096):
    rng=np.random.default_rng(seed)
    if len(x_np)>max_points: x_np=x_np[rng.choice(len(x_np),max_points,replace=False)]
    x=np.asarray(x_np,np.float32); g=torch.Generator().manual_seed(seed)
    if len(x)<k: raise ValueError(f'need >=k points ({len(x)}<{k})')
    c=x[torch.randperm(len(x),generator=g)[:k].numpy()].copy()
    for _ in range(iters):
        idx=cKDTree(c).query(x,k=1,workers=-1)[1].astype(np.int64)
        counts=np.bincount(idx,minlength=k).astype(np.float32)
        sums=np.zeros((k,2),np.float64)
        np.add.at(sums,idx,x)
        nz=counts>0; c[nz]=(sums[nz]/counts[nz,None]).astype(np.float32)
        if (~nz).any():
            repl=torch.randint(0,len(x),(int((~nz).sum()),),generator=g).numpy(); c[~nz]=x[repl]
    return c.astype(np.float32)

def nearest(flat,cb,chunk=8192):
    x=np.asarray(flat,np.float32); c=np.asarray(cb,np.float32)
    return cKDTree(c).query(x,k=1,workers=-1)[1].astype(np.int64)

def fit_rvq(points,k1,k2,s1,s2,iters=20,max_points=60000):
    c1=fit_kmeans(points,k1,s1,iters,max_points); i1=nearest(points,c1); r=points-c1[i1]; c2=fit_kmeans(r,k2,s2,iters,max_points); return c1,c2

def rvq_flat(flat,c1,c2):
    i1=nearest(flat,c1); r=flat-c1[i1]; i2=nearest(r,c2); return c1[i1]+c2[i2]

def rvq_array(x,c1,c2):
    shp=x.shape; q=rvq_flat(flat2(x),c1,c2); lead=shp[:-3]; h,w=shp[-2:]
    return q.reshape(*lead,h,w,2).transpose(*range(len(lead)),len(lead)+2,len(lead),len(lead)+1).astype(np.float32)

def q_state(x,m,s,c1,c2):
    z=((x-m[None,:,None,None])/s[None,:,None,None]).astype(np.float32); q=rvq_array(z,c1,c2)
    return (q*s[None,:,None,None]+m[None,:,None,None]).astype(np.float32)

def q_plain_delta(d,m,s,c1,c2):
    z=((d-m[None,:,None,None])/s[None,:,None,None]).astype(np.float32); q=rvq_array(z,c1,c2)
    return (q*s[None,:,None,None]+m[None,:,None,None]).astype(np.float32)

# Frozen Phase-10 compander: two 9-bit channel indices -> 18-bit packed token.
LEVELS=512; ZMAX=16.0; YMAX=float(np.arcsinh(ZMAX))
def companded_indices(d,m,s):
    z=(d-m[None,:,None,None])/s[None,:,None,None]
    y=np.arcsinh(z); yc=np.clip(y,-YMAX,YMAX)
    idx=np.rint((yc+YMAX)*(LEVELS-1)/(2*YMAX)).astype(np.int32)
    return np.clip(idx,0,LEVELS-1), z

def decode_companded_indices(idx,m,s):
    y=-YMAX + idx.astype(np.float32)*(2*YMAX/(LEVELS-1)); z=np.sinh(y)
    return (m[None,:,None,None]+s[None,:,None,None]*z).astype(np.float32)

def pack_indices(idx):
    return (idx[:,0].astype(np.int32)*LEVELS + idx[:,1].astype(np.int32)).astype(np.int32)

def unpack_tokens(tok):
    return np.stack([tok//LEVELS,tok%LEVELS],1).astype(np.int32)

def q_comp_delta(d,m,s,return_diag=False):
    idx,z=companded_indices(d,m,s); packed=pack_indices(idx); out=decode_companded_indices(unpack_tokens(packed),m,s)
    if return_diag:
        return out, {'clip_fraction':float(np.mean(np.abs(z)>ZMAX)),'token_min':int(packed.min()),'token_max':int(packed.max())}
    return out

def encode_state_seq(seq,codec):
    m,s,c1,c2=codec; return np.stack([q_state(seq[:,j],m,s,c1,c2) for j in range(seq.shape[1])],1)

def encode_plain_seq(seq,state,plain):
    sm,ss,sc1,sc2=state; dm,ds,dc1,dc2=plain; out=[q_state(seq[:,0],sm,ss,sc1,sc2)]
    for j in range(1,seq.shape[1]):
        p=out[-1]; out.append((p+q_plain_delta(seq[:,j]-p,dm,ds,dc1,dc2)).astype(np.float32))
    return np.stack(out,1)

def encode_comp_seq(seq,state,dm,ds,diagnostics=False):
    sm,ss,sc1,sc2=state; out=[q_state(seq[:,0],sm,ss,sc1,sc2)]; clips=[]; tmins=[]; tmaxs=[]
    for j in range(1,seq.shape[1]):
        p=out[-1]; q,diag=q_comp_delta(seq[:,j]-p,dm,ds,True); out.append((p+q).astype(np.float32)); clips.append(diag['clip_fraction']); tmins.append(diag['token_min']); tmaxs.append(diag['token_max'])
    ans=np.stack(out,1)
    if diagnostics: return ans, {'clip_fraction':float(np.mean(clips)) if clips else 0.,'token_min':min(tmins) if tmins else 0,'token_max':max(tmaxs) if tmaxs else 0}
    return ans

def features(s):
    cols=[np.ones_like(s[:,0])]
    for ch in range(2):
        z=s[:,ch]; cols += [z,
            np.roll(z,1,-1)+np.roll(z,-1,-1)+np.roll(z,1,-2)+np.roll(z,-1,-2),
            np.roll(np.roll(z,1,-2),1,-1)+np.roll(np.roll(z,1,-2),-1,-1)+np.roll(np.roll(z,-1,-2),1,-1)+np.roll(np.roll(z,-1,-2),-1,-1),
            np.roll(z,2,-1)+np.roll(z,-2,-1)+np.roll(z,2,-2)+np.roll(z,-2,-2)]
    u,v=s[:,0],s[:,1]; cols += [u*u,u*v,v*v,u*u*u,u*u*v,u*v*v,v*v*v]
    return np.stack(cols,-1).astype(np.float32)

def fit_maps(ctx,ridge=.001):
    B=ctx.shape[0]; maps=np.empty((B,2,16),np.float32)
    for b in range(B):
        X=[]; Y=[]
        for t in range(2): X.append(features(ctx[b:b+1,t])[0].reshape(-1,16)); Y.append(ctx[b,t+1].transpose(1,2,0).reshape(-1,2))
        X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y).astype(np.float64); sc=np.sqrt(np.mean(X*X,axis=0)+1e-12); sc[0]=1.; Xs=X/sc
        W=np.linalg.solve(Xs.T@Xs+ridge*np.eye(16),Xs.T@Y).T; maps[b]=(W/sc[None,:]).astype(np.float32)
    return maps

def predict(s,maps): return np.einsum('bhwf,bcf->bchw',features(s),maps,optimize=True).astype(np.float32)
def mse(a,b): return float(np.mean((a-b)**2))
def rollout(ctx,maps,h,project):
    cur=ctx[:,-1].copy(); diags=[]
    for _ in range(h):
        cur,d=project(predict(cur,maps),cur); diags.append(d)
    return cur,diags

def eval_split(raw,state,plain,dm,ds,h,ridge=.001):
    sm,ss,sc1,sc2=state; pdm,pds,pc1,pc2=plain
    qS=encode_state_seq(raw[:,:3],state); qP=encode_plain_seq(raw[:,:3],state,plain); qC,cctxdiag=encode_comp_seq(raw[:,:3],state,dm,ds,True)
    mS,mP,mC=fit_maps(qS,ridge),fit_maps(qP,ridge),fit_maps(qC,ridge)
    def pS(y,prev): return q_state(y,sm,ss,sc1,sc2),{}
    def pP(y,prev): return (prev+q_plain_delta(y-prev,pdm,pds,pc1,pc2)).astype(np.float32),{}
    def pC(y,prev):
        q,d=q_comp_delta(y-prev,dm,ds,True); return (prev+q).astype(np.float32),d
    yS,_=rollout(qS,mS,h,pS); yP,_=rollout(qP,mP,h,pP); yC,croll=rollout(qC,mC,h,pC)
    target=raw[:,3+h-1]; persist=raw[:,2]; pm=mse(persist,target)
    mr=fit_maps(raw[:,:3],ridge); crS,_=rollout(qS,mr,h,pS); crP,_=rollout(qP,mr,h,pP); crC,_=rollout(qC,mr,h,pC)
    s_or=q_state(target,sm,ss,sc1,sc2)
    pp=qP[:,-1].copy(); cc=qC[:,-1].copy(); oracle_clips=[]
    for j in range(h):
        truth=raw[:,3+j]; pp=(pp+q_plain_delta(truth-pp,pdm,pds,pc1,pc2)).astype(np.float32); q,d=q_comp_delta(truth-cc,dm,ds,True); cc=(cc+q).astype(np.float32); oracle_clips.append(d['clip_fraction'])
    rawctx=raw[:,:3]; dr=rawctx[:,1:]-rawctx[:,:-1]
    def delta_err(q): return mse(q[:,1:]-q[:,:-1],dr)
    return {
      'persistence_mse':pm,
      'state_ratio':mse(yS,target)/pm,'plain_innovation_ratio':mse(yP,target)/pm,'companded_ratio':mse(yC,target)/pm,
      'companded_over_plain':mse(yC,target)/mse(yP,target),'companded_over_state':mse(yC,target)/mse(yS,target),
      'state_mse':mse(yS,target),'plain_innovation_mse':mse(yP,target),'companded_mse':mse(yC,target),
      'raw_fit_state_projection_ratio':mse(crS,target)/pm,'raw_fit_plain_projection_ratio':mse(crP,target)/pm,'raw_fit_companded_projection_ratio':mse(crC,target)/pm,
      'state_target_oracle_ratio':mse(s_or,target)/pm,'plain_target_oracle_ratio':mse(pp,target)/pm,'companded_target_oracle_ratio':mse(cc,target)/pm,
      'state_context_mse':mse(qS,rawctx),'plain_context_mse':mse(qP,rawctx),'companded_context_mse':mse(qC,rawctx),
      'state_context_delta_mse':delta_err(qS),'plain_context_delta_mse':delta_err(qP),'companded_context_delta_mse':delta_err(qC),
      'companded_context_clip_fraction':cctxdiag['clip_fraction'],
      'companded_rollout_clip_fraction':float(np.mean([d.get('clip_fraction',0.) for d in croll])) if croll else 0.,
      'companded_target_oracle_clip_fraction':float(np.mean(oracle_clips)) if oracle_clips else 0.,
      'companded_context_token_min':cctxdiag['token_min'],'companded_context_token_max':cctxdiag['token_max']}

@dataclass
class Seeds:
    wave:int; gray:int; sc1:int; sc2:int; ic1:int; ic2:int; combined:int; id:int; spectral:int

REGISTERED=[
Seeds(2026091401,2026091402,2026091403,2026091404,2026091405,2026091406,2026091407,2026091408,2026091409),
Seeds(2026091411,2026091412,2026091413,2026091414,2026091415,2026091416,2026091417,2026091418,2026091419),
Seeds(2026091421,2026091422,2026091423,2026091424,2026091425,2026091426,2026091427,2026091428,2026091429)]

def run_rep(seeds,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time(); k1,k2=code_sizes
    w,_=gen_wave(train_n,8,grid,(.6,1.),seeds.wave,3); g=gen_gray(train_n,8,grid,(.025,.045),(.055,.065),seeds.gray); joint=np.concatenate([w,g],0)
    sm,ss=stats(joint); sc1,sc2=fit_rvq(flat2(norm5(joint,sm,ss)),k1,k2,seeds.sc1,seeds.sc2,iters,max_points); state=(sm,ss,sc1,sc2)
    rawdiff=joint[:,1:]-joint[:,:-1]; dm,ds=stats(rawdiff); ic1,ic2=fit_rvq(flat2(norm5(rawdiff,dm,ds)),k1,k2,seeds.ic1,seeds.ic2,iters,max_points); plain=(dm,ds,ic1,ic2)
    comb,_=gen_wave(48,8,grid,(1.15,1.35),seeds.combined,6); ids,_=gen_wave(48,16,grid,(.6,1.),seeds.id,3); spec,_=gen_wave(48,8,grid,(.6,1.),seeds.spectral,6)
    return {'seeds':asdict(seeds),'codec':{'state_mean':sm.tolist(),'state_std':ss.tolist(),'innovation_mean':dm.tolist(),'innovation_std':ds.tolist(),'state_coarse':k1,'state_residual':k2,'companded_levels_per_channel':LEVELS,'companded_zmax':ZMAX,'nominal_bits':18},
            'primary_combined_ood_h3':eval_split(comb,state,plain,dm,ds,3),'secondary_id_h8':eval_split(ids,state,plain,dm,ds,8),'secondary_spectral_h3':eval_split(spec,state,plain,dm,ds,3),'runtime_seconds':time.time()-t0}

def validate_invariants():
    m=np.array([0.,0.],np.float32); s=np.array([1.,1.],np.float32)
    idx=np.array([[[[0]],[[0]]],[[[511]],[[511]]]],np.int32); dec=decode_companded_indices(idx,m,s)
    assert np.allclose(dec[0,:,0,0],[-16.,-16.],rtol=0,atol=2e-5); assert np.allclose(dec[1,:,0,0],[16.,16.],rtol=0,atol=2e-5)
    packed=pack_indices(np.concatenate([np.zeros((1,2,1,1),np.int32),np.full((1,2,1,1),511,np.int32)],0)); assert int(packed.min())==0 and int(packed.max())==262143
    raw=np.zeros((2,4,2,3,3),np.float32); changed=raw.copy(); changed[:,3]=7.
    dummy=(m,s,np.array([[0.,0.]],np.float32),np.array([[0.,0.]],np.float32))
    a=encode_comp_seq(raw,dummy,m,s); b=encode_comp_seq(changed,dummy,m,s); assert np.array_equal(a[:,:3],b[:,:3])
    assert float(np.max(np.abs(a))) < .05
    return {'packed_min':0,'packed_max':262143,'nominal_bits':18,'endpoint_z':[-16.,16.],'prefix_causal':True,'static_max_abs':float(np.max(np.abs(a)))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    inv=validate_invariants()
    if a.smoke:
        r=run_rep(Seeds(2718281801,2718281802,2718281803,2718281804,2718281805,2718281806,2718281807,2718281808,2718281809),grid=8,train_n=12,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE10_PAIRED_ENGINEERING_SCREEN'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
