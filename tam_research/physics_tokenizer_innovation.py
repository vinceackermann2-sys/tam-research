from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch


def lap(x):
    return np.roll(x,1,-1)+np.roll(x,-1,-1)+np.roll(x,1,-2)+np.roll(x,-1,-2)-4*x


def wave_step(u,v,c,dt=0.12):
    a=(c*c)*lap(u); vh=v+0.5*dt*a; u2=u+dt*vh; a2=(c*c)*lap(u2); v2=vh+0.5*dt*a2
    return u2.astype(np.float32),v2.astype(np.float32)


def smooth_field(rng,n,max_mode=3):
    yy,xx=np.meshgrid(np.arange(n),np.arange(n),indexing='ij'); f=np.zeros((n,n),np.float32)
    for _ in range(int(rng.integers(2,6))):
        kx=int(rng.integers(0,max_mode+1)); ky=int(rng.integers(0,max_mode+1))
        if kx==0 and ky==0: kx=1
        amp=float(rng.normal())/(1+kx*kx+ky*ky); phase=float(rng.uniform(0,2*np.pi))
        f += amp*np.cos(2*np.pi*(kx*xx+ky*yy)/n+phase)
    f-=f.mean(); f/=f.std()+1e-6
    return f.astype(np.float32)


def gen_wave(num,seq_len,grid,c_range,seed,max_mode=3,dt=0.12):
    rng=np.random.default_rng(seed); xs=np.empty((num,seq_len,2,grid,grid),np.float32); cs=np.empty(num,np.float32)
    for i in range(num):
        c=float(rng.uniform(*c_range)); cs[i]=c
        u=smooth_field(rng,grid,max_mode)*float(rng.uniform(.5,1.0)); v=smooth_field(rng,grid,max_mode)*float(rng.uniform(.05,.25))
        for t in range(seq_len):
            xs[i,t,0]=u; xs[i,t,1]=v; u,v=wave_step(u,v,c,dt)
    return xs,cs


def gray_step(u,v,F,k,du=.16,dv=.08,dt=1.0):
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
            dx=np.minimum(np.abs(xx-cx),grid-np.abs(xx-cx)); dy=np.minimum(np.abs(yy-cy),grid-np.abs(yy-cy))
            blob=np.exp(-(dx*dx+dy*dy)/(2*r*r)).astype(np.float32)
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
    shape=(1,)*(x.ndim-3)+(2,1,1)
    return ((x-m.reshape(shape))/s.reshape(shape)).astype(np.float32)


def flat2(x):
    return x.transpose(*range(x.ndim-3),x.ndim-2,x.ndim-1,x.ndim-3).reshape(-1,2).astype(np.float32)


def fit_kmeans(x_np,k,seed,iters=20,max_points=60000,chunk=4096):
    rng=np.random.default_rng(seed)
    if len(x_np)>max_points: x_np=x_np[rng.choice(len(x_np),max_points,replace=False)]
    x=torch.from_numpy(np.asarray(x_np,np.float32)); g=torch.Generator().manual_seed(seed)
    if len(x)<k: raise ValueError(f'need >=k points ({len(x)}<{k})')
    c=x[torch.randperm(len(x),generator=g)[:k]].clone()
    for _ in range(iters):
        sums=torch.zeros_like(c); counts=torch.zeros(k,dtype=torch.float32); c2=c.pow(2).sum(1)[None,:]
        for i in range(0,len(x),chunk):
            xb=x[i:i+chunk]; idx=(xb.pow(2).sum(1,keepdim=True)+c2-2*xb@c.t()).argmin(1)
            sums.index_add_(0,idx,xb); counts.index_add_(0,idx,torch.ones(len(xb)))
        nz=counts>0; c[nz]=sums[nz]/counts[nz,None]
        if (~nz).any(): c[~nz]=x[torch.randint(0,len(x),(int((~nz).sum()),),generator=g)]
    return c.numpy().astype(np.float32)


def nearest(flat,cb,chunk=8192):
    x=torch.from_numpy(np.asarray(flat,np.float32)); c=torch.from_numpy(np.asarray(cb,np.float32)); out=[]; c2=c.pow(2).sum(1)[None,:]
    for i in range(0,len(x),chunk):
        xb=x[i:i+chunk]; out.append((xb.pow(2).sum(1,keepdim=True)+c2-2*xb@c.t()).argmin(1))
    return torch.cat(out).numpy()


def fit_rvq(points,k1,k2,s1,s2,iters=20,max_points=60000):
    cb1=fit_kmeans(points,k1,s1,iters,max_points); i1=nearest(points,cb1); r=points-cb1[i1]
    return cb1,fit_kmeans(r,k2,s2,iters,max_points)


def rvq_flat(flat,cb1,cb2):
    i1=nearest(flat,cb1); r=flat-cb1[i1]; i2=nearest(r,cb2); return cb1[i1]+cb2[i2]


def rvq_array(x_norm,cb1,cb2):
    shp=x_norm.shape; q=rvq_flat(flat2(x_norm),cb1,cb2); lead=shp[:-3]; h,w=shp[-2:]
    return q.reshape(*lead,h,w,2).transpose(*range(len(lead)),len(lead)+2,len(lead),len(lead)+1).astype(np.float32)


def q_state(x,m,s,cb1,cb2):
    n=((x-m[None,:,None,None])/s[None,:,None,None]).astype(np.float32); q=rvq_array(n,cb1,cb2)
    return (q*s[None,:,None,None]+m[None,:,None,None]).astype(np.float32)


def q_delta(d,m,s,cb1,cb2):
    n=((d-m[None,:,None,None])/s[None,:,None,None]).astype(np.float32); q=rvq_array(n,cb1,cb2)
    return (q*s[None,:,None,None]+m[None,:,None,None]).astype(np.float32)


def encode_state_seq(seq,m,s,cb1,cb2):
    return np.stack([q_state(seq[:,j],m,s,cb1,cb2) for j in range(seq.shape[1])],1)


def encode_innov_seq(seq,sm,ss,sc1,sc2,dm,ds,dc1,dc2):
    out=[q_state(seq[:,0],sm,ss,sc1,sc2)]
    for j in range(1,seq.shape[1]):
        prev=out[-1]; out.append((prev+q_delta(seq[:,j]-prev,dm,ds,dc1,dc2)).astype(np.float32))
    return np.stack(out,1)


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
        for t in range(2):
            X.append(features(ctx[b:b+1,t])[0].reshape(-1,16)); Y.append(ctx[b,t+1].transpose(1,2,0).reshape(-1,2))
        X=np.concatenate(X).astype(np.float64); Y=np.concatenate(Y).astype(np.float64)
        sc=np.sqrt(np.mean(X*X,axis=0)+1e-12); sc[0]=1.; Xs=X/sc
        W=np.linalg.solve(Xs.T@Xs+ridge*np.eye(16),Xs.T@Y).T
        maps[b]=(W/sc[None,:]).astype(np.float32)
    return maps


def predict(s,maps):
    return np.einsum('bhwf,bcf->bchw',features(s),maps,optimize=True).astype(np.float32)


def rollout(ctx,maps,h,project):
    cur=ctx[:,-1].copy()
    for _ in range(h): cur=project(predict(cur,maps),cur)
    return cur


def mse(a,b): return float(np.mean((a-b)**2))


def eval_split(raw,state_codec,innov_codec,h,ridge=.001):
    sm,ss,sc1,sc2=state_codec; dm,ds,dc1,dc2=innov_codec
    qS=encode_state_seq(raw[:,:3],sm,ss,sc1,sc2); qI=encode_innov_seq(raw[:,:3],sm,ss,sc1,sc2,dm,ds,dc1,dc2)
    mS=fit_maps(qS,ridge); mI=fit_maps(qI,ridge)
    pS=lambda y,prev:q_state(y,sm,ss,sc1,sc2)
    pI=lambda y,prev:(prev+q_delta(y-prev,dm,ds,dc1,dc2)).astype(np.float32)
    yS=rollout(qS,mS,h,pS); yI=rollout(qI,mI,h,pI); target=raw[:,3+h-1]; persist=raw[:,2]
    pm=mse(persist,target); sr=mse(yS,target)/pm; ir=mse(yI,target)/pm
    mr=fit_maps(raw[:,:3],ridge); crS=rollout(qS,mr,h,pS); crI=rollout(qI,mr,h,pI)
    s_or=q_state(target,sm,ss,sc1,sc2); prev=qI[:,-1].copy()
    for j in range(h):
        truth=raw[:,3+j]; prev=(prev+q_delta(truth-prev,dm,ds,dc1,dc2)).astype(np.float32)
    rawctx=raw[:,:3]; trans_raw=rawctx[:,1:]-rawctx[:,:-1]; transS=qS[:,1:]-qS[:,:-1]; transI=qI[:,1:]-qI[:,:-1]
    return {'persistence_mse':pm,'state_ratio':sr,'innovation_ratio':ir,'innovation_over_state':ir/sr,
        'state_mse':mse(yS,target),'innovation_mse':mse(yI,target),
        'raw_fit_state_projection_ratio':mse(crS,target)/pm,'raw_fit_innovation_projection_ratio':mse(crI,target)/pm,
        'state_target_oracle_ratio':mse(s_or,target)/pm,'innovation_target_oracle_ratio':mse(prev,target)/pm,
        'state_context_mse':mse(qS,rawctx),'innovation_context_mse':mse(qI,rawctx),
        'state_context_delta_mse':mse(transS,trans_raw),'innovation_context_delta_mse':mse(transI,trans_raw)}


@dataclass
class Seeds:
    wave:int; gray:int; sc1:int; sc2:int; ic1:int; ic2:int; combined:int; id:int; spectral:int


def run_rep(seeds,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time(); k1,k2=code_sizes
    w,_=gen_wave(train_n,8,grid,(.6,1.0),seeds.wave,3); g=gen_gray(train_n,8,grid,(.025,.045),(.055,.065),seeds.gray)
    joint=np.concatenate([w,g],0)
    sm,ss=stats(joint); sc1,sc2=fit_rvq(flat2(norm5(joint,sm,ss)),k1,k2,seeds.sc1,seeds.sc2,iters,max_points)
    rawdiff=joint[:,1:]-joint[:,:-1]; dm,ds=stats(rawdiff); ic1,ic2=fit_rvq(flat2(norm5(rawdiff,dm,ds)),k1,k2,seeds.ic1,seeds.ic2,iters,max_points)
    state=(sm,ss,sc1,sc2); innov=(dm,ds,ic1,ic2)
    comb,_=gen_wave(48,8,grid,(1.15,1.35),seeds.combined,6); ids,_=gen_wave(48,16,grid,(.6,1.0),seeds.id,3); spec,_=gen_wave(48,8,grid,(.6,1.0),seeds.spectral,6)
    return {'seeds':seeds.__dict__,'codec':{'state_mean':sm.tolist(),'state_std':ss.tolist(),'innovation_mean':dm.tolist(),'innovation_std':ds.tolist(),'coarse':k1,'residual':k2},
        'primary_combined_ood_h3':eval_split(comb,state,innov,3),'secondary_id_h8':eval_split(ids,state,innov,8),'secondary_spectral_h3':eval_split(spec,state,innov,3),'runtime_seconds':time.time()-t0}


REGISTERED=[
    Seeds(2026091301,2026091302,2026091303,2026091304,2026091305,2026091306,2026091307,2026091308,2026091309),
    Seeds(2026091311,2026091312,2026091313,2026091314,2026091315,2026091316,2026091317,2026091318,2026091319),
    Seeds(2026091321,2026091322,2026091323,2026091324,2026091325,2026091326,2026091327,2026091328,2026091329)]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    if a.smoke:
        r=run_rep(Seeds(31415901,31415902,31415903,31415904,31415905,31415906,31415907,31415908,31415909),grid=8,train_n=12,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE9_PAIRED_ENGINEERING_SCREEN'
    Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))


if __name__=='__main__': main()
