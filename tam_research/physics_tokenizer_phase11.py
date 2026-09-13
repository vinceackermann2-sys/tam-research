from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10


def gen_gray(num, seq_len, grid, F_range, k_range, seed, sharp=False, substeps=4):
    rng=np.random.default_rng(seed)
    out=np.empty((num,seq_len,2,grid,grid),np.float32)
    Fs=rng.uniform(*F_range,size=num).astype(np.float32)
    ks=rng.uniform(*k_range,size=num).astype(np.float32)
    yy,xx=np.meshgrid(np.arange(grid),np.arange(grid),indexing='ij')
    for i in range(num):
        u=np.ones((grid,grid),np.float32); v=np.zeros((grid,grid),np.float32)
        for _ in range(int(rng.integers(1,4))):
            cx,cy=float(rng.uniform(0,grid)),float(rng.uniform(0,grid))
            r=float(rng.uniform(.8,1.8) if sharp else rng.uniform(1.5,3.0))
            dx=np.minimum(np.abs(xx-cx),grid-np.abs(xx-cx)); dy=np.minimum(np.abs(yy-cy),grid-np.abs(yy-cy))
            blob=np.exp(-(dx*dx+dy*dy)/(2*r*r)).astype(np.float32)
            u-=float(rng.uniform(.25,.5))*blob; v+=float(rng.uniform(.18,.35))*blob
        u+=rng.normal(0,.02 if sharp else .01,(grid,grid)).astype(np.float32)
        v+=rng.normal(0,.01 if sharp else .005,(grid,grid)).astype(np.float32)
        for t in range(seq_len):
            out[i,t,0]=u; out[i,t,1]=v
            for _ in range(substeps): u,v=p10.gray_step(u,v,Fs[i],ks[i])
    return out,Fs,ks


@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; pc1:int; pc2:int
    wave_id:int; wave_spectral:int; wave_combined:int
    gray_id:int; gray_sharp:int; gray_parameter:int; gray_combined:int


REGISTERED=[
    Seeds(611370101,611370102,611370103,611370104,611370105,611370106,611370111,611370112,611370113,611370121,611370122,611370123,611370124),
    Seeds(611370201,611370202,611370203,611370204,611370205,611370206,611370211,611370212,611370213,611370221,611370222,611370223,611370224),
    Seeds(611370301,611370302,611370303,611370304,611370305,611370306,611370311,611370312,611370313,611370321,611370322,611370323,611370324),
]


def build_codecs(seeds,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    k1,k2=code_sizes
    w,_=p10.gen_wave(train_n,8,grid,(.6,1.0),seeds.wave_train,3)
    g,_,_=gen_gray(train_n,8,grid,(.025,.045),(.055,.065),seeds.gray_train,sharp=False)
    joint=np.concatenate([w,g],0)
    sm,ss=p10.stats(joint)
    sc1,sc2=p10.fit_rvq(p10.flat2(p10.norm5(joint,sm,ss)),k1,k2,seeds.sc1,seeds.sc2,iters,max_points)
    rawdiff=joint[:,1:]-joint[:,:-1]
    dm,ds=p10.stats(rawdiff)
    pc1,pc2=p10.fit_rvq(p10.flat2(p10.norm5(rawdiff,dm,ds)),k1,k2,seeds.pc1,seeds.pc2,iters,max_points)
    return (sm,ss,sc1,sc2),(dm,ds,pc1,pc2),dm,ds


def run_rep(seeds,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time()
    state,plain,dm,ds=build_codecs(seeds,grid,train_n,code_sizes,iters,max_points)
    wave_id,_=p10.gen_wave(eval_n,16,grid,(.6,1.0),seeds.wave_id,3)
    wave_spec,_=p10.gen_wave(eval_n,8,grid,(.6,1.0),seeds.wave_spectral,6)
    wave_comb,_=p10.gen_wave(eval_n,8,grid,(1.15,1.35),seeds.wave_combined,6)
    gray_id,_,_=gen_gray(eval_n,16,grid,(.025,.045),(.055,.065),seeds.gray_id,False)
    gray_sharp,_,_=gen_gray(eval_n,8,grid,(.025,.045),(.055,.065),seeds.gray_sharp,True)
    gray_param,_,_=gen_gray(eval_n,8,grid,(.050,.060),(.045,.055),seeds.gray_parameter,False)
    gray_comb,_,_=gen_gray(eval_n,8,grid,(.050,.060),(.045,.055),seeds.gray_combined,True)
    splits={
      'wave_id_h8': p10.eval_split(wave_id,state,plain,dm,ds,8),
      'wave_spectral_ood_h3': p10.eval_split(wave_spec,state,plain,dm,ds,3),
      'wave_combined_ood_h3': p10.eval_split(wave_comb,state,plain,dm,ds,3),
      'gray_id_h8': p10.eval_split(gray_id,state,plain,dm,ds,8),
      'gray_sharp_ood_h3': p10.eval_split(gray_sharp,state,plain,dm,ds,3),
      'gray_parameter_ood_h3': p10.eval_split(gray_param,state,plain,dm,ds,3),
      'gray_combined_ood_h3': p10.eval_split(gray_comb,state,plain,dm,ds,3),
    }
    primary={name:float(row['companded_ratio']) for name,row in splits.items()}
    return {
      'seeds':asdict(seeds),
      'frozen_compander':{'levels_per_channel':p10.LEVELS,'zmax':p10.ZMAX,'nominal_bits':18},
      'splits':splits,
      'primary_companded_ratios':primary,
      'all_seven_below_persistence':bool(all(v<1.0 for v in primary.values())),
      'runtime_seconds':time.time()-t0,
    }


def validate_invariants():
    base=p10.validate_invariants()
    assert p10.LEVELS==512 and p10.ZMAX==16.0
    smooth,Fs,ks=gen_gray(4,5,8,(.025,.045),(.055,.065),971001,False)
    sharp,Fs2,ks2=gen_gray(4,5,8,(.025,.045),(.055,.065),971001,True)
    assert smooth.shape==sharp.shape==(4,5,2,8,8)
    assert np.all((Fs>=.025)&(Fs<=.045)) and np.all((ks>=.055)&(ks<=.065))
    assert np.all((Fs2>=.025)&(Fs2<=.045)) and np.all((ks2>=.055)&(ks2<=.065))
    assert not np.array_equal(smooth,sharp)
    return {**base,'gray_generators_validated':True}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    inv=validate_invariants()
    if a.smoke:
        seeds=Seeds(731942101,731942102,731942103,731942104,731942105,731942106,731942111,731942112,731942113,731942121,731942122,731942123,731942124)
        r=run_rep(seeds,grid=8,train_n=12,eval_n=8,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE11_CROSSLAW_ENGINEERING_VALIDATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))

if __name__=='__main__': main()
