from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from tam_research import physics_tokenizer_phase10 as p10
from tam_research import physics_tokenizer_phase12 as p12
from tam_research import physics_tokenizer_phase13 as p13


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
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    wave_id:int; wave_spectral:int; wave_combined:int
    gray_id:int; gray_sharp:int; gray_parameter:int; gray_combined:int
    nls_id:int; nls_spectral:int; nls_parameter:int; nls_combined:int


REGISTERED=[
    Seeds(514140001,514140002,514140003,514140004,514140005,514140006,514140011,514140012,514140013,514140021,514140022,514140023,514140024,514140031,514140032,514140033,514140034),
    Seeds(514140101,514140102,514140103,514140104,514140105,514140106,514140111,514140112,514140113,514140121,514140122,514140123,514140124,514140131,514140132,514140133,514140134),
    Seeds(514140201,514140202,514140203,514140204,514140205,514140206,514140211,514140212,514140213,514140221,514140222,514140223,514140224,514140231,514140232,514140233,514140234),
]


def build_codecs(seeds,grid=16,train_n=64,code_sizes=(2048,128),iters=20,max_points=60000):
    # Exact Phase-10/11/12 representation-training recipe: wave + smooth Gray only.
    k1,k2=code_sizes
    w,_=p10.gen_wave(train_n,8,grid,(.6,1.0),seeds.wave_train,3)
    g,_,_=gen_gray(train_n,8,grid,(.025,.045),(.055,.065),seeds.gray_train,False)
    joint=np.concatenate([w,g],0)
    sm,ss=p10.stats(joint)
    sc1,sc2=p10.fit_rvq(p10.flat2(p10.norm5(joint,sm,ss)),k1,k2,seeds.sc1,seeds.sc2,iters,max_points)
    state=(sm,ss,sc1,sc2)
    rawdiff=joint[:,1:]-joint[:,:-1]
    dm,ds=p10.stats(rawdiff)
    ic1,ic2=p10.fit_rvq(p10.flat2(p10.norm5(rawdiff,dm,ds)),k1,k2,seeds.ic1,seeds.ic2,iters,max_points)
    plain=(dm,ds,ic1,ic2)
    return state,plain,dm,ds


def run_rep(seeds,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time(); state,plain,dm,ds=build_codecs(seeds,grid,train_n,code_sizes,iters,max_points)
    raw={
      'wave_id_h8':p10.gen_wave(eval_n,16,grid,(.6,1.0),seeds.wave_id,3)[0],
      'wave_spectral_ood_h3':p10.gen_wave(eval_n,11,grid,(.6,1.0),seeds.wave_spectral,6)[0],
      'wave_combined_ood_h3':p10.gen_wave(eval_n,11,grid,(1.15,1.35),seeds.wave_combined,6)[0],
      'gray_id_h8':gen_gray(eval_n,16,grid,(.025,.045),(.055,.065),seeds.gray_id,False)[0],
      'gray_sharp_ood_h3':gen_gray(eval_n,11,grid,(.025,.045),(.055,.065),seeds.gray_sharp,True)[0],
      'gray_parameter_ood_h3':gen_gray(eval_n,11,grid,(.050,.060),(.045,.055),seeds.gray_parameter,False)[0],
      'gray_combined_ood_h3':gen_gray(eval_n,11,grid,(.050,.060),(.045,.055),seeds.gray_combined,True)[0],
      'nls_id_h8':p12.gen_nls(eval_n,16,grid,(.04,.08),(.10,.25),seeds.nls_id,3)[0],
      'nls_spectral_ood_h3':p12.gen_nls(eval_n,11,grid,(.04,.08),(.10,.25),seeds.nls_spectral,6)[0],
      'nls_parameter_ood_h3':p12.gen_nls(eval_n,11,grid,(.10,.14),(.30,.45),seeds.nls_parameter,3)[0],
      'nls_combined_ood_h3':p12.gen_nls(eval_n,11,grid,(.10,.14),(.30,.45),seeds.nls_combined,6)[0],
    }
    splits={}
    for name,x in raw.items():
        splits[name]=p13.eval_companded_context8(x,state,dm,ds,8 if name.endswith('h8') else 3)
    primary={k:float(v['companded_ratio']) for k,v in splits.items()}
    return {
      'seeds':asdict(seeds),
      'context_frames':8,'observed_transitions_used':7,
      'representation_training_families':['wave','gray_scott'],
      'nls_used_in_representation_training':False,
      'frozen_compander':{'levels_per_channel':p10.LEVELS,'zmax':p10.ZMAX,'nominal_bits':18},
      'primary_companded_ratios':primary,
      'all_eleven_below_persistence':bool(all(v<1.0 for v in primary.values())),
      'splits':splits,
      'runtime_seconds':time.time()-t0,
    }


def validate_invariants():
    assert p10.LEVELS==512 and p10.ZMAX==16.0
    assert p13.CONTEXT==8 and p13.RIDGE==0.001
    assert p10.features(np.zeros((1,2,4,4),np.float32)).shape[-1]==16
    smooth,Fs,ks=gen_gray(4,5,8,(.025,.045),(.055,.065),971401,False)
    sharp,Fs2,ks2=gen_gray(4,5,8,(.025,.045),(.055,.065),971401,True)
    assert smooth.shape==sharp.shape==(4,5,2,8,8); assert not np.array_equal(smooth,sharp)
    assert np.all((Fs>=.025)&(Fs<=.045)) and np.all((ks>=.055)&(ks<=.065))
    assert np.all((Fs2>=.025)&(Fs2<=.045)) and np.all((ks2>=.055)&(ks2<=.065))
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    return {'phase10_levels':p10.LEVELS,'phase10_zmax':p10.ZMAX,'feature_dim':16,'context_frames':8,'transitions_used':7,'ridge':.001,'gray_generators_validated':True,'registered_seed_count':len(vals)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    inv=validate_invariants()
    if a.smoke:
        s=Seeds(731953001,731953002,731953003,731953004,731953005,731953006,731953011,731953012,731953013,731953021,731953022,731953023,731953024,731953031,731953032,731953033,731953034)
        r=run_rep(s,grid=8,train_n=12,eval_n=6,code_sizes=(32,8),iters=3,max_points=3000); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE14_UNIFIED_CONTEXT8_REGISTERED_CONFIRMATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))

if __name__=='__main__': main()
