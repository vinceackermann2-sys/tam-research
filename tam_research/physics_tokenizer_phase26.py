from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from tam_research import physics_tokenizer_phase25 as p25

CONTEXT=p25.CONTEXT
RIDGE=p25.RIDGE

@dataclass
class Seeds:
    wave_train:int; gray_train:int; sc1:int; sc2:int; ic1:int; ic2:int
    burg_id:int; burg_spectral:int; burg_parameter:int; burg_combined:int

REGISTERED=[
    Seeds(826260001,826260002,826260003,826260004,826260005,826260006,826260011,826260012,826260013,826260014),
    Seeds(826260101,826260102,826260103,826260104,826260105,826260106,826260111,826260112,826260113,826260114),
    Seeds(826260201,826260202,826260203,826260204,826260205,826260206,826260211,826260212,826260213,826260214),
]

def build_eval(s,grid=16,eval_n=48):
    return {
      'burgers_id_h8':p25.gen_burgers(eval_n,16,grid,(.01,.03),(.15,.35),s.burg_id,3)[0],
      'burgers_spectral_ood_h3':p25.gen_burgers(eval_n,11,grid,(.01,.03),(.15,.35),s.burg_spectral,6)[0],
      'burgers_parameter_ood_h3':p25.gen_burgers(eval_n,11,grid,(.04,.06),(.15,.35),s.burg_parameter,3)[0],
      'burgers_combined_ood_h3':p25.gen_burgers(eval_n,11,grid,(.04,.06),(.35,.55),s.burg_combined,6)[0],
    }

def run_rep(s,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    t0=time.time(); state,dm,ds,basis=p25.build_all(s,grid,train_n,code_sizes,iters,max_points); raw=build_eval(s,grid,eval_n)
    splits={name:p25.eval_split(x,state,dm,ds,basis,8 if name.endswith('h8') else 3) for name,x in raw.items()}
    primary={k:float(v['token62_rollout_ratio']) for k,v in splits.items()}
    control26={k:float(v['primary_token26_ratio']) for k,v in splits.items()}
    return {
      'seeds':asdict(s),'family':'two_component_viscous_burgers_2d','dt':p25.BURGERS_DT,
      'primary_feature_library':'plus_center_times_spatial_62','secondary_control_library':'full_fourier_linear_plus_center_polynomial_26',
      'context_frames':CONTEXT,'ridge':RIDGE,'representation_training_families':['wave','gray_scott'],
      'burgers_used_in_representation_training':False,
      'primary_token62_ratios':primary,'primary_all_four_below_persistence':bool(all(v<1 for v in primary.values())),
      'secondary_token26_ratios':control26,'splits':splits,'runtime_seconds':time.time()-t0,
    }

def validate_invariants():
    z=np.zeros((2,2,8,8),np.float32); b=p25.p22.fourier_basis()
    assert p25.interaction62_features(z,b).shape[-1]==62
    vals=[]
    for s in REGISTERED: vals.extend(asdict(s).values())
    assert len(vals)==len(set(vals))
    return {'primary_feature_dim':62,'secondary_feature_dim':26,'context_frames':CONTEXT,'ridge':RIDGE,'registered_seed_count':len(vals),'phase25_eval_path_reused':True}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--rep',type=int,default=0); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); inv=validate_invariants()
    if a.smoke:
        s=Seeds(999260001,999260002,999260003,999260004,999260005,999260006,999260011,999260012,999260013,999260014)
        r=run_rep(s,grid=8,train_n=10,eval_n=4,code_sizes=(16,4),iters=2,max_points=1500); r['status']='SMOKE_ONLY_NONREGISTERED_SEED'
    else:
        if a.rep not in (1,2,3): raise SystemExit('--rep must be 1,2,3')
        r=run_rep(REGISTERED[a.rep-1]); r['replicate']=a.rep; r['status']='PHASE26_BURGERS_INTERACTION_REGISTERED_CONFIRMATION'
    r['invariants']=inv; Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
