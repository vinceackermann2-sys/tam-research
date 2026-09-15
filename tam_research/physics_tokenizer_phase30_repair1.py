from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path
from dataclasses import asdict
from tam_research import physics_tokenizer_phase30 as p30

REPAIR_BASES=(830310000,830310100,830310200)
FAMILIES=('wave','gray_scott','nls','advection','burgers','hamilton_jacobi')

def seeds_for_rep(rep:int):
    if rep not in (1,2,3): raise ValueError(rep)
    return p30._seedset(REPAIR_BASES[rep-1])

def prepare(rep:int,out:str,grid=16,train_n=64,eval_n=48,code_sizes=(2048,128),iters=20,max_points=60000):
    s=seeds_for_rep(rep); t=time.time()
    state,dm,ds,basis=p30.build_all(s,grid,train_n,code_sizes,iters,max_points)
    raw=p30.build_eval(s,grid,eval_n)
    obj={'replicate':rep,'seed_base':REPAIR_BASES[rep-1],'seeds':asdict(s),'state':state,'dm':dm,'ds':ds,'basis':basis,'raw':raw,
         'scientific_config':'phase30_exact','runtime_seconds':time.time()-t}
    Path(out).write_bytes(pickle.dumps(obj,protocol=pickle.HIGHEST_PROTOCOL))
    return {'replicate':rep,'seed_base':REPAIR_BASES[rep-1],'split_count':len(raw),'runtime_seconds':obj['runtime_seconds']}

def split_family(name:str)->str:
    if name.startswith('wave_'): return 'wave'
    if name.startswith('gray_'): return 'gray_scott'
    if name.startswith('nls_'): return 'nls'
    if name.startswith('adv_'): return 'advection'
    if name.startswith('burgers_'): return 'burgers'
    if name.startswith('hj_'): return 'hamilton_jacobi'
    raise KeyError(name)

def eval_family(prep_path:str,family:str,out:str):
    if family not in FAMILIES: raise ValueError(family)
    obj=pickle.loads(Path(prep_path).read_bytes()); t=time.time()
    names=[k for k in obj['raw'] if split_family(k)==family]
    splits={k:p30.eval_split(obj['raw'][k],obj['state'],obj['dm'],obj['ds'],obj['basis'],8 if k.endswith('h8') else 3) for k in names}
    ratios={k:float(v['selected_ratio']) for k,v in splits.items()}
    r={'status':'PHASE30_REPAIR1_REGISTERED_FAMILY_RESULT','replicate':obj['replicate'],'seed_base':obj['seed_base'],'family':family,
       'primary_ratios':ratios,'passing_cells':int(sum(v<1 for v in ratios.values())),'total_cells':len(ratios),'all_below_persistence':bool(all(v<1 for v in ratios.values())),
       'splits':splits,'runtime_seconds':time.time()-t,'scientific_config':'phase30_exact'}
    Path(out).write_text(json.dumps(r,indent=2)); return r

def validate_invariants():
    vals=[]
    for rep in (1,2,3): vals.extend(asdict(seeds_for_rep(rep)).values())
    assert len(vals)==87 and len(vals)==len(set(vals))
    assert set(FAMILIES)=={'wave','gray_scott','nls','advection','burgers','hamilton_jacobi'}
    assert p30.MEDIAN_GAIN_MIN==.10
    return {'fresh_registered_seed_count':len(vals),'families':list(FAMILIES),'scientific_config':'phase30_exact','execution_only_change':True}

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--rep',type=int,required=True); a.add_argument('--out',required=True); a.add_argument('--smoke',action='store_true')
    b=sub.add_parser('eval-family'); b.add_argument('--prep',required=True); b.add_argument('--family',required=True); b.add_argument('--out',required=True)
    z=ap.parse_args(); inv=validate_invariants()
    if z.cmd=='prepare':
        if z.smoke:
            s=p30._seedset(999302000); t=time.time(); state,dm,ds,basis=p30.build_all(s,8,8,(16,4),2,1200); raw=p30.build_eval(s,8,3)
            obj={'replicate':0,'seed_base':999302000,'seeds':asdict(s),'state':state,'dm':dm,'ds':ds,'basis':basis,'raw':raw,'scientific_config':'smoke','runtime_seconds':time.time()-t}
            Path(z.out).write_bytes(pickle.dumps(obj,protocol=pickle.HIGHEST_PROTOCOL)); print(json.dumps({'status':'SMOKE','split_count':len(raw),'invariants':inv}))
        else: print(json.dumps({'status':'PREPARED',**prepare(z.rep,z.out),'invariants':inv},indent=2))
    else:
        r=eval_family(z.prep,z.family,z.out); r['invariants']=inv; Path(z.out).write_text(json.dumps(r,indent=2)); print(json.dumps({'family':z.family,'pass':r['passing_cells'],'total':r['total_cells'],'ratios':r['primary_ratios']},indent=2))
if __name__=='__main__': main()
