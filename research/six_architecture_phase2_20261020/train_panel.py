"""Prospective stage-gated ~100M panel. Exclusive claims prevent duplicates."""
import gc,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from phase2_aligned_v2 import make,count,VOCAB

ROOT=Path('/content/tam-phase2-20261020');OUT=ROOT/'results';DATA=ROOT/'data'
SEED=202610291;BATCH=16;CONTEXT=512;UNIT=BATCH*CONTEXT;MAX=2_000_000_000
STAGES=[n//UNIT for n in (100_000_000,300_000_000,1_000_000_000,2_000_000_000)]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def atomic(path,data):
    p=path.with_name(path.name+'.tmp');p.write_text(json.dumps(data,indent=2,allow_nan=False));p.replace(path)
def exclusive(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
def status(**kw):atomic(ROOT/'STATUS.json',{'utc':time.time(),**kw})
def sync():torch.cuda.synchronize()

@torch.no_grad()
def evaluate(model,split,tokens):
    model.eval();data=np.memmap(DATA/(split+'.bin'),mode='r',dtype=np.uint16)
    blocks=min((len(data)-1)//CONTEXT,tokens//CONTEXT);losses=[]
    for start in range(0,blocks,BATCH):
        positions=np.arange(start,min(start+BATCH,blocks))*CONTEXT
        z=np.stack([data[p:p+CONTEXT+1] for p in positions]).astype(np.int64)
        x=torch.from_numpy(z[:,:-1]).cuda();y=torch.from_numpy(z[:,1:]).cuda()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
        per=F.cross_entropy(logits.float().flatten(0,1),y.flatten(),reduction='none').view(len(z),CONTEXT).mean(-1)
        losses.extend(per.cpu().tolist())
    model.train();return {'nll':float(np.mean(losses)),'tokens':blocks*CONTEXT,'block_nll':losses}

def checkpoint(model,opt,arm,step,seconds,history):
    name=f'{arm}-s{step}';directory=OUT/name;directory.mkdir()
    p=directory/'state.pt'
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'tokens_seen':step*UNIT,
        'training_seconds':seconds,'history':history,'seed':SEED,'variant':arm,'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all()},p)
    digest=sha(p);chunks=[]
    with p.open('rb') as f:
        for i,b in enumerate(iter(lambda:f.read(64*1024**2),b'')):
            q=directory/f'chunk{i:03d}.bin';q.write_bytes(b)
            chunks.append({'path':str(q.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    manifest={'variant':arm,'step':step,'tokens_seen':step*UNIT,'checkpoint_sha256':digest,'chunks':chunks,
        'bytes':p.stat().st_size,'includes_optimizer':True,'source_sha256':sha(Path(__file__))}
    exclusive(directory/'MANIFEST.json',manifest);p.unlink()
    return manifest

def load_checkpoint(manifest):
    directory=OUT/f"{manifest['variant']}-s{manifest['step']}";p=directory/'restore.tmp.pt'
    with p.open('xb') as f:
        for row in manifest['chunks']:
            q=ROOT/row['path'];assert sha(q)==row['sha256']
            with q.open('rb') as src:
                for b in iter(lambda:src.read(8*1024**2),b''):f.write(b)
    assert sha(p)==manifest['checkpoint_sha256']
    value=torch.load(p,map_location='cuda',weights_only=False);p.unlink();return value

def segment(arm,end,previous,evidence):
    start=previous['step'] if previous else 0;ident=f'{arm}-s{end}'
    exclusive(OUT/(ident+'.CLAIM.json'),{'variant':arm,'seed':SEED,'start_step':start,'end_step':end,
        'reserved_exposures':(end-start)*UNIT,'cumulative_upper_bound':end*UNIT,'ambiguous_retry_allowed':False})
    assert end*UNIT<=MAX
    torch.manual_seed(SEED);model=make(arm).cuda();n=count(model);assert n<=100_000_000
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,betas=(.9,.95),weight_decay=.1,fused=True)
    seconds=0.;history=[]
    if previous:
        state=load_checkpoint(previous);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        seconds=state['training_seconds'];history=state['history'];torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([z.cpu() for z in state['cuda_rng']]);del state
    train=np.memmap(DATA/'train.bin',dtype=np.uint16,mode='r')
    assert len(train)>end*UNIT
    torch.cuda.reset_peak_memory_stats();wall=time.monotonic();segment_seconds=0.
    for step in range(start,end):
        lo=step*UNIT;z=np.array(train[lo:lo+UNIT+1],dtype=np.int64)
        x=torch.from_numpy(z[:-1].reshape(BATCH,CONTEXT)).cuda();y=torch.from_numpy(z[1:].reshape(BATCH,CONTEXT)).cuda()
        seen=(step+1)*UNIT
        if seen>MAX:raise ValueError('USER_TOKEN_CAP')
        lr=.0003*seen/2_000_000 if seen<=2_000_000 else .00003+.00027*.5*(1+math.cos(math.pi*(seen-2_000_000)/(STAGES[-1]*UNIT-2_000_000)))
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True);sync();t=time.perf_counter()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(x,y)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();sync()
        elapsed=time.perf_counter()-t;seconds+=elapsed;segment_seconds+=elapsed
        if (step+1)%256==0 or step==end-1:
            row={'step':step+1,'tokens':seen,'loss':float(loss.detach()),'lr':lr,'train_seconds':seconds,'grad_norm':float(grad)}
            history.append(row);status(phase='training',current=arm,stage=end,progress=row,complete=False)
            print(arm,json.dumps(row),flush=True)
    validation=evaluate(model,'validation',524_288)
    manifest=checkpoint(model,opt,arm,end,seconds,history)
    value={'variant':arm,'seed':SEED,'parameters':n,'width':model.d,'ff_hidden':model.ff,'steps':end,
        'tokens_seen':end*UNIT,'validation':validation,'train_seconds':seconds,'train_tokens_per_second':end*UNIT/seconds,
        'segment_train_seconds':segment_seconds,'segment_wall_seconds':time.monotonic()-wall,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'checkpoint':manifest,
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'source_sha256':sha(Path(__file__)),
        'model_source_sha256':{p:sha(ROOT/p) for p in ('phase2_model.py','phase2_aligned_model.py','phase2_aligned_v2.py')},
        'classification':'ONE_SEED_STAGE_GATED_DEVELOPMENT','breakthrough_demonstrated':False}
    exclusive(OUT/(ident+'.RESULT.json'),value);evidence['results'][ident]=value;atomic(ROOT/'EVIDENCE.json',evidence)
    del model,opt,loss,x,y;gc.collect();torch.cuda.empty_cache()
    return value

def final_test(arm,result,evidence):
    torch.manual_seed(SEED);model=make(arm).cuda();state=load_checkpoint(result['checkpoint']);model.load_state_dict(state['model']);del state
    test=evaluate(model,'test',2_097_152)
    value={'variant':arm,'steps':result['steps'],'tokens_seen':result['tokens_seen'],'test':test,
        'selection':'single terminal checkpoint selected by prospective development gate','breakthrough_demonstrated':False}
    exclusive(OUT/(arm+'.TERMINAL.json'),value);evidence['terminal'][arm]=value;atomic(ROOT/'EVIDENCE.json',evidence)
    del model;gc.collect();torch.cuda.empty_cache()

def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    exclusive(ROOT/'TRAINING_CLAIM.json',{'seed':SEED,'stages':STAGES,'protocol_sha256':sha(ROOT/'PROTOCOL.md')})
    for _ in range(180):
        if (DATA/'DATA.json').exists():break
        status(phase='waiting_for_corpus',complete=False);time.sleep(10)
    else:raise TimeoutError('corpus preparation30minute deadline')
    data=json.loads((DATA/'DATA.json').read_text())
    for split in ('train','validation','test'):
        assert sha(DATA/(split+'.bin'))==data['splits'][split]['token_sha256']
    evidence={'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'data':data,'results':{},'gates':[],'terminal':{},'complete':False}
    atomic(ROOT/'EVIDENCE.json',evidence)
    active=['transformer','attention8','conv7','cpw_local'];latest={};prev={}
    for stage_index,end in enumerate(STAGES):
        for arm in active:
            result=segment(arm,end,prev.get(arm),evidence);latest[arm]=result;prev[arm]=result['checkpoint']
        control=latest['transformer'];keep=[]
        for arm in ('conv7','cpw_local'):
            if arm not in active:continue
            r=latest[arm];delta=r['validation']['nll']-control['validation']['nll'];speed=r['train_tokens_per_second']/control['train_tokens_per_second']
            parent_benefit=latest['attention8']['validation']['nll']-r['validation']['nll'] if arm=='conv7' else None
            passed=((delta<=.05 or (delta<=.10 and speed>=1.03)) if stage_index==0 else (delta<=-.03 or (delta<=.015 and speed>=1.03)))
            if parent_benefit is not None:passed=passed and parent_benefit>=.005
            gate={'stage_tokens':end*UNIT,'variant':arm,'delta_nll_to_transformer':delta,'speed_ratio':speed,'parent_benefit':parent_benefit,'passed':passed}
            evidence['gates'].append(gate)
            if passed and stage_index<len(STAGES)-1:keep.append(arm)
            else:final_test(arm,r,evidence)
        if 'attention8' in active and 'conv7' not in keep:final_test('attention8',latest['attention8'],evidence)
        atomic(ROOT/'EVIDENCE.json',evidence)
        if not keep:
            final_test('transformer',control,evidence);break
        active=['transformer']+(['attention8'] if 'conv7' in keep else [])+keep
    evidence['complete']=True;atomic(ROOT/'EVIDENCE.json',evidence)
    status(phase='finished',complete=True,terminal={k:{'steps':v['steps'],'nll':v['test']['nll']} for k,v in evidence['terminal'].items()})

if __name__=='__main__':
    try:main()
    except BaseException as e:
        status(phase='error',complete=True,success=False,error=repr(e),traceback=traceback.format_exc());raise
