"""Predeclared fresh 20M-class / 25M-token development horizon after tiny pilot.

One new seed, three arms. No 2B inference, no scientific replication claim.
"""
import gc,hashlib,json,math,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from conv7_language import ScheduledLM,count,dump,sync,VOCAB

ROOT=Path('/content/tam-six-20261020')
OUT=ROOT/'results/conv7_recovery'
DATA=ROOT/'fineweb25m'
SEED=202610262
BATCH,CONTEXT=16,256
STEPS=25_000_000//(BATCH*CONTEXT)

def prepare():
    import requests,tiktoken
    from datasets import load_dataset
    DATA.mkdir(exist_ok=True)
    api=requests.get('https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu',timeout=30);api.raise_for_status()
    rev=api.json()['sha'];enc=tiktoken.get_encoding('gpt2')
    stream=load_dataset('HuggingFaceFW/fineweb-edu','sample-10BT',split='train',streaming=True,revision=rev)
    train=[];val=[];ntrain=nval=0;seen=set();tdocs=[];vdocs=[]
    for row in stream:
        text=row['text'];h=hashlib.sha256(text.encode()).hexdigest()
        if h in seen:continue
        seen.add(h);is_val=int(h[:8],16)%20==0
        if (is_val and nval>=250_000) or (not is_val and ntrain>=25_100_000):continue
        tokens=enc.encode_ordinary(text)+[50256]
        if is_val:val.extend(tokens);nval+=len(tokens);vdocs.append(h)
        else:train.extend(tokens);ntrain+=len(tokens);tdocs.append(h)
        if ntrain>=25_100_000 and nval>=250_000:break
    if ntrain<25_100_000 or nval<250_000:raise RuntimeError('insufficient source data')
    meta={'dataset':'HuggingFaceFW/fineweb-edu','config':'sample-10BT','revision':rev,
          'split_rule':'first8hex(SHA256(UTF8(document))) mod20 ==0 gives validation; duplicate document hashes skipped',
          'tokenizer':'gpt2','eos_between_documents':50256}
    for name,tokens,docs in [('train',train[:25_100_000],tdocs),('validation',val[:250_000],vdocs)]:
        a=np.asarray(tokens,dtype=np.uint16);a.tofile(DATA/(name+'.bin'))
        meta[name]={'tokens':len(a),'token_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                    'document_count':len(docs),'document_hash_root':hashlib.sha256('\n'.join(sorted(docs)).encode()).hexdigest()}
    if set(tdocs)&set(vdocs):raise AssertionError('document overlap')
    dump(DATA/'DATA.json',meta)
    return meta

def batches(data,seed,steps):
    rng=np.random.default_rng(seed)
    for _ in range(steps):
        starts=rng.integers(0,len(data)-CONTEXT-1,size=BATCH)
        z=np.stack([data[p:p+CONTEXT+1] for p in starts]).astype(np.int64)
        yield torch.from_numpy(z[:,:-1]).cuda(),torch.from_numpy(z[:,1:]).cuda()

@torch.no_grad()
def evaluate(model,val):
    model.eval();ls=[]
    for x,y in batches(val,34_000_000,16):
        with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
        ls.append(float(F.cross_entropy(z.float().reshape(-1,VOCAB),y.reshape(-1))))
    model.train();return float(np.mean(ls))

def main():
    OUT.mkdir(exist_ok=True,parents=True);torch.set_num_threads(2)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if (ROOT/'status.json').exists() and not json.loads((ROOT/'status.json').read_text()).get('complete'):
        raise RuntimeError('prior GPU queue still active')
    recipe={'width':192,'layers':24,'context':CONTEXT,'batch':BATCH,'steps':STEPS,'tokens':STEPS*BATCH*CONTEXT,
            'seed':SEED,'optimizer':'AdamW','lr':.0003,'betas':[.9,.95],'wd':.1,'warmup_steps':max(1,STEPS//50),
            'end_lr':.00003,'arms':['transformer','attention8','conv7'],'retries':0,'resume':False,
            'interpretation':'ONE-SEED DEVELOPMENT SCREEN; no 100M/2B or breakthrough inference',
            'gate':'Conv7 NLL-T <= .015 AND TPS/T >=1.03; mechanism benefit requires attention8 NLL-Conv7 NLL >=.005'}
    dump(OUT/'PREREGISTRATION.json',recipe)
    meta=prepare();train=np.fromfile(DATA/'train.bin',dtype=np.uint16);val=np.fromfile(DATA/'validation.bin',dtype=np.uint16)
    for arm in recipe['arms']:
        ident=f'conv7-recovery-w192-{arm}-{SEED}';torch.manual_seed(SEED)
        model=ScheduledLM(192,arm).cuda()
        if count(model)>100_000_000 or recipe['tokens']>2_000_000_000:raise ValueError('USER_CEILING')
        record={'attempt':ident,'recipe':recipe,'parameters':count(model),'seed':SEED,'variant':arm,'width':192,
                'data':meta,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'model_source_sha256':hashlib.sha256((ROOT/'conv7_language.py').read_bytes()).hexdigest()}
        dump(OUT/(ident+'.ATTEMPT.json'),record)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,betas=(.9,.95),weight_decay=.1)
        torch.cuda.reset_peak_memory_stats();seen=0;seconds=0;history=[]
        initial=evaluate(model,val)
        for step,(x,y) in enumerate(batches(train,33_000_000,STEPS)):
            if seen+x.numel()>2_000_000_000 or seen+x.numel()>25_000_000:raise ValueError('TOKEN_CEILING')
            warm=recipe['warmup_steps']
            lr=.0003*(step+1)/warm if step<warm else .00003+.00027*.5*(1+math.cos(math.pi*(step-warm)/(STEPS-1-warm)))
            for group in optimizer.param_groups:group['lr']=lr
            sync();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
            loss=F.cross_entropy(logits.float().reshape(-1,VOCAB),y.reshape(-1))
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            sync();seconds+=time.perf_counter()-start;seen+=x.numel()
            if (step+1)%1024==0 or step==STEPS-1:
                history.append({'step':step+1,'tokens':seen,'training_seconds':seconds,'nll':evaluate(model,val)})
                print(ident,json.dumps(history[-1]),flush=True)
                (OUT/(ident+'.PROGRESS.json')).write_text(json.dumps({'tokens':seen,'history':history}))
            if seconds>1800:raise TimeoutError('30_MINUTE_MODEL_HARD_STOP')
        record.update({'tokens_seen':seen,'validation_nll':history[-1]['nll'],'initial_validation_nll':initial,
                       'train_seconds':seconds,'train_tokens_per_second':seen/seconds,
                       'history':history,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                       'classification':'FRESH_ONE_SEED_DEVELOPMENT_SCREEN','breakthrough_demonstrated':False})
        path=OUT/(ident+'.pt')
        with path.open('xb') as f:torch.save(model.state_dict(),f)
        record['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        dump(OUT/(ident+'.RESULT.json'),record);print('RESULT',json.dumps(record),flush=True)
        del model,optimizer,logits,loss;gc.collect();torch.cuda.empty_cache()
    (OUT/'COMPLETE.json').write_text(json.dumps({'complete':True}))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True)
        dump(OUT/'ERROR.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
