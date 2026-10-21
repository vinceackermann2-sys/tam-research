"""Source-faithful scalable Conv7/attention8/full-attention development panel.

This is a NEW width-generalized implementation. It never alters frozen code.
See PROTOCOL.md. The 1M-token pilot is not a 100M/2B scientific result.
"""
from __future__ import annotations
import argparse, gc, hashlib, io, json, math, os, platform, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MAX_PARAMS=100_000_000
MAX_TOKENS=2_000_000_000
SEEDS=(202610201,202610202,202610203)
VOCAB=50257
LAYERS=24
CONTEXT=128
POSITIONS=1024
SCHEDULE=tuple(range(3,25,3))

def count(model):return sum(p.numel() for p in model.parameters())
def dump(path,data):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False)
def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()

class Attention(nn.Module):
    def __init__(self,d):
        super().__init__();self.d=d;self.heads=8
        self.qkv=nn.Linear(d,3*d,bias=False);self.out=nn.Linear(d,d,bias=False)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=[z.view(b,t,self.heads,d//self.heads).transpose(1,2) for z in self.qkv(x).chunk(3,-1)]
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.out(y.transpose(1,2).contiguous().view(b,t,d))

class Block(nn.Module):
    def __init__(self,d,ff,attention,conv):
        super().__init__()
        self.norm_attention=nn.LayerNorm(d) if attention else None
        self.attention=Attention(d) if attention else None
        self.norm_ff=nn.LayerNorm(d)
        self.feedforward=nn.Sequential(nn.Linear(d,ff,bias=False),nn.GELU(),nn.Linear(ff,d,bias=False))
        self.conv=nn.Conv1d(d,d,7,groups=d,bias=False) if conv else None
        if conv:self.conv_gain=nn.Parameter(torch.zeros(()))
    def forward(self,x):
        if self.attention is not None:x=x+self.attention(self.norm_attention(x))
        if self.conv is not None:
            z=self.norm_ff(x).transpose(1,2)
            x=x+self.conv_gain*self.conv(F.pad(z,(6,0))).transpose(1,2)
        return x+self.feedforward(self.norm_ff(x))

class ScheduledLM(nn.Module):
    def __init__(self,d,variant):
        super().__init__();self.d=d;self.variant=variant
        attention_count=24 if variant=='transformer' else 8
        # Match the full-attention/4D-FFN model using LIVE FFN capacity.
        saved=(24-attention_count)*(4*d*d+2*d)
        added=(16*(7*d+1)) if variant=='conv7' else 0
        ff=4*d+round((saved-added)/(24*2*d))
        self.ff=ff
        self.token_emb=nn.Embedding(VOCAB,d);self.pos_emb=nn.Embedding(POSITIONS,d)
        self.blocks=nn.ModuleList([Block(d,ff,variant=='transformer' or i+1 in SCHEDULE,
                                       variant=='conv7' and i+1 not in SCHEDULE) for i in range(24)])
        self.norm=nn.LayerNorm(d);self.lm_head=nn.Linear(d,VOCAB,bias=False);self.lm_head.weight=self.token_emb.weight
        self.apply(self.init_weights)
    @staticmethod
    def init_weights(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,mean=0,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None:nn.init.zeros_(m.bias)
    def forward(self,tokens):
        if tokens.size(1)>POSITIONS:raise ValueError('context cap')
        x=self.token_emb(tokens)+self.pos_emb(torch.arange(tokens.size(1),device=tokens.device))[None]
        for b in self.blocks:x=b(x)
        return self.lm_head(self.norm(x))

def preflight():
    records=[]
    for d in (128,504):
        for name in ('transformer','attention8','conv7'):
            with torch.device('meta'):m=ScheduledLM(d,name)
            n=count(m)
            if n>MAX_PARAMS:raise ValueError('parameter cap violated')
            records.append({'width':d,'variant':name,'parameters':n,'ff_hidden':m.ff})
    torch.manual_seed(89123);m=ScheduledLM(16,'conv7');m.eval()
    x=torch.randint(VOCAB,(1,12));changed=x.clone();changed[:,6:]=torch.randint(VOCAB,(1,6))
    with torch.no_grad():
        delta=float((m(x)[:,:6]-m(changed)[:,:6]).abs().max())
    if delta>1e-6:raise AssertionError('future leakage')
    optimizer=torch.optim.SGD(m.parameters(),lr=.01)
    convolution=m.blocks[0].conv
    grads=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(m(x).reshape(-1,VOCAB),torch.roll(x,-1,1).reshape(-1))
        loss.backward();grads.append(float(convolution.weight.grad.norm()));optimizer.step()
    if grads[0]!=0 or grads[1]<=0:raise AssertionError('zero-gate wakeup failure')
    return {'parameter_counts':records,'future_leakage_max':delta,'conv_weight_gradient_steps':grads}

def prepare_data(folder):
    import requests, pyarrow.parquet as pq, tiktoken
    folder.mkdir(parents=True,exist_ok=True)
    meta_path=folder/'DATA.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        for split in ('train','validation'):
            raw=(folder/(split+'.bin')).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=meta[split]['token_sha256']:raise ValueError('token file drift')
        return meta
    api=requests.get('https://huggingface.co/api/datasets/Salesforce/wikitext',timeout=30);api.raise_for_status()
    rev=api.json()['sha'];enc=tiktoken.get_encoding('gpt2');meta={'dataset':'Salesforce/wikitext','config':'wikitext-2-raw-v1','revision':rev,'tokenizer':'gpt2','tokenizer_vocab':enc.n_vocab}
    for split in ('train','validation'):
        url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{rev}/wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
        r=requests.get(url,timeout=120);r.raise_for_status()
        texts=pq.read_table(io.BytesIO(r.content),columns=['text'])['text'].to_pylist()
        stream='\n'.join(texts)
        tokens=np.asarray(enc.encode_ordinary(stream),dtype=np.uint16)
        tokens.tofile(folder/(split+'.bin'))
        meta[split]={'tokens':len(tokens),'text_sha256':hashlib.sha256(stream.encode()).hexdigest(),'parquet_sha256':hashlib.sha256(r.content).hexdigest(),'token_sha256':hashlib.sha256(tokens.tobytes()).hexdigest()}
    dump(meta_path,meta);return meta

def batches(data,seed,steps=512):
    g=np.random.default_rng(seed)
    for _ in range(steps):
        positions=g.integers(0,len(data)-CONTEXT-1,size=16)
        z=np.stack([data[p:p+CONTEXT+1] for p in positions]).astype(np.int64)
        yield torch.from_numpy(z[:,:-1]),torch.from_numpy(z[:,1:])

@torch.no_grad()
def evaluate(model,data,device,amp_dtype):
    model.eval();losses=[]
    for x,y in batches(data,8_000_000,8):
        x=x.to(device);y=y.to(device)
        with torch.autocast('cuda',dtype=amp_dtype):z=model(x)
        losses.append(float(F.cross_entropy(z.float().reshape(-1,VOCAB),y.reshape(-1))))
    model.train();return sum(losses)/len(losses)

def train(out,data_dir,pre):
    if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED_NO_CPU_FALLBACK')
    device=torch.device('cuda');dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    meta=prepare_data(data_dir);train_data=np.fromfile(data_dir/'train.bin',dtype=np.uint16);val=np.fromfile(data_dir/'validation.bin',dtype=np.uint16)
    for seed in SEEDS:
        for name in ('transformer','attention8','conv7'):
            identity=f'conv7-w128-{name}-{seed}'
            torch.manual_seed(seed);model=ScheduledLM(128,name).to(device)
            if count(model)>MAX_PARAMS:raise ValueError('PARAMETER_CAP')
            record={'attempt':identity,'variant':name,'parameters':count(model),'width':128,'layers':24,
                'ff_hidden':model.ff,'seed':seed,'updates':512,'batch':16,'context':128,'planned_tokens':1_048_576,
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'data':meta,
                'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'precision':str(dtype),
                'preflight':pre,'breakthrough_demonstrated':False,'classification':'EXPLORATORY_LANGUAGE_PILOT'}
            dump(out/(identity+'.ATTEMPT.json'),record)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.95),weight_decay=.1)
            scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
            torch.cuda.reset_peak_memory_stats();seen=0;history=[];seconds=0
            initial=evaluate(model,val,device,dtype)
            for step,(x,y) in enumerate(batches(train_data,8_100_000+seed,512)):
                if seen+x.numel()>MAX_TOKENS:raise ValueError('TOKEN_CAP')
                lr=.001*(step+1)/10 if step<10 else .0001+.0009*.5*(1+math.cos(math.pi*(step-10)/(511-10)))
                for group in opt.param_groups:group['lr']=lr
                x=x.to(device);y=y.to(device);sync();start=time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=dtype):z=model(x)
                loss=F.cross_entropy(z.float().reshape(-1,VOCAB),y.reshape(-1))
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
                scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt);scaler.update();sync();seconds+=time.perf_counter()-start;seen+=x.numel()
                if (step+1)%128==0:
                    nll=evaluate(model,val,device,dtype)
                    history.append({'step':step+1,'tokens':seen,'nll':nll,'training_seconds':seconds})
                    print(identity,json.dumps(history[-1]),flush=True)
                if seconds>1800:raise TimeoutError('30_MINUTE_MODEL_CAP')
            record.update({'tokens_seen':seen,'initial_validation_nll':initial,'validation_nll':history[-1]['nll'],
                'train_seconds':seconds,'train_tokens_per_second':seen/seconds,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'history':history})
            checkpoint=out/(identity+'.pt')
            with checkpoint.open('xb') as f:torch.save(model.state_dict(),f)
            record['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            dump(out/(identity+'.RESULT.json'),record);print('RESULT',json.dumps(record),flush=True)
            del model,opt,z,loss;gc.collect();torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--data',required=True);p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    try:
        pre=preflight();print('PREFLIGHT',json.dumps(pre),flush=True)
        if not (out/'PREFLIGHT.json').exists():dump(out/'PREFLIGHT.json',pre)
        if not a.preflight_only:train(out,Path(a.data),pre)
    except BaseException as exc:
        dump(out/'ERROR.json',{'error':repr(exc),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
