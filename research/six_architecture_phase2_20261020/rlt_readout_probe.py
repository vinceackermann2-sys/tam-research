import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
from experiments.rlt.model_streaming_bounded import BoundedChunkGatedRLT,BoundedStreamState,tiny_streaming_config
from experiments.rlt.model_gated_scan import associative_affine_scan
from experiments.rlt.cpu_lastwrite_probe_ar import SET0,SET1,QUERY
ROOT=Path(__file__).parent;OUT=ROOT/'results/rlt_probe';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
    with (OUT/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def samples(length,namespace):
    g=torch.Generator().manual_seed(namespace+length);x=torch.randint(4,16,(256,length),generator=g).repeat_interleave(2,0)
    y=torch.tensor([0,1]*256);x[:,0]=SET0+y;x[:,-1]=QUERY;order=torch.randperm(512,generator=g)
    return x[order],y[order]

@torch.no_grad()
def state_and_logits(model,x,carry=True):
    b,length=x.shape;state=model.init_stream(b);model.carry_between_chunks=carry
    for start in range(0,length-8,8):
        memory=model.encode(x[:,start:start+8]);d=memory.size(-1)
        keep=torch.sigmoid(F.linear(memory,model.merge.weight[:,d:])+model.GATE_LOGIT_BIAS)
        writes=(1-keep)*torch.tanh(F.linear(memory,model.merge.weight[:,:d]))
        previous=state.recurrent if carry else model.start_state.expand(b,-1)
        value=associative_affine_scan(keep,writes,previous.float())[:,-1]
        state=BoundedStreamState(value,start+8)
    logits,final=model.forward_chunk(x[:,-8:],state)
    return final.recurrent,logits[:,-1]

@torch.no_grad()
def features(model,namespace,carry=True):
    rows=[];states=[];labels=[]
    for length in (128,256,512,1024):
        x,y=samples(length,namespace);ss=[];ls=[]
        for z in x.split(64):
            s,l=state_and_logits(model,z,carry);ss.append(s);ls.append(l)
        s=torch.cat(ss);l=torch.cat(ls);states.append(s);labels.append(y)
        pred=l[:,SET0:SET1+1].argmax(-1)
        rows.append({'length':length,'original_accuracy':float((pred==y).float().mean()),'state_first':sum(len(v) for v in states[:-1]),'count':len(y)})
    return torch.cat(states),torch.cat(labels),rows

def main():
    dump('PREREGISTRATION.json',{'protocol_sha256':sha(ROOT/'RLT_PROBE_PROTOCOL.md'),'source_sha256':sha(Path(__file__))})
    for seed in (202610311,202610312,202610313):
        for arm in ('native','public_gate_prior'):
            ck=ROOT/'results/rlt_gate'/f'rlt-{arm}-{seed}.pt';model=BoundedChunkGatedRLT(tiny_streaming_config());model.load_state_dict(torch.load(ck,weights_only=True));model.eval()
            for param in model.parameters():param.requires_grad_(False)
            fixture,_=samples(128,75_000_000)
            with torch.no_grad():
                model.carry_between_chunks=True;expected=model(fixture[:4])[:,-1];_,actual=state_and_logits(model,fixture[:4]);delta=float((expected-actual).abs().max());assert delta<1e-6
            started=time.monotonic();train,y,_=features(model,76_000_000);mean=train.mean(0);std=train.std(0).clamp_min(1e-6);z=(train-mean)/std
            torch.manual_seed(seed+1337);probe=nn.Linear(32,2);opt=torch.optim.AdamW(probe.parameters(),lr=.03,weight_decay=0)
            params=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in probe.parameters());assert params<=100_000_000
            for step in range(1000):
                loss=F.cross_entropy(probe(z),y);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            evaluation,target,rows=features(model,77_000_000);erased,ey,_=features(model,77_000_000,False);assert torch.equal(target,ey)
            with torch.no_grad():pred=probe((evaluation-mean)/std).argmax(-1);ep=probe((erased-mean)/std).argmax(-1)
            for row in rows:
                sl=slice(row['state_first'],row['state_first']+row['count']);truth=target[sl];pr=pred[sl];er=ep[sl]
                row.update({'probe_accuracy':float((pr==truth).float().mean()),'class0':float((pr[truth==0]==0).float().mean()),'class1':float((pr[truth==1]==1).float().mean()),'erased_carry_probe_accuracy':float((er==truth).float().mean())})
                assert abs(row['erased_carry_probe_accuracy']-.5)<=1/512
            path=OUT/f'rlt-probe-{arm}-{seed}.pt';torch.save({'probe':probe.state_dict(),'mean':mean,'std':std,'optimizer':opt.state_dict(),'base_checkpoint_sha256':sha(ck)},path)
            record={'seed':seed,'arm':arm,'parameters_including_frozen_backbone':params,'accounted_training_positions':3_031_040,'rows':rows,'integration_max_logit_delta':delta,
                'gate_passed':all(r[c]>=.95 for r in rows for c in ('class0','class1')),'source_sha256':sha(Path(__file__)),'checkpoint_sha256':sha(path),
                'base_checkpoint_sha256':sha(ck),'seconds':time.monotonic()-started,'scope':'frozen-state long-context supervised linear probe; not unmodified model extrapolation','breakthrough_demonstrated':False}
            dump(f'rlt-probe-{arm}-{seed}.RESULT.json',record);print('RESULT',json.dumps(record),flush=True)
    dump('COMPLETE.json',{'complete':True})
if __name__=='__main__':main()
