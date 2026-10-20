import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
from experiments.rlt.model_streaming_bounded import BoundedChunkGatedRLT,tiny_streaming_config
from experiments.rlt.cpu_lastwrite_probe_ar import make_batch,SET0,SET1,QUERY,last_write_labels
ROOT=Path(__file__).parent;OUT=ROOT/'results/rlt_gate';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,v):
    with (OUT/name).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
class Instrumented(BoundedChunkGatedRLT):
    def encode(self,x):
        m=super().encode(x)
        if self.training:self.memories.append(m)
        return m
    def forward(self,x):self.memories=[];return super().forward(x)

panels={}
for length in (128,256,512,1024):
    g=torch.Generator().manual_seed(72_000_000+length);prefix=torch.randint(4,16,(128,length),generator=g);prefix[:,-1]=QUERY
    x=prefix.repeat_interleave(2,0);y=torch.tensor([0,1]*128);x[:,0]=SET0+y
    assert torch.equal(last_write_labels(x),y);panels[length]=(x,y)

@torch.no_grad()
def evaluate(model):
    model.eval();rows=[]
    for keep in (True,False):
        model.carry_between_chunks=keep
        for length,(x,y) in panels.items():
            ls=torch.cat([model(z)[:,-1,SET0:SET1+1] for z in x.split(32)]);pred=ls.argmax(-1)
            row={'length':length,'carry':keep,'accuracy':float((pred==y).float().mean()),
                'class0':float((pred[y==0]==0).float().mean()),'class1':float((pred[y==1]==1).float().mean()),'nll':float(F.cross_entropy(ls,y))}
            if not keep:assert row['accuracy']==.5
            rows.append(row)
    model.carry_between_chunks=True;return rows

def main():
    dump('PREREGISTRATION.json',{'protocol_sha256':sha(ROOT/'RLT_PROTOCOL.md'),'source_sha256':sha(Path(__file__))})
    for seed in (202610311,202610312,202610313):
        for arm in ('native','public_gate_prior'):
            torch.manual_seed(seed);model=Instrumented(tiny_streaming_config());params=sum(p.numel() for p in model.parameters());assert params<=100_000_000
            opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01);start=time.monotonic();seen=0
            for stage,n,lr,length,gaps in [('easy',2100,.003,8,(1,)),('long',1200,.001,128,(1,4,16,64,127))]:
                for group in opt.param_groups:group['lr']=lr
                for step in range(n):
                    x,y,_=make_batch(8,gaps,(70_000_000 if stage=='easy' else 71_000_000)+step,seq_len=length)
                    model.train();opt.zero_grad(set_to_none=True);logits=model(x)[:,-1,SET0:SET1+1]
                    memory=torch.cat(model.memories,1);gate_logits=F.linear(memory,model.merge.weight[:,model.cfg.d_model:])+model.GATE_LOGIT_BIAS
                    keep=((x!=SET0)&(x!=SET1)).to(memory.dtype).unsqueeze(-1).expand_as(gate_logits)
                    auxiliary=F.binary_cross_entropy_with_logits(gate_logits,keep)
                    loss=F.cross_entropy(logits,y)+(.01 if arm=='public_gate_prior' else 0.)*auxiliary
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();seen+=x.numel()
                    if (step+1)%500==0:print(seed,arm,stage,step+1,float(loss.detach()),flush=True)
            rows=evaluate(model);ck=OUT/f'rlt-{arm}-{seed}.pt';torch.save(model.state_dict(),ck)
            record={'seed':seed,'arm':arm,'parameters':params,'training_exposures':seen,'rows':rows,'seconds':time.monotonic()-start,
                'gate_passed':all(r[c]>=.95 for r in rows if r['carry'] for c in ('class0','class1')),
                'checkpoint_sha256':sha(ck),'source_sha256':sha(Path(__file__)),'extra_visible_command_supervision':arm!='native','breakthrough_demonstrated':False}
            dump(f'rlt-{arm}-{seed}.RESULT.json',record);print('RESULT',json.dumps(record),flush=True)
    dump('COMPLETE.json',{'complete':True})
if __name__=='__main__':main()
