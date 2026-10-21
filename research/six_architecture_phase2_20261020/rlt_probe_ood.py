import argparse,hashlib,json,sys
from pathlib import Path
import torch
from torch import nn
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
from rlt_readout_probe import state_and_logits
from experiments.rlt.model_streaming_bounded import BoundedChunkGatedRLT,tiny_streaming_config
from experiments.rlt.cpu_lastwrite_probe_ar import make_batch,last_write_labels,SET0,SET1,QUERY
ROOT=Path(__file__).parent;OUT=ROOT/'results/rlt_probe_ood';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
    with (OUT/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

def panels():
    cases=[]
    for length in (2048,4096):
        g=torch.Generator().manual_seed(78_000_000+length);x=torch.randint(4,16,(128,length),generator=g).repeat_interleave(2,0)
        y=torch.tensor([0,1]*128);x[:,0]=SET0+y;x[:,-1]=QUERY;order=torch.randperm(256,generator=g)
        cases.append((f'first_set_length{length}',length-1,x[order],y[order]))
    for gap in (1,16,127,511,1023):
        base,_,_=make_batch(64,(gap,),79_000_000+gap,seq_len=1024);x=base.repeat_interleave(2,0);y=torch.tensor([0,1]*64);x[:,1023-gap]=SET0+y
        assert torch.equal(last_write_labels(x),y);order=torch.randperm(128,generator=torch.Generator().manual_seed(79_100_000+gap))
        cases.append((f'last_write_gap{gap}',gap,x[order],y[order]))
    return cases

@torch.no_grad()
def main():
    assert (ROOT/'results/rlt_probe/COMPLETE.json').exists();cases=panels()
    dump('PREREGISTRATION.json',{'protocol_sha256':sha(ROOT/'RLT_PROBE_OOD_PROTOCOL.md'),'source_sha256':sha(Path(__file__)),'training_exposures':0})
    for seed in (202610311,202610312,202610313):
        for arm in ('native','public_gate_prior'):
            base=ROOT/'results/rlt_gate'/f'rlt-{arm}-{seed}.pt';head=ROOT/'results/rlt_probe'/f'rlt-probe-{arm}-{seed}.pt'
            model=BoundedChunkGatedRLT(tiny_streaming_config());model.load_state_dict(torch.load(base,weights_only=True));model.eval()
            saved=torch.load(head,weights_only=False);probe=nn.Linear(32,2);probe.load_state_dict(saved['probe']);rows=[]
            for name,gap,x,y in cases:
                states=[];original=[];erased=[]
                for part in x.split(64):
                    s,l=state_and_logits(model,part,True);states.append(s);original.append(l[:,SET0:SET1+1].argmax(-1))
                    es,_=state_and_logits(model,part,False);erased.append(es)
                z=torch.cat(states);ez=torch.cat(erased);pr=probe((z-saved['mean'])/saved['std']).argmax(-1);ep=probe((ez-saved['mean'])/saved['std']).argmax(-1)
                row={'case':name,'gap':gap,'examples':len(y),'original_accuracy':float((torch.cat(original)==y).float().mean()),'probe_accuracy':float((pr==y).float().mean()),
                    'class0':float((pr[y==0]==0).float().mean()),'class1':float((pr[y==1]==1).float().mean()),'erased_carry_probe_accuracy':float((ep==y).float().mean())}
                if gap>=16:assert abs(row['erased_carry_probe_accuracy']-.5)<=1/len(y)
                rows.append(row)
            record={'seed':seed,'arm':arm,'rows':rows,'gate_passed':all(r[c]>=.95 for r in rows for c in ('class0','class1')),
                'base_checkpoint_sha256':sha(base),'probe_checkpoint_sha256':sha(head),'source_sha256':sha(Path(__file__)),'training_exposures':0,'breakthrough_demonstrated':False}
            dump(f'rlt-probe-ood-{arm}-{seed}.DIAGNOSTIC.json',record);print(seed,arm,[(r['case'],r['probe_accuracy']) for r in rows],flush=True)
    dump('COMPLETE.json',{'complete':True})
if __name__=='__main__':main()
