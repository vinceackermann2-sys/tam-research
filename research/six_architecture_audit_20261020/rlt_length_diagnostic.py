"""Inference-only length/causal-state panel; see RLT_LENGTH_PROTOCOL.md."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.repo.resolve()))
from experiments.rlt.model_streaming_bounded import BoundedChunkGatedRLT,tiny_streaming_config
from experiments.rlt.cpu_lastwrite_probe_ar import SET0,SET1,QUERY,last_write_labels
root=Path(__file__).parent;out=root/'results'/'rlt_length';out.mkdir(exist_ok=True)
torch.set_num_threads(2)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(name,value):
    with (out/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
panels={};meta={}
for length in (256,512,1024):
    gen=torch.Generator().manual_seed(202610242+length)
    prefixes=torch.randint(4,16,(128,length),generator=gen);prefixes[:,-1]=QUERY
    x=prefixes.repeat_interleave(2,dim=0);y=torch.tensor([0,1]*128);x[:,0]=y+SET0
    assert torch.equal(last_write_labels(x),y)
    assert torch.equal(x[::2,1:],x[1::2,1:])
    panels[length]=(x,y);meta[length]={'clusters':128,'examples':256,'sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()}
dump('PANEL.json',{'panels':meta,'protocol_sha256':sha(root/'RLT_LENGTH_PROTOCOL.md'),'script_sha256':sha(Path(__file__)),'new_training_exposures':0})
for seed in (202610201,202610202,202610203):
    for arm in ('carry','bias4'):
        ck=root/'results'/'rlt'/f'rlt-{arm}-{seed}.pt'
        assert ck.exists(),'Wait for the original training queue to finish before this diagnostic.'
        model=BoundedChunkGatedRLT(tiny_streaming_config());model.GATE_LOGIT_BIAS=4. if arm=='bias4' else 2.
        model.load_state_dict(torch.load(ck,weights_only=True));model.eval();rows=[]
        with torch.no_grad():
            for keep in (True,False):
                model.carry_between_chunks=keep
                for length,(x,y) in panels.items():
                    logits=torch.cat([model(z)[:,-1,SET0:SET1+1] for z in x.split(64)]);pred=logits.argmax(-1)
                    acc=float((pred==y).float().mean())
                    if not keep and acc!=.5:raise AssertionError('erased-state pair leakage')
                    rows.append({'length':length,'gap':length-1,'carry_enabled':keep,'accuracy':acc,
                        'class0':float((pred[y==0]==0).float().mean()),'class1':float((pred[y==1]==1).float().mean()),
                        'nll':float(F.cross_entropy(logits,y)),'predictions':pred.tolist()})
        dump(f'rlt-length-{arm}-{seed}.DIAGNOSTIC.json',{'seed':seed,'arm':arm,'checkpoint_sha256':sha(ck),'rows':rows,
            'classification':'POST_TRAINING_DEVELOPMENT_LENGTH_AND_CAUSAL_STATE_DIAGNOSTIC','breakthrough_demonstrated':False})
        print(seed,arm,[(r['gap'],r['carry_enabled'],r['accuracy']) for r in rows],flush=True)
