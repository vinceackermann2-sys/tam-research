"""Post-hoc localization after initial long-distance failures; no optimizer.

The encoder sees independent fixed-size chunks. For paired prompts differing
only in the first SET, later input-conditioned gates and writes are identical.
State differences can therefore only survive through the diagonal gate products.
This measures actual checkpoint state separation rather than hypothesizing it.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
from experiments.rlt.model_streaming_bounded import BoundedChunkGatedRLT,tiny_streaming_config
from experiments.rlt.cpu_lastwrite_probe_ar import SET0,QUERY
ROOT=Path(__file__).parent;OUT=ROOT/'results/rlt_state';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
    with (OUT/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

@torch.no_grad()
def main():
    assert (ROOT/'results/rlt_gate/COMPLETE.json').exists(),'Wait for all six predeclared models'
    for seed in (202610311,202610312,202610313):
        for arm in ('native','public_gate_prior'):
            ck=ROOT/'results/rlt_gate'/f'rlt-{arm}-{seed}.pt';model=BoundedChunkGatedRLT(tiny_streaming_config());model.load_state_dict(torch.load(ck,weights_only=True));model.eval()
            g=torch.Generator().manual_seed(73_000_000);x=torch.randint(4,16,(32,1024),generator=g).repeat_interleave(2,0);y=torch.tensor([0,1]*32);x[:,0]=SET0+y;x[:,-1]=QUERY
            state=model.init_stream(64);initial=None;log_products=torch.zeros(64,32);rows=[]
            for start in range(0,1024,8):
                chunk=x[:,start:start+8];memory=model.encode(chunk)
                gate=torch.sigmoid(F.linear(memory,model.merge.weight[:,32:])+model.GATE_LOGIT_BIAS)
                if start>=8:
                    assert torch.equal(gate[::2],gate[1::2]);log_products+=gate.log().sum(1)
                logits,state=model.forward_chunk(chunk,state)
                delta=(state.recurrent[::2]-state.recurrent[1::2]).abs()
                if start==0:initial=delta.clone()
                if start+8 in (128,256,512,1024):
                    predicted=initial*log_products[::2].exp()
                    rows.append({'length':start+8,'mean_absolute_bit_state_difference':float(delta.mean()),'max_absolute_bit_state_difference':float(delta.max()),
                        'exactly_identical_pair_fraction':float((delta.max(-1).values==0).float().mean()),
                        'mean_largest_gate_survival_product':float(log_products[::2].exp().max(-1).values.mean()),
                        'mean_theoretical_bit_state_difference':float(predicted.mean()),
                        'interpretation':'tiny state separation relative to accumulated filler writes may be lost numerically; readout accuracy is separately measured'})
            dump(f'rlt-state-{arm}-{seed}.DIAGNOSTIC.json',{'seed':seed,'arm':arm,'rows':rows,'checkpoint_sha256':sha(ck),'source_sha256':sha(Path(__file__)),
                'optimizer_updates':0,'classification':'POSTHOC_STATE_ATTENUATION_DIAGNOSTIC','new_examples_namespace':73000000})
            print(seed,arm,json.dumps(rows[-1]),flush=True)
    dump('COMPLETE.json',{'complete':True})
if __name__=='__main__':main()
