"""Post-hoc component restoration on already consumed SI evaluation samples.

Restore one untrained baseline head into the repaired system, keeping all other
weights fixed. This diagnoses reliance with coadaptation/distribution-shift
limits; it does not attribute the benefit to an isolated training intervention.
No optimizer updates, no new held-out efficacy claim.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
from tam_research.si_original_world_scoped_v2 import SIOriginalWorldScopedV2
from tam_research.si_original_crossmodal_learning_v1 import _public_batch
ROOT=Path(__file__).parent;OUT=ROOT/'results/si_components';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
    with (OUT/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

@torch.no_grad()
def main():
    assert (ROOT/'results/si_repair/COMPLETE.json').exists()
    for seed in (202610301,202610302,202610303):
        base=ROOT/'results/si_repair'/f'si-{seed}-policy.pt';repair=ROOT/'results/si_repair'/f'si-{seed}-repair.pt'
        baseline=torch.load(base,weights_only=False,map_location='cpu');repaired=torch.load(repair,weights_only=False,map_location='cpu')
        model=SIOriginalWorldScopedV2(width=24,layers=4,experts=4,max_steps=3);model.load_state_dict(repaired['weights']);model.eval();batches=[]
        for i in range(64):
            x,y,_=_public_batch(62_000_000+i,32);o=model(x,world='calibration',enable_memory=False,enable_fast=False,enable_search=False)
            batches.append((o['state'].clone(),o['direct_action'].clone(),y))
        rows=[]
        for condition in ('baseline','repaired','restore_transition','restore_reward_head','restore_value_head','restore_verifier'):
            weights=dict(repaired['weights'])
            if condition=='baseline':weights=baseline['weights']
            elif condition.startswith('restore_'):
                prefix=condition.removeprefix('restore_')+'.'
                weights={k:baseline['weights'][k] if k.startswith(prefix) else v for k,v in weights.items()}
            model.load_state_dict(weights);correct=greedy=gated=searched=0
            for state,direct,y in batches:
                pred,_=model._search(state);scores=torch.stack([model._world_step(state,j)[1] for j in range(4)],-1)
                flags=torch.sigmoid(model.verifier(state)).flatten()>.5
                correct+=int((pred==y).sum());greedy+=int((scores.argmax(-1)==y).sum())
                gated+=int((torch.where(flags,pred,direct)==y).sum());searched+=int(flags.sum())
            rows.append({'condition':condition,'forced_depth2_accuracy':correct/2048,'forced_depth1_accuracy':greedy/2048,'gated_accuracy':gated/2048,'searched_fraction':searched/2048})
        old=json.loads((ROOT/'results/si_repair'/f'si-{seed}.RESULT.json').read_text())
        assert rows[0]['forced_depth2_accuracy']==old['baseline']['accuracy']['forced_depth2']
        assert rows[1]['forced_depth2_accuracy']==old['repaired']['accuracy']['forced_depth2']
        dump(f'si-components-{seed}.DIAGNOSTIC.json',{'seed':seed,'rows':rows,'baseline_checkpoint_sha256':sha(base),'repaired_checkpoint_sha256':sha(repair),
            'source_sha256':sha(Path(__file__)),'optimizer_updates':0,'evaluation_samples_reused':True,'representation_fixed_and_identical':True})
        print(seed,json.dumps(rows),flush=True)
    dump('COMPLETE.json',{'complete':True})
if __name__=='__main__':main()
