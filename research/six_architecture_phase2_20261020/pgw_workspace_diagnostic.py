import argparse,hashlib,itertools,json,math,sys,types
from pathlib import Path
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
from tam_research.pgw_v3_stage2a.model import PGWV3Stage2A,batch_verified_examples
from tam_research.pgw_v3_stage1b.oracle import make_example,POSITIONS,OVERWRITES,DELAYS
ROOT=Path(__file__).parent;OLD=ROOT.parent/'research_20261020';OUT=ROOT/'results/pgw_trace';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
    with (OUT/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

@torch.no_grad()
def main():
    dump('PREREGISTRATION.json',{'protocol_sha256':sha(ROOT/'MEMORY_DIAGNOSTICS_PROTOCOL.md'),'source_sha256':sha(Path(__file__))})
    factors=list(itertools.product(POSITIONS,OVERWRITES,(False,True),(False,True)))
    for seed in (202610201,202610202,202610203):
        model=PGWV3Stage2A();ck=OLD/'results/pgw'/f'pgw-hybrid-{seed}.pt';model.load_state_dict(torch.load(ck,weights_only=True));model.eval();rows=[]
        workspace=model.workspace;original_write=workspace._write
        for delay in DELAYS:
            examples=[make_example(split='validation',index=74_000_000+j,last_position=q[0],overwrite_count=q[1],interference=q[2],delay_chunks=delay,missing=q[3]) for j in range(180) for q in [factors[j%len(factors)]]]
            x,anchor,y=batch_verified_examples(examples);batch,length=x.shape;events=(length//8)*2
            coefficients=torch.zeros(batch,4,events);trace=[];reads=[]
            def write(self,event,keys,values):
                key=self.event_key(event);scores=torch.einsum('bd,bsd->bs',key,keys)/math.sqrt(self.cfg.key_width)
                slot=scores.argmax(-1);gate=torch.sigmoid(self.write_gate(event)).flatten();amount=F.one_hot(slot,4).float()*gate[:,None]
                coefficients.mul_(1-amount[:,:,None]);coefficients[:,:,len(trace)].add_(amount);trace.append((slot.clone(),gate.clone()))
                return original_write(event,keys,values)
            def read(self,chunk,keys,values):
                weights=F.softmax(self.query_key(chunk)@keys.transpose(1,2)/math.sqrt(self.cfg.key_width),-1)
                reads.append((weights[:,2:3]@coefficients).squeeze(1).clone())
                return self.read_out(weights@values)
            workspace._write=types.MethodType(write,workspace);workspace._read=types.MethodType(read,workspace)
            logits=model(x,anchor);pred=logits.argmax(-1);selected=workspace.last_stats['selected_positions']
            slots=torch.stack([s for s,g in trace[:-2]],1);coefficient=reads[-1]
            for i,e in enumerate(examples):
                row={'delay':delay,'missing':e.missing,'correct':int(pred[i])==int(y[i]),'slots_used':int(slots[i].unique().numel()),
                    'most_used_slot_fraction':float(F.one_hot(slots[i],4).sum(0).max()/slots.shape[1])}
                if not e.missing:
                    event_ids=[2*e.last_target_position+j for j in range(2) if int(selected[i,e.last_target_position,j])>=2]
                    row.update({'target_value_visible_to_selected_state':bool(event_ids),
                        'target_query_read_coefficient':float(coefficient[i,event_ids].sum()) if event_ids else 0.})
                rows.append(row)
        summary={}
        for delay in DELAYS:
            rs=[r for r in rows if r['delay']==delay];present=[r for r in rs if not r['missing']]
            summary[delay]={'accuracy':sum(r['correct'] for r in rs)/len(rs),
                'target_state_selected':sum(r['target_value_visible_to_selected_state'] for r in present)/len(present),
                'mean_target_query_coefficient':sum(r['target_query_read_coefficient'] for r in present)/len(present),
                'target_coefficient_below_1e_3':sum(r['target_query_read_coefficient']<1e-3 for r in present)/len(present),
                'mean_slots_used':sum(r['slots_used'] for r in rs)/len(rs),
                'mean_most_used_slot_fraction':sum(r['most_used_slot_fraction'] for r in rs)/len(rs)}
        dump(f'pgw-trace-{seed}.DIAGNOSTIC.json',{'seed':seed,'summary':summary,'rows':rows,'checkpoint_sha256':sha(ck),
            'optimizer_updates':0,'source_sha256':sha(Path(__file__)),'breakthrough_demonstrated':False})
        print(seed,json.dumps(summary),flush=True)
    dump('COMPLETE.json',{'complete':True})
if __name__=='__main__':main()
