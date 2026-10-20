import argparse,hashlib,json,math,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
from tam_research.chm_v4_neural_hard_copy_train_1430 import TinyV4HardCopyPointer,hard_copy_prediction,encode_visible_v4,DIM
from tam_research.chm_v4_independent_binding_benchmark_1421 import generate_episode,seal_input,oracle_read,FAMILIES
ROOT=Path(__file__).parent;OLD=ROOT.parent/'research_20261020';OUT=ROOT/'results/chm_address';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
    with (OUT/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

@torch.no_grad()
def main():
    dump('PREREGISTRATION.json',{'protocol_sha256':sha(ROOT/'MEMORY_DIAGNOSTICS_PROTOCOL.md'),'source_sha256':sha(Path(__file__)),'test_opened':False})
    for seed in (202610271,202610272,202610273):
        ck=OLD/'results/chm_broad'/f'chm-broad-{seed}.pt';model=TinyV4HardCopyPointer();model.load_state_dict(torch.load(ck,weights_only=True));model.eval();rows=[]
        for family in FAMILIES:
            for entity in range(16):
                for variant in (range(8) if family!='no_match' else (0,)):
                    e=generate_episode('development',family,entity,variant);view=seal_input(e);encoded=encode_visible_v4(view)
                    out=model(view);pred=hard_copy_prediction(out);gold,link,value=oracle_read(e)
                    row={'family':family,'entity':entity,'variant':variant,'native_correct':pred.predicted_answer_id==gold,'native_present':not pred.did_abstain,
                        'gate_correct':(not pred.did_abstain)==(gold is not None)}
                    if gold is not None:
                        h=model.base(encoded.model_tokens);memory=h[:encoded.model_tokens.memory_length];question=h[-1]
                        starts=[pos for _,pos in encoded.model_tokens.line_anchors];ends=[(starts[k+1] if k+1<len(starts) else len(memory))-1 for k in range(len(starts))]
                        lines=memory[ends];first_physical=link if link is not None else value;first_index=encoded.first_source_physical.index(first_physical)
                        soft=torch.softmax(out.first_line_logits,0)@lines
                        replay=(memory[list(encoded.code_candidate_positions)]@model.second_query(question+soft))/math.sqrt(DIM)
                        assert torch.equal(replay,out.second_code_logits)
                        teacher=(memory[list(encoded.code_candidate_positions)]@model.second_query(question+lines[first_index]))/math.sqrt(DIM)
                        row.update({'first_correct':pred.first_source_physical==first_physical,
                            'gold_gate_only_correct':encoded.code_candidate_ids[int(replay.argmax())]==gold,
                            'gold_gate_and_first_context_correct':encoded.code_candidate_ids[int(teacher.argmax())]==gold})
                    rows.append(row)
        summary={f:{k:sum(r[k] for r in rows if r['family']==f)/sum(r['family']==f for r in rows)
            for k in (('native_correct','gate_correct') if f=='no_match' else ('native_correct','gate_correct','first_correct','gold_gate_only_correct','gold_gate_and_first_context_correct'))} for f in FAMILIES}
        record={'seed':seed,'summary':summary,'rows':rows,'checkpoint_sha256':sha(ck),'source_sha256':sha(Path(__file__)),
            'optimizer_updates':0,'development_reused':True,'test_opened':False,'gold_intervention_is_not_model_accuracy':True}
        dump(f'chm-address-{seed}.DIAGNOSTIC.json',record);print(seed,json.dumps(summary),flush=True)
    dump('COMPLETE.json',{'complete':True,'test_opened':False})
if __name__=='__main__':main()
