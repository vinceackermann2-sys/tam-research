"""Inference-only diagnostic; see CHM_GENERALIZATION_PROTOCOL.md."""
import argparse, hashlib, json, sys
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
from tam_research.chm_v4_neural_hard_copy_train_1430 import TinyV4HardCopyPointer,hard_copy_prediction
from tam_research.chm_v4_independent_binding_benchmark_1421 import generate_episode,seal_input,oracle_read,FAMILIES
root=Path(__file__).parent;out=root/'results'/'chm_unseen';out.mkdir(exist_ok=True)
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,d):
    with (out/name).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
episodes=[generate_episode('train',family,index,variant)
          for family in FAMILIES for index in range(64) if index!=48
          for variant in (range(8) if family!='no_match' else (0,))]
panel=hashlib.sha256('\n'.join(repr(e) for e in episodes).encode()).hexdigest()
write('PANEL.json',{'protocol_sha256':sha(root/'CHM_GENERALIZATION_PROTOCOL.md'),
      'source_sha256':sha(Path(__file__)),'panel_sha256':panel,'examples':len(episodes),
      'entities_per_family':63,'namespace':'native TRAIN, optimizer excluded entities',
      'reserved_development_test_opened':False,'new_training_exposures':0})
for seed in (202610201,202610202,202610203):
    ck=root/'results'/'chm'/f'chm-train-capacity-{seed}.pt'
    model=TinyV4HardCopyPointer();model.load_state_dict(torch.load(ck,weights_only=True));model.eval()
    assert sum(p.numel() for p in model.parameters())<=100_000_000
    rows=[]
    with torch.no_grad():
        for e in episodes:
            pred=hard_copy_prediction(model(seal_input(e)))
            gold,relation,payload=oracle_read(e)
            assert gold==e.gold_answer
            rows.append({'family':e.family,'entity':e.entity_index,'variant':e.variant,
                'correct':pred.predicted_answer_id==gold,'gold':gold,
                'answer':pred.predicted_answer_id,'abstained':pred.did_abstain,
                'first_correct':pred.first_source_physical==(relation if relation is not None else payload),
                'payload_correct':pred.payload_source_physical==payload})
    clusters=[]
    for family in FAMILIES:
        for index in range(64):
            rs=[r for r in rows if r['family']==family and r['entity']==index]
            if rs:clusters.append({'family':family,'entity':index,'accuracy':sum(r['correct'] for r in rs)/len(rs)})
    summaries={}
    for family in FAMILIES:
        rs=[r for r in rows if r['family']==family]
        summaries[family]={k:sum(r[k] for r in rs)/len(rs) for k in ('correct','first_correct','payload_correct','abstained')}
    write(f'chm-unseen-{seed}.DIAGNOSTIC.json',{'seed':seed,'checkpoint_sha256':sha(ck),
          'panel_sha256':panel,'rows':rows,'clusters':clusters,'summary':summaries,
          'classification':'EXPLORATORY_INFERENCE_ONLY_UNSEEN_OPTIMIZER_ENTITIES',
          'fair_trained_baseline':False,'breakthrough_demonstrated':False})
    print(seed,json.dumps(summaries),flush=True)
