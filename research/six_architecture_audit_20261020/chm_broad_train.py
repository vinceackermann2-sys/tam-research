"""Fresh multi-entity training; first native DEV access, never native TEST."""
import argparse,hashlib,json,random,subprocess,sys,time,traceback
from pathlib import Path
import torch
from torch import nn
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
from tam_research.chm_v4_neural_hard_copy_train_1430 import TinyV4HardCopyPointer,hard_copy_prediction,train_only_supervised_loss,encode_visible_v4
from tam_research.chm_v4_independent_binding_benchmark_1421 import generate_episode,seal_input,oracle_read,FAMILIES
root=Path(__file__).parent;out=root/'results/chm_broad';out.mkdir(exist_ok=True)
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,d):
    with (out/name).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
def reset(model,seed):
    torch.manual_seed(seed)
    def f(m):
        if isinstance(m,(nn.Linear,nn.Embedding,nn.LayerNorm)):m.reset_parameters()
        if isinstance(m,nn.MultiheadAttention):m._reset_parameters()
    model.apply(f)
def episodes(split):
    return [generate_episode(split,f,i,v) for f in FAMILIES for i in range(64 if split=='train' else 16)
            for v in (range(8) if f!='no_match' else (0,))]
@torch.no_grad()
def evaluate(model,examples):
    model.eval();rows=[];clusters=[];summary={}
    for e in examples:
        pred=hard_copy_prediction(model(seal_input(e)));gold,link,value=oracle_read(e);assert gold==e.gold_answer
        rows.append({'family':e.family,'entity':e.entity_index,'variant':e.variant,'gold':gold,'answer':pred.predicted_answer_id,
            'correct':pred.predicted_answer_id==gold,'first_correct':pred.first_source_physical==(link if link is not None else value),
            'payload_correct':pred.payload_source_physical==value,'abstained':pred.did_abstain})
    for f in FAMILIES:
        rs=[r for r in rows if r['family']==f]
        summary[f]={k:sum(r[k] for r in rs)/len(rs) for k in ('correct','first_correct','payload_correct','abstained')}
        for entity in sorted({r['entity'] for r in rs}):
            group=[r for r in rs if r['entity']==entity]
            clusters.append({'family':f,'entity':entity,'accuracy':sum(r['correct'] for r in group)/len(group)})
    return {'summary':summary,'clusters':clusters,'rows':rows}
def main():
    dump('PREREGISTRATION.json',{'protocol_sha256':sha(root/'CHM_BROAD_TRAIN_PROTOCOL.md'),'seeds':[202610271,202610272,202610273],
        'steps':4096,'native_test_opened':False,'development_opened_only_after_training':True})
    train=episodes('train');train_hash=hashlib.sha256('\n'.join(repr(e) for e in train).encode()).hexdigest();development=None;records=[]
    schedule=[]
    for epoch in range(2):
        order=[(f,i,v) for f in FAMILIES for i in range(64) for v in range(8)]
        random.Random(202610270+epoch).shuffle(order);schedule.extend(order)
    assert len(schedule)==4096
    for seed in (202610271,202610272,202610273):
        model=TinyV4HardCopyPointer();reset(model,seed);ident=f'chm-broad-{seed}';params=sum(p.numel() for p in model.parameters())
        assert params<=100_000_000
        record={'attempt':ident,'parameters':params,'recipe':{'seed':seed,'updates':4096,'lr':.005,'batch':1,'source_address_supervision':True},
            'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip(),
            'script_sha256':sha(Path(__file__)),'torch':torch.__version__,'device':'cpu','train_panel_sha256':train_hash,
            'classification':'FRESH_BROAD_TRAIN_FIRST_RESERVED_DEVELOPMENT_STUDY','breakthrough_demonstrated':False}
        dump(ident+'.ATTEMPT.json',record);print('START',ident,flush=True)
        opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=0);seen=0;started=time.perf_counter()
        for step,(family,index,variant) in enumerate(schedule):
            e=generate_episode('train',family,index,variant);view=seal_input(e);tokens=len(encode_visible_v4(view).model_tokens.token_ids)
            if seen+tokens>2_000_000_000:raise ValueError('TOKEN_CEILING')
            if time.perf_counter()-started>900:raise TimeoutError('15_MINUTE_ATTEMPT_CAP')
            model.train();opt.zero_grad(set_to_none=True);loss,_=train_only_supervised_loss(model(view),e)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();seen+=tokens
            if (step+1)%512==0:
                print(ident,step+1,float(loss.detach()),flush=True)
                (out/(ident+'.PROGRESS.json')).write_text(json.dumps({'updates':step+1,'training_token_exposures':seen,'loss':float(loss.detach())}))
        train_scores=evaluate(model,train)
        if development is None:
            development=episodes('development')
            dump('DEVELOPMENT_PANEL.json',{'sha256':hashlib.sha256('\n'.join(repr(e) for e in development).encode()).hexdigest(),
                'entities_per_family':16,'examples':len(development),'native_test_opened':False})
        dev=evaluate(model,development);gate=all(dev['summary'][f]['correct']>=.8 for f in FAMILIES[:3]) and dev['summary']['no_match']['correct']>=.9
        ck=out/(ident+'.pt')
        with ck.open('xb') as f:torch.save(model.state_dict(),f)
        record.update({'training_token_exposures':seen,'total_seconds':time.perf_counter()-started,'checkpoint_sha256':sha(ck),
            'result':{'train':train_scores,'development':dev,'preliminary_learnability_gate':gate,'fair_trained_comparator':False,'native_test_opened':False}})
        dump(ident+'.RESULT.json',record);records.append(record)
        print('RESULT',ident,json.dumps({'train':train_scores['summary'],'development':dev['summary'],'gate':gate}),flush=True)
    dump('COMPLETE.json',{'complete':True,'all_seed_gates_passed':all(r['result']['preliminary_learnability_gate'] for r in records),'native_test_opened':False})
if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump('ERROR.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
