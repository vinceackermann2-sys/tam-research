"""New source/seed/data: finite corrected-binding positive-control calibration.

No rescue of the consumed 500-update pilot. Candidate runs occur only if the
new positive control passes the precommitted endpoint. No retuning or retry.
"""
import gc,hashlib,json,sys,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path('/content/tam-six-20261020');OUT=ROOT/'results/cpw_recovery'
sys.path.insert(0,str(ROOT/'cpw'))
from tam_research.cpw_v1.model import CPWV1Config
from tam_research.cpw_v3_sparse_world.model import SparseWorldCPWResearchLM
from tam_research.cpw_v5_afm.model import AFMCPWResearchLM
from tam_research.cpw_binding_placement.model import EarlyAFMResearchLM
from tam_research.cpw_r1.model import CPWR1ResearchLM
from tam_research.cpw_binding_v2.task import make_binding_batch,counterfactual_query,explicit_key_lookup
from tam_research.cpw_v5_afm.train import query_logits
from tam_research.models import ResearchLM,ModelConfig

SEED=202610261;STEPS=5000;BATCH=32
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

@torch.no_grad()
def evaluate(model):
    model.eval();result=[]
    for delay in (32,64,128,256):
        b=make_binding_batch(batch_size=512,generator=torch.Generator().manual_seed(31_000_000+delay),delay=delay)
        logits=torch.cat([query_logits(model,z.cuda()) for z in b.tokens.split(32)])
        result.append({'delay':delay,'accuracy':float((logits.argmax(-1).cpu()==b.targets).float().mean()),
                       'nll':float(F.cross_entropy(logits,b.targets.cuda()))})
    base=make_binding_batch(batch_size=128,generator=torch.Generator().manual_seed(32_000_000),delay=32)
    correct=[]
    for delay in (32,64,128,256):
        x,y=counterfactual_query(base,new_delay=delay)
        assert torch.equal(x[:,:-1],base.tokens[:,:-1]) and torch.equal(explicit_key_lookup(x),y)
        z=torch.cat([query_logits(model,part.cuda()) for part in x.split(32)])
        correct.append(z.argmax(-1).cpu()==y)
    model.train()
    return {'by_delay':result,'all_four':float(torch.stack(correct).all(0).float().mean()),'independent_source_groups':128}

def main():
    OUT.mkdir(exist_ok=True,parents=True);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    recipe={'seed':SEED,'steps':STEPS,'batch':BATCH,'tokens_each':STEPS*BATCH*320,'lr':.001,'weight_decay':.01,
            'input_ids':'original corrected generator, unchanged IDs; vocab4096',
            'positive_control_gate':'minimum accuracy across all four distances >=.80 AND all_four>=.70 at fixed step5000',
            'arms_if_valid':['sequence_only','afm_last1','afm_first1','r1_final'],
            'purpose':'ONE-SEED DEVELOPMENT CALIBRATION, capacity/control diagnostic, not matched architecture science',
            'retry':False,'resume':False,'candidate_checkpoint_selection':False}
    dump(OUT/'PREREGISTRATION.json',recipe)
    for arm in ('transformer','sequence_only','afm_last1','afm_first1','r1_final'):
        torch.manual_seed(SEED);cfg=CPWV1Config(vocab_size=4096)
        model=(ResearchLM(ModelConfig(vocab_size=4096)) if arm=='transformer' else
               SparseWorldCPWResearchLM('sequence_only',cfg) if arm=='sequence_only' else
               AFMCPWResearchLM(cfg) if arm=='afm_last1' else
               EarlyAFMResearchLM(cfg) if arm=='afm_first1' else CPWR1ResearchLM(cfg)).cuda()
        n=sum(p.numel() for p in model.parameters())
        if n>100_000_000 or recipe['tokens_each']>2_000_000_000:raise ValueError('USER_CEILING')
        ident=f'cpw-recovery-{arm}-{SEED}'
        record={'attempt':ident,'parameters':n,'recipe':recipe,'gpu':torch.cuda.get_device_name(),
                'torch':torch.__version__,'source_sha':'aad622d1eea591545a3482618ca8b269e4d9d43f','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        dump(OUT/(ident+'.ATTEMPT.json'),record)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        seen=0;history=[];start=time.perf_counter()
        for step in range(STEPS):
            batch=make_binding_batch(batch_size=BATCH,generator=torch.Generator().manual_seed(30_000_000+step))
            if seen+batch.tokens.numel()>2_000_000_000:raise ValueError('TOKEN_CEILING')
            x,y=batch.tokens.cuda(),batch.targets.cuda();opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(query_logits(model,x),y)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();seen+=x.numel()
            if (step+1)%1000==0:
                history.append({'step':step+1,'train_loss':float(loss.detach()),'tokens':seen})
                print(ident,json.dumps(history[-1]),flush=True)
                (OUT/(ident+'.PROGRESS.json')).write_text(json.dumps(history))
            if time.perf_counter()-start>1800:raise TimeoutError('30_MINUTE_MODEL_STOP')
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        result=evaluate(model)
        record.update({'tokens_seen':seen,'train_seconds':seconds,'train_tokens_per_second':seen/seconds,
                       'result':result,'history':history,'breakthrough_demonstrated':False})
        p=OUT/(ident+'.pt')
        with p.open('xb') as f:torch.save(model.state_dict(),f)
        record['checkpoint_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        dump(OUT/(ident+'.RESULT.json'),record);print('RESULT',json.dumps(record),flush=True)
        if arm=='transformer':
            valid=min(d['accuracy'] for d in result['by_delay'])>=.8 and result['all_four']>=.7
            dump(OUT/'CONTROL_GATE.json',{'passed':valid,'result':result,'candidate_training_started':valid})
            if not valid:
                print('POSITIVE_CONTROL_FAILED: candidate arms not started; efficacy verdict INCONCLUSIVE',flush=True)
                break
        del model,opt,loss;gc.collect();torch.cuda.empty_cache()
    dump(OUT/'COMPLETE.json',{'complete':True})

if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(exist_ok=True,parents=True);dump(OUT/'ERROR.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
