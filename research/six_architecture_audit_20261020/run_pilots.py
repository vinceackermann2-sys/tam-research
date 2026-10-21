"""Exploratory continuation. See PROTOCOL.md; never overwrite an attempt."""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, random, subprocess, sys, time, traceback
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

MAX_PARAMS=100_000_000
MAX_TOKENS=2_000_000_000
SEEDS=(202610201,202610202,202610203)

def count(model):
    return sum(p.numel() for p in model.parameters())

def dump(path, value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def fresh_init(model,seed):
    torch.manual_seed(seed)
    def reset(m):
        if isinstance(m,(nn.Linear,nn.Embedding,nn.LayerNorm)): m.reset_parameters()
        if isinstance(m,nn.MultiheadAttention): m._reset_parameters()
    model.apply(reset)

class Attempt:
    def __init__(self,out,name,model,recipe):
        self.out=out; self.name=name; self.tokens=0; self.started=time.perf_counter()
        self.parameters=count(model)
        if self.parameters>MAX_PARAMS: raise ValueError('PARAMETER_CAP_EXCEEDED')
        self.base={'attempt':name,'parameters':self.parameters,'recipe':recipe,
                   'torch':torch.__version__,'device':str(next(model.parameters()).device),
                   'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                   'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        dump(out/(name+'.ATTEMPT.json'),self.base)
        print('START',name,self.parameters,flush=True)
    def expose(self,n):
        if self.tokens+n>MAX_TOKENS: raise ValueError('TOKEN_CAP_EXCEEDED')
        if time.perf_counter()-self.started>1800: raise TimeoutError('30_MINUTE_ATTEMPT_WALL_LIMIT')
        self.tokens+=n
    def finish(self,result,model=None):
        sync(); seconds=time.perf_counter()-self.started
        rec={**self.base,'training_token_exposures':self.tokens,'total_seconds':seconds,
             'total_input_tokens_per_second_including_eval':self.tokens/max(seconds,1e-9),
             'result':result,'classification':'EXPLORATORY_DEVELOPMENT_ONLY',
             'breakthrough_demonstrated':False}
        if model is not None:
            p=self.out/(self.name+'.pt')
            with p.open('xb') as f: torch.save(model.state_dict(),f)
            rec['checkpoint_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        dump(self.out/(self.name+'.RESULT.json'),rec)
        print('RESULT',json.dumps(rec),flush=True)
        return rec

def optimize(model,opt,loss):
    if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite loss')
    opt.zero_grad(set_to_none=True); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

def grad_groups(model):
    return {name:bool(p.grad is not None and torch.count_nonzero(p.grad))
            for name,p in model.named_parameters()}

def run_chm(out):
    from tam_research.chm_v4_neural_hard_copy_train_1430 import TinyV4HardCopyPointer, hard_copy_prediction, train_only_supervised_loss,encode_visible_v4
    from tam_research.chm_v4_independent_binding_benchmark_1421 import train_counterfactual_group, seal_input, oracle_read
    examples=[e for family in ('direct','overwrite','two_hop') for e in train_counterfactual_group(family,48)]
    examples += [train_counterfactual_group('no_match',48)[0]]
    rows=[]
    for seed in SEEDS:
        model=TinyV4HardCopyPointer(); fresh_init(model,seed)
        run=Attempt(out,f'chm-train-capacity-{seed}',model,{'updates':256,'lr':.005,'entity':48,'evaluation':'TRAIN ONLY','seed':seed})
        optimizer=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=0)
        first=None
        for step in range(256):
            episode=examples[step%25]; view=seal_input(episode)
            output=model(view); loss,parts=train_only_supervised_loss(output,episode)
            run.expose(len(encode_visible_v4(view).model_tokens.token_ids))
            if first is None:first=float(loss.detach())
            optimize(model,optimizer,loss)
        model.eval(); pred=[]
        with torch.no_grad():
            for e in examples:
                p=hard_copy_prediction(model(seal_input(e))); gold,relation,payload=oracle_read(e)
                pred.append({'family':e.family,'variant':e.variant,'gold':gold,'answer':p.predicted_answer_id,
                    'first_correct':p.first_source_physical==(relation if relation is not None else payload),
                    'payload_correct':p.payload_source_physical==payload,'abstained':p.did_abstain})
        result={'initial_loss':first,'final_loss':float(loss.detach()),'predictions':pred,
                'positive_correct':sum(p['answer']==p['gold'] for p in pred[:24]),
                'positive_total':24,'missing_correct':pred[-1]['answer'] is None,
                'interpretation':'TRAIN memorization only; cannot establish generalization or architecture superiority.'}
        rows.append(run.finish(result,model))
    return rows

def run_pgw(out):
    from tam_research.pgw_v3_stage2a.model import PGWV3Stage2A,PGWV3Stage2AConfig,batch_verified_examples
    from tam_research.pgw_v3_stage2b3.position import GlobalPositionCausalReference
    from tam_research.pgw_v3_stage2b2.reference import CausalReference,CausalReferenceConfig
    from tam_research.pgw_v3_stage2b0.auxiliary import predictor_auxiliary_loss
    from tam_research.pgw_v3_stage1b.oracle import make_example,POSITIONS,OVERWRITES,DELAYS
    factors=list(itertools.product(POSITIONS,OVERWRITES,(False,True),(False,True)))
    def examples(index,delay,split):
        vals=[]
        for j in range(12):
            pos,over,inter,missing=factors[(index*12+j)%len(factors)]
            vals.append(make_example(split=split,index=index*12+j,last_position=pos,overwrite_count=over,
                interference=inter,delay_chunks=delay,missing=missing))
        return batch_verified_examples(vals)
    evals={}
    for delay in DELAYS:
        cases=[make_example(split='validation',index=2_000_000+j,last_position=p,overwrite_count=o,
                    interference=i,delay_chunks=delay,missing=m)
               for j,(p,o,i,m) in enumerate(factors)]
        evals[delay]=batch_verified_examples(cases)
    rows=[]
    for seed in SEEDS:
        for arm in ('hybrid','full_global','chunk_only'):
            torch.manual_seed(seed)
            model=(PGWV3Stage2A(PGWV3Stage2AConfig()) if arm=='hybrid' else
                  GlobalPositionCausalReference() if arm=='full_global' else
                  CausalReference(CausalReferenceConfig(attention_scope='chunk')))
            run=Attempt(out,f'pgw-{arm}-{seed}',model,{'updates':256,'batch':12,'lr':.003,'aux_lambda':.01,'seed':seed})
            opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
            started=time.perf_counter()
            for step in range(256):
                x,a,y=examples(1_000_000+step,DELAYS[step%3],'train')
                if arm=='hybrid': logits=model(x,a); aux=predictor_auxiliary_loss(model,x,a)
                else: logits,aux=model.forward_with_aux(x,a)
                loss=F.cross_entropy(logits,y)+.01*aux
                run.expose(x.numel()); optimize(model,opt,loss)
            train_seconds=time.perf_counter()-started
            per=[]; model.eval()
            with torch.no_grad():
                for delay,(x,a,y) in evals.items():
                    logits=model(x,a); pred=logits.argmax(-1); present=y!=32; missing=~present
                    per.append({'delay':delay,'examples':len(y),'accuracy':float((pred==y).float().mean()),
                        'present_accuracy':float((pred[present]==y[present]).float().mean()),
                        'missing_accuracy':float((pred[missing]==y[missing]).float().mean()),
                        'nll':float(F.cross_entropy(logits,y))})
            rows.append(run.finish({'by_delay':per,'train_seconds':train_seconds,'final_loss':float(loss.detach()),
                'objective_parity':False,'active_parameter_parity':False,
                'caveat':'Native hidden-target auxiliary losses have differing accessible histories. Full-global must learn before efficacy can be judged.'},model))
    return rows

def run_si(out):
    from tam_research.si_original_world_scoped_v2 import SIOriginalWorldScopedV2
    from tam_research.si_original_hybrid_v0 import DenseBlock
    from tam_research.si_original_crossmodal_learning_v1 import _public_batch
    class TransformerV2(SIOriginalWorldScopedV2):
        def __init__(self):
            super().__init__(width=24,layers=4,experts=4,max_steps=3)
            self.blocks=nn.ModuleList([DenseBlock(self.cfg) for _ in range(4)])
        def _backbone(self,steps):
            x=self.encode(steps)
            for block in self.blocks:x=block(x)
            return self.norm(x[:,-1]),x.new_zeros(()),[]
    def eval_model(model,options):
        correct=direct=total=expansions=0; iters=0; bymod=[[0,0] for _ in range(4)]
        with torch.no_grad():
            for i in range(8):
                x,y,q=_public_batch(4_000_000+i,32)
                o=model(x,world='development',**options)
                hits=o['action']==y
                correct+=int(hits.sum());direct+=int((o['direct_action']==y).sum());total+=len(y)
                expansions+=o['search_expansions']; iters+=int(o['latent_steps'].sum())
                for k in range(4):
                    m=q==k;bymod[k][0]+=int(hits[m].sum());bymod[k][1]+=int(m.sum())
        return {'accuracy':correct/total,'direct_accuracy':direct/total,'examples':total,
                'search_expansions':expansions,'mean_latent_steps':iters/total,
                'by_modality':{k:c/n for k,(c,n) in zip(('text','image','audio','video'),bymod)}}
    rows=[]
    for seed in SEEDS:
        torch.manual_seed(seed); original=SIOriginalWorldScopedV2(width=24,layers=4,experts=4,max_steps=3)
        torch.manual_seed(seed); control=TransformerV2()
        hs=original.state_dict();ts=control.state_dict()
        control.load_state_dict({k:hs[k].clone() if not k.startswith('blocks.') and k in hs and hs[k].shape==v.shape else v for k,v in ts.items()})
        for name,model in [('world_scoped_hybrid',original),('transformer_system',control)]:
            run=Attempt(out,f'si-{name}-{seed}',model,{'updates':128,'batch':16,'lr':.003,'seed':seed,'feedback':'actual fixed logged random actions in source-authored cue environment'})
            opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0)
            coverage={}; first=None
            for step in range(128):
                x,y,_=_public_batch(3_000_000+step,16)
                o=model(x,world='training',enable_search=True)
                loss=F.cross_entropy(o['logits'],y)+.001*o['balance']
                run.expose(80)
                if first is None:first=float(loss.detach())
                optimize(model,opt,loss)
                for k,v in grad_groups(model).items():coverage[k]=coverage.get(k,False) or v
                if step%4==0:
                    before=[(k,v[:1].clone()) for k,v in x]
                    action=(step//4)%4
                    # Execute the chosen behavior-policy action, then observe reward.
                    reward=float(action==int(y[0]))
                    after=before+[('text',torch.tensor([100+int(reward)]))]
                    model.ingest_executed(before,action,reward,after,world='training')
                    replay=model.consolidate_verified_replay(opt,count=min(8,len(model.replay)))
                    for k,v in grad_groups(model).items():coverage[k]=coverage.get(k,False) or v
            model.eval()
            full=eval_model(model,{})
            ablations={n:eval_model(model,kw) for n,kw in [('no_search',{'enable_search':False}),('no_memory',{'enable_memory':False}),('no_fast',{'enable_fast':False}),('one_latent',{'reason_budget':1})]}
            rows.append(run.finish({'initial_loss':first,'final_loss':float(loss.detach()),'all_paths':full,'ablations':ablations,
                'gradient_coverage':coverage,'value_head_ever_trained':any(v for k,v in coverage.items() if k.startswith('value_head.')),
                'factual_updates':model.fast_updates,'replay_loss':replay,'parameters_and_compute_matched':False,
                'caveat':'New development world has no factual adaptation records. Source-authored task; not a full SI evaluation.'},model))
    return rows

def run_cpw(out):
    from tam_research.cpw_v1.model import CPWV1Config
    from tam_research.cpw_v3_sparse_world.model import SparseWorldCPWResearchLM
    from tam_research.cpw_v5_afm.model import AFMCPWResearchLM
    from tam_research.cpw_binding_placement.model import EarlyAFMResearchLM
    from tam_research.cpw_binding_v2.task import make_binding_batch,counterfactual_query,explicit_key_lookup
    from tam_research.cpw_v5_afm.train import query_logits
    from tam_research.models import ResearchLM,ModelConfig
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[];seed=SEEDS[0]
    for arm in ('transformer','sequence_only','afm_last1','afm_first1'):
        torch.manual_seed(seed)
        cfg=CPWV1Config(vocab_size=4096)
        model=(ResearchLM(ModelConfig(vocab_size=4096)) if arm=='transformer' else
               SparseWorldCPWResearchLM('sequence_only',cfg) if arm=='sequence_only' else
               AFMCPWResearchLM(cfg) if arm=='afm_last1' else EarlyAFMResearchLM(cfg)).to(device)
        run=Attempt(out,f'cpw-{arm}-{seed}',model,{'updates':500,'batch':16,'lr':.001,'seed':seed,'vocab':4096,'ids_unchanged':True})
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        start=time.perf_counter()
        for step in range(500):
            batch=make_binding_batch(batch_size=16,generator=torch.Generator().manual_seed(5_000_000+step))
            run.expose(batch.tokens.numel())
            logits=query_logits(model,batch.tokens.to(device))
            optimize(model,opt,F.cross_entropy(logits,batch.targets.to(device)))
        sync();train_seconds=time.perf_counter()-start
        model.eval(); results=[]
        with torch.no_grad():
            for delay in (32,64,128,256):
                b=make_binding_batch(batch_size=128,generator=torch.Generator().manual_seed(6_000_000+delay),delay=delay)
                ls=torch.cat([query_logits(model,x.to(device)) for x in b.tokens.split(16)])
                results.append({'delay':delay,'accuracy':float((ls.argmax(-1).cpu()==b.targets).float().mean()),'nll':float(F.cross_entropy(ls,b.targets.to(device)))})
            base=make_binding_batch(batch_size=32,generator=torch.Generator().manual_seed(6_001_000),delay=32)
            correct=[]
            for d in (32,64,128,256):
                x,y=counterfactual_query(base,new_delay=d)
                assert torch.equal(explicit_key_lookup(x),y)
                ls=torch.cat([query_logits(model,t.to(device)) for t in x.split(16)])
                correct.append(ls.argmax(-1).cpu()==y)
            all_four=float(torch.stack(correct).all(0).float().mean())
        rows.append(run.finish({'by_delay':results,'counterfactual_all_four':all_four,'train_seconds':train_seconds,
            'parameter_matched_to_transformer':False,'positive_control_required':True,
            'caveat':'First learnability screen; compact vocabulary differs from historical scientific models.'},model))
        del model,opt
        if device.type=='cuda':torch.cuda.empty_cache()
    return rows

def run_rlt(out):
    from experiments.rlt.model_streaming_bounded import BoundedChunkGatedRLT,tiny_streaming_config
    from experiments.rlt.cpu_lastwrite_probe_ar import make_batch,SET0,SET1
    from experiments.rlt.cpu_bounded_streaming_aw import balanced_holdout
    class Bias4(BoundedChunkGatedRLT):GATE_LOGIT_BIAS=4.0
    rows=[]
    for seed in SEEDS:
        for arm in ('carry','bias4','reset'):
            torch.manual_seed(seed)
            model=(Bias4 if arm=='bias4' else BoundedChunkGatedRLT)(tiny_streaming_config(),carry_between_chunks=arm!='reset')
            run=Attempt(out,f'rlt-{arm}-{seed}',model,{'easy_updates':2100,'long_updates':1200,'batch':8,'seed':seed,'gate_bias':4 if arm=='bias4' else 2})
            opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
            for stage,n,lr,length,gaps in [('easy',2100,.003,8,(1,)),('long',1200,.001,128,(1,4,16,64,127))]:
                for group in opt.param_groups:group['lr']=lr
                for step in range(n):
                    x,y,_=make_batch(8,gaps,7_000_000+(0 if stage=='easy' else 100_000)+step,seq_len=length)
                    run.expose(x.numel())
                    optimize(model,opt,F.cross_entropy(model(x)[:,-1,SET0:SET1+1],y))
            model.eval();by_gap={}
            with torch.no_grad():
                for gap in (1,4,16,64,127):
                    x,y=balanced_holdout(gap,seed=7_200_000+gap)
                    ls=torch.cat([model(z)[:,-1,SET0:SET1+1] for z in x.split(8)])
                    pred=ls.argmax(-1)
                    by_gap[str(gap)]={'accuracy':float((pred==y).float().mean()),
                        'class0':float((pred[y==0]==0).float().mean()),'class1':float((pred[y==1]==1).float().mean()),'nll':float(F.cross_entropy(ls,y))}
            passed=all(by_gap[str(g)][c]>=.95 for g in (16,64,127) for c in ('class0','class1'))
            rows.append(run.finish({'by_gap':by_gap,'long_retention_pass':passed,
                'caveat':'Fixed synthetic state task. Reset control has inaccessible history; not a general Transformer superiority test.'},model))
    return rows

def main():
    global REPO
    p=argparse.ArgumentParser();p.add_argument('--track',required=True,choices=['chm','pgw','si','cpw','rlt'])
    p.add_argument('--repo',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();REPO=Path(a.repo).resolve();sys.path.insert(0,str(REPO))
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    try: globals()['run_'+a.track](out)
    except BaseException as exc:
        dump(out/(a.track+'.ERROR.json'),{'error':repr(exc),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
