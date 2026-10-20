"""Finite fresh strong-control study; see PGW_CONTROL_PROTOCOL.md."""
import argparse,hashlib,itertools,json,subprocess,sys,time,traceback
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.repo.resolve()))
from tam_research.pgw_v3_stage2a.model import PGWV3Stage2A,batch_verified_examples
from tam_research.pgw_v3_stage1b.oracle import make_example,POSITIONS,OVERWRITES,DELAYS,KEY_TOKENS,oracle_latest_value_or_not_found
ROOT=Path(__file__).parent;OUT=ROOT/'results'/'pgw_control';OUT.mkdir(exist_ok=True)
SEEDS=(202610231,202610232,202610233);STEPS=5000;BATCH=24
torch.set_num_threads(2)
def dump(path,d):
    with path.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
class StrongControl(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(256,64);self.position=nn.Embedding(128,64)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(64,4,256,dropout=0,batch_first=True,norm_first=True) for _ in range(3)])
        self.norm=nn.LayerNorm(64);self.head=nn.Linear(64,33)
    def forward(self,x,anchor):
        h=self.embedding(x)+self.position(torch.arange(x.size(1)))[None]
        mask=torch.triu(torch.ones(x.size(1),x.size(1),dtype=torch.bool),diagonal=1)
        for block in self.blocks:h=block(h,src_mask=mask)
        return self.head(self.norm(h[torch.arange(len(x)),anchor]))
factors=list(itertools.product(POSITIONS,OVERWRITES,(False,True),(False,True)))
def make(index,delay,split,n):
    return [make_example(split=split,index=index+j,last_position=q[0],overwrite_count=q[1],
        interference=q[2],delay_chunks=delay,missing=q[3])
        for j in range(n) for q in [factors[(index+j)%len(factors)]]]
evals={d:batch_verified_examples(make(21_000_000,d,'validation',180)) for d in DELAYS}
@torch.no_grad()
def evaluate(model):
    model.eval();rows=[];groups=[]
    for delay,(x,anchor,y) in evals.items():
        logits=torch.cat([model(xx,aa) for xx,aa in zip(x.split(24),anchor.split(24))]);pred=logits.argmax(-1)
        rows.append({'delay':delay,'examples':len(y),'accuracy':float((pred==y).float().mean()),
             'present_accuracy':float((pred[y!=32]==y[y!=32]).float().mean()),
             'missing_accuracy':float((pred[y==32]==32).float().mean()),'nll':float(F.cross_entropy(logits,y))})
        for original in x[:24]:
            swapped=original.repeat(8,1);swapped[:,-7]=torch.tensor(KEY_TOKENS)
            gold=torch.tensor([oracle_latest_value_or_not_found(tuple(z.tolist()))-32 for z in swapped])
            predicted=model(swapped,torch.full((8,),x.size(1)-6)).argmax(-1)
            groups.append({'delay':delay,'correct':int((predicted==gold).sum()),'all_eight':bool((predicted==gold).all())})
    return {'by_delay':rows,'query_swap_prefixes':groups,
        'all_eight_fraction':sum(g['all_eight'] for g in groups)/len(groups)}
def run(arm,seed):
    torch.manual_seed(seed);model=StrongControl() if arm=='strong_transformer' else PGWV3Stage2A()
    ident=f'pgw-control-{arm}-{seed}';params=sum(p.numel() for p in model.parameters());assert params<=100_000_000
    base={'attempt':ident,'parameters':params,'recipe':{'seed':seed,'updates':STEPS,'batch':BATCH,'lr':.001,'aux_lambda':0},
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':'cpu','torch':torch.__version__,
      'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip(),
      'fair_parameter_compute_parity':False,'classification':'EXPLORATORY_LEARNABILITY_CONTROL','breakthrough_demonstrated':False}
    dump(OUT/(ident+'.ATTEMPT.json'),base);print('START',ident,params,flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);seen=0;coverage={};datahash=hashlib.sha256();start=time.perf_counter()
    for step in range(STEPS):
        examples=make(20_000_000+step*BATCH,DELAYS[step%3],'train',BATCH)
        x,anchor,y=batch_verified_examples(examples)
        if seen+x.numel()>2_000_000_000:raise ValueError('TOKEN_CEILING')
        if time.perf_counter()-start>900:raise TimeoutError('15_MINUTE_ATTEMPT_CAP')
        datahash.update(''.join(e.sha256 for e in examples).encode());seen+=x.numel()
        model.train();opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x,anchor),y)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward()
        for name,param in model.named_parameters():coverage[name]=coverage.get(name,False) or bool(param.grad is not None and torch.count_nonzero(param.grad))
        nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if (step+1)%500==0:
            print(ident,step+1,float(loss.detach()),flush=True)
            (OUT/(ident+'.PROGRESS.json')).write_text(json.dumps({'updates':step+1,'training_token_exposures':seen,'loss':float(loss.detach())}))
    result=evaluate(model);elapsed=time.perf_counter()-start
    path=OUT/(ident+'.pt')
    with path.open('xb') as f:torch.save(model.state_dict(),f)
    record={**base,'result':result,'training_token_exposures':seen,'total_seconds':elapsed,
      'data_stream_sha256':datahash.hexdigest(),'gradient_coverage':coverage,
      'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    dump(OUT/(ident+'.RESULT.json'),record);print('RESULT',ident,json.dumps(result),flush=True)
    return record
def main():
    torch.manual_seed(509872);pre=StrongControl().eval();x=torch.randint(0,256,(2,88));y=x.clone();y[:,83:]=torch.randint(0,256,(2,5))
    with torch.no_grad():delta=float((pre(x,torch.tensor([82,82]))-pre(y,torch.tensor([82,82]))).abs().max())
    if delta>1e-6:raise AssertionError('control future leakage')
    dump(OUT/'PREFLIGHT.json',{'future_delta':delta,'actual_parameters':sum(p.numel() for p in pre.parameters())})
    dump(OUT/'PREREGISTRATION.json',{'protocol_sha256':hashlib.sha256((ROOT/'PGW_CONTROL_PROTOCOL.md').read_bytes()).hexdigest(),'seeds':SEEDS})
    controls=[run('strong_transformer',seed) for seed in SEEDS]
    passed=all(r[k]>=.8 for c in controls for r in c['result']['by_delay'] for k in ('present_accuracy','missing_accuracy'))
    dump(OUT/'CONTROL_GATE.json',{'passed':passed,'candidate_runs_authorized_by_gate':passed})
    if passed:
        for seed in SEEDS:run('native_hybrid',seed)
    dump(OUT/'COMPLETE.json',{'complete':True,'control_gate_passed':passed})
if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(OUT/'ERROR.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
