"""Isolate whether factual planning-head training repairs measured search harm."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()))
from tam_research.si_original_world_scoped_v2 import SIOriginalWorldScopedV2
from tam_research.si_original_crossmodal_learning_v1 import _public_batch

ROOT=Path(__file__).parent;OUT=ROOT/'results/si_repair';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
def dump(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forward(model,x):return model(x,world='calibration',enable_memory=False,enable_fast=False,enable_search=False)

@torch.no_grad()
def evaluate(model):
    model.eval();correct={k:0 for k in ('direct','native_gated','forced_depth2','forced_depth1')};squares=0.;n=0;logits=[]
    for i in range(64):
        x,y,_=_public_batch(62_000_000+i,32);o=forward(model,x);s=o['state'];direct=o['direct_action']
        native,_=model._search(s);scores=[]
        for action in range(4):scores.append(model._world_step(s,action)[1])
        scores=torch.stack(scores,-1);depth1=scores.argmax(-1)
        flags=o['uncertainty']>.5;gated=torch.where(flags,native,direct)
        for name,pred in [('direct',direct),('native_gated',gated),('forced_depth2',native),('forced_depth1',depth1)]:correct[name]+=int((pred==y).sum())
        # Calibration uses independently selected actually executed evaluation
        # actions; no optimization uses this evaluation feedback.
        actions=torch.randint(4,(32,),generator=torch.Generator().manual_seed(62_100_000+i))
        rewards=(actions==y).float();pred=scores[torch.arange(32),actions];squares+=float((pred-rewards).square().sum());n+=32
        logits.append(o['logits'].clone())
    return {'accuracy':{k:v/n for k,v in correct.items()},'executed_action_reward_mse':squares/n,'examples':n},torch.cat(logits)

def main():
    dump(OUT/'PREREGISTRATION.json',{'protocol_sha256':sha(ROOT/'SI_PROTOCOL.md'),'source_sha256':sha(Path(__file__))})
    for seed in (202610301,202610302,202610303):
        torch.manual_seed(seed);model=SIOriginalWorldScopedV2(width=24,layers=4,experts=4,max_steps=3)
        params=sum(p.numel() for p in model.parameters());assert params<=100_000_000
        start=time.monotonic();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
        for step in range(2000):
            model.train();x,y,_=_public_batch(60_000_000+step,32);o=forward(model,x)
            loss=F.cross_entropy(o['logits'],y)+.001*o['balance'];opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        baseline,logits_before=evaluate(model)
        basepath=OUT/f'si-{seed}-policy.pt';torch.save({'weights':model.state_dict(),'persistent':model.export_persistent_state(),'optimizer':opt.state_dict()},basepath)
        for name,param in model.named_parameters():param.requires_grad_(name.startswith(('transition.','reward_head.','value_head.','verifier.')))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=0)
        ledger=hashlib.sha256();coverage=set();final_components={}
        for step in range(1000):
            model.eval();x,y,_=_public_batch(61_000_000+step,32)
            actions=torch.randint(4,(32,),generator=torch.Generator().manual_seed(61_100_000+step))
            rewards=(actions==y).float();after=list(x)+[('text',100+rewards.long())]
            with torch.no_grad():before=forward(model,x)['state'];terminal=forward(model,after)['state']
            onehot=F.one_hot(actions,4).to(before);predicted=before+model.transition(torch.cat([before,onehot],-1))
            errors=(predicted-terminal).square().mean(-1)
            transition=errors.mean();reward=F.mse_loss(model.reward_head(predicted).flatten(),rewards)
            value=.5*(F.mse_loss(model.value_head(before).flatten(),rewards)+model.value_head(terminal).square().mean())
            verifier=F.mse_loss(torch.sigmoid(model.verifier(before)).flatten(),errors.detach().clamp(0,1))
            loss=transition+reward+value+.1*verifier
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            coverage.update(n for n,p in model.named_parameters() if p.grad is not None and bool(p.grad.count_nonzero()))
            opt.step();ledger.update(actions.numpy().tobytes()+rewards.numpy().tobytes())
            final_components={k:float(v.detach()) for k,v in [('transition',transition),('reward',reward),('value',value),('verifier',verifier)]}
        repaired,logits_after=evaluate(model);delta=float((logits_after-logits_before).abs().max());assert delta==0
        checkpoint=OUT/f'si-{seed}-repair.pt';torch.save({'weights':model.state_dict(),'persistent':model.export_persistent_state(),'optimizer':opt.state_dict()},checkpoint)
        record={'seed':seed,'parameters':params,'accounted_training_positions':736000,'baseline':baseline,'repaired':repaired,
            'policy_logits_max_delta':delta,'gradient_coverage':sorted(coverage),'executed_feedback_digest':ledger.hexdigest(),
            'last_training_components':final_components,'seconds':time.monotonic()-start,'checkpoint_sha256':sha(checkpoint),
            'baseline_checkpoint_sha256':sha(basepath),'source_sha256':sha(Path(__file__)),'scope':'frozen-policy one-step factual planning calibration; memory/fast disabled',
            'breakthrough_demonstrated':False}
        record['gate_passed']=repaired['accuracy']['forced_depth1']>=repaired['accuracy']['direct']-.02 and repaired['executed_action_reward_mse']<baseline['executed_action_reward_mse']
        dump(OUT/f'si-{seed}.RESULT.json',record);print('RESULT',json.dumps(record),flush=True)
    dump(OUT/'COMPLETE.json',{'complete':True})
if __name__=='__main__':main()
