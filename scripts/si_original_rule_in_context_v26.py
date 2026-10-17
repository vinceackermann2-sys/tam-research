"""v26: source-attested original hybrid versus matched controls on unseen rules.

Synthetic in-context adaptation: eight (input A, input B, noisy output) exemplars
followed by a query with NO answer. Episodic rule identity is latent, NEVER a model
input. Train with observed query feedback; no Bayes teacher or privileged task IDs.
"""
from __future__ import annotations
import argparse, json, math, time
import torch
import torch.nn.functional as F
from scripts.si_original_active_experiment_v17 import SequentialController, verify_original

# Rules encoded as deterministic truth tables at A,B = 00,01,10,11.
# Train XOR, XNOR, AND, NAND. Four OOD holdout rules are withheld by identity.
RULES=torch.tensor([[0,1,1,0], [1,0,0,1], [0,0,0,1], [1,1,1,0],
                    [0,1,1,1], [1,0,0,0], [1,1,0,1], [1,0,1,1]], dtype=torch.float32)
FAMILIES=('seen','unseen_rules','noisy_labels','long_context')


def episodes(n,seed,family='seen',length=None):
    if family not in FAMILIES: raise ValueError(family)
    g=torch.Generator().manual_seed(int(seed))
    support=12 if family=='long_context' else 8 if length is None else length
    if support+1>64: raise ValueError('model max sequence')
    if family=='unseen_rules': rule=torch.randint(4,8,(n,),generator=g)
    else: rule=torch.randint(0,4,(n,),generator=g)
    pair=torch.randint(0,4,(n,support+1),generator=g)
    # The query is uniform, not cherry-picked from calibration examples.
    a=(pair//2).float(); b=(pair%2).float()
    latent=RULES[rule[:,None],pair]
    corrupt=torch.rand(n,support,generator=g)<(.27 if family=='noisy_labels' else .08)
    observed=torch.where(corrupt,1-latent[:,:support],latent[:,:support])
    token=torch.zeros(n,support+1,7)
    token[:,:,0]=2*a-1;token[:,:,1]=2*b-1
    token[:,:support,2]=2*observed-1
    token[:,:support,3]=1.  # known-label support indicator
    token[:,-1,4]=1.  # query marker; label always zero, not encoded
    token[:,:,5]=torch.arange(support+1,dtype=torch.float32)[None]/max(support,1)
    token[:,:,6]=1.
    return token,latent[:,-1].float(),pair,observed,rule


def empirical_lookup(pair,observed):
    """No task-family knowledge: Beta(1,1) counts matching query input only."""
    target=pair[:,-1,None]
    matches=(pair[:,:-1]==target).float()
    good=(matches*observed).sum(1)
    count=matches.sum(1)
    return (1+good)/(2+count)


def known_rule_bayes(pair,observed,family='all',noise=.08):
    """STRUCTURALLY PRIVILEGED known finite rule family, likelihood prior.

    This is a hand-specified reference, NOT a learning architecture or teacher.
    """
    ids=torch.arange(8 if family=='all' else 4)
    hypotheses=RULES[ids[:,None,None],pair[None,:,:-1]]
    agreements=1-(hypotheses-observed[None,:,:]).abs()
    loglik=(agreements*math.log(1-noise)+(1-agreements)*math.log(noise)).sum(-1)
    posterior=torch.softmax(loglik,dim=0)
    pred=RULES[ids[:,None],pair[:,-1][None,:]]
    return (posterior*pred).sum(0)


def train(seed,arch,steps=180,batch=96):
    torch.manual_seed(seed);torch.set_num_threads(2)
    model=SequentialController(arch)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    start=time.perf_counter()
    for i in range(steps):
        X,y,_,_,_=episodes(batch,seed*1000000+110000+i,'seen')
        logits=model(X)[:,0]
        loss=F.binary_cross_entropy_with_logits(logits,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    return model,time.perf_counter()-start


@torch.no_grad()
def evaluate(model,X,y,pair,observed):
    logits=model(X)[:,0]
    prob=logits.sigmoid()
    pred=prob>=.5
    return dict(accuracy=float((pred==y.bool()).float().mean()),
                brier=float((prob-y).square().mean()),
                cross_entropy=float(F.binary_cross_entropy_with_logits(logits,y)))


@torch.no_grad()
def classical_metrics(pair,y,observed,family):
    look=empirical_lookup(pair,observed)
    all8=known_rule_bayes(pair,observed,'all',.27 if family=='noisy_labels' else .08)
    known4=known_rule_bayes(pair,observed,'train',.27 if family=='noisy_labels' else .08)
    def metric(p):return {'accuracy':float(((p>=.5)==y.bool()).float().mean()),
                          'brier':float((p-y).square().mean())}
    return {'lookup':metric(look),'bayes_known8':metric(all8),'bayes_train4':metric(known4)}


def run(seed=118000,steps=180,batch=96,n_test=1024):
    torch.set_num_threads(2)
    canonical=verify_original()
    env={f:episodes(n_test,seed*1000000+85000+i*733,f) for i,f in enumerate(FAMILIES)}
    out={'seed':seed,'canonical_model_blob':canonical,'steps':steps,'batch':batch,'n_test':n_test,
         'family_rules':{'train':[0,1,2,3],'unseen':[4,5,6,7]},'architectures':{},
         'classical':{f:classical_metrics(e[2],e[1],e[3],f) for f,e in env.items()}}
    for arch in ('hybrid','transformer','gru'):
        net,t=train(seed,arch,steps,batch)
        out['architectures'][arch]={'parameters':sum(p.numel() for p in net.parameters()),
          'train_cpu_seconds':t,
          'families':{f:evaluate(net,*e[:4]) for f,e in env.items()}}
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=118000)
    p.add_argument('--steps',type=int,default=180)
    p.add_argument('--batch',type=int,default=96)
    p.add_argument('--n-test',type=int,default=1024)
    a=p.parse_args();print(json.dumps(run(a.seed,a.steps,a.batch,a.n_test),indent=2))
