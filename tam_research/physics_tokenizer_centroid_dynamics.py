"""Centroid-aware dynamics for the fixed Phase-0 wave-token benchmark.
Every predicted state is requantized to the 256-code direct-state VQ vocabulary before the next autoregressive step.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from . import physics_tokenizer_phase0 as base
from . import physics_tokenizer_direct_vq as dvq

class CentroidResidualDynamics(nn.Module):
    def __init__(self, context=3, width=64):
        super().__init__(); self.context=context
        self.net=nn.Sequential(
            nn.Conv2d(2*context,width,3,padding=1,padding_mode='circular'),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1,padding_mode='circular'),nn.GELU(),
            nn.Conv2d(width,width//2,3,padding=1,padding_mode='circular'),nn.GELU(),
            nn.Conv2d(width//2,2,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        b,c,ch,h,w=x.shape
        return x[:,-1]+self.net(x.reshape(b,c*ch,h,w))

def tokens_to_norm(ids,cb,side):
    b,t,_=ids.shape
    return cb[ids.reshape(-1)].reshape(b,t,side,side,2).permute(0,1,4,2,3).contiguous()

def norm_to_tokens(x,cb,chunk=16384):
    b,c,h,w=x.shape; flat=x.permute(0,2,3,1).reshape(-1,c); cb2=cb.pow(2).sum(1)[None,:]; out=[]
    for i in range(0,len(flat),chunk):
        z=flat[i:i+chunk]; out.append((z.pow(2).sum(1,keepdim=True)+cb2-2*z@cb.t()).argmin(1))
    return torch.cat(out).reshape(b,h*w)

def make_pairs(toks,norm,context,cb,side):
    xs=[]; ys=[]
    for t in range(context,toks.shape[1]):
        xs.append(toks[:,t-context:t]); ys.append(torch.from_numpy(norm[:,t].astype(np.float32)))
    return tokens_to_norm(torch.cat(xs),cb,side),torch.cat(ys)

def train_model(model,x,y,epochs,batch,seed):
    g=torch.Generator().manual_seed(seed); dl=DataLoader(TensorDataset(x,y),batch_size=batch,shuffle=True,generator=g)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5); hist=[]
    for _ in range(epochs):
        model.train(); tot=n=0
        for xb,yb in dl:
            loss=F.mse_loss(model(xb),yb); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tot+=loss.item()*len(xb); n+=len(xb)
        hist.append(tot/n)
    return hist

def physical(x,means,stds):
    m=torch.as_tensor(means)[None,:,None,None]; s=torch.as_tensor(stds)[None,:,None,None]
    return (x*s+m).detach().cpu().numpy()

def evaluate(model,toks,raw,cb,means,stds,context):
    side=raw.shape[-1]; ctx=tokens_to_norm(toks[:,:context].clone(),cb,side); frozen=toks[:,context-1]; frozen_dec=dvq.decode_direct(frozen,cb,means,stds,side); rows=[]
    for h in range(1,toks.shape[1]-context+1):
        target=toks[:,context+h-1]; true=raw[:,context+h-1]; persist=raw[:,context-1]
        with torch.no_grad(): cont=model(ctx[:,-context:]); ids=norm_to_tokens(cont,cb); q=tokens_to_norm(ids[:,None],cb,side)[:,0]
        qm=float(np.mean((physical(q,means,stds)-true)**2)); cm=float(np.mean((physical(cont,means,stds)-true)**2)); om=float(np.mean((dvq.decode_direct(target,cb,means,stds,side)-true)**2)); pm=float(np.mean((persist-true)**2)); tm=float(np.mean((frozen_dec-true)**2))
        rows.append({'horizon':h,'model_discrete_rollout_mse':qm,'model_continuous_diagnostic_mse':cm,'oracle_ground_truth_token_decode_mse':om,'raw_persistence_mse':pm,'decoded_token_persistence_mse':tm,'model_over_raw_persistence':qm/pm,'continuous_over_raw_persistence':cm/pm,'oracle_over_raw_persistence':om/pm,'model_token_accuracy':float((ids==target).float().mean()),'token_persistence_accuracy':float((frozen==target).float().mean())})
        ctx=torch.cat([ctx,q[:,None]],1)
    return {'horizons':rows,'fixed_h3_discrete_beats_raw_persistence':rows[2]['model_over_raw_persistence']<1.0,'fixed_h3_ratio':rows[2]['model_over_raw_persistence']}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='physics_tokenizer_centroid_dynamics.json'); p.add_argument('--epochs',type=int,default=30); p.add_argument('--model-seed',type=int,default=9132601); a=p.parse_args()
    cfg=base.Config(seed=74193); t0=time.time(); base.set_seed(cfg.seed); torch.set_num_threads(min(8,torch.get_num_threads()))
    train,_=base.generate_sequences(cfg.train_n,cfg.seq_len,cfg.grid,(.6,1.0),cfg.dt,cfg.seed+1); val,_=base.generate_sequences(cfg.val_n,cfg.seq_len,cfg.grid,(.6,1.0),cfg.dt,cfg.seed+2); ood,_=base.generate_sequences(cfg.ood_n,cfg.seq_len,cfg.grid,(1.15,1.35),cfg.dt,cfg.seed+3)
    means,stds=base.channel_stats(train); trn=base.normalize(train,means,stds); van=base.normalize(val,means,stds); oon=base.normalize(ood,means,stds)
    cb=base.fit_kmeans(trn.transpose(0,1,3,4,2).reshape(-1,2),cfg.codebook_size,cfg.seed,iters=20,max_points=60000); tr=dvq.encode_direct(trn,cb); va=dvq.encode_direct(van,cb); oo=dvq.encode_direct(oon,cb)
    counts=torch.bincount(tr.reshape(-1),minlength=cfg.codebook_size).float(); probs=counts/counts.sum(); nz=probs[probs>0]; perp=float(torch.exp(-(nz*torch.log(nz)).sum())); active=int((counts>0).sum())
    rec=dvq.decode_direct(va.reshape(-1,cfg.grid*cfg.grid),cb,means,stds,cfg.grid).reshape(val.shape); recn=(rec-means[None,None,:,None,None])/stds[None,None,:,None,None]; nmse=float(np.mean((recn-van)**2)); pmse=float(np.mean((rec-val)**2))
    x,y=make_pairs(tr,trn,cfg.context,cb,cfg.grid); base.set_seed(a.model_seed); model=CentroidResidualDynamics(cfg.context); hist=train_model(model,x,y,a.epochs,cfg.batch,a.model_seed); ID=evaluate(model,va,val,cb,means,stds,cfg.context); OOD=evaluate(model,oo,ood,cb,means,stds,cfg.context)
    gates={'tokenizer_usage_pass':active>=64 and perp>=4,'tokenizer_reconstruction_pass':nmse<=.20,'id_rollout_beats_persistence':ID['fixed_h3_ratio']<1.0}; gates['phase0_pass']=all(gates.values())
    res={'experiment':'direct-state-vq-centroid-residual-dynamics-v1','scientific_status':'new dynamics ablation; not retry of interrupted categorical 80-epoch attempt','locked_benchmark':{'benchmark_seed':74193,'codec_seed':74193,'model_seed':a.model_seed,'grid':cfg.grid,'seq_len':cfg.seq_len,'dt':cfg.dt,'train_n':cfg.train_n,'val_n':cfg.val_n,'ood_n':cfg.ood_n,'codebook_size':256,'context':3,'fixed_gate_horizon':3,'fixed_gate':'ID discrete autoregressive h3 decoded MSE < raw persistence h3 MSE'},'change_from_direct_vq_baseline':'centroid/state-space residual ConvNet; every predicted step requantized to same 256-token vocabulary','tokenizer':{'active_codes':active,'perplexity':perp,'reconstruction_normalized_mse':nmse,'reconstruction_physical_mse':pmse},'training':{'epochs':a.epochs,'lr':2e-3,'train_mse_first':hist[0],'train_mse_last':hist[-1],'train_mse_history':hist},'diagnostics':{'id':ID,'ood':OOD},'gates':gates,'runtime_seconds':time.time()-t0}
    Path(a.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
