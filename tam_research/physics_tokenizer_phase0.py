"""Phase-0 non-text physics tokenization experiment.

The experiment trains directly on numeric 2-D wave-equation fields (u, du/dt),
learns a discrete vector-quantized codebook, and predicts future token grids.
It intentionally does not convert scientific states into natural-language text.

This module is an engineering feasibility test, not evidence of new physics.
"""

import argparse, json, math, random, time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def laplacian_periodic(u):
    return (np.roll(u,1,-1)+np.roll(u,-1,-1)+np.roll(u,1,-2)+np.roll(u,-1,-2)-4*u)


def step_wave(u,v,c,dt):
    a=(c*c)*laplacian_periodic(u)
    vh=v+0.5*dt*a
    u2=u+dt*vh
    a2=(c*c)*laplacian_periodic(u2)
    v2=vh+0.5*dt*a2
    return u2,v2


def smooth_periodic_field(rng,n,max_mode=3):
    yy,xx=np.meshgrid(np.arange(n),np.arange(n), indexing='ij')
    f=np.zeros((n,n),np.float32)
    nm=int(rng.integers(2,6))
    for _ in range(nm):
        kx=int(rng.integers(0,max_mode+1)); ky=int(rng.integers(0,max_mode+1))
        if kx==0 and ky==0: kx=1
        amp=float(rng.normal())/(1.0+kx*kx+ky*ky)
        phase=float(rng.uniform(0,2*np.pi))
        f += amp*np.cos(2*np.pi*(kx*xx+ky*yy)/n + phase)
    f -= f.mean(); s=f.std()+1e-6; f=f/s
    return f.astype(np.float32)


def generate_sequences(num, seq_len, n, c_range, dt, seed):
    rng=np.random.default_rng(seed)
    xs=np.empty((num,seq_len,2,n,n),np.float32)
    cs=np.empty((num,),np.float32)
    for i in range(num):
        c=float(rng.uniform(*c_range)); cs[i]=c
        u=smooth_periodic_field(rng,n)*float(rng.uniform(0.5,1.0))
        v=smooth_periodic_field(rng,n)*float(rng.uniform(0.05,0.25))
        for t in range(seq_len):
            xs[i,t,0]=u; xs[i,t,1]=v
            u,v=step_wave(u,v,c,dt)
    return xs,cs


def fit_kmeans(x_np, k, seed, iters=12, max_points=30000, chunk=4096):
    """Deterministic Lloyd k-means using only torch/numpy (repo core deps)."""
    rng=np.random.default_rng(seed)
    if len(x_np)>max_points:
        x_np=x_np[rng.choice(len(x_np),max_points,replace=False)]
    x=torch.from_numpy(np.asarray(x_np,np.float32))
    g=torch.Generator().manual_seed(seed)
    perm=torch.randperm(len(x),generator=g)[:k]
    c=x[perm].clone()
    for _ in range(iters):
        sums=torch.zeros_like(c); counts=torch.zeros(k,dtype=torch.float32)
        for i in range(0,len(x),chunk):
            xb=x[i:i+chunk]
            dist=xb.pow(2).sum(1,keepdim=True)+c.pow(2).sum(1)[None,:]-2*xb@c.t()
            idx=dist.argmin(1)
            sums.index_add_(0,idx,xb)
            counts.index_add_(0,idx,torch.ones(len(xb)))
        nz=counts>0
        c[nz]=sums[nz]/counts[nz,None]
        if (~nz).any():
            repl=torch.randint(0,len(x),(int((~nz).sum()),),generator=g)
            c[~nz]=x[repl]
    return c


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=24):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv2d(2,32,3,1,1), nn.GELU(),
            nn.Conv2d(32,48,3,1,1), nn.GELU(),
            nn.Conv2d(48,latent_dim,3,1,1),
        )
        self.dec=nn.Sequential(
            nn.Conv2d(latent_dim,48,3,1,1), nn.GELU(),
            nn.Conv2d(48,32,3,1,1), nn.GELU(),
            nn.Conv2d(32,2,3,1,1),
        )
    def encode(self,x): return self.enc(x)
    def decode(self,z): return self.dec(z)
    def forward(self,x): return self.decode(self.encode(x))


def nearest_codes(z, codebook):
    # z B,D,H,W -> idx B,H,W and quantized B,D,H,W
    b,d,h,w=z.shape
    flat=z.permute(0,2,3,1).reshape(-1,d)
    # ||x-c||^2 = x^2 + c^2 -2xc
    dist=(flat.pow(2).sum(1,keepdim=True)+codebook.pow(2).sum(1)[None,:]-2*flat@codebook.t())
    idx=dist.argmin(1)
    q=codebook[idx].reshape(b,h,w,d).permute(0,3,1,2).contiguous()
    return idx.reshape(b,h,w),q


class TokenDynamics(nn.Module):
    """Spatial token world model: embeds discrete physics tokens and predicts the next token grid.
    The 2-D convolution is an intentional Phase-0 inductive bias for field dynamics; no text is used.
    """
    def __init__(self,k,context,spatial=64,d_model=24,layers=0,heads=0):
        super().__init__(); self.k=k; self.context=context; self.spatial=spatial
        side=int(math.sqrt(spatial)); assert side*side==spatial; self.side=side
        self.tok=nn.Embedding(k,d_model)
        self.net=nn.Sequential(
            nn.Conv2d(context*d_model,96,3,1,1), nn.GELU(),
            nn.Conv2d(96,96,3,1,1), nn.GELU(),
            nn.Conv2d(96,k,1),
        )
    def forward(self,x): # B,C,S
        b,c,s=x.shape
        e=self.tok(x).reshape(b,c,self.side,self.side,-1).permute(0,1,4,2,3).reshape(b,-1,self.side,self.side)
        logits=self.net(e)
        return logits.permute(0,2,3,1).reshape(b,s,self.k)


def channel_stats(train):
    # train N,T,C,H,W
    means=train.mean(axis=(0,1,3,4)); stds=train.std(axis=(0,1,3,4))+1e-6
    return means.astype(np.float32),stds.astype(np.float32)

def normalize(x,means,stds): return (x-means[None,None,:,None,None])/stds[None,None,:,None,None]
def denorm_torch(x,means,stds):
    m=torch.as_tensor(means,device=x.device)[None,:,None,None]; s=torch.as_tensor(stds,device=x.device)[None,:,None,None]
    return x*s+m


def train_ae(model, frames, epochs, batch, lr, device):
    dl=DataLoader(TensorDataset(torch.from_numpy(frames)),batch_size=batch,shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    hist=[]
    model.train()
    for ep in range(epochs):
        tot=n=0
        for (x,) in dl:
            x=x.to(device); y=model(x); loss=F.mse_loss(y,x)
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()*len(x); n+=len(x)
        hist.append(tot/n)
    return hist


def collect_latents(model, frames, batch, device):
    out=[]; model.eval()
    with torch.no_grad():
        for i in range(0,len(frames),batch):
            z=model.encode(torch.from_numpy(frames[i:i+batch]).to(device)).cpu()
            out.append(z)
    return torch.cat(out,0)


def finetune_quantized(model, codebook, frames, epochs, batch, lr, beta, device):
    # codebook fixed after kmeans; train encoder+decoder with STE and commitment.
    dl=DataLoader(TensorDataset(torch.from_numpy(frames)),batch_size=batch,shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5); hist=[]
    cb=codebook.to(device)
    for ep in range(epochs):
        tot=n=0
        model.train()
        for (x,) in dl:
            x=x.to(device); z=model.encode(x); _,q=nearest_codes(z,cb)
            qst=z+(q-z).detach(); rec=model.decode(qst)
            loss=F.mse_loss(rec,x)+beta*F.mse_loss(z,q.detach())
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()*len(x); n+=len(x)
        hist.append(tot/n)
    return hist


def tokenize_sequences(model, codebook, seqs_norm, batch, device):
    n,t,c,h,w=seqs_norm.shape
    frames=seqs_norm.reshape(n*t,c,h,w)
    toks=[]; model.eval(); cb=codebook.to(device)
    with torch.no_grad():
        for i in range(0,len(frames),batch):
            z=model.encode(torch.from_numpy(frames[i:i+batch]).to(device)); idx,_=nearest_codes(z,cb); toks.append(idx.cpu())
    toks=torch.cat(toks,0).reshape(n,t,-1)
    return toks


def decode_tokens(model, codebook, ids, device):
    # ids B,S where S must be a square spatial token grid
    model.eval(); cb=codebook.to(device); ids=ids.to(device)
    b,s=ids.shape; side=int(math.sqrt(s)); d=cb.shape[1]
    q=cb[ids.reshape(-1)].reshape(b,side,side,d).permute(0,3,1,2).contiguous()
    with torch.no_grad(): return model.decode(q)


def build_pairs(tokens, context):
    xs=[]; ys=[]
    for t in range(context,tokens.shape[1]):
        xs.append(tokens[:,t-context:t,:]); ys.append(tokens[:,t,:])
    return torch.cat(xs,0),torch.cat(ys,0)


def train_dyn(model,x,y,epochs,batch,lr,device):
    dl=DataLoader(TensorDataset(x,y),batch_size=batch,shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); hist=[]
    for ep in range(epochs):
        model.train(); tot=n=0
        for xb,yb in dl:
            xb=xb.to(device); yb=yb.to(device); logits=model(xb)
            loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),yb.reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tot+=loss.item()*len(xb); n+=len(xb)
        hist.append(tot/n)
    return hist


def energy_np(state,c):
    # state B,2,H,W c B
    u=state[:,0]; v=state[:,1]
    dx=.5*(np.roll(u,-1,-1)-np.roll(u,1,-1)); dy=.5*(np.roll(u,-1,-2)-np.roll(u,1,-2))
    return np.mean(.5*v*v+.5*(c[:,None,None]**2)*(dx*dx+dy*dy),axis=(1,2))


def evaluate(name, dyn, ae, cb, toks, raw, cs, means, stds, context, horizon, device):
    dyn.eval(); n=toks.shape[0]; start=context
    # one-step token acc at t=context
    with torch.no_grad():
        logits=dyn(toks[:,start-context:start,:].to(device)); pred=logits.argmax(-1).cpu()
    tok_acc=(pred==toks[:,start,:]).float().mean().item()
    pred_raw=denorm_torch(decode_tokens(ae,cb,pred,device),means,stds).cpu().numpy()
    true1=raw[:,start]
    one_mse=float(np.mean((pred_raw-true1)**2)); persist1=float(np.mean((raw[:,start-1]-true1)**2))

    # recursive rollout horizon starting after context
    ctx=toks[:,:context,:].clone()
    pred_states=[]
    for h in range(horizon):
        with torch.no_grad(): p=dyn(ctx[:,-context:,:].to(device)).argmax(-1).cpu()
        pred_states.append(denorm_torch(decode_tokens(ae,cb,p,device),means,stds).cpu().numpy())
        ctx=torch.cat([ctx,p[:,None,:]],1)
    pred_h=pred_states[-1]; true_h=raw[:,context+horizon-1]
    persist_h=raw[:,context-1]
    roll_mse=float(np.mean((pred_h-true_h)**2)); persist_mse=float(np.mean((persist_h-true_h)**2))
    e_true=energy_np(true_h,cs); e_pred=energy_np(pred_h,cs)
    energy_rel=float(np.mean(np.abs(e_pred-e_true)/(np.abs(e_true)+1e-8)))
    return {"split":name,"token_accuracy_1step":tok_acc,"mse_1step":one_mse,"persistence_mse_1step":persist1,"ratio_1step":one_mse/persist1,
            f"mse_rollout_h{horizon}":roll_mse,f"persistence_mse_h{horizon}":persist_mse,f"ratio_h{horizon}":roll_mse/persist_mse,"energy_relative_error_h":energy_rel}


@dataclass
class Config:
    seed:int=74193; grid:int=16; seq_len:int=8; dt:float=0.12
    train_n:int=128; val_n:int=48; ood_n:int=48
    latent_dim:int=24; codebook_size:int=256
    ae_epochs:int=6; vq_epochs:int=3; dyn_epochs:int=20
    batch:int=64; context:int=3; horizon:int=3


def main():
    p=argparse.ArgumentParser(description='Train/test a non-text wave-field tokenizer and token world model.')
    p.add_argument('--out', default='physics_tokenizer_phase0_results.json')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--seed', type=int, default=74193)
    p.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    args=p.parse_args(); cfg=Config(seed=args.seed)
    if args.smoke:
        cfg.train_n=96; cfg.val_n=24; cfg.ood_n=24; cfg.ae_epochs=2; cfg.vq_epochs=1; cfg.dyn_epochs=2; cfg.batch=32
    set_seed(cfg.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is false')
    device = torch.device('cuda' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg.seed)
    t0=time.time()
    train,ctr=generate_sequences(cfg.train_n,cfg.seq_len,cfg.grid,(0.6,1.0),cfg.dt,cfg.seed+1)
    val,cva=generate_sequences(cfg.val_n,cfg.seq_len,cfg.grid,(0.6,1.0),cfg.dt,cfg.seed+2)
    ood,coo=generate_sequences(cfg.ood_n,cfg.seq_len,cfg.grid,(1.15,1.35),cfg.dt,cfg.seed+3)
    means,stds=channel_stats(train); trn=normalize(train,means,stds); van=normalize(val,means,stds); oon=normalize(ood,means,stds)
    frames=trn.reshape(-1,2,cfg.grid,cfg.grid)
    ae=AutoEncoder(cfg.latent_dim).to(device)
    ae_hist=train_ae(ae,frames,cfg.ae_epochs,cfg.batch,2e-3,device)
    z=collect_latents(ae,frames,cfg.batch,device)
    flat=z.permute(0,2,3,1).reshape(-1,cfg.latent_dim).numpy()
    # bounded sample for clustering
    if len(flat)>60000:
        rng=np.random.default_rng(cfg.seed); flat_fit=flat[rng.choice(len(flat),60000,replace=False)]
    else: flat_fit=flat
    cb=fit_kmeans(flat_fit,cfg.codebook_size,cfg.seed)
    vq_hist=finetune_quantized(ae,cb,frames,cfg.vq_epochs,cfg.batch,8e-4,.25,device)

    # tokenizer metrics val
    va_toks=tokenize_sequences(ae,cb,van,cfg.batch,device); tr_toks=tokenize_sequences(ae,cb,trn,cfg.batch,device); oo_toks=tokenize_sequences(ae,cb,oon,cfg.batch,device)
    counts=torch.bincount(tr_toks.reshape(-1),minlength=cfg.codebook_size).float(); probs=counts/counts.sum(); nz=probs[probs>0]
    perplex=float(torch.exp(-(nz*torch.log(nz)).sum()).item()); active=int((counts>0).sum().item())
    # quantized val reconstruction
    ids=va_toks.reshape(-1,va_toks.shape[-1]); rec=[]
    for i in range(0,len(ids),cfg.batch): rec.append(decode_tokens(ae,cb,ids[i:i+cfg.batch],device).cpu())
    rec=torch.cat(rec,0).numpy().reshape(van.shape)
    recon_nmse=float(np.mean((rec-van)**2)); recon_phys=float(np.mean((rec*stds[None,None,:,None,None]+means[None,None,:,None,None]-val)**2))

    xtr,ytr=build_pairs(tr_toks,cfg.context)
    dyn=TokenDynamics(cfg.codebook_size,cfg.context,spatial=tr_toks.shape[-1]).to(device)
    dyn_hist=train_dyn(dyn,xtr,ytr,cfg.dyn_epochs,cfg.batch,2e-3,device)
    id_metrics=evaluate('ID',dyn,ae,cb,va_toks,val,cva,means,stds,cfg.context,cfg.horizon,device)
    ood_metrics=evaluate('OOD-speed',dyn,ae,cb,oo_toks,ood,coo,means,stds,cfg.context,cfg.horizon,device)

    gates={
      'tokenizer_usage_pass': bool(active>=cfg.codebook_size//4 and perplex>=4.0),
      'tokenizer_reconstruction_pass': bool(recon_nmse<=0.20),
      'id_rollout_beats_persistence': bool(id_metrics[f'ratio_h{cfg.horizon}']<1.0),
    }
    gates['phase0_pass']=all(gates.values())
    res={'config':asdict(cfg),'normalization':{'means':means.tolist(),'stds':stds.tolist()},
         'tokenizer':{'active_codes':active,'active_fraction':active/cfg.codebook_size,'perplexity':perplex,'reconstruction_normalized_mse':recon_nmse,'reconstruction_physical_mse':recon_phys,'ae_loss_last':ae_hist[-1],'vq_loss_last':vq_hist[-1]},
         'dynamics':{'train_ce_last':dyn_hist[-1],'id':id_metrics,'ood':ood_metrics},'gates':gates,'runtime_seconds':time.time()-t0}
    Path(args.out).write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2))

if __name__=='__main__': main()
