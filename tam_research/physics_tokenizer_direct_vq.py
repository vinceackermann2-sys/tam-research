"""Phase-0 direct-state VQ baseline for non-text wave-field tokenization.

Keeps the same data, splits, context, dynamics class, and fixed h3 raw-persistence gate
as physics_tokenizer_phase0.py. The only experimental change is the codec: each
normalized [u, du/dt] pixel is quantized directly to a learned 2-D centroid and
decoded by centroid lookup, eliminating the neural autoencoder bottleneck.
"""
import argparse, json, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from . import physics_tokenizer_phase0 as base


def encode_direct(seqs_norm, codebook, chunk=16384):
    n,t,c,h,w=seqs_norm.shape
    x=torch.from_numpy(seqs_norm.transpose(0,1,3,4,2).reshape(-1,c).astype(np.float32))
    out=[]
    for i in range(0,len(x),chunk):
        xb=x[i:i+chunk]
        dist=xb.pow(2).sum(1,keepdim=True)+codebook.pow(2).sum(1)[None,:]-2*xb@codebook.t()
        out.append(dist.argmin(1))
    return torch.cat(out).reshape(n,t,h*w)


def decode_direct(ids, codebook, means, stds, side):
    vals=codebook[ids.reshape(-1)].reshape(ids.shape[0],side,side,2).permute(0,3,1,2)
    m=torch.as_tensor(means)[None,:,None,None]; s=torch.as_tensor(stds)[None,:,None,None]
    return (vals*s+m).numpy()


def diagnostics(dyn, toks, raw, codebook, means, stds, context, device):
    dyn.eval(); side=raw.shape[-1]; max_h=toks.shape[1]-context
    frozen_ids=toks[:,context-1,:]
    frozen_dec=decode_direct(frozen_ids,codebook,means,stds,side)
    ctx=toks[:,:context,:].clone(); rows=[]
    for h in range(1,max_h+1):
        target_ids=toks[:,context+h-1,:]; true=raw[:,context+h-1]; raw_p=raw[:,context-1]
        with torch.no_grad(): pred_ids=dyn(ctx[:,-context:,:].to(device)).argmax(-1).cpu()
        model=decode_direct(pred_ids,codebook,means,stds,side)
        oracle=decode_direct(target_ids,codebook,means,stds,side)
        mm=float(np.mean((model-true)**2)); om=float(np.mean((oracle-true)**2)); rpm=float(np.mean((raw_p-true)**2)); tpm=float(np.mean((frozen_dec-true)**2))
        rows.append({'horizon':h,'model_decoded_mse':mm,'oracle_ground_truth_token_decode_mse':om,'raw_persistence_mse':rpm,'decoded_token_persistence_mse':tpm,'model_over_raw_persistence':mm/rpm,'oracle_over_raw_persistence':om/rpm,'model_token_accuracy':float((pred_ids==target_ids).float().mean()),'token_persistence_accuracy':float((frozen_ids==target_ids).float().mean())})
        ctx=torch.cat([ctx,pred_ids[:,None,:]],1)
    h3=rows[2]
    return {'exploratory_only':True,'horizons':rows,'fixed_gate_feasible_given_codec_h3':bool(h3['oracle_ground_truth_token_decode_mse']<h3['raw_persistence_mse']),'oracle_over_raw_persistence_h3':h3['oracle_over_raw_persistence']}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='physics_tokenizer_direct_vq.json'); ap.add_argument('--seed',type=int,default=74193); ap.add_argument('--dyn-epochs',type=int,default=20)
    args=ap.parse_args(); cfg=base.Config(seed=args.seed); cfg.dyn_epochs=args.dyn_epochs
    base.set_seed(cfg.seed); torch.set_num_threads(min(8,torch.get_num_threads())); device=torch.device('cpu'); t0=time.time()
    train,_=base.generate_sequences(cfg.train_n,cfg.seq_len,cfg.grid,(0.6,1.0),cfg.dt,cfg.seed+1)
    val,_=base.generate_sequences(cfg.val_n,cfg.seq_len,cfg.grid,(0.6,1.0),cfg.dt,cfg.seed+2)
    ood,_=base.generate_sequences(cfg.ood_n,cfg.seq_len,cfg.grid,(1.15,1.35),cfg.dt,cfg.seed+3)
    means,stds=base.channel_stats(train); trn=base.normalize(train,means,stds); van=base.normalize(val,means,stds); oon=base.normalize(ood,means,stds)
    flat=trn.transpose(0,1,3,4,2).reshape(-1,2)
    cb=base.fit_kmeans(flat,cfg.codebook_size,cfg.seed,iters=20,max_points=60000)
    tr_toks=encode_direct(trn,cb); va_toks=encode_direct(van,cb); oo_toks=encode_direct(oon,cb)
    counts=torch.bincount(tr_toks.reshape(-1),minlength=cfg.codebook_size).float(); probs=counts/counts.sum(); nz=probs[probs>0]
    perplex=float(torch.exp(-(nz*torch.log(nz)).sum())); active=int((counts>0).sum())
    rec=decode_direct(va_toks.reshape(-1,cfg.grid*cfg.grid),cb,means,stds,cfg.grid).reshape(val.shape)
    recon_phys=float(np.mean((rec-val)**2)); rec_norm=(rec-means[None,None,:,None,None])/stds[None,None,:,None,None]; recon_nmse=float(np.mean((rec_norm-van)**2))
    xtr,ytr=base.build_pairs(tr_toks,cfg.context); dyn=base.TokenDynamics(cfg.codebook_size,cfg.context,spatial=cfg.grid*cfg.grid).to(device)
    hist=base.train_dyn(dyn,xtr,ytr,cfg.dyn_epochs,cfg.batch,2e-3,device)
    idd=diagnostics(dyn,va_toks,val,cb,means,stds,cfg.context,device); oodd=diagnostics(dyn,oo_toks,ood,cb,means,stds,cfg.context,device); idh3=idd['horizons'][2]
    gates={'tokenizer_usage_pass':bool(active>=cfg.codebook_size//4 and perplex>=4.0),'tokenizer_reconstruction_pass':bool(recon_nmse<=0.20),'id_rollout_beats_persistence':bool(idh3['model_over_raw_persistence']<1.0)}; gates['phase0_pass']=all(gates.values())
    res={'experiment':'direct-state-vq-baseline','change_from_locked_phase0':'codec_only: normalized [u,v] pixel -> 256-centroid VQ -> centroid lookup; data/dynamics/gate unchanged','config':asdict(cfg),'normalization':{'means':means.tolist(),'stds':stds.tolist()},'tokenizer':{'active_codes':active,'active_fraction':active/cfg.codebook_size,'perplexity':perplex,'reconstruction_normalized_mse':recon_nmse,'reconstruction_physical_mse':recon_phys},'dynamics':{'train_ce_last':hist[-1]},'diagnostics':{'id':idd,'ood':oodd},'gates':gates,'runtime_seconds':time.time()-t0}
    Path(args.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))

if __name__=='__main__': main()
