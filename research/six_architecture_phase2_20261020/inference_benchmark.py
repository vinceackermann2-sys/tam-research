import gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from phase2_aligned_v2 import make,count,VOCAB
from train_panel import load_checkpoint,evaluate
ROOT=Path('/content/tam-phase2-20261020')
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    while True:
        status=json.loads((ROOT/'STATUS.json').read_text()) if (ROOT/'STATUS.json').exists() else {}
        if status.get('phase')=='error':raise RuntimeError('training queue failed; inference not started')
        if (ROOT/'EVIDENCE.json').exists():
            evidence=json.loads((ROOT/'EVIDENCE.json').read_text())
            if evidence.get('complete'):break
        time.sleep(20)
    dump(ROOT/'INFERENCE_CLAIM.json',{'protocol_sha256':hashlib.sha256((ROOT/'INFERENCE_PROTOCOL.md').read_bytes()).hexdigest()})
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'records':[]}
    for arm,terminal in evidence['terminal'].items():
        record=evidence['results'][f"{arm}-s{terminal['steps']}"];model=make(arm).cuda();state=load_checkpoint(record['checkpoint']);model.load_state_dict(state['model']);del state;model.eval()
        row={'variant':arm,'steps':terminal['steps'],'parameters':count(model),'prefill':[],'checkpoint_sha256':record['checkpoint']['checkpoint_sha256']}
        if arm=='conv7':
            base=evaluate(model,'validation',524288);assert abs(base['nll']-record['validation']['nll'])<1e-6
            gains=[(b,b.conv_gain.detach().clone()) for b in model.blocks if b.conv is not None]
            for b,gain in gains:b.conv_gain.zero_()
            erased=evaluate(model,'validation',524288)
            for b,gain in gains:b.conv_gain.copy_(gain)
            row['conv_erasure']={'replay_nll':base['nll'],'zero_gain_nll':erased['nll'],'delta_nll':erased['nll']-base['nll']}
        model.eval()
        if arm=='cpw_local':
            x=torch.arange(32,device='cuda')[None]+100;y=x.clone();y[:,:16]+=200
            with torch.autocast('cuda',dtype=torch.bfloat16):delta=float((model(x)[:,-1]-model(y)[:,-1]).abs().max())
            assert delta==0;row['distant_prefix_delta']=delta
        for length in (128,512,1024):
            for batch in (1,8):
                x=(torch.arange(batch*length,device='cuda').reshape(batch,length)%40000)+100
                for _ in range(5):
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
                for _ in range(20):
                    torch.cuda.synchronize();t=time.perf_counter()
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
                    torch.cuda.synchronize();times.append(time.perf_counter()-t)
                median=float(np.median(times));row['prefill'].append({'context':length,'batch':batch,'median_ms':median*1000,'p90_ms':float(np.quantile(times,.9))*1000,'tokens_per_second':batch*length/median,'peak_gib':torch.cuda.max_memory_allocated()/2**30})
                del z,x
        prompt=torch.arange(128,device='cuda')[None]+100;times=[]
        for iteration in range(6):
            x=prompt.clone();torch.cuda.synchronize();t=time.perf_counter()
            for _ in range(8):
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):nxt=model(x)[:,-1].argmax(-1,keepdim=True)
                x=torch.cat([x,nxt],-1)
            torch.cuda.synchronize()
            if iteration:times.append(time.perf_counter()-t)
        row['naive_generation_without_kv_cache']={'new_tokens':8,'batch':1,'prompt':128,'median_total_ms':float(np.median(times))*1000,'new_tokens_per_second':8/float(np.median(times))}
        result['records'].append(row);dump(ROOT/(arm+'.INFERENCE.json'),row);print('INFERENCE',arm,flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    result['complete']=True;result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(ROOT/'INFERENCE.json',result)
if __name__=='__main__':main()
