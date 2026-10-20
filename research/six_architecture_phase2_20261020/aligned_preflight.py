import gc,hashlib,json,time
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from phase2_aligned_model import make,count,VOCAB

def main():
    torch.set_num_threads(4);records=[]
    for arm in ('transformer','attention8','conv7','cpw_local'):
        torch.manual_seed(202610282);model=make(arm).cuda();assert count(model)<=100_000_000
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1,fused=True)
        x=torch.randint(VOCAB,(16,512),device='cuda');y=torch.randint(VOCAB,(16,512),device='cuda')
        times=[];torch.cuda.reset_peak_memory_stats()
        for step in range(8):
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter()
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(x,y)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-t)
        records.append({'variant':arm,'parameters':count(model),'ff_hidden':model.ff,'times':times,
            'tokens_per_second':8192/(sum(times[3:])/5),'peak_gib':torch.cuda.max_memory_allocated()/2**30,
            'training_exposures':65536,'language_efficacy':False})
        del model,opt,loss,x,y;gc.collect();torch.cuda.empty_cache()
    value={'gpu':torch.cuda.get_device_name(),'records':records,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path('/content/tam-phase2-20261020/ALIGNED_PREFLIGHT.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
if __name__=='__main__':main()
