import hashlib,json,platform,time
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from phase2_model import LanguageModel,count,VOCAB

def main():
    torch.set_num_threads(4)
    records=[]
    for arm in ('transformer','attention8','conv7'):
        torch.manual_seed(202610281)
        model=LanguageModel(arm).cuda();n=count(model);assert n<=100_000_000
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1,fused=True)
        x=torch.randint(VOCAB,(16,512),device='cuda');y=torch.randint(VOCAB,(16,512),device='cuda')
        times=[];torch.cuda.reset_peak_memory_stats()
        for step in range(8):
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter()
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(x,y)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            torch.cuda.synchronize();times.append(time.perf_counter()-t)
        records.append({'variant':arm,'parameters':n,'ff_hidden':model.ff,'batch':16,'context':512,
            'step_seconds':times,'steady_tokens_per_second':8192/(sum(times[3:])/5),
            'peak_gib':torch.cuda.max_memory_allocated()/2**30,'flash_attention_required':True,
            'tokens_seen':8*8192,'purpose':'synthetic hardware preflight; discarded model; no efficacy claim'})
        del model,opt,loss,x,y;torch.cuda.empty_cache()
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'python':platform.python_version(),
        'records':records,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path('/content/tam-phase2-20261020/PREFLIGHT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
