import gc,hashlib,json,math,time
from pathlib import Path
import numpy as np
import requests,tiktoken,torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from phase2_aligned_v2 import make
from train_panel import load_checkpoint
ROOT=Path('/content/tam-phase2-20261020')
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def fetch(url):
    r=requests.get(url,timeout=60);r.raise_for_status();return r.content

@torch.no_grad()
def score(model,examples):
    rows=[];forwards=0
    for lo in range(0,len(examples),8):
        batch=examples[lo:lo+8];sequences=[];starts=[]
        for ctx,tgt in batch:
            ctx=ctx[-(1024-len(tgt)):]
            if not ctx:ctx=[50256]
            sequences.append(ctx+tgt);starts.append(len(ctx))
        width=max(len(z) for z in sequences);x=torch.full((len(batch),width),50256,dtype=torch.long,device='cuda')
        for i,z in enumerate(sequences):x[i,:len(z)]=torch.tensor(z,device='cuda')
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION),torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
        for i,((ctx,tgt),start) in enumerate(zip(batch,starts)):
            part=logits[i,start-1:start-1+len(tgt)].float();target=torch.tensor(tgt,device='cuda')
            losses=F.cross_entropy(part,target,reduction='none')
            rows.append({'sum_nll':float(losses.sum()),'mean_token_nll':float(losses.mean()),'tokens':len(tgt),'all_greedy_correct':bool((part.argmax(-1)==target).all())})
        forwards+=len(batch)*width
    return rows,forwards

def main():
    torch.set_num_threads(4)
    class Echo(torch.nn.Module):
        def forward(self,x):return F.one_hot((x+1)%50257,50257).float()*10
    fixture,_=score(Echo(),[([1,2],[3,4]),([12],[13])])
    assert all(r['all_greedy_correct'] for r in fixture)
    assert abs(fixture[0]['mean_token_nll']-fixture[1]['mean_token_nll'])<1e-5
    while not (ROOT/'INFERENCE.json').exists():
        if (ROOT/'STATUS.json').exists() and json.loads((ROOT/'STATUS.json').read_text()).get('phase')=='error':raise RuntimeError('training failed')
        time.sleep(20)
    dump(ROOT/'BENCHMARK_CLAIM.json',{'protocol_sha256':hashlib.sha256((ROOT/'BENCHMARK_PROTOCOL.md').read_bytes()).hexdigest()})
    sources=json.loads((ROOT/'BENCHMARK_SOURCES.json').read_text());enc=tiktoken.get_encoding('gpt2')
    prefix=f"https://raw.githubusercontent.com/ybisk/ybisk.github.io/{sources['piqa_git_revision']}/piqa/data/"
    raw_piqa=fetch(prefix+'valid.jsonl');raw_labels=fetch(prefix+'valid-labels.lst')
    raw_lambada=fetch(f"https://huggingface.co/datasets/EleutherAI/lambada_openai/resolve/{sources['lambada_hf_revision']}/data/lambada_test.jsonl")
    piqa=[json.loads(z) for z in raw_piqa.decode().splitlines()];labels=[int(z) for z in raw_labels.decode().splitlines()]
    lambada=[json.loads(z) for z in raw_lambada.decode().splitlines()]
    assert len(piqa)==len(labels)==1838 and len(lambada)==5153
    p_examples=[];p_char=[]
    for d in piqa:
        ctx=enc.encode_ordinary('Question: '+d['goal']+'\nAnswer:')
        for key in ('sol1','sol2'):
            p_examples.append((ctx,enc.encode_ordinary(' '+d[key])));p_char.append(len(d[key]))
    l_examples=[]
    for d in lambada:
        ctx,word=d['text'].rsplit(' ',1);l_examples.append((enc.encode_ordinary(ctx),enc.encode_ordinary(' '+word)))
    evidence=json.loads((ROOT/'EVIDENCE.json').read_text());results=[]
    for arm,terminal in evidence['terminal'].items():
        record=evidence['results'][f"{arm}-s{terminal['steps']}"];model=make(arm).cuda();state=load_checkpoint(record['checkpoint']);model.load_state_dict(state['model']);del state;model.eval()
        p_rows,p_tokens=score(model,p_examples);l_rows,l_tokens=score(model,l_examples);p_result=[]
        for i,gold in enumerate(labels):
            pair=p_rows[2*i:2*i+2]
            predictions={'token_normalized':int(np.argmin([z['mean_token_nll'] for z in pair])),
                'summed':int(np.argmin([z['sum_nll'] for z in pair])),
                'character_normalized':int(np.argmin([z['sum_nll']/max(1,p_char[2*i+j]) for j,z in enumerate(pair)]))}
            p_result.append({'index':i,'gold':gold,'predictions':predictions,'option_scores':pair})
        row={'variant':arm,'training_tokens':record['tokens_seen'],'checkpoint_sha256':record['checkpoint']['checkpoint_sha256'],
            'piqa':{'examples':len(piqa),'accuracy':{key:sum(z['predictions'][key]==z['gold'] for z in p_result)/len(p_result) for key in ('token_normalized','summed','character_normalized')},'rows':p_result},
            'lambada':{'examples':len(lambada),'accuracy':sum(z['all_greedy_correct'] for z in l_rows)/len(l_rows),'mean_word_nll':float(np.mean([z['sum_nll'] for z in l_rows])),'rows':l_rows},
            'evaluation_input_forwards_including_padding':p_tokens+l_tokens,'optimizer_exposures':0}
        row['lambada']['word_perplexity']=math.exp(row['lambada']['mean_word_nll'])
        dump(ROOT/(arm+'.BENCHMARKS.json'),row);results.append(row);print('BENCHMARK',arm,row['piqa']['accuracy'],row['lambada']['accuracy'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    result={'complete':True,'results':results,'source_files':{'piqa_valid_sha256':hashlib.sha256(raw_piqa).hexdigest(),'piqa_labels_sha256':hashlib.sha256(raw_labels).hexdigest(),'lambada_sha256':hashlib.sha256(raw_lambada).hexdigest()},
        'sources':sources,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(ROOT/'BENCHMARKS.json',result)
if __name__=='__main__':main()
