"""Frozen source, document hash split, >2B unique-position training stream.

Build disk streams incrementally. Exact duplicate documents are removed. This
does not certify removal of near duplicates or data from other pretraining.
"""
import hashlib,json,os,time
from pathlib import Path
import numpy as np

REV='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
ROOT=Path('/content/tam-phase2-20261020');DATA=ROOT/'data'
LIMITS={'train':2_000_001_024,'validation':1_048_576,'test':2_097_152}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def main():
    import requests,pyarrow.parquet as pq,tiktoken
    from huggingface_hub import hf_hub_download
    DATA.mkdir(parents=True,exist_ok=True)
    with (DATA/'CLAIM.json').open('x') as f:json.dump({'limits':LIMITS,'revision':REV},f)
    files=requests.get(f'https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/tree/{REV}/sample/10BT',params={'recursive':'true','limit':1000},timeout=30)
    files.raise_for_status();inventory={r['path']:r for r in files.json()}
    enc=tiktoken.get_encoding('gpt2');counts={s:0 for s in LIMITS};docs={s:0 for s in LIMITS}
    digests={s:hashlib.sha256() for s in LIMITS};seen=set();sources=[];t=time.monotonic()
    handles={s:(DATA/(s+'.bin.part')).open('xb') for s in LIMITS}
    # Separate later shard for validation/test; training uses only buckets2..19.
    for index in (13,0,1,2,3):
        name=f'sample/10BT/{index:03d}_00000.parquet'
        path=Path(hf_hub_download('HuggingFaceFW/fineweb-edu',name,repo_type='dataset',revision=REV))
        expected=inventory[name]['lfs']['oid'];actual=sha(path)
        if actual!=expected:raise ValueError('source parquet SHA mismatch')
        sources.append({'path':name,'bytes':path.stat().st_size,'sha256':actual})
        print('SOURCE_VERIFIED',name,flush=True)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024,columns=['text']):
            texts=[];splits=[];hashes=[]
            for text in batch.column(0).to_pylist():
                h=hashlib.sha256(text.encode()).digest();bucket=int.from_bytes(h[:4],'big')%20
                split=('validation' if bucket==0 else 'test' if bucket==1 else None) if index==13 else ('train' if bucket>=2 else None)
                if split is None or counts[split]>=LIMITS[split] or h in seen:continue
                seen.add(h);texts.append(text);splits.append(split);hashes.append(h)
            encoded=enc.encode_ordinary_batch(texts,num_threads=4) if texts else []
            for tokens,split,h in zip(encoded,splits,hashes):
                room=LIMITS[split]-counts[split]
                if room<=0:continue
                a=np.asarray((tokens+[50256])[:room],dtype=np.uint16)
                handles[split].write(a.tobytes());counts[split]+=len(a);docs[split]+=1;digests[split].update(h)
            (ROOT/'DATA_PROGRESS.json').write_text(json.dumps({'tokens':counts,'documents':docs,'elapsed_seconds':time.monotonic()-t,'current_source':name}))
            if (index==13 and counts['validation']>=LIMITS['validation'] and counts['test']>=LIMITS['test']) or (index!=13 and counts['train']>=LIMITS['train']):break
        if index!=13 and counts['train']>=LIMITS['train']:break
    if counts!=LIMITS:raise ValueError('incomplete corpus')
    result={'dataset':'HuggingFaceFW/fineweb-edu','config':'sample-10BT','revision':REV,'sources':sources,
        'tokenizer':'gpt2 tiktoken','eos':50256,'split_rule':'SHA256(document) first32bits mod20: validation0/test1 from shard013, train2..19 from shards000 onward',
        'deduplication':'exact document SHA only; retained shared set across all processed source shards',
        'train_order':'source order, one pass; no random-window replacement','splits':{}}
    for split,f in handles.items():
        f.flush();os.fsync(f.fileno());f.close();p=DATA/(split+'.bin.part');p.replace(DATA/(split+'.bin'))
        result['splits'][split]={'tokens':counts[split],'documents':docs[split],'document_order_digest':digests[split].hexdigest(),'token_sha256':sha(DATA/(split+'.bin'))}
    result['preparation_seconds']=time.monotonic()-t
    result['source_sha256']=sha(Path(__file__))
    with (DATA/'DATA.json').open('x') as f:json.dump(result,f,indent=2)
    print('CORPUS_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
