"""Transport-only: private signed downloads, then exact checkpoint verification.

No optimizer or training import. Frozen restore_worker independently waits for
RESTORE_COMPLETE and reproduces/validates the data before the original controller.
"""
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

ROOT=Path('/content/tam-l4-panel-20261020')
PRIVATE=ROOT/'work/RESTORE_DOWNLOADS.private.json'


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for raw in iter(lambda:stream.read(8*1024**2),b''):result.update(raw)
    return result.hexdigest()


def restore(entry):
    manifest=entry['manifest'];url=entry['signed_url']
    parsed=urlsplit(url)
    assert parsed.scheme=='https' and parsed.hostname.endswith('.githubusercontent.com')
    directory=(ROOT/'results'/f"{manifest['variant']}-s{manifest['step']}").resolve()
    assert directory.is_relative_to((ROOT/'results').resolve())
    directory.mkdir(exist_ok=True)
    state=directory/'cache-download.private.pt'
    total=sum(row['bytes'] for row in manifest['chunks'])
    expected=manifest['checkpoint_sha256']
    if not(state.exists() and state.stat().st_size==total and digest(state)==expected):
        temp=state.with_suffix('.downloading')
        for attempt in range(3):
            try:
                with requests.get(url,stream=True,allow_redirects=False,timeout=(15,120)) as response:
                    assert response.status_code==200,'Signed asset download failed'
                    h=hashlib.sha256();count=0
                    with temp.open('wb') as stream:
                        for raw in response.iter_content(8*1024**2):
                            stream.write(raw);h.update(raw);count+=len(raw)
                        stream.flush();os.fsync(stream.fileno())
                    assert count==total and h.hexdigest()==expected,'Cached optimizer state failed integrity check'
                break
            except Exception as error:
                print('CACHE_DOWNLOAD_RETRY',manifest['variant'],attempt+1,type(error).__name__,flush=True)
                if attempt==2:raise
                time.sleep(3)
        temp.replace(state)
    with state.open('rb') as stream:
        for row in manifest['chunks']:
            target=(ROOT/row['path']).resolve()
            assert target.is_relative_to(directory)
            raw=stream.read(row['bytes'])
            assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
            temp=target.with_suffix('.restoring')
            with temp.open('wb') as output:
                output.write(raw);output.flush();os.fsync(output.fileno())
            temp.replace(target)
        assert not stream.read(1)
    # Verify the exact files the immutable trainer will consume, not just input.
    whole=hashlib.sha256()
    for row in manifest['chunks']:
        path=ROOT/row['path'];assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
        with path.open('rb') as stream:
            for raw in iter(lambda:stream.read(8*1024**2),b''):whole.update(raw)
    assert whole.hexdigest()==expected
    (directory/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    state.unlink()
    print('CACHE_CHECKPOINT_VERIFIED',manifest['variant'],manifest['step'],flush=True)
    return {'variant':manifest['variant'],'step':manifest['step'],'sha256':expected,'asset_id':entry['asset_id']}


def main():
    start=time.monotonic()
    with (ROOT/'CACHE_DOWNLOAD_CLAIM.json').open('x') as stream:
        json.dump({'utc':time.time(),'source_sha256':digest(Path(__file__))},stream)
    private=json.loads(PRIVATE.read_text())
    assert len(private['entries'])==4
    assert not(ROOT/'RESTORE_COMPLETE.json').exists()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        verified=list(pool.map(restore,private['entries']))
    result={'complete':True,'verified':verified,'seconds':time.monotonic()-start,
            'release_id':private['release_id'],'source_sha256':digest(Path(__file__))}
    (ROOT/'RESTORE_CACHE_VERIFICATION.json').write_text(json.dumps(result,indent=2))
    PRIVATE.unlink()
    temp=ROOT/'RESTORE_COMPLETE.tmp'
    temp.write_text(json.dumps({'verified':[row['sha256'] for row in verified]}))
    temp.replace(ROOT/'RESTORE_COMPLETE.json')
    print('ALL_CACHED_OPTIMIZER_STATES_VERIFIED_CONTINUATION_RELEASED',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        (ROOT/'CACHE_RESTORE_ERROR.json').write_text(json.dumps({'utc':time.time(),'type':type(error).__name__}))
        print('CACHE_RESTORE_ERROR',type(error).__name__,flush=True)
        sys.exit(1)
