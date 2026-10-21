# Run this ONE cell in the EXISTING A100 notebook if the agent link is unavailable.
# It exports files only. It never starts, restarts or changes training.
from pathlib import Path
import hashlib, json, zipfile
from google.colab import files
root=Path('/content/tam-six-20261020')
assert root.is_dir(), 'Wrong runtime: research directory is absent. Do not rerun training.'
status={p.name:p.read_text() for p in root.glob('*status.json')}
print('Recorded worker status:',status)
selected=[]
for p in root.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(root)
    if rel.parts[0] in ('main','cpw','si','rlt','wikitext2','fineweb25m'):continue
    if p.suffix in ('.json','.pt','.log','.py','.md'):selected.append(p)
for folder in ('wikitext2','fineweb25m'):
    p=root/folder/'DATA.json'
    if p.exists():selected.append(p)
dest=Path('/content/tam-six-20261020-export.zip')
manifest={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in selected}
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_STORED) as z:
    z.writestr('EXPORT_MANIFEST.json',json.dumps(manifest,indent=2))
    for p in selected:z.write(p,str(p.relative_to(root)))
print('EXPORT',dest.stat().st_size,'bytes',len(selected),'files')
print('If training is still running this is a partial snapshot; wait for completion and export again.')
files.download(str(dest))
