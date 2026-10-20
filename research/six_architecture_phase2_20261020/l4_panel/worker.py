import importlib.metadata,json,pathlib,subprocess,sys,time
r=pathlib.Path('/content/tam-l4-panel-20261020')
subprocess.run([sys.executable,'-m','pip','install','-q','tiktoken','huggingface_hub','pyarrow'],check=True)
import torch
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
(r/'ENVIRONMENT.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'python':sys.version,
 'packages':{n:importlib.metadata.version(n) for n in ('numpy','tiktoken','huggingface_hub','pyarrow')}}))
subprocess.run([sys.executable,'-u','-c',"import prepare_corpus as p;from pathlib import Path;p.ROOT=Path('/content/tam-l4-panel-20261020');p.DATA=p.ROOT/'data';p.main()"],check=True,cwd=r)
subprocess.run([sys.executable,'-u','l4_controller.py'],check=True,cwd=r)
