import json,pathlib,subprocess,sys,time
r=pathlib.Path('/content/tam-l4-panel-20261020')
(r/'STATUS.json').write_text(json.dumps({'phase':'recovery_preparing_corpus','complete':False}))
subprocess.run([sys.executable,'-m','pip','install','-q','tiktoken==0.14.0','huggingface_hub==1.33.0','pyarrow==23.0.1'],check=True)
import torch
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
(r/'ENVIRONMENT.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'python':sys.version,'recovery':True}))
subprocess.run([sys.executable,'-u','-c',"import prepare_corpus as p;from pathlib import Path;p.ROOT=Path('/content/tam-l4-panel-20261020');p.DATA=p.ROOT/'data';p.main()"],check=True,cwd=r)
(r/'STATUS.json').write_text(json.dumps({'phase':'awaiting_checkpoint_restore','complete':False}))
while not (r/'RESTORE_COMPLETE.json').exists():time.sleep(5)
resume=json.loads((r/'RESUME.json').read_text())
if resume['evidence'].get('complete'):
    (r/'EVIDENCE.json').write_text(json.dumps(resume['evidence']));(r/'STATUS.json').write_text(json.dumps({'phase':'finished','complete':True}))
else:
    subprocess.run([sys.executable,'-u','l4_controller.py'],check=True,cwd=r)
import train_panel as p
p.ROOT=r;p.OUT=r/'results';p.DATA=r/'data'
import inference_benchmark as i
i.ROOT=r;i.main()
import language_benchmarks as b
b.ROOT=r;b.main()
