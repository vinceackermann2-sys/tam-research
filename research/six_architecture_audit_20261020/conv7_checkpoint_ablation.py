"""Inference-only development erasure; never overlaps the timed training."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from conv7_language import ScheduledLM
from conv7_fineweb_recovery import evaluate
root=Path('/content/tam-six-20261020');out=root/'results/conv7_recovery'
assert json.loads((root/'recovery_status.json').read_text())['complete'],'Training is still active'
name='conv7-recovery-w192-conv7-202610262'
record=json.loads((out/(name+'.RESULT.json')).read_text());checkpoint=out/(name+'.pt')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(checkpoint)==record['checkpoint_sha256']
assert sha(root/'conv7_language.py')==record['model_source_sha256']
assert sha(root/'fineweb25m/validation.bin')==record['data']['validation']['token_sha256']
torch.set_num_threads(2);model=ScheduledLM(192,'conv7').cuda()
model.load_state_dict(torch.load(checkpoint,weights_only=True));val=np.fromfile(root/'fineweb25m/validation.bin',dtype=np.uint16)
normal=evaluate(model,val);gains={str(i+1):float(b.conv_gain.detach()) for i,b in enumerate(model.blocks) if b.conv is not None}
with torch.no_grad():
    for b in model.blocks:
        if b.conv is not None:b.conv_gain.zero_()
erased=evaluate(model,val)
result={'checkpoint_sha256':record['checkpoint_sha256'],'original_gains':gains,'recorded_terminal_nll':record['validation_nll'],
    'replayed_nll':normal,'replay_delta':normal-record['validation_nll'],'gains_zero_nll':erased,'erasure_nll_delta':erased-normal,
    'source_sha256':sha(Path(__file__)),'classification':'SAME_DEVELOPMENT_CHECKPOINT_CAUSAL_DEPENDENCE_NOT_TRAINING_ATTRIBUTION',
    'new_training_exposures':0,'breakthrough_demonstrated':False}
with (out/(name+'.ABLATION.json')).open('x') as f:json.dump(result,f,indent=2)
print('ABLATION',json.dumps(result))
