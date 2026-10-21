"""Posthoc CE decomposition on the consumed development panel. No training."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);args=p.parse_args()
root=Path(__file__).parent
# Reuse the exact frozen control class/evaluation inputs; __main__ never runs.
sys.argv=['pgw_strong_control.py','--repo',str(args.repo.resolve())]
spec=importlib.util.spec_from_file_location('pgw_frozen_control',root/'pgw_strong_control.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
out=root/'results'/'pgw_control_diagnostic';out.mkdir(exist_ok=True)
for path in sorted((root/'results'/'pgw_control').glob('*strong_transformer*.RESULT.json')):
    record=json.loads(path.read_text());ident=record['attempt'];dest=out/(ident+'.DIAGNOSTIC.json')
    if dest.exists():continue
    ck=path.with_name(path.name.replace('.RESULT.json','.pt'))
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==record['checkpoint_sha256']
    model=module.StrongControl();model.load_state_dict(torch.load(ck,weights_only=True));model.eval();rows=[]
    with torch.no_grad():
        for delay,(x,anchor,y) in module.evals.items():
            logits=torch.cat([model(xx,aa) for xx,aa in zip(x.split(24),anchor.split(24))]);lp=logits.log_softmax(-1)
            present=y!=32;missing=~present
            binary_ce=torch.where(missing,-lp[:,32],-torch.logsumexp(lp[:,:32],-1)).mean()
            payload_ce=F.cross_entropy(logits[present,:32],y[present])
            total=F.cross_entropy(logits,y)
            torch.testing.assert_close(total,binary_ce+present.float().mean()*payload_ce)
            predicted_missing=lp[:,32].exp()>=.5
            rows.append({'delay':delay,'answer_accuracy':float((logits.argmax(-1)==y).float().mean()),
               'binary_presence_accuracy_at_fixed_half_threshold':float((predicted_missing==missing).float().mean()),
               'conditional_payload_accuracy_excluding_NOT_FOUND':float((logits[present,:32].argmax(-1)==y[present]).float().mean()),
               'binary_presence_nll':float(binary_ce),'conditional_payload_nll':float(payload_ce),'total_nll':float(total),
               'mean_not_found_probability_present':float(lp[present,32].exp().mean()),
               'mean_not_found_probability_missing':float(lp[missing,32].exp().mean())})
    result={'attempt':ident,'checkpoint_sha256':record['checkpoint_sha256'],'rows':rows,
       'classification':'POSTHOC_SAME_CONSUMED_DEVELOPMENT_PANEL_NO_TUNING',
       'conditional_payload_uses_known_presence_for_diagnosis_only':True,'new_training_exposures':0}
    with dest.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
