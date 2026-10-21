"""Read-only result/checkpoint audit. No training and no result mutation."""
import hashlib,json
from pathlib import Path
root=Path(__file__).parent;completed=[];missing=[];verified=[]
for path in sorted((root/'results').glob('**/*.RESULT.json')):
    d=json.loads(path.read_text());params=d['parameters'];tokens=d.get('tokens_seen',d.get('training_token_exposures'))
    assert 0<params<=100_000_000,(path,'parameter ceiling')
    assert 0<tokens<=2_000_000_000,(path,'token ceiling')
    assert d.get('breakthrough_demonstrated',False) is False,(path,'unexpected breakthrough claim')
    ck=path.with_name(path.name.replace('.RESULT.json','.pt'))
    if ck.exists():
        assert hashlib.sha256(ck.read_bytes()).hexdigest()==d['checkpoint_sha256'],(path,'checkpoint hash')
        verified.append(str(ck.relative_to(root)))
    else:missing.append(str(ck.relative_to(root)))
    source=d.get('model_source_sha256')
    if source:assert hashlib.sha256((root/'conv7_language.py').read_bytes()).hexdigest()==source
    source=d.get('script_sha256')
    if source:
        if d['attempt'].startswith('conv7-w128'):script=root/'conv7_language.py'
        elif d['attempt'].startswith('chm-broad'):script=root/'chm_broad_train.py'
        elif d['attempt'].startswith('conv7-recovery'):script=root/'conv7_fineweb_recovery.py'
        elif d['attempt'].startswith('cpw-recovery'):script=root/'cpw_binding_recovery.py'
        elif d['attempt'].startswith('conv7-fineweb'):script=root/'conv7_fineweb.py'
        elif d['attempt'].startswith('cpw-calibration'):script=root/'cpw_binding_calibration.py'
        else:script=root/'run_pilots.py'
        assert hashlib.sha256(script.read_bytes()).hexdigest()==source,(path,'source hash')
    if d['attempt'].startswith('pgw-control'):
        assert hashlib.sha256((root/'pgw_strong_control.py').read_bytes()).hexdigest()==d['source_sha256']
    completed.append(d)
report={'completed_models':len(completed),'largest_trained_parameters':max(d['parameters'] for d in completed),
 'largest_recorded_training_exposures':max(d.get('tokens_seen',d.get('training_token_exposures')) for d in completed),
 'total_recorded_training_exposures':sum(d.get('tokens_seen',d.get('training_token_exposures')) for d in completed),
 'unverified_original_extension_maximum_planned_exposures':5*51_200_000+3*24_997_888,
 'verified_local_checkpoints':len(verified),'unretrieved_checkpoint_paths':missing,
 'all_recorded_models_within_user_ceilings':True,
 'si_extra_fast_and_latent_replay_accounting':'See EXECUTION_NOTES.md; direct modality-input counts are not language tokens.'}
print(json.dumps(report,indent=2))
