"""Audit recorded caps, immutable trained sources and available checkpoint bytes."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(p.read_text())
def main():
    source_checks=[];weight_checks=[];caps=[]
    def source(record,file,key='source_sha256'):
        assert record[key]==sha(ROOT/file),(file,record.get('seed'));source_checks.append(file)
    for p in (ROOT/'results/si_repair').glob('*.RESULT.json'):
        r=load(p);source(r,'si_head_repair.py');caps.append((r['parameters'],r['accounted_training_positions']))
        for suffix,key in [('repair','checkpoint_sha256'),('policy','baseline_checkpoint_sha256')]:
            q=p.parent/f"si-{r['seed']}-{suffix}.pt";assert sha(q)==r[key];weight_checks.append(str(q.relative_to(ROOT)))
    for p in (ROOT/'results/rlt_gate').glob('*.RESULT.json'):
        r=load(p);source(r,'rlt_gate_repair.py');caps.append((r['parameters'],r['training_exposures']))
        q=p.parent/f"rlt-{r['arm']}-{r['seed']}.pt";assert sha(q)==r['checkpoint_sha256'];weight_checks.append(str(q.relative_to(ROOT)))
    for p in (ROOT/'results/rlt_probe').glob('*.RESULT.json'):
        r=load(p);source(r,'rlt_readout_probe.py');caps.append((r['parameters_including_frozen_backbone'],r['accounted_training_positions']+1_363_200))
        q=p.parent/f"rlt-probe-{r['arm']}-{r['seed']}.pt";assert sha(q)==r['checkpoint_sha256'];weight_checks.append(str(q.relative_to(ROOT)))
    for folder,file in [('chm_address','chm_address_diagnostic.py'),('pgw_trace','pgw_workspace_diagnostic.py'),('si_components','si_component_diagnostic.py'),('rlt_state','rlt_state_diagnostic_v2.py'),('rlt_probe_ood','rlt_probe_ood.py')]:
        for p in (ROOT/'results'/folder).glob('*.DIAGNOSTIC.json'):
            r=load(p);source(r,file);assert r.get('optimizer_updates',r.get('training_exposures',0))==0
    language=[]
    for trial in (ROOT,ROOT/'l4_panel'):
        if not (trial/'EVIDENCE.json').exists():continue
        e=load(trial/'EVIDENCE.json')
        for r in e['results'].values():
            source(r,'train_panel.py');caps.append((r['parameters'],r['tokens_seen']+e.get('charges',{}).get(r['variant'],0)));language.append(r)
            for name,digest in r['model_source_sha256'].items():assert sha(ROOT/name)==digest
            if trial!=ROOT:assert r['controller_source_sha256']==sha(trial/'l4_controller.py')
        if trial==ROOT:
            manifests=[r['checkpoint'] for r in e['results'].values() if (trial/'results'/f"{r['variant']}-s{r['steps']}"/'MANIFEST.json').exists()]
        else:
            catalog=load(trial/'CHECKPOINT_CATALOG.json') if (trial/'CHECKPOINT_CATALOG.json').exists() else {}
            manifests=[m for m in catalog.values() if not m.get('pruned')]
        for m in manifests:
            whole=hashlib.sha256()
            for row in m['chunks']:
                q=(trial/row['path']).resolve();assert q.is_relative_to(trial.resolve()) and q.stat().st_size==row['bytes'] and sha(q)==row['sha256']
                with q.open('rb') as f:
                    for b in iter(lambda:f.read(8*1024**2),b''):whole.update(b)
            assert whole.hexdigest()==m['checkpoint_sha256'];weight_checks.append(f"{trial.name}:{m['variant']}-s{m['step']} full optimizer chunks")
    for parameters,tokens in caps:assert parameters<=100_000_000 and tokens<=2_000_000_000
    if (ROOT/'l4_panel/data/DATA.json').exists():
        a=load(ROOT/'data/DATA.json');b=load(ROOT/'l4_panel/data/DATA.json');assert a['splits']==b['splits']
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_checked_caps_passed':True,'immutable_source_checks':len(source_checks),
        'checkpoint_sets_checked':len(weight_checks),'checkpoint_sets':weight_checks,'source_files':sorted(set(source_checks)),
        'largest_recorded_language_parameters':max(r['parameters'] for r in language),'largest_recorded_language_exposures':max(r['tokens_seen'] for r in language),
        'language_confirmation_complete':all((ROOT/'l4_panel'/n).exists() and load(ROOT/'l4_panel'/n).get('complete',False) for n in ('EVIDENCE.json','INFERENCE.json','BENCHMARKS.json')),'infrastructure_unknowns_preserved':'INTERRUPTED_PANEL.json; missing parent binary is not a failure outcome'}
    (ROOT/'AUDIT_CURRENT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
