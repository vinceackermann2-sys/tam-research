"""Audit recorded caps, immutable trained sources and available checkpoint bytes."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(p.read_text())
def main():
    source_checks=[];weight_checks=[];caps=[];reservation_checks=[];charge_ledgers=[]
    def source(record,file,key='source_sha256'):
        assert record[key]==sha(ROOT/file),(file,record.get('seed'));source_checks.append(file)
    if (ROOT/'results/si_memory_revision/PREREGISTRATION.json').exists():
        prereg=load(ROOT/'results/si_memory_revision/PREREGISTRATION.json')
        native=Path(os.environ.get('SI_REPO',str(ROOT.parents[1]/'work/si')))
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=native,text=True).strip()=='9298c117e736676269107374a402519e45e52f71'
        for name,digest in prereg['native_source_sha256'].items():
            assert sha(native/'tam_research'/name)==digest;source_checks.append('native_si/'+name)
    for p in (ROOT/'results/si_repair').glob('*.RESULT.json'):
        r=load(p);source(r,'si_head_repair.py');caps.append((r['parameters'],r['accounted_training_positions']))
        for suffix,key in [('repair','checkpoint_sha256'),('policy','baseline_checkpoint_sha256')]:
            q=p.parent/f"si-{r['seed']}-{suffix}.pt";assert sha(q)==r[key];weight_checks.append(str(q.relative_to(ROOT)))
    for p in (ROOT/'results/si_memory_revision').glob('*.DIAGNOSTIC.json'):
        r=load(p);source(r,'si_memory_revision.py')
        assert r['protocol_sha256']==sha(ROOT/'SI_MEMORY_REVISION_PROTOCOL.md')
        assert r['optimizer_updates']==0 and r['training_exposures']==0
        assert r['total_accounted_training_and_feedback_positions']==r['prior_accounted_training_positions']+r['additional_online_feedback_raw_positions']
        caps.append((r['parameters'],r['total_accounted_training_and_feedback_positions']))
        q=p.parent/f"si-memory-{r['seed']}.pt";assert sha(q)==r['checkpoint_sha256'];weight_checks.append(str(q.relative_to(ROOT)))
        original=ROOT/f"results/si_repair/si-{r['seed']}-repair.pt";assert sha(original)==r['input_checkpoint_sha256']
        assert r['neural_parameters_unchanged'] and r['feedback_events']==56
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
        if trial!=ROOT:
            controller_digest=sha(trial/'l4_controller.py')
            assert e['controller_source_sha256']==controller_digest
            ledger=e.get('lost_reservations',[]);seen=set();totals={arm:0 for arm in e['charges']}
            for row in ledger:
                reservation=dict(row['reservation']);digest=reservation.pop('digest')
                assert hashlib.sha256(json.dumps(reservation,sort_keys=True).encode()).hexdigest()==digest
                assert digest not in seen and digest in e.get('charged_reservation_digests',[])
                seen.add(digest)
                expected=(reservation['end']-max(row['verified_step_at_loss'],reservation['start']))*8192
                assert expected==row['charged_exposures'] and expected>0
                totals[reservation['variant']]+=expected
            assert totals==e['charges']
            charge_ledgers.append({'trial':e['trial'],'charges':totals,'lost_reservations':len(ledger)})
            reservations={p.name:load(p) for p in (trial/'reservations').glob('*.json')}
            if (trial/'ACTIVE_SEGMENT.json').exists():reservations['active']=load(trial/'ACTIVE_SEGMENT.json')
            for key,reservation in reservations.items():
                payload=dict(reservation);digest=payload.pop('digest')
                assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==digest
                assert reservation['controller_source_sha256']==controller_digest
                assert reservation['reserved_exposures']==(reservation['end']-reservation['start'])*8192
                assert reservation['cumulative_upper_bound']==reservation['end']*8192+reservation['charged_lost_exposures']<=2_000_000_000
                reservation_checks.append({'trial':e['trial'],'reservation':reservation['id'],
                    'digest':digest,'cumulative_upper_bound':reservation['cumulative_upper_bound']})
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
    recovery_checks=[]
    for path in (ROOT/'l4_panel/recovery_archive').glob('*/RESTORE_CACHE_VERIFICATION.json'):
        verified=load(path);directory=path.parent
        assert verified['complete'] and verified['source_sha256']==sha(directory/'cache_download_worker.py')
        plan=load(directory.with_suffix('.json'))
        expected={m['checkpoint_sha256'] for m in plan['manifests']}
        assert {m['sha256'] for m in verified['verified']}==expected
        cache=load(directory/'RELEASE_CACHE.json');assert cache['draft']
        assert {m['sha256'] for m in cache['assets']}==expected
        remote=load(directory/'REMOTE_SOURCE_AUDIT.json')
        for name,digest in remote['source_sha256'].items():
            q=ROOT/'l4_panel'/name if name in ('l4_controller.py','RECOVERY_PROTOCOL.md','INFERENCE_PROTOCOL.md') else ROOT/name
            assert sha(q)==digest
        assert remote['restore_worker_sha256']==sha(directory/'restore_worker.py')
        recovery_checks.append({'archive':str(directory.relative_to(ROOT)),
            'full_checkpoint_hashes':sorted(expected),'download_seconds':verified['seconds'],
            'scientific_sources_verified':len(remote['source_sha256']),'cache_remains_draft':True})
    if (ROOT/'l4_panel/data/DATA.json').exists():
        a=load(ROOT/'data/DATA.json');b=load(ROOT/'l4_panel/data/DATA.json');assert a['splits']==b['splits']
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_checked_caps_passed':True,'immutable_source_checks':len(source_checks),
        'checkpoint_sets_checked':len(weight_checks),'checkpoint_sets':weight_checks,'source_files':sorted(set(source_checks)),
        'largest_recorded_language_parameters':max(r['parameters'] for r in language),'largest_recorded_language_exposures':max(r['tokens_seen'] for r in language),
        'largest_committed_plus_lost_exposure_bound':max(tokens for parameters,tokens in caps),
        'largest_reserved_cumulative_exposure_bound':max((r['cumulative_upper_bound'] for r in reservation_checks),default=0),
        'reservation_checks':reservation_checks,'charged_lost_reservation_ledgers':charge_ledgers,
        'cached_runtime_restoration_checks':recovery_checks,
        'language_confirmation_complete':all((ROOT/'l4_panel'/n).exists() and load(ROOT/'l4_panel'/n).get('complete',False) for n in ('EVIDENCE.json','INFERENCE.json','BENCHMARKS.json')),'infrastructure_unknowns_preserved':'INTERRUPTED_PANEL.json; missing parent binary is not a failure outcome'}
    (ROOT/'AUDIT_CURRENT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
