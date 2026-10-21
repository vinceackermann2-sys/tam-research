"""Summarize immutable experimental records; never rewrite executed sources."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
def write(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,allow_nan=False))
def load(name):return json.loads((ROOT/name).read_text())
def records(pattern):return [json.loads(p.read_text()) for p in sorted(ROOT.glob(pattern))]
def paired(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    assert a.ndim==1 and a.shape==b.shape and len(a)>0
    assert np.isfinite(a).all() and np.isfinite(b).all()
    difference=a-b
    rng=np.random.default_rng(202610399);samples=[]
    for _ in range(2000):samples.append(float(difference[rng.integers(0,len(difference),len(difference))].mean()))
    return {'delta_mean_nll':float(difference.mean()),'approximate_95_percent_block_bootstrap_interval':np.quantile(samples,[.025,.975]).tolist(),
        'blocks':len(difference),'interpretation':'paired token-block resampling; adjacent blocks/documents dependent; descriptive, no multiplicity correction'}

def main():
    from analyze_l4_panel import main as analyze_l4
    l4=analyze_l4(ROOT)
    evidence=load('EVIDENCE.json') if (ROOT/'EVIDENCE.json').exists() else {'results':{},'gates':[],'terminal':{},'complete':False}
    si=records('results/si_repair/*.RESULT.json');rlt=records('results/rlt_gate/*.RESULT.json')
    si_memory=records('results/si_memory_revision/*.DIAGNOSTIC.json')
    pgw=records('results/pgw_trace/*.DIAGNOSTIC.json');chm=records('results/chm_address/*.DIAGNOSTIC.json')
    probes=records('results/rlt_probe/*.RESULT.json');probe_ood=records('results/rlt_probe_ood/*.DIAGNOSTIC.json')
    comparisons=[];ledger=[]
    for key,r in evidence['results'].items():
        assert r['parameters']<=100_000_000 and r['tokens_seen']<=2_000_000_000
        ledger.append({'lineage':r['variant'],'parameters':r['parameters'],'tokens':r['tokens_seen'],'step':r['steps'],'validation_nll':r['validation']['nll'],'train_tps':r['train_tokens_per_second']})
        control=evidence['results'].get(f"transformer-s{r['steps']}")
        if control and r['variant']!='transformer':
            comparisons.append({'variant':r['variant'],'tokens':r['tokens_seen'],**paired(r['validation']['block_nll'],control['validation']['block_nll']),
                'training_speed_ratio':r['train_tokens_per_second']/control['train_tokens_per_second'],
                'parameters':r['parameters'],'control_parameters':control['parameters']})
    for r in si:assert r['parameters']<=100_000_000 and r['accounted_training_positions']<=2_000_000_000 and r['policy_logits_max_delta']==0
    for r in rlt:assert r['parameters']<=100_000_000 and r['training_exposures']<=2_000_000_000
    write('LANGUAGE_COMPARISONS.json',comparisons)
    with (ROOT/'LANGUAGE_RESULTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['lineage','parameters','tokens','step','validation_nll','train_tps']);w.writeheader();w.writerows(ledger)
    l4_rows=l4['segment_records'] if l4 else []
    all_parameters=[r['parameters'] for r in ledger]+[r['parameters'] for r in l4_rows]
    all_exposures=[r['tokens'] for r in ledger]+[r['tokens_seen'] for r in l4_rows]
    confirmation_complete=bool(l4 and l4['complete']) and all((ROOT/'l4_panel'/name).exists() and load('l4_panel/'+name).get('complete',False) for name in ('INFERENCE.json','BENCHMARKS.json'))
    result={'language_complete':confirmation_complete,'language_training_complete':bool(l4 and l4['complete']),
        'interrupted_a100_complete':evidence.get('complete',False),
        'language_lineages':len({('interrupted_a100',r['lineage']) for r in ledger}|{('l4_paired',r['variant']) for r in l4_rows}),
        'language_segments_complete':len(ledger)+len(l4_rows),'interrupted_a100_segments_complete':len(ledger),
        'language_max_parameters':max(all_parameters,default=0),
        'language_max_exposures':max(all_exposures,default=0),'si_lineages':len(si),'rlt_lineages':len(rlt),
        'si_native_memory_revision_diagnostics':len(si_memory),
        'pgw_diagnostics':len(pgw),'chm_diagnostics':len(chm),'all_recorded_caps_passed':True,'breakthrough_demonstrated':False,
        'rlt_frozen_state_probes':len(probes),'rlt_probe_ood_gates_passed':sum(r['gate_passed'] for r in probe_ood),
        'gates':l4['gates'] if l4 else [],'interrupted_a100_gates':evidence['gates'],
        'l4_scientific_comparisons':len(l4['scientific_comparisons']) if l4 else 0,
        'l4_segment_records':len(l4['segment_records']) if l4 else 0,'l4_gates':l4['gates'] if l4 else []}
    write('RESEARCH_STATUS.json',result)
    lines=['# Phase-two measured results','',f"L4 training complete: {bool(l4 and l4['complete'])}. See [L4_RESULTS.md](L4_RESULTS.md) for the separate paired trial.",
        '', 'The following language table belongs to the interrupted A100 trial. Candidate outcomes are unknown.', '',
        '| Model | Parameters | Training exposures | Development NLL | Training tokens/s |','|---|---:|---:|---:|---:|']
    for r in ledger:lines.append(f"| {r['lineage']} | {r['parameters']:,} | {r['tokens']:,} | {r['validation_nll']:.6f} | {r['train_tps']:,.0f} |")
    lines+=['','Comparisons at the same exposure count are saved in LANGUAGE_COMPARISONS.json.',
        'Intervals resample token blocks and are descriptive; a one-seed study is not a robust superiority result.','',
        '| SI seed | Direct policy | Native gated search before | Native gated after | Forced depth2 after | Reward MSE before/after |','|---|---:|---:|---:|---:|---:|']
    for r in si:
        b=r['baseline'];a=r['repaired'];lines.append(f"| {r['seed']} | {a['accuracy']['direct']:.2%} | {b['accuracy']['native_gated']:.2%} | {a['accuracy']['native_gated']:.2%} | {a['accuracy']['forced_depth2']:.2%} | {b['executed_action_reward_mse']:.6f}/{a['executed_action_reward_mse']:.6f} |")
    lines+=['','All SI non-head weights and direct logits are unchanged by the repair. This small finite cue task does not demonstrate an advantage over the direct policy.','',
        '| RLT seed | Training recipe | Accuracy at gaps127/255/511/1023 | All-length gate |','|---|---|---|']
    components=records('results/si_components/*.DIAGNOSTIC.json')
    if components:
        marker=lines.index('| RLT seed | Training recipe | Accuracy at gaps127/255/511/1023 | All-length gate |')
        insert=['SI post-hoc head restoration: the calibrated gate selects no search in all three repaired systems. Forced search still scores100%, so this is not merely hiding a bad branch.',
            'Restoring the untrained transition reduces forced-search accuracy, while restoring the untrained reward or value head individually preserves100% in this task. The transition is the critical reliance found by these interventions; value-head training is not separately demonstrated to help this one-step problem. Coadaptation limits attribution.','',
            '| SI seed | Restored head | Forced depth2 accuracy | Fraction searched by calibrated gate |','|---|---|---:|---:|']
        for r in components:
            for z in r['rows']:
                if z['condition'].startswith('restore_'):insert.append(f"| {r['seed']} | {z['condition']} | {z['forced_depth2_accuracy']:.2%} | {z['searched_fraction']:.2%} |")
        lines[marker:marker]=insert+['']
    for r in rlt:
        scores='/'.join(f"{z['accuracy']:.1%}" for z in r['rows'] if z['carry'])
        lines.append(f"| {r['seed']} | {r['arm']} | {scores} | {r['gate_passed']} |")
    lines+=['','The RLT gate prior adds public instruction-type supervision; it is not an architecture-only gain. Carry erasure must score50% on every paired prefix panel.','',
        'PGW: the following traces quantify slot collapse in the original tiny hybrid checkpoints.','',
        '| Seed | Delay | Most-used slot write share | Target contribution to final query read |','|---|---:|---:|---:|']
    for r in pgw:
        for delay,z in r['summary'].items():lines.append(f"| {r['seed']} | {delay} | {z['mean_most_used_slot_fraction']:.2%} | {z['mean_target_query_coefficient']:.6g} |")
    lines+=['','CHM: evaluator-gold source context is a failure-localization intervention, not model accuracy.','',
        '| Seed | Family | Native accuracy | Gold gate only | Gold gate and first-hop context |','|---|---|---:|---:|---:|']
    for r in chm:
        for family,z in r['summary'].items():
            if family=='no_match':continue
            lines.append(f"| {r['seed']} | {family} | {z['native_correct']:.2%} | {z['gold_gate_only_correct']:.2%} | {z['gold_gate_and_first_context_correct']:.2%} |")
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if probes:
        extra=['','Frozen RLT state readout: additional supervised long-context linear-head training, not unmodified-model extrapolation. Composite27,778parameters;3,031,040 additional positions plus1,363,200 backbone pretraining positions.','',
            '| Seed | Base recipe | Probe accuracy at128/256/512/1024 | New length/overwrite gate |','|---|---|---|---|']
        for r in probes:
            follow=next((q for q in probe_ood if q['seed']==r['seed'] and q['arm']==r['arm']),None)
            extra.append(f"| {r['seed']} | {r['arm']} | {'/'.join(format(z['probe_accuracy'],'.2%') for z in r['rows'])} | {follow['gate_passed'] if follow else 'pending'} |")
        extra+=['','All three native-state probes decode the constrained first-SET task through1024 perfectly, with erased carry50%. All OOD follow-up gates fail: extrapolation to4096 is near chance and many last-write gaps fail. This identifies recoverable information without establishing a general readout fix.']
        with (ROOT/'RESULTS.md').open('a') as f:f.write('\n'.join(extra)+'\n')
    if si_memory:
        extra=['','SI native memory revision and persistence: frozen trained representations, no optimizer updates; association correctness is not action utility. Each checkpoint receives56 actual fixture feedback events/616 raw positions, on top of736,000 prior positions.','',
            '| SI seed | A after learning B/C | A after8 revised outcomes | Native replay restored | Explicit full replay restored |','|---|---:|---:|---:|---:|']
        for r in si_memory:
            stages=r['stage_associations']
            extra.append(f"| {r['seed']} | {stages['after_C']['A']['native_latest_association_accuracy']:.0%} | {stages['after_A_revision']['A']['native_latest_association_accuracy']:.0%} | {r['native_export_replay_after']}/{r['native_export_replay_before']} | {r['explicit_full_restore_replay_after']} |")
        extra+=['','All eight revised A associations return old deltas in every seed; the eight unchanged associations remain correct. Native append-only memory retains duplicate keys, and cosine argmax chooses the first tied key. The same-feedback latest-record dictionary returns all current deltas by construction. No capacity overflow occurs. World A logits are bitwise unchanged after B/C feedback, and native weight+fast/memory import preserves outputs exactly, but native export omits replay. The separate diagnostic checkpoint saves/restores all56 replay records. These identify changing-fact and lifetime-checkpoint repair targets, without establishing continual decision improvement or family-wide failure. See SI_MEMORY_REVISION_PROTOCOL.md and CONTINUAL_LEARNING_STATUS.md.']
        with (ROOT/'RESULTS.md').open('a') as f:f.write('\n'.join(extra)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
