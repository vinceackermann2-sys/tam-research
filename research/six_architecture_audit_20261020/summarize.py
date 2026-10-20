"""Generate compact tables/plots from recorded results; no model execution."""
import csv,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
records=[];rows=[]
for f in sorted((ROOT/'results').glob('**/*.RESULT.json')):
    d=json.loads(f.read_text());records.append(d);r=d.get('result',{})
    track=f.parent.name;tokens=d.get('tokens_seen',d.get('training_token_exposures'))
    row={'track':track,'attempt':d['attempt'],'parameters':d['parameters'],'tokens':tokens,
         'seed':d.get('seed',d.get('recipe',{}).get('seed','')),'nll':d.get('validation_nll',''),
         'accuracy':'','throughput':d.get('train_tokens_per_second',''),
         'seconds':d.get('train_seconds',d.get('total_seconds','')),'file':str(f.relative_to(ROOT))}
    if 'positive_correct' in r:row['accuracy']=r['positive_correct']/r['positive_total']
    elif 'all_paths' in r:row['accuracy']=r['all_paths']['accuracy']
    elif 'by_delay' in r:row['accuracy']=statistics.mean(v['accuracy'] for v in r['by_delay'])
    elif 'by_gap' in r:row['accuracy']=statistics.mean(v['accuracy'] for v in r['by_gap'].values())
    elif 'development' in r:row['accuracy']=statistics.mean(v['correct'] for v in r['development']['summary'].values())
    rows.append(row)
with (ROOT/'EXPERIMENTS.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['attempt']);writer.writeheader();writer.writerows(rows)

small=[d for d in records if d['attempt'].startswith('conv7-w128')]
colors={'transformer':'#495565','attention8':'#e19a38','conv7':'#168778'}
labels={'transformer':'Transformer','attention8':'Attention8 parent','conv7':'Conv7'}
if len(small)==9:
    fig,axs=plt.subplots(1,3,figsize=(13,3.6))
    for name in colors:
        ds=[d for d in small if d['variant']==name]
        x=[h['tokens']/1e6 for h in ds[0]['history']]
        ys=[[h['nll'] for h in d['history']] for d in ds]
        means=[statistics.mean(v) for v in zip(*ys)]
        axs[0].plot(x,means,label=labels[name],color=colors[name],marker='o')
    axs[0].set(xlabel='Training exposures (million)',ylabel='Validation NLL',title='New WikiText2 pilot • three seeds')
    axs[0].legend(fontsize=8)
    for i,name in enumerate(colors):
        ds=[d for d in small if d['variant']==name]
        vals=[d['validation_nll'] for d in ds]
        axs[1].bar(i,statistics.mean(vals),color=colors[name],alpha=.8)
        axs[1].scatter([i]*3,vals,color='black',s=12,zorder=3)
        axs[2].bar(i,statistics.mean(d['train_tokens_per_second'] for d in ds)/1000,color=colors[name],alpha=.8)
    for ax in axs[1:]:ax.set_xticks(range(3),['Transformer','Parent','Conv7'])
    axs[1].set(title='Final NLL (dots = individual seeds)',ylabel='NLL');axs[1].set_ylim(6.32,6.48)
    axs[2].set(title='Measured A100 training throughput',ylabel='Thousand tokens / second')
    for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.suptitle('11.3M parameters, 1.048576M exposures/model — exploratory, no breakthrough claim',fontsize=12)
    fig.tight_layout();fig.savefig(ROOT/'conv7_pilot.png',dpi=180);plt.close(fig)

summary=['# Recorded experiment results','',f'Completed result records: **{len(records)}**. Every row is a recorded attempt, not an independent breakthrough confirmation.','',
         '| Track | Completed | Largest trained model | Largest per-model exposure count |','|---|---:|---:|---:|']
for track in sorted({r['track'] for r in rows}):
    rs=[r for r in rows if r['track']==track]
    summary.append(f"| {track} | {len(rs)} | {max(r['parameters'] for r in rs):,} | {max(r['tokens'] for r in rs):,} |")
summary += ['','## First language pilot','', '| Arm | NLL mean | Seed SD | Mean tokens/s |','|---|---:|---:|---:|']
for name in colors:
    ds=[d for d in small if d['variant']==name]
    if ds:
        vals=[d['validation_nll'] for d in ds]
        summary.append(f"| {labels[name]} | {statistics.mean(vals):.6f} | {statistics.stdev(vals) if len(vals)>1 else 0:.6f} | {statistics.mean(d['train_tokens_per_second'] for d in ds):,.0f} |")
summary += ['','Three wins from three initialization seeds do not establish a population-level significance claim; the one-sided sign-test probability under equal chances is1/8. The corpus and development evaluation remain shared.','',
            '![Conv7 pilot](conv7_pilot.png)','', '## Longer screens and diagnostics','']
for d in records:
    ident=d['attempt'];r=d.get('result',{})
    if ident.startswith(('conv7-fineweb','conv7-recovery')):
        summary.append(f"- **{ident}:** NLL {d['validation_nll']:.6f}, {d['train_tokens_per_second']:,.0f} tok/s, {d['tokens_seen']:,} exposures; one seed.")
    elif ident.startswith(('cpw-calibration','cpw-recovery')):
        acc=[x['accuracy'] for x in r['by_delay']]
        summary.append(f"- **{ident}:** per-delay accuracy {acc}; all-four {r['all_four']:.4f}; {d['tokens_seen']:,} exposures.")
    elif ident.startswith('rlt-'):
        summary.append(f"- **{ident}:** retention pass {r['long_retention_pass']}; gap127 accuracy {r['by_gap']['127']['accuracy']:.4f} (class0 {r['by_gap']['127']['class0']:.4f}, class1 {r['by_gap']['127']['class1']:.4f}).")
pgw_heading = ['','## PGW deeper-control calibration','',
            'Fresh three-layer 176,801-parameter controls, answer CE alone, 5,000 updates. A passing control is required before the conditional native-hybrid extension. These controls have a different parameter/compute budget from the native PGW; this is learnability work, not a superiority comparison.','',
            '| Seed | Delay | Present answer accuracy | Missing rejection | NLL |','|---|---:|---:|---:|---:|']
long_panel=[d for d in records if d['attempt'].startswith('conv7-recovery')]
if len(long_panel)==3:
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for name in colors:
        d=next(d for d in long_panel if d['variant']==name)
        axs[0].plot([r['tokens']/1e6 for r in d['history']],[r['nll'] for r in d['history']],
                    label=labels[name],color=colors[name],marker='o')
    for i,name in enumerate(colors):
        d=next(d for d in long_panel if d['variant']==name)
        axs[1].bar(i,d['train_tokens_per_second']/1000,color=colors[name])
    axs[0].set(xlabel='Training exposures (million)',ylabel='Development NLL',title='FineWeb-Edu learning curves')
    axs[0].legend(fontsize=8);axs[1].set(xticks=range(3),xticklabels=['Transformer','Parent','Conv7'],ylabel='Thousand tokens / second',title='A100 80GB training throughput')
    for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.suptitle('20.5M parameters, 25M exposures/model • one fresh seed, exploratory')
    fig.tight_layout();fig.savefig(ROOT/'conv7_fineweb.png',dpi=180);plt.close(fig)
    transformer=next(d for d in long_panel if d['variant']=='transformer');candidate=next(d for d in long_panel if d['variant']=='conv7');parent=next(d for d in long_panel if d['variant']=='attention8')
    delta=candidate['validation_nll']-transformer['validation_nll'];ratio=candidate['train_tokens_per_second']/transformer['train_tokens_per_second'];benefit=parent['validation_nll']-candidate['validation_nll']
    summary += ['','### FineWeb development gate','',
                f'Conv7 minus Transformer NLL: **{delta:+.6f}**; training throughput ratio: **{ratio:.4f}**; parent minus Conv7 NLL: **{benefit:.6f}**. Frozen endpoint quality/speed/mechanism gates passed: **{delta<=.015 and ratio>=1.03 and benefit>=.005}**. This is one-seed development evidence, not an independent replicated breakthrough.','',
                '![Longer Conv7 screen](conv7_fineweb.png)','']
summary += pgw_heading
for d in records:
    if d['attempt'].startswith('pgw-control'):
        for r in d['result']['by_delay']:
            summary.append(f"| {d['recipe']['seed']} | {r['delay']} | {r['present_accuracy']:.4f} | {r['missing_accuracy']:.4f} | {r['nll']:.6f} |")
summary += ['','## CHM optimizer-unseen entity diagnostic','',
            'Inference only, all 63 native TRAIN entities excluded from the capacity optimizer. Eight rotations per positive entity are correlated. Reserved DEVELOPMENT/TEST remain unopened. Values below are initialization-seed means and ranges, not confidence intervals.','',
            '| Family | Mean answer accuracy | Seed range | Mean abstention |','|---|---:|---:|---:|']
chm=[json.loads(p.read_text()) for p in (ROOT/'results/chm_unseen').glob('*.DIAGNOSTIC.json')]
for family in ('direct','overwrite','two_hop','no_match'):
    if chm:
        vals=[d['summary'][family]['correct'] for d in chm]
        abst=[d['summary'][family]['abstained'] for d in chm]
        summary.append(f'| {family} | {statistics.mean(vals):.4f} | {min(vals):.4f}–{max(vals):.4f} | {statistics.mean(abst):.4f} |')
summary += ['','## CHM broad TRAIN / first reserved DEVELOPMENT study','',
            'New initialization/data-order protocol, 4,096 updates across all 64 TRAIN entities. Native DEVELOPMENT: 16 independent entities/family; native TEST remains closed. This flat Transformer-based pointer has extra gold-address training supervision; no superiority or hierarchical acceleration claim.','',
            '| Seed | Family | TRAIN answer accuracy | DEVELOPMENT answer accuracy | DEV first source | DEV payload source | DEV abstention |','|---|---|---:|---:|---:|---:|---:|']
for d in records:
    if d['attempt'].startswith('chm-broad'):
        for f in ('direct','overwrite','two_hop','no_match'):
            t=d['result']['train']['summary'][f];v=d['result']['development']['summary'][f]
            summary.append(f"| {d['recipe']['seed']} | {f} | {t['correct']:.4f} | {v['correct']:.4f} | {v['first_correct']:.4f} | {v['payload_correct']:.4f} | {v['abstained']:.4f} |")
summary += ['','## RLT longer retention and causal erasure','',
            'No additional optimization. New paired noise prefixes, lengths beyond the training maximum of 128. Opposite-label pairs share every token after the initial SET; same-weight state erasure must give 50% balanced accuracy. This tests one-bit retention, not general language quality.','',
            '| Arm | Seed | Gap | Carry | Accuracy | Class 0 | Class 1 |','|---|---:|---:|---|---:|---:|---:|']
lengths=[json.loads(p.read_text()) for p in (ROOT/'results/rlt_length').glob('*.DIAGNOSTIC.json')]
for d in lengths:
    for r in d['rows']:
        summary.append(f"| {d['arm']} | {d['seed']} | {r['gap']} | {r['carry_enabled']} | {r['accuracy']:.4f} | {r['class0']:.4f} | {r['class1']:.4f} |")
if lengths:
    fig,ax=plt.subplots(figsize=(7,4))
    for d in lengths:
        rs=[r for r in d['rows'] if r['carry_enabled']]
        ax.plot([r['gap'] for r in rs],[r['accuracy'] for r in rs],marker='o',
                linestyle='-' if d['arm']=='carry' else '--',label=f"{d['arm']} / {d['seed']}")
    ax.axhline(.5,color='black',linestyle=':',label='Same-weight state erased')
    ax.set(xlabel='Last SET to QUERY gap (tokens)',ylabel='Balanced paired-prefix accuracy',ylim=(0,1.03),
           title='One-bit retention at longer lengths • same checkpoints')
    ax.legend(fontsize=7);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(ROOT/'rlt_length.png',dpi=180);plt.close(fig)
    summary += ['','![RLT length diagnostic](rlt_length.png)','']
unfinished=[p for p in (ROOT/'results').glob('**/*.ATTEMPT.json') if not p.with_name(p.name.replace('.ATTEMPT.json','.RESULT.json')).exists()]
summary += ['','## Incomplete / error records','']+[f'- Incomplete: `{p.relative_to(ROOT)}`' for p in unfinished]
summary += [f'- Error: `{p.relative_to(ROOT)}`' for p in (ROOT/'results').glob('**/*ERROR.json')]
if not unfinished:summary.append('No started attempt currently lacks a terminal RESULT in this local snapshot; inspect error and remote-worker records separately.')
(ROOT/'RESULTS.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
print('Summarized',len(records),'completed attempts;',len(unfinished),'started without local terminal result')
