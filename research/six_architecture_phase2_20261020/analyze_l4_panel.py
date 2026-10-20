"""Report the L4 trial separately; intermediate backups are not decision endpoints."""
import csv
import json
from pathlib import Path

from analyze_phase2 import paired

ROOT = Path(__file__).parent


def summarize(evidence):
    rows = []
    comparisons = []
    unavailable = []
    results = list(evidence.get('results', {}).values())
    charges = evidence.get('charges', {})
    for r in sorted(results, key=lambda z: (z['tokens_seen'], z['variant'])):
        lost = charges.get(r['variant'], 0)
        assert 0 < r['parameters'] <= 100_000_000
        assert 0 <= lost and r['tokens_seen'] + lost <= 2_000_000_000
        rows.append({
            'trial': evidence['trial'], 'seed': r['seed'], 'gpu': r['gpu'],
            'variant': r['variant'], 'parameters': r['parameters'],
            'tokens_seen': r['tokens_seen'], 'step': r['steps'],
            'scientific_endpoint': bool(r.get('scientific_endpoint', False)),
            'charged_lost_exposures': lost,
            'charged_exposure_upper_bound': r['tokens_seen'] + lost,
            'validation_nll': r['validation']['nll'],
            'training_tokens_per_second': r['train_tokens_per_second'],
        })
        if not r.get('scientific_endpoint') or r['variant'] == 'transformer':
            continue
        control = next((q for q in results if q['variant'] == 'transformer'
                        and q['tokens_seen'] == r['tokens_seen']
                        and q.get('scientific_endpoint')), None)
        if control is None:
            unavailable.append({'variant': r['variant'], 'tokens_seen': r['tokens_seen'],
                                'reason': 'No Transformer scientific endpoint at the same exposure count.'})
            continue
        if (r['gpu'], r['seed'], r['torch']) != (control['gpu'], control['seed'], control['torch']):
            unavailable.append({'variant': r['variant'], 'tokens_seen': r['tokens_seen'],
                                'reason': 'Recorded device, seed or PyTorch version differs.'})
            continue
        comparison = {
            'trial': evidence['trial'], 'seed': r['seed'], 'gpu': r['gpu'],
            'variant': r['variant'], 'tokens_seen': r['tokens_seen'],
            'parameters': r['parameters'], 'control_parameters': control['parameters'],
            **paired(r['validation']['block_nll'], control['validation']['block_nll']),
            'training_speed_ratio': r['train_tokens_per_second'] / control['train_tokens_per_second'],
        }
        if r['variant'] == 'conv7':
            parent = next((q for q in results if q['variant'] == 'attention8'
                           and q['tokens_seen'] == r['tokens_seen']
                           and q.get('scientific_endpoint')
                           and (q['gpu'], q['seed'], q['torch']) == (r['gpu'], r['seed'], r['torch'])), None)
            if parent:
                comparison['delta_nll_to_attention8'] = paired(
                    r['validation']['block_nll'], parent['validation']['block_nll'])
        comparisons.append(comparison)
    terminal = []
    transformer = evidence.get('terminal', {}).get('transformer')
    for arm, value in sorted(evidence.get('terminal', {}).items()):
        row = {'variant': arm, 'tokens_seen': value['tokens_seen'],
               'test_nll': value['test']['nll'], 'test_tokens': value['test']['tokens'],
               'matched_terminal_transformer': bool(transformer and
                   value['tokens_seen'] == transformer['tokens_seen'])}
        if arm != 'transformer' and row['matched_terminal_transformer']:
            row['delta_test_nll_to_transformer'] = paired(
                value['test']['block_nll'], transformer['test']['block_nll'])
        terminal.append(row)
    return {'trial': evidence['trial'], 'seed': evidence['seed'],
            'complete': evidence.get('complete', False),
            'segment_records': rows, 'scientific_comparisons': comparisons,
            'unavailable_comparisons': unavailable, 'gates': evidence.get('gates', []),
            'terminal': terminal, 'breakthrough_demonstrated': False,
            'limitations': [
                'One paired seed; token-block bootstrap intervals are descriptive.',
                'Intermediate backup validations do not select checkpoints or authorize continuation.',
                'Terminal models stopped at different budgets are not matched-budget comparisons.',
                'A100 measurements belong to the separate interrupted trial.',
                'Training throughput excludes input preparation, transfers, evaluation and backup time.',
            ]}


def main(root=ROOT):
    path = root / 'l4_panel/EVIDENCE.json'
    if not path.exists():
        return None
    summary = summarize(json.loads(path.read_text()))
    (root / 'L4_SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    fields = ['trial', 'seed', 'gpu', 'variant', 'parameters', 'tokens_seen', 'step',
              'scientific_endpoint', 'charged_lost_exposures', 'charged_exposure_upper_bound',
              'validation_nll', 'training_tokens_per_second']
    with (root / 'L4_LANGUAGE_RESULTS.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary['segment_records'])
    lines = ['# L4 paired language confirmation', '',
             f"Seed: {summary['seed']}. Training complete: {summary['complete']}.", '',
             'Only scientific endpoints enter the comparison table. Segment backup metrics remain in L4_LANGUAGE_RESULTS.csv.', '',
             '| Candidate | Exposures | Development NLL minus Transformer | Training speed / Transformer |',
             '|---|---:|---:|---:|']
    for r in summary['scientific_comparisons']:
        lines.append(f"| {r['variant']} | {r['tokens_seen']:,} | {r['delta_mean_nll']:+.6f} | {r['training_speed_ratio']:.4f} |")
    if not summary['scientific_comparisons']:
        lines += ['', 'No matched scientific endpoint is complete yet.']
    lines += ['', '| Terminal model | Exposures | Test NLL | Matched terminal Transformer budget |',
              '|---|---:|---:|---|']
    for r in summary['terminal']:
        lines.append(f"| {r['variant']} | {r['tokens_seen']:,} | {r['test_nll']:.6f} | {r['matched_terminal_transformer']} |")
    lines += ['', *summary['limitations'], '']
    (root / 'L4_RESULTS.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    main()
