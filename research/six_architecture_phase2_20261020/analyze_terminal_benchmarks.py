"""Analyze terminal evaluations without changing the frozen training decision."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def paired_accuracy(candidate, control):
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(control, dtype=float)
    assert a.shape == b.shape and a.ndim == 1 and len(a)
    assert np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()
    difference = a - b
    rng = np.random.default_rng(202610399)
    samples = [float(difference[rng.integers(0, len(a), len(a))].mean())
               for _ in range(2000)]
    return {
        'candidate_minus_control_accuracy': float(difference.mean()),
        'descriptive_95_percent_example_bootstrap_interval':
            np.quantile(samples, [.025, .975]).tolist(),
        'candidate_only_correct': int(((a == 1) & (b == 0)).sum()),
        'control_only_correct': int(((a == 0) & (b == 1)).sum()),
        'examples': len(a),
        'interpretation': 'One trained seed, paired task-example resampling; no multiplicity correction.',
    }


def summarize(evidence, inference, benchmarks):
    assert all(x.get('complete') for x in (evidence, inference, benchmarks))
    arms = set(evidence['terminal'])
    assert arms == {'transformer', 'attention8', 'conv7', 'cpw_local'}
    task = {r['variant']: r for r in benchmarks['results']}
    speed = {r['variant']: r for r in inference['records']}
    assert len(task) == len(benchmarks['results']) == len(arms)
    assert len(speed) == len(inference['records']) == len(arms)
    assert set(task) == set(speed) == arms
    rows = []
    for arm in sorted(arms):
        terminal = evidence['terminal'][arm]
        trained = evidence['results'][f"{arm}-s{terminal['steps']}"]
        b, i = task[arm], speed[arm]
        checkpoint = trained['checkpoint']['checkpoint_sha256']
        assert b['checkpoint_sha256'] == i['checkpoint_sha256'] == checkpoint
        assert b['training_tokens'] == terminal['tokens_seen'] == trained['tokens_seen']
        assert b['optimizer_exposures'] == 0
        assert i['parameters'] == trained['parameters'] <= 100_000_000
        charged = evidence['charges'][arm]
        assert trained['tokens_seen'] + charged <= 2_000_000_000
        assert b['piqa']['examples'] == len(b['piqa']['rows']) == 1838
        assert b['lambada']['examples'] == len(b['lambada']['rows']) == 5153
        assert [r['index'] for r in b['piqa']['rows']] == list(range(1838))
        for r in b['piqa']['rows']:
            assert r['gold'] in (0, 1)
            for field, normalization in [('summed', 'sum_nll'),
                                         ('token_normalized', 'mean_token_nll')]:
                values = [v[normalization] for v in r['option_scores']]
                assert len(values) == 2 and all(math.isfinite(v) for v in values)
                assert r['predictions'][field] == int(np.argmin(values))
        for key, value in b['piqa']['accuracy'].items():
            expected = np.mean([r['predictions'][key] == r['gold']
                                for r in b['piqa']['rows']])
            assert abs(value - expected) < 1e-12
        assert abs(b['lambada']['accuracy'] - np.mean([
            r['all_greedy_correct'] for r in b['lambada']['rows']])) < 1e-12
        for r in b['lambada']['rows']:
            assert r['tokens'] > 0 and math.isfinite(r['sum_nll'])
        assert {(r['context'], r['batch']) for r in i['prefill']} == {
            (length, batch) for length in (128, 512, 1024) for batch in (1, 8)}
        assert len(i['prefill']) == 6
        for r in i['prefill']:
            assert 0 < r['median_ms'] <= r['p90_ms']
            assert 0 < r['tokens_per_second'] and math.isfinite(r['peak_gib'])
        rows.append({
            'variant': arm, 'parameters': trained['parameters'],
            'training_exposures': trained['tokens_seen'], 'charged_lost_exposures': charged,
            'committed_plus_conservative_charge': trained['tokens_seen'] + charged,
            'checkpoint_sha256': checkpoint, 'development_nll': trained['validation']['nll'],
            'test_nll': terminal['test']['nll'], 'test_tokens': terminal['test']['tokens'],
            'timed_training_tokens_per_second': trained['train_tokens_per_second'],
            'piqa_primary_token_normalized_accuracy': b['piqa']['accuracy']['token_normalized'],
            'piqa_secondary_accuracies': b['piqa']['accuracy'],
            'lambada_teacher_forced_exact_word_accuracy': b['lambada']['accuracy'],
            'lambada_mean_word_nll': b['lambada']['mean_word_nll'],
            'lambada_word_perplexity': b['lambada']['word_perplexity'],
            'prefill': i['prefill'],
            'naive_generation_without_kv_cache': i['naive_generation_without_kv_cache'],
        })
    comparisons = []
    control = task['transformer']
    for arm in sorted(arms - {'transformer'}):
        candidate = task[arm]
        if candidate['training_tokens'] != control['training_tokens']:
            continue
        assert [r['gold'] for r in candidate['piqa']['rows']] == [
            r['gold'] for r in control['piqa']['rows']]
        assert [r['tokens'] for r in candidate['lambada']['rows']] == [
            r['tokens'] for r in control['lambada']['rows']]
        comparisons.append({
            'variant': arm, 'control': 'transformer',
            'matched_committed_training_exposures': candidate['training_tokens'],
            'piqa_primary': paired_accuracy(
                [r['predictions']['token_normalized'] == r['gold'] for r in candidate['piqa']['rows']],
                [r['predictions']['token_normalized'] == r['gold'] for r in control['piqa']['rows']]),
            'lambada': paired_accuracy(
                [r['all_greedy_correct'] for r in candidate['lambada']['rows']],
                [r['all_greedy_correct'] for r in control['lambada']['rows']]),
        })
    conv = speed['conv7']['conv_erasure']
    assert abs(conv['replay_nll'] - evidence['results'][
        f"conv7-s{evidence['terminal']['conv7']['steps']}"]['validation']['nll']) < 1e-6
    assert speed['cpw_local']['distant_prefix_delta'] == 0
    return {
        'complete': True, 'gpu': inference['gpu'], 'torch': inference['torch'],
        'records': rows, 'matched_budget_task_comparisons': comparisons,
        'conv7_trained_branch_erasure': conv,
        'cpw_local_distant_prefix_logit_max_delta': speed['cpw_local']['distant_prefix_delta'],
        'breakthrough_demonstrated': False,
        'limitations': [
            'One paired seed; example/block intervals do not measure training-seed uncertainty.',
            'CPW stopped at 100M; other terminal models stopped at 300M. No paired CPW-versus-300M claim.',
            'Parent has a separate conservative lost-exposure charge; matched committed tokens do not imply equal total compute.',
            'LAMBADA uses teacher-forced all-target-token accuracy. PIQA primary scoring normalizes by target tokens.',
            'No benchmark contamination audit or independent-seed replication.',
            'Prefill returns dense vocabulary logits; generation is an eight-token implementation benchmark without KV caching.',
            'Post-hoc Conv7 gain erasure measures trained-branch reliance, not the effect of training a model without that branch.',
            'Terminal task benchmarks cannot overturn the predeclared development continuation gate.',
        ],
    }


def main(root=ROOT):
    paths = [root / 'l4_panel' / name for name in (
        'EVIDENCE.json', 'INFERENCE.json', 'BENCHMARKS.json')]
    if not all(p.exists() for p in paths):
        return None
    inputs = [json.loads(p.read_text()) for p in paths]
    if not all(r.get('complete') for r in inputs):
        return None
    for value, name in zip(inputs[1:], ['inference_benchmark.py', 'language_benchmarks.py']):
        assert value['source_sha256'] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    summary = summarize(*inputs)
    summary['input_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (root / 'TERMINAL_SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = ['# Terminal language and inference benchmarks', '',
             'Training stopped by the prospective development gates; these terminal tasks did not select checkpoints.', '',
             '| Model | Training exposures | Test NLL | PIQA primary | LAMBADA exact word |',
             '|---|---:|---:|---:|---:|']
    for r in summary['records']:
        lines.append(f"| {r['variant']} | {r['training_exposures']:,} | {r['test_nll']:.6f} | "
                     f"{r['piqa_primary_token_normalized_accuracy']:.2%} | {r['lambada_teacher_forced_exact_word_accuracy']:.2%} |")
    lines += ['', 'PIQA: 1,838 validation questions. LAMBADA: 5,153 examples. Secondary PIQA scoring rules and per-example records are retained in BENCHMARKS.json.', '',
              '| Matched candidate versus Transformer | PIQA difference (percentage points) | LAMBADA difference (percentage points) |',
              '|---|---:|---:|']
    for r in summary['matched_budget_task_comparisons']:
        cells = []
        for key in ('piqa_primary', 'lambada'):
            z = r[key]
            low, high = z['descriptive_95_percent_example_bootstrap_interval']
            cells.append(f"{100*z['candidate_minus_control_accuracy']:+.3f} [{100*low:+.3f}, {100*high:+.3f}]")
        lines.append(f"| {r['variant']} | {' | '.join(cells)} |")
    lines += ['', 'Intervals are descriptive paired example resampling, not independent-seed confirmation.', '',
              '| Model | Context | Batch | Median prefill ms | p90 ms | Tokens/s | Peak allocated GiB |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['records']:
        for p in r['prefill']:
            lines.append(f"| {r['variant']} | {p['context']} | {p['batch']} | {p['median_ms']:.3f} | {p['p90_ms']:.3f} | {p['tokens_per_second']:,.0f} | {p['peak_gib']:.3f} |")
    lines += ['', '| Model | Eight-token generation ms, prompt128/batch1/no KV cache | New tokens/s |',
              '|---|---:|---:|']
    for r in summary['records']:
        g = r['naive_generation_without_kv_cache']
        lines.append(f"| {r['variant']} | {g['median_total_ms']:.3f} | {g['new_tokens_per_second']:.2f} |")
    conv = summary['conv7_trained_branch_erasure']
    lines += ['', f"Conv7 gain erasure: replay NLL {conv['replay_nll']:.6f}; erased NLL {conv['zero_gain_nll']:.6f}; change {conv['delta_nll']:+.6f}.",
              'CPW local: changing the distant prefix outside its active window gives exactly zero final-logit change.', '',
              *summary['limitations'], '']
    (root / 'TERMINAL_BENCHMARKS.md').write_text('\n'.join(lines))
    print('TERMINAL_BENCHMARKS_ANALYZED', len(summary['records']))
    return summary


if __name__ == '__main__':
    main()
