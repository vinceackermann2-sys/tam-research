"""Guard against premature or unmatched comparisons in the research report."""
import pytest
from analyze_l4_panel import summarize
from analyze_phase2 import paired


def record(arm, tokens, endpoint):
    return {'variant': arm, 'parameters': 99_000_000, 'tokens_seen': tokens,
            'steps': tokens // 8192, 'scientific_endpoint': endpoint,
            'seed': 7, 'gpu': 'NVIDIA L4', 'torch': '2.11.0+cu130',
            'validation': {'nll': 5., 'block_nll': [4., 6.]},
            'train_tokens_per_second': 30_000.}


def trial(*rows):
    return {'trial': 'L4_NEW_PAIRED_SEED', 'seed': 7,
            'results': {str(i): r for i, r in enumerate(rows)},
            'charges': {}, 'gates': [], 'terminal': {}, 'complete': False}


def test_only_matched_scientific_endpoints_are_compared():
    e = trial(record('transformer', 20_004_864, False), record('conv7', 20_004_864, False),
              record('transformer', 99_999_744, True), record('conv7', 99_999_744, True))
    s = summarize(e)
    assert len(s['segment_records']) == 4
    assert [r['tokens_seen'] for r in s['scientific_comparisons']] == [99_999_744]
    e['results']['3']['seed'] = 8
    s = summarize(e)
    assert s['scientific_comparisons'] == [] and len(s['unavailable_comparisons']) == 1


def test_unequal_terminal_budgets_do_not_gain_paired_test_comparison():
    e = trial()
    for arm, tokens in [('transformer', 299_999_232), ('conv7', 99_999_744)]:
        e['terminal'][arm] = {'tokens_seen': tokens,
                              'test': {'nll': 5., 'tokens': 1024, 'block_nll': [4., 6.]}}
    candidate = next(r for r in summarize(e)['terminal'] if r['variant'] == 'conv7')
    assert not candidate['matched_terminal_transformer']
    assert 'delta_test_nll_to_transformer' not in candidate
    e['terminal']['conv7']['tokens_seen'] = 299_999_232
    candidate = next(r for r in summarize(e)['terminal'] if r['variant'] == 'conv7')
    assert candidate['delta_test_nll_to_transformer']['delta_mean_nll'] == 0


def test_lost_reservations_count_toward_the_lineage_ceiling():
    e = trial(record('conv7', 1_999_994_880, True))
    e['charges']['conv7'] = 8192
    with pytest.raises(AssertionError):
        summarize(e)


def test_block_alignment_is_required_including_single_block_input():
    with pytest.raises(AssertionError):
        paired([1., 2.], [1.])


def test_current_gate_reporting_and_completion_require_terminal_benchmarks(tmp_path, monkeypatch):
    import json
    import analyze_phase2
    e = trial(record('transformer', 99_999_744, True), record('conv7', 99_999_744, True))
    e['complete'] = True
    e['gates'] = [{'variant': 'conv7', 'stage_tokens': 99_999_744, 'passed': True}]
    (tmp_path/'l4_panel').mkdir()
    (tmp_path/'l4_panel/EVIDENCE.json').write_text(json.dumps(e))
    (tmp_path/'EVIDENCE.json').write_text(json.dumps({'results': {}, 'gates': [], 'complete': False}))
    monkeypatch.setattr(analyze_phase2, 'ROOT', tmp_path)
    analyze_phase2.main()
    status = json.loads((tmp_path/'RESEARCH_STATUS.json').read_text())
    assert status['gates'] == e['gates']
    assert status['interrupted_a100_gates'] == []
    assert status['language_training_complete'] and not status['language_complete']
    assert status['language_segments_complete'] == 2
    for name in ('INFERENCE.json', 'BENCHMARKS.json'):
        (tmp_path/'l4_panel'/name).write_text(json.dumps({'complete': True}))
    analyze_phase2.main()
    assert json.loads((tmp_path/'RESEARCH_STATUS.json').read_text())['language_complete']
