from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

from tam_research import physics_tokenizer_phase30 as p30
from tam_research import physics_tokenizer_phase32 as p32

ATTEMPT_ID = 'phase32-registered-rep1-attempt1'
REPLICATE = 1
SEED_BASE = 832320000
SEEDSET = p30._seedset(SEED_BASE)
LOCKED_FUTURE_BASES = (832320100, 832320200)
AUTH_PATH = Path('results/physics_tokens/phase32_rep1_execution_authorization.json')
FAMILIES = ('wave', 'gray_scott', 'nls', 'advection', 'burgers', 'hamilton_jacobi')
EXPECTED_SPLITS = (
    'wave_id_h8',
    'wave_spectral_ood_h3',
    'wave_combined_ood_h3',
    'gray_id_h8',
    'gray_sharp_ood_h3',
    'gray_parameter_ood_h3',
    'gray_combined_ood_h3',
    'nls_id_h8',
    'nls_spectral_ood_h3',
    'nls_parameter_ood_h3',
    'nls_combined_ood_h3',
    'adv_id_h8',
    'adv_spectral_ood_h3',
    'adv_speed_ood_h3',
    'adv_combined_ood_h3',
    'burgers_id_h8',
    'burgers_spectral_ood_h3',
    'burgers_parameter_ood_h3',
    'burgers_combined_ood_h3',
    'hj_id_h8',
    'hj_spectral_ood_h3',
    'hj_parameter_ood_h3',
    'hj_combined_ood_h3',
)


def split_family(name: str) -> str:
    if name.startswith('wave_'):
        return 'wave'
    if name.startswith('gray_'):
        return 'gray_scott'
    if name.startswith('nls_'):
        return 'nls'
    if name.startswith('adv_'):
        return 'advection'
    if name.startswith('burgers_'):
        return 'burgers'
    if name.startswith('hj_'):
        return 'hamilton_jacobi'
    raise KeyError(name)


def validate_invariants():
    seed_values = list(asdict(SEEDSET).values())
    future_values = []
    for base in LOCKED_FUTURE_BASES:
        future_values.extend(asdict(p30._seedset(base)).values())
    assert SEED_BASE == p32.REGISTERED_BASES[0]
    assert LOCKED_FUTURE_BASES == p32.REGISTERED_BASES[1:]
    assert len(seed_values) == 29 and len(set(seed_values)) == 29
    assert not set(seed_values).intersection(future_values)
    assert FAMILIES == p32.FAMILIES
    assert len(EXPECTED_SPLITS) == 23 and len(set(EXPECTED_SPLITS)) == 23
    assert {split_family(x) for x in EXPECTED_SPLITS} == set(FAMILIES)
    assert p32.DAMPING_GRID == (1.0, 0.5, 0.25, 0.125)
    assert tuple(target - start for _, start, target in p32.GUARD_WINDOWS) == (5, 4, 3)
    return {
        'attempt_id': ATTEMPT_ID,
        'replicate': REPLICATE,
        'seed_base': SEED_BASE,
        'registered_seed_count': len(seed_values),
        'future_registered_bases_locked': list(LOCKED_FUTURE_BASES),
        'families': list(FAMILIES),
        'expected_primary_cells': len(EXPECTED_SPLITS),
        'authorization_required': True,
        'frozen_pass_rule': 'all 23 replicate-1 cells must be < 1.0; any >= 1.0 makes overall Phase-32 69/69 gate decisively FAIL',
    }


def _load_authorization() -> dict:
    if not AUTH_PATH.exists():
        raise RuntimeError(f'missing authorization ledger: {AUTH_PATH}')
    return json.loads(AUTH_PATH.read_text())


def _require_authorization() -> dict:
    auth = _load_authorization()
    if auth.get('status') != 'AUTHORIZED_FOR_REGISTERED_EXECUTION':
        raise RuntimeError('Phase-32 replicate 1 is not authorized for registered execution')
    scope = auth.get('authorization_scope', {})
    if scope.get('replicate') != REPLICATE:
        raise RuntimeError('authorization replicate mismatch')
    if scope.get('seed_base') != SEED_BASE:
        raise RuntimeError('authorization seed-base mismatch')
    if scope.get('attempt_id') != ATTEMPT_ID:
        raise RuntimeError('authorization attempt-id mismatch')
    seeds = auth.get('registered_seed_bases', {})
    if seeds.get('replicate_1', {}).get('status') != 'AUTHORIZED_UNCONSUMED':
        raise RuntimeError('replicate-1 authorization ledger is not in AUTHORIZED_UNCONSUMED state')
    if seeds.get('replicate_2', {}).get('status') not in ('LOCKED_UNCONSUMED', 'LOCKED_UNCONSUMED_DO_NOT_RUN'):
        raise RuntimeError('replicate 2 must remain locked')
    if seeds.get('replicate_3', {}).get('status') not in ('LOCKED_UNCONSUMED', 'LOCKED_UNCONSUMED_DO_NOT_RUN'):
        raise RuntimeError('replicate 3 must remain locked')
    return auth


def _write_json(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))


def prepare(out: str, marker: str):
    inv = validate_invariants()
    _require_authorization()
    marker_payload = {
        'experiment': 'physics-token-phase32-multihorizon-context-guard-six-family-v1',
        'status': 'REGISTERED_REP1_CONSUMED_PREPARATION_STARTED',
        'attempt_id': ATTEMPT_ID,
        'replicate': REPLICATE,
        'seed_base': SEED_BASE,
        'seeds': asdict(SEEDSET),
        'future_registered_bases_locked': list(LOCKED_FUTURE_BASES),
        'scientific_evidence': False,
        'note': 'Written immediately before registered build_all/build_eval. From this point the replicate and namespace are consumed and must never be retried.',
        'invariants': inv,
    }
    _write_json(marker, marker_payload)

    t0 = time.time()
    state, dm, ds, basis = p30.build_all(
        SEEDSET, grid=16, train_n=64, code_sizes=(2048, 128), iters=20, max_points=60000
    )
    raw = p30.build_eval(SEEDSET, grid=16, eval_n=48)
    if set(raw) != set(EXPECTED_SPLITS):
        raise RuntimeError(
            f'generated split set mismatch: expected={sorted(EXPECTED_SPLITS)} got={sorted(raw)}'
        )

    obj = {
        'attempt_id': ATTEMPT_ID,
        'replicate': REPLICATE,
        'seed_base': SEED_BASE,
        'seeds': asdict(SEEDSET),
        'state': state,
        'dm': dm,
        'ds': ds,
        'basis': basis,
        'raw': raw,
        'scientific_config': 'phase32_frozen_multihorizon_context_guard_exact',
        'runtime_seconds': time.time() - t0,
    }
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    marker_payload['status'] = 'REGISTERED_REP1_PREPARATION_COMPLETE'
    marker_payload['split_count'] = len(raw)
    marker_payload['runtime_seconds'] = obj['runtime_seconds']
    _write_json(marker, marker_payload)
    return marker_payload


def eval_family(prep_path: str, family: str, out: str):
    if family not in FAMILIES:
        raise ValueError(family)
    obj = pickle.loads(Path(prep_path).read_bytes())
    if obj['attempt_id'] != ATTEMPT_ID or obj['replicate'] != REPLICATE or obj['seed_base'] != SEED_BASE:
        raise RuntimeError('prep provenance mismatch')

    names = [x for x in EXPECTED_SPLITS if split_family(x) == family]
    if any(name not in obj['raw'] for name in names):
        raise RuntimeError(f'missing expected {family} splits in prep')

    t0 = time.time()
    splits = {
        name: p32.eval_split(
            obj['raw'][name], obj['state'], obj['dm'], obj['ds'], obj['basis'],
            8 if name.endswith('h8') else 3,
        )
        for name in names
    }
    ratios = {name: float(value['selected_ratio']) for name, value in splits.items()}
    result = {
        'experiment': 'physics-token-phase32-multihorizon-context-guard-six-family-v1',
        'status': 'PHASE32_REGISTERED_REP1_FAMILY_RESULT',
        'attempt_id': ATTEMPT_ID,
        'replicate': REPLICATE,
        'seed_base': SEED_BASE,
        'family': family,
        'primary_ratios': ratios,
        'passing_cells': int(sum(x < 1.0 for x in ratios.values())),
        'total_cells': len(ratios),
        'all_below_persistence': bool(all(x < 1.0 for x in ratios.values())),
        'splits': splits,
        'runtime_seconds': time.time() - t0,
        'scientific_config': 'phase32_frozen_multihorizon_context_guard_exact',
    }
    _write_json(out, result)
    return result


def aggregate(paths: list[str], out: str):
    if len(paths) != len(FAMILIES):
        raise RuntimeError(f'exactly {len(FAMILIES)} family files required')
    records = [json.loads(Path(path).read_text()) for path in paths]
    families = [x.get('family') for x in records]
    if set(families) != set(FAMILIES) or len(families) != len(set(families)):
        raise RuntimeError(f'family set mismatch: {families}')

    all_splits = {}
    family_summary = {}
    for record in records:
        if record.get('attempt_id') != ATTEMPT_ID:
            raise RuntimeError('attempt provenance mismatch')
        if record.get('replicate') != REPLICATE or record.get('seed_base') != SEED_BASE:
            raise RuntimeError('registered replicate provenance mismatch')
        family = record['family']
        ratios = record.get('primary_ratios', {})
        for name, ratio in ratios.items():
            if name in all_splits:
                raise RuntimeError(f'duplicate split: {name}')
            all_splits[name] = float(ratio)
        family_summary[family] = {
            'passing_cells': int(record['passing_cells']),
            'total_cells': int(record['total_cells']),
            'all_below_persistence': bool(record['all_below_persistence']),
            'worst_ratio': float(max(ratios.values())),
        }

    if set(all_splits) != set(EXPECTED_SPLITS) or len(all_splits) != len(EXPECTED_SPLITS):
        missing = sorted(set(EXPECTED_SPLITS) - set(all_splits))
        extra = sorted(set(all_splits) - set(EXPECTED_SPLITS))
        raise RuntimeError(f'primary split set mismatch missing={missing} extra={extra}')

    failing = {name: ratio for name, ratio in all_splits.items() if ratio >= 1.0}
    passing = len(all_splits) - len(failing)
    if failing:
        status = 'PHASE32_DECISIVE_FAIL_REGISTERED_REP1'
        overall_gate = 'FAIL_DECISIVE'
        next_action = 'STOP_REGISTERED_EXECUTION_PRESERVE_REP2_REP3_UNCONSUMED'
    else:
        status = 'PHASE32_REGISTERED_REP1_PASS_23_OF_23_OVERALL_GATE_UNRESOLVED'
        overall_gate = 'UNRESOLVED_REQUIRES_SEPARATE_REP2_AUTHORIZATION'
        next_action = 'STOP_REP1_COMPLETE_REP2_REMAINS_LOCKED'

    summary = {
        'experiment': 'physics-token-phase32-multihorizon-context-guard-six-family-v1',
        'status': status,
        'attempt_id': ATTEMPT_ID,
        'replicate': REPLICATE,
        'seed_base': SEED_BASE,
        'registered_seed_status': 'CONSUMED_VALID_REGISTERED_ATTEMPT',
        'primary_ratios': {name: all_splits[name] for name in EXPECTED_SPLITS},
        'primary_passing_cells': passing,
        'primary_total_cells': len(all_splits),
        'failing_cells': failing,
        'replicate_all_23_below_persistence': not failing,
        'family_summary': family_summary,
        'frozen_overall_gate': '69/69 across three fresh registered replicates; any ratio >=1.0 is FAIL',
        'overall_gate_status': overall_gate,
        'next_action': next_action,
        'replicate_2_seed_base': 832320100,
        'replicate_2_status': 'LOCKED_UNCONSUMED',
        'replicate_3_seed_base': 832320200,
        'replicate_3_status': 'LOCKED_UNCONSUMED',
        'retry_authorized': False,
        'claim_limit': 'Synthetic benchmark evidence only; not evidence of new physics, unrestricted scientific-law discovery, AGI, or SSI.',
    }
    _write_json(out, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    sub.add_parser('validate')
    sub.add_parser('check-authorization')

    prep = sub.add_parser('prepare')
    prep.add_argument('--out', required=True)
    prep.add_argument('--marker', required=True)

    fam = sub.add_parser('eval-family')
    fam.add_argument('--prep', required=True)
    fam.add_argument('--family', required=True, choices=FAMILIES)
    fam.add_argument('--out', required=True)

    agg = sub.add_parser('aggregate')
    agg.add_argument('--family-file', action='append', required=True)
    agg.add_argument('--out', required=True)

    args = parser.parse_args()
    if args.cmd == 'validate':
        print(json.dumps(validate_invariants(), indent=2))
    elif args.cmd == 'check-authorization':
        print(json.dumps(_require_authorization(), indent=2))
    elif args.cmd == 'prepare':
        print(json.dumps(prepare(args.out, args.marker), indent=2))
    elif args.cmd == 'eval-family':
        result = eval_family(args.prep, args.family, args.out)
        print(json.dumps({
            'family': args.family,
            'passing_cells': result['passing_cells'],
            'total_cells': result['total_cells'],
            'primary_ratios': result['primary_ratios'],
        }, indent=2))
    else:
        print(json.dumps(aggregate(args.family_file, args.out), indent=2))


if __name__ == '__main__':
    main()
