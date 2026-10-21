"""Inspect terminal model/Adam/RNG contents without running optimizer updates."""
import datetime
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from phase2_aligned_v2 import count, make

ROOT = Path(__file__).parent
BASE = ROOT / 'l4_panel'


def main():
    torch.set_num_threads(2)
    evidence_bytes = (BASE / 'EVIDENCE.json').read_bytes()
    evidence = json.loads(evidence_bytes)
    assert evidence['complete'] and set(evidence['terminal']) == {
        'transformer', 'attention8', 'conv7', 'cpw_local'}
    catalog = json.loads((BASE / 'CHECKPOINT_CATALOG.json').read_text())
    records = []
    for arm, terminal in sorted(evidence['terminal'].items()):
        result = evidence['results'][f"{arm}-s{terminal['steps']}"]
        manifest = result['checkpoint']
        verified = catalog[manifest['checkpoint_sha256']]
        assert verified['scientific_endpoint'] and not verified.get('pruned')
        with tempfile.TemporaryDirectory(prefix='terminal-audit-') as directory:
            path = Path(directory) / 'state.pt'
            whole = hashlib.sha256()
            with path.open('xb') as output:
                for chunk in manifest['chunks']:
                    source = (BASE / chunk['path']).resolve()
                    assert source.is_relative_to(BASE.resolve())
                    digest = hashlib.sha256()
                    size = 0
                    with source.open('rb') as stream:
                        for raw in iter(lambda: stream.read(8 * 1024**2), b''):
                            whole.update(raw)
                            digest.update(raw)
                            output.write(raw)
                            size += len(raw)
                    assert size == chunk['bytes']
                    assert digest.hexdigest() == chunk['sha256']
                output.flush()
                os.fsync(output.fileno())
            assert whole.hexdigest() == manifest['checkpoint_sha256']
            state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            assert state['seed'] == evidence['seed'] == 202610321
            assert state['variant'] == arm
            assert state['step'] == terminal['steps'] == manifest['step']
            assert state['tokens_seen'] == manifest['tokens_seen'] == state['step'] * 8192
            assert state['training_seconds'] > 0
            with torch.device('meta'):
                model = make(arm)
            expected = model.state_dict()
            assert set(expected) == set(state['model'])
            for name, tensor in state['model'].items():
                assert tensor.shape == expected[name].shape
                assert bool(torch.isfinite(tensor).all())
            parameters = count(model)
            assert parameters == result['parameters'] <= 100_000_000
            optimizer = state['optimizer']
            ids = [i for group in optimizer['param_groups'] for i in group['params']]
            assert len(ids) == len(list(model.parameters())) == len(optimizer['state'])
            for ident, parameter in zip(ids, model.parameters()):
                row = optimizer['state'][ident]
                assert int(row['step']) == state['step']
                for name in ('exp_avg', 'exp_avg_sq'):
                    assert row[name].shape == parameter.shape
                    assert bool(torch.isfinite(row[name]).all())
            assert state['torch_rng'].numel() > 0 and len(state['cuda_rng']) == 1
            charge = evidence['charges'][arm]
            assert state['tokens_seen'] + charge <= 2_000_000_000
            records.append({
                'variant': arm, 'seed': state['seed'], 'step': state['step'],
                'actual_parameters_from_frozen_constructor': parameters,
                'committed_training_exposures': state['tokens_seen'],
                'charged_lost_exposures': charge,
                'committed_plus_conservative_charge': state['tokens_seen'] + charge,
                'checkpoint_sha256': manifest['checkpoint_sha256'],
                'individual_chunks_and_whole_hash_verified': True,
                'all_model_and_optimizer_tensors_finite': True,
                'all_optimizer_steps_match_checkpoint': True,
                'optimizer_parameter_tensors': len(ids), 'rng_states_present': True,
            })
            print('TERMINAL_OPTIMIZER_VERIFIED', arm, state['step'], parameters, flush=True)
            del state, optimizer, expected, model, tensor, row
            gc.collect()
    report = {
        'complete': True, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evidence_sha256': hashlib.sha256(evidence_bytes).hexdigest(),
        'optimizer_updates': 0, 'records': records,
    }
    (ROOT / 'TERMINAL_OPTIMIZER_AUDIT.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
