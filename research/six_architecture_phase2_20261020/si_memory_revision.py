"""Unmodified SI memory: factual revision, world isolation and lifetime restore."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent
OUT = ROOT / 'results/si_memory_revision'
PROTOCOL = ROOT / 'SI_MEMORY_REVISION_PROTOCOL.md'
SEEDS = (202610301, 202610302, 202610303)
TOLERANCE = 2e-6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def history_digest(history):
    digest = hashlib.sha256()
    for kind, value in history:
        digest.update(kind.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main(repo):
    sys.path.insert(0, str(repo.resolve()))
    from tam_research.si_original_world_scoped_v2 import SIOriginalWorldScopedV2
    from tam_research.si_original_crossmodal_learning_v1 import _public_batch

    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    dump(OUT / 'PREREGISTRATION.json', {
        'utc': time.time(), 'protocol_sha256': sha(PROTOCOL), 'source_sha256': sha(Path(__file__)),
        'native_source_sha256': {name: sha(repo / 'tam_research' / name) for name in
            ('si_original_world_scoped_v2.py', 'si_original_integrated_v1.py',
             'si_original_hybrid_v0.py', 'si_original_crossmodal_learning_v1.py')},
        'seeds': SEEDS, 'optimizer_updates': 0})
    histories = []
    seen = set()
    offset = 0
    while len(histories) < 16:
        history, _, _ = _public_batch(79_000_000 + offset, 1)
        offset += 1
        digest = history_digest(history)
        if digest not in seen:
            seen.add(digest)
            histories.append(history)

    def fresh():
        return SIOriginalWorldScopedV2(width=24, layers=4, experts=4, max_steps=3).eval()

    def output(model, history, world):
        return model(history, world=world, enable_search=False)

    with torch.no_grad():
        for seed in SEEDS:
            start = time.monotonic()
            path = ROOT / f'results/si_repair/si-{seed}-repair.pt'
            original = torch.load(path, map_location='cpu', weights_only=False)
            model = fresh()
            model.load_state_dict(original['weights'])
            model.import_persistent_state(original['persistent'])
            assert not model.replay and not model.memory.rows and not model._world_fast
            parameters = sum(p.numel() for p in model.parameters())
            assert parameters == 62366 and parameters <= 100_000_000
            assert 736_000 + 616 <= 2_000_000_000
            weights_before = {name: p.clone() for name, p in model.named_parameters()}
            latest = {}
            public_ledger = hashlib.sha256()
            feedback_events = 0

            def ingest(world, index, reward):
                nonlocal feedback_events
                history = histories[index]
                # The fixture actually executes action0 and reveals its reward.
                after = list(history) + [('text', torch.tensor([100 + reward]))]
                before_state = model._backbone(history)[0][0]
                after_state = model._backbone(after)[0][0]
                latest[world, index] = (before_state.clone(), (after_state - before_state).clone())
                model.ingest_executed(history, 0, float(reward), after, world=world)
                event = {'world': world, 'context': history_digest(history), 'action': 0,
                         'reward': reward, 'after': history_digest(after)}
                public_ledger.update(json.dumps(event, sort_keys=True).encode())
                feedback_events += 1

            def associations():
                records = {}
                for world in ('A', 'B', 'C'):
                    rows = [(index, data) for (name, index), data in latest.items() if name == world]
                    if not rows:
                        continue
                    errors = []
                    for index, (key, expected) in rows:
                        actual = model.memory.read(world, key)
                        errors.append(float((actual - expected).abs().max()))
                        # The public latest-record positive control gets exactly
                        # these same factual deltas, without another execution.
                        assert torch.equal(latest[world, index][1], expected)
                    records[world] = {'associations': len(rows),
                        'native_latest_association_accuracy': sum(e <= TOLERANCE for e in errors) / len(errors),
                        'latest_record_control_accuracy': 1.0,
                        'maximum_delta_error': max(errors), 'per_context_delta_error': errors}
                return records

            stages = {}
            for index in range(16):
                ingest('A', index, 0)
            stages['after_A'] = associations()
            a_logits = torch.cat([output(model, history, 'A')['logits'] for history in histories])
            isolation = {}
            for world, reward in (('B', 1), ('C', 0)):
                for index in range(16):
                    ingest(world, index, reward)
                stages['after_' + world] = associations()
                logits = torch.cat([output(model, history, 'A')['logits'] for history in histories])
                isolation['after_' + world] = float((a_logits - logits).abs().max())
            for index in range(8):
                ingest('A', index, 1)
            stages['after_A_revision'] = associations()
            assert feedback_events == 56 and len(model.memory.rows) == 56
            for name, p in model.named_parameters():
                assert torch.equal(p, weights_before[name]), 'Diagnostic changed a neural parameter'

            native_state = model.export_persistent_state()
            restored = fresh()
            restored.load_state_dict(model.state_dict())
            restored.import_persistent_state(native_state)
            roundtrip = {key: 0.0 for key in ('logits', 'state', 'retrieval')}
            for world in ('A', 'B', 'C'):
                for history in histories:
                    before = output(model, history, world)
                    after = output(restored, history, world)
                    for key in roundtrip:
                        roundtrip[key] = max(roundtrip[key], float((before[key] - after[key]).abs().max()))
            assert max(roundtrip.values()) <= TOLERANCE
            native_replay_count = len(restored.replay)
            checkpoint = OUT / f'si-memory-{seed}.pt'
            with checkpoint.open('xb') as stream:
                torch.save({'weights': model.state_dict(), 'persistent': native_state,
                    'replay': list(model.replay), 'protocol_sha256': sha(PROTOCOL),
                    'representation_weights_sha256': sha(path)}, stream)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            restored.replay.extend(copy.deepcopy(saved['replay']))
            assert len(restored.replay) == 56
            assert all(torch.equal(a.before, b.before) and torch.equal(a.after, b.after)
                       and (a.world, a.action, a.reward, a.skill, a.provenance) ==
                           (b.world, b.action, b.reward, b.skill, b.provenance)
                       for a, b in zip(restored.replay, model.replay))
            result = {'seed': seed, 'parameters': parameters, 'optimizer_updates': 0,
                'training_exposures': 0, 'additional_online_feedback_raw_positions': 616,
                'prior_accounted_training_positions': 736_000,
                'total_accounted_training_and_feedback_positions': 736_616,
                'feedback_events': feedback_events, 'public_feedback_digest': public_ledger.hexdigest(),
                'stage_associations': stages, 'world_A_logits_max_change_after_other_worlds': isolation,
                'native_export_roundtrip_max_change': roundtrip,
                'native_export_replay_before': len(model.replay), 'native_export_replay_after': native_replay_count,
                'explicit_full_restore_replay_after': len(restored.replay),
                'external_state': {'memory_records': len(model.memory.rows),
                    'memory_float_scalars': sum(k.numel() + v.numel() for _, k, v in model.memory.rows),
                    'world_fast_float_scalars': sum(u.numel() + v.numel() for u, v, _ in model._world_fast.values()),
                    'replay_latent_float_scalars': sum(r.before.numel() + r.after.numel() for r in model.replay),
                    'native_json_bytes': len(json.dumps(native_state).encode()),
                    'explicit_full_checkpoint_bytes': checkpoint.stat().st_size},
                'neural_parameters_unchanged': True, 'seconds': time.monotonic() - start,
                'source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(PROTOCOL),
                'input_checkpoint_sha256': sha(path), 'checkpoint_sha256': sha(checkpoint),
                'scope': 'frozen trained-representation memory association and persistence, not action utility',
                'breakthrough_demonstrated': False}
            dump(OUT / f'si-memory-{seed}.DIAGNOSTIC.json', result)
            print('SI_MEMORY_RESULT', seed, json.dumps({
                'revision_A_accuracy': stages['after_A_revision']['A']['native_latest_association_accuracy'],
                'other_world_isolation': isolation, 'native_replay_restored': native_replay_count}), flush=True)
    dump(OUT / 'COMPLETE.json', {'complete': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    main(args.repo)
