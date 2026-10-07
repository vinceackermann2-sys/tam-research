from __future__ import annotations

from pathlib import Path

from tam_research.cpw_v4_memory import protocol


def test_protocol_identities_and_thresholds_are_frozen() -> None:
    assert protocol.ISSUE == 1275
    assert protocol.SMOKE_SEED == 1_273_001
    assert protocol.REPLICATION_SEEDS == (1_273_101, 1_273_102, 1_273_103)
    assert protocol.RESULT_ROOT == '/vol/cpw-v4/delayed-associative-recall-v1'
    assert protocol.SMOKE_STEPS == 250
    assert protocol.REPLICATION_STEPS == 1_500
    assert protocol.SEQUENCE_LEAKAGE_ALARM_LONG_ACCURACY == 0.05
    assert protocol.TRANSFORMER_PARAMETERS == 24_940_288
    assert protocol.CPW_PARAMETERS == 21_745_408


def test_h100_runner_is_one_shot_and_package_complete() -> None:
    text = Path('modal_cpw_v4_memory_1275.py').read_text()
    assert 'gpu="H100!"' in text
    assert 'retries=0' in text
    assert ".add_local_python_source('tam_research')" in text
    assert ".add_local_python_source('architectures')" in text
    assert 'ATTEMPT.json' in text
    assert 'RESULT.json' in text
    assert 'FAILURE.json' in text
    assert 'ensure_data.remote()' not in text
    assert 'prepare_fineweb' not in text
    assert "'retry_authorized': False" in text
    assert "'resume_authorized': False" in text
    assert "'checkpoint_reuse_authorized': False" in text


def test_runner_uses_exact_preregistered_classification_gates() -> None:
    text = Path('modal_cpw_v4_memory_1275.py').read_text()
    assert 'TRANSFORMER_VALID_LONG_MEAN_ACCURACY' in text
    assert 'TRANSFORMER_VALID_DISTANCE_256_ACCURACY' in text
    assert 'WORLD_LONG_MEAN_ACCURACY' in text
    assert 'WORLD_DISTANCE_256_ACCURACY' in text
    assert 'WORLD_SEQUENCE_MARGIN' in text
    assert 'SEQUENCE_LEAKAGE_ALARM_LONG_ACCURACY' in text
    assert "supported_seeds >= 2" in text
    assert "supported_seeds == len(REPLICATION_SEEDS)" in text
    assert "abs(world_long - transformer_long) <= 0.05" in text


def test_query_only_training_does_not_project_unused_positions() -> None:
    text = Path('tam_research/cpw_v4_memory/train.py').read_text()
    assert 'def query_logits(' in text
    assert 'last = model.norm(x[:, -1])' in text
    assert 'return model.lm_head(last)' in text
    assert 'model(tokens)[:, -1' not in text
