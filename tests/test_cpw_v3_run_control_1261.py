from __future__ import annotations

from pathlib import Path

from tam_research.cpw_v3_sparse_world import protocol


def test_protocol_identities_are_frozen() -> None:
    assert protocol.ISSUE == 1261
    assert protocol.SMOKE_SEED == 1_261_001
    assert protocol.REPLICATION_SEEDS == (1_261_101, 1_261_102, 1_261_103)
    assert protocol.RESULT_ROOT == "/vol/cpw-v3/sparse-world-panel-v1"
    assert protocol.CPW_V3_PARAMETERS == 21_745_408
    assert protocol.TRANSFORMER_PARAMETERS == 24_940_288


def test_scientific_launcher_is_one_shot_and_package_complete() -> None:
    text = Path("modal_cpw_v3_sparse_world_1261.py").read_text()
    assert 'gpu="H100!"' in text
    assert "retries=0" in text
    assert '.add_local_python_source("tam_research")' in text
    assert '.add_local_python_source("architectures")' in text
    assert "ATTEMPT.json" in text
    assert "RESULT.json" in text
    assert "FAILURE.json" in text
    assert "ensure_data.remote()" not in text
    assert "prepare_fineweb" not in text


def test_launcher_requires_exact_frozen_data_boundary() -> None:
    text = Path("modal_cpw_v3_sparse_world_1261.py").read_text()
    assert '"train_tokens": 25_000_000' in text
    assert '"val_tokens": 2_000_000' in text
    assert '"seed": 1234' in text
    assert '"tokenizer": "gpt2"' in text
    assert "25_000_000 * 2" in text
    assert "2_000_000 * 2" in text
    assert '"writes_performed": False' in text
