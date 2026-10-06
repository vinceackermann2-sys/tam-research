from __future__ import annotations

from pathlib import Path

import modal_cpw_v2_sequence_1254 as launcher
from tam_research.cpw_v2_sequence import protocol


def test_protocol_is_frozen() -> None:
    assert protocol.ISSUE == 1254
    assert protocol.SMOKE_SEED == 1_254_001
    assert protocol.REPLICATION_SEEDS == (1_254_101, 1_254_102, 1_254_103)
    assert protocol.SMOKE_TOKENS == 1_000_000
    assert protocol.REPLICATION_TOKENS == 25_000_000
    assert protocol.SEQUENCE_PARAMETERS == 21_745_408
    assert protocol.TRANSFORMER_PARAMETERS == 24_940_288


def test_launcher_is_one_shot_h100() -> None:
    text = Path("modal_cpw_v2_sequence_1254.py").read_text()
    assert 'gpu="H100!"' in text
    assert "retries=0" in text
    assert "ATTEMPT.json" in text
    assert "RESULT.json" in text
    assert "FAILURE.json" in text
    assert "resume_authorized" in text
    assert launcher.APP_NAME == "tam-research-cpw-v2-sequence-1254"


def test_data_guard_reuses_exact_25m_shard() -> None:
    text = Path("modal_cpw_v2_sequence_1254.py").read_text()
    assert '"train_tokens": 25_000_000' in text
    assert '"val_tokens": 2_000_000' in text
    assert '"seed": 1234' in text
    assert '"tokenizer": "gpt2"' in text
    assert 'train_path.stat().st_size != 25_000_000 * 2' in text
    assert 'val_path.stat().st_size != 2_000_000 * 2' in text


def test_launcher_has_no_pre_h100_remote_data_hop() -> None:
    text = Path("modal_cpw_v2_sequence_1254.py").read_text()
    assert "ensure_data.remote()" not in text
    assert "def ensure_data(" not in text
    assert "datasets>=4.0" not in text
    assert "transformers>=4.55" not in text
    assert "def _verify_frozen_data()" in text
    assert '"train_tokens": 25_000_000' in text
    assert '"val_tokens": 2_000_000' in text
    assert 'train_path.stat().st_size != 25_000_000 * 2' in text
    assert 'val_path.stat().st_size != 2_000_000 * 2' in text
    assert "data_guard = _verify_frozen_data()" in text


def test_modal_image_ships_transitive_architectures_package() -> None:
    text = Path("modal_cpw_v2_sequence_1254.py").read_text()
    assert '.add_local_python_source("tam_research")' in text
    assert '.add_local_python_source("architectures")' in text
