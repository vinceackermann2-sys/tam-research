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
