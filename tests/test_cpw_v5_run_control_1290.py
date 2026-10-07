from __future__ import annotations

from pathlib import Path

from tam_research.cpw_v5_afm import protocol


def test_protocol_is_frozen() -> None:
    assert protocol.ISSUE == 1290
    assert protocol.SMOKE_SEED == 1_291_001
    assert protocol.REPLICATION_SEEDS == (1_291_101, 1_291_102, 1_291_103)
    assert protocol.RESULT_ROOT == "/vol/cpw-v5/associative-fast-memory-v1"
    assert protocol.MEMORY_RANK == 32
    assert protocol.AFM_LAYER == 14
    assert protocol.EXPECTED_AFM_PARAMETERS == 21_721_344
    assert protocol.EXPECTED_AFM_PARAMETERS < protocol.TRANSFORMER_PARAMETERS


def test_launcher_is_one_shot_h100_and_no_resume() -> None:
    text = Path("modal_cpw_v5_afm_1290.py").read_text()
    assert 'gpu="H100!"' in text
    assert "retries=0" in text
    assert "ATTEMPT.json" in text
    assert "RESULT.json" in text
    assert "FAILURE.json" in text
    assert '"retry_authorized": False' in text
    assert '"resume_authorized": False' in text
    assert '"checkpoint_reuse_authorized": False' in text
    assert '"language_panel_authorized": False' in text
    assert 'APP_NAME = "tam-research-cpw-v5-afm-1290"' in text


def test_modal_image_contains_full_import_closure() -> None:
    text = Path("modal_cpw_v5_afm_1290.py").read_text()
    assert '.add_local_python_source("tam_research")' in text
    assert '.add_local_python_source("architectures")' in text


def test_no_task_specific_tokens_are_encoded_in_model() -> None:
    text = Path("tam_research/cpw_v5_afm/model.py").read_text()
    for literal in ("QUERY_TOKEN", "PAIR_TOKEN", "KEY_START", "VALUE_START"):
        assert literal not in text
