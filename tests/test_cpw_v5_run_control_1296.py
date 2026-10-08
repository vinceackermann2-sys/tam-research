from __future__ import annotations

from pathlib import Path

from tam_research.cpw_v5_afm import protocol


def test_preregistered_seeds_and_parameter_contracts() -> None:
    assert protocol.ISSUE == 1296
    assert protocol.SMOKE_SEED == 1_291_001
    assert protocol.REPLICATION_SEEDS == (1_291_101, 1_291_102, 1_291_103)
    assert protocol.SMOKE_STEPS == 250
    assert protocol.REPLICATION_STEPS == 1500
    assert protocol.RESULT_ROOT == "/vol/cpw-v5/associative-fast-memory-v1"
    assert protocol.TRANSFORMER_PARAMETERS == 24_940_288
    assert protocol.SEQUENCE_PARAMETERS == 21_745_408
    assert protocol.AFM_PARAMETERS == protocol.SEQUENCE_PARAMETERS
    assert protocol.ARMS == ("transformer", "sequence_only", "afm_first1")


def test_runner_one_shot_and_package_complete() -> None:
    text = Path("modal_cpw_v5_afm_1296.py").read_text()
    for required in (
        'gpu="H100!"',
        "retries=0",
        '.add_local_python_source("tam_research")',
        '.add_local_python_source("architectures")',
        "ATTEMPT.json",
        "RESULT.json",
        "FAILURE.json",
        "volume.commit()",
        '"retry_authorized": False',
        '"resume_authorized": False',
        '"checkpoint_reuse_authorized": False',
        '"breakthrough_claim_allowed": False',
        '"scale_up_authorized": False',
    ):
        assert required in text, required
    assert "ensure_data.remote()" not in text
    assert "prepare_fineweb" not in text
    assert "if any(root.iterdir())" in text
    assert "source_sha" in text


def test_runner_uses_preregistered_gates() -> None:
    text = Path("modal_cpw_v5_afm_1296.py").read_text()
    for required in (
        "AFM_LONG_MEAN_ACCURACY",
        "AFM_DISTANCE_256_ACCURACY",
        "AFM_SEQUENCE_MARGIN",
        "TRANSFORMER_VALID_LONG_MEAN_ACCURACY",
        "TRANSFORMER_VALID_DISTANCE_256_ACCURACY",
        "SEQUENCE_LEAKAGE_ALARM_LONG_ACCURACY",
        "supported >= 2",
        "supported == len(REPLICATION_SEEDS)",
    ):
        assert required in text, required


def test_query_only_training_and_frozen_generator_import() -> None:
    train = Path("tam_research/cpw_v5_afm/train.py").read_text()
    assert "from tam_research.cpw_v4_memory.task import make_associative_batch" in train
    assert "last = model.norm(x[:, -1])" in train
    assert "return model.lm_head(last)" in train
    assert "model(tokens)[:, -1" not in train
