from __future__ import annotations

ISSUE = 1253
SMOKE_SEED = 1_253_001
REPLICATION_SEEDS = (1_253_101, 1_253_102, 1_253_103)
RESULT_ROOT = "/vol/cpw-v2/single-predictor-panel-v1"

ARM_ORDER = (
    "transformer",
    "sequence_memory",
    "sequence_only",
    "memory_only",
    "world_only",
)

SMOKE_TOKENS = 500_000
REPLICATION_TOKENS = 3_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2

EXPECTED_PARAMETERS = {
    "transformer": 24_940_288,
    "sequence_memory": 22_236_928,
    "sequence_only": 21_745_408,
    "memory_only": 21_499_648,
    "world_only": 21_745_408,
}
