from __future__ import annotations

ISSUE = 1252
SMOKE_SEED = 1_252_001
REPLICATION_SEEDS = (1_252_101, 1_252_102, 1_252_103)
RESULT_ROOT = "/vol/cpw-v1/pairwise-core-panel-v1"

ARM_ORDER = (
    "transformer",
    "full_triad_fast",
    "sequence_memory",
    "world_memory",
    "world_sequence",
)

SMOKE_TOKENS = 500_000
REPLICATION_TOKENS = 3_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2

EXPECTED_PARAMETERS = {
    "transformer": 24_940_288,
    "full_triad_fast": 22_974_208,
    "sequence_memory": 22_236_928,
    "world_memory": 22_236_928,
    "world_sequence": 22_482_688,
}
