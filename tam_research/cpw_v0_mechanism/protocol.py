from __future__ import annotations

ISSUE = 1247
SOURCE_ARCHITECTURE_HEAD = "311d501a732379afdadcdd2f07a41e42ba9d4819"

SMOKE_SEED = 1_247_001
REPLICATION_SEEDS = (1_247_101, 1_247_102, 1_247_103)
RESULT_ROOT = "/vol/cpw-v0/mechanism-panel-v1"

ARM_ORDER = (
    "transformer",
    "full",
    "no_aux",
    "uniform_all",
    "no_memory",
    "no_sequence",
    "no_world",
    "no_attention",
)

SMOKE_TOKENS = 500_000
REPLICATION_TOKENS = 3_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2
EFFECT_THRESHOLD = 0.005

CPW_PARAMETERS = 24_955_648
TRANSFORMER_PARAMETERS = 24_940_288

ABLATION_MAP = {
    "predictive_aux": "no_aux",
    "sparse_router": "uniform_all",
    "memory_branch": "no_memory",
    "sequence_branch": "no_sequence",
    "world_branch": "no_world",
    "attention_branch": "no_attention",
}
