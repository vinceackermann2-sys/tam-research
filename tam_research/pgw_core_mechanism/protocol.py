from __future__ import annotations

ISSUE = 1201
CLASSIFICATION = "PGW_LOCAL_PREDICTOR_CORE_ATTRIBUTION_PREREGISTRATION"

SOURCE_SHA = "2fd4be2941b3219a3e6a42de701250abe994a058"
SOURCE_TREE = "46b02894b0f2ec16f02da8cc2dd413cd75dc2151"

ARMS = (
    "transformer",
    "chunk_local_256",
    "local224_predictor_carry",
    "local224_predictor_reset",
    "local224_only",
)

SMOKE_SEED = 1_200_001
REPLICATION_SEEDS = (1_200_101, 1_200_102, 1_200_103)
RESULT_ROOT = "/vol/pgw-v2/local-predictor-panel-v1"

VOCAB_SIZE = 50_257
D_MODEL = 256
N_LAYERS = 15
N_HEADS = 8
MAX_SEQ_LEN = 1024
FF_MULT = 4
CHUNK_SIZE = 64
LOCAL_ATTN_INNER = 224
FULL_LOCAL_ATTN_INNER = 256
WORKSPACE_LAYERS = (4, 9, 14)

SMOKE_TOKENS = 1_000_000
REPLICATION_TOKENS = 10_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2

EXPECTED_PARAMETERS = 24_940_288
EFFECT_THRESHOLD_NLL = 0.005


def protocol_manifest() -> dict[str, object]:
    return {
        "issue": ISSUE,
        "classification": CLASSIFICATION,
        "source_sha": SOURCE_SHA,
        "source_tree": SOURCE_TREE,
        "arms": list(ARMS),
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "result_root": RESULT_ROOT,
        "expected_parameters": EXPECTED_PARAMETERS,
        "effect_threshold_nll": EFFECT_THRESHOLD_NLL,
        "chunk_size": CHUNK_SIZE,
        "local_attn_inner": LOCAL_ATTN_INNER,
        "full_local_attn_inner": FULL_LOCAL_ATTN_INNER,
        "workspace_layers": list(WORKSPACE_LAYERS),
        "smoke_tokens": SMOKE_TOKENS,
        "replication_tokens": REPLICATION_TOKENS,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "breakthrough_claim_allowed": False,
        "scale_up_authorized": False,
    }
