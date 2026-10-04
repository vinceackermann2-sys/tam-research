from __future__ import annotations

ISSUE = 1238
CLASSIFICATION = "ATTENTION_WIDTH_GLOBAL_100M_REPLICATION_PREREGISTRATION"

SOURCE_SHA = "ee01b5a5d123c381dee861a3e202e93e5eeb49d0"
SOURCE_TREE = "af2fde0a72c1b32154345395ab0c1a85135ef2a9"

ARMS = (
    "global512_ff2048",
    "global448_ff2176",
)

SMOKE_SEED = 1_240_001
REPLICATION_SEEDS = (1_240_101, 1_240_102, 1_240_103)
RESULT_ROOT = "/vol/attention-width-global/100m-replication-v1"

VOCAB_SIZE = 50_257
D_MODEL = 512
N_LAYERS = 24
N_HEADS = 16
MAX_SEQ_LEN = 1024

BASE_ATTN_INNER = 512
BASE_FF_HIDDEN = 2048
REDUCED_ATTN_INNER = 448
REDUCED_FF_HIDDEN = 2176

SMOKE_TOKENS = 1_000_000
REPLICATION_TOKENS = 10_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2

EXPECTED_PARAMETERS = 101_803_520
EFFECT_THRESHOLD_NLL = 0.005
MIN_THROUGHPUT_RATIO = 0.80


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
        "min_throughput_ratio": MIN_THROUGHPUT_RATIO,
        "geometry": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "base_attention_inner": BASE_ATTN_INNER,
            "base_ff_hidden": BASE_FF_HIDDEN,
            "reduced_attention_inner": REDUCED_ATTN_INNER,
            "reduced_ff_hidden": REDUCED_FF_HIDDEN,
        },
        "smoke_tokens": SMOKE_TOKENS,
        "replication_tokens": REPLICATION_TOKENS,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "breakthrough_claim_allowed": False,
        "combination_authorized": False,
        "production_claim_allowed": False,
    }
