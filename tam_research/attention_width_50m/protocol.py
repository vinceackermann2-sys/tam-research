from __future__ import annotations

ISSUE = 1229
CLASSIFICATION = "ATTENTION_WIDTH_GLOBAL_50M_REPLICATION_PREREGISTRATION"

SOURCE_SHA = "7e84c5ce4d8e2da3a6d412f7e80def6dcf80bac3"
SOURCE_TREE = "28c488e28bc24f274457d6bde8c8152cba31a36d"

ARMS = (
    "global384_ff1536",
    "global336_ff1632",
)

SMOKE_SEED = 1_230_001
REPLICATION_SEEDS = (1_230_101, 1_230_102, 1_230_103)
RESULT_ROOT = "/vol/attention-width-global/50m-replication-v1"

VOCAB_SIZE = 50_257
D_MODEL = 384
N_LAYERS = 17
N_HEADS = 12
MAX_SEQ_LEN = 1024

BASE_ATTN_INNER = 384
BASE_FF_HIDDEN = 1536
REDUCED_ATTN_INNER = 336
REDUCED_FF_HIDDEN = 1632

SMOKE_TOKENS = 1_000_000
REPLICATION_TOKENS = 10_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2

EXPECTED_PARAMETERS = 49_799_808
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
