from __future__ import annotations

ISSUE = 1218
CLASSIFICATION = "ATTENTION_WIDTH_LOCALITY_FACTORIAL_25M_PREREGISTRATION"

SOURCE_SHA = "cd93cdcfa0afd4c4a592b6c9872a4328f1bb9ba3"
SOURCE_TREE = "4b59d7daebbe97f58bd49bf1977ca6bd1c9bbe31"

ARMS = (
    "global256_ff1024",
    "local256_ff1024",
    "global224_ff1088",
    "local224_ff1088",
)

SMOKE_SEED = 1_220_001
REPLICATION_SEEDS = (1_220_101, 1_220_102, 1_220_103)
RESULT_ROOT = "/vol/attention-width-locality/factorial-25m-v1"

VOCAB_SIZE = 50_257
D_MODEL = 256
N_LAYERS = 15
N_HEADS = 8
MAX_SEQ_LEN = 1024
CHUNK_SIZE = 64

BASE_ATTN_INNER = 256
BASE_FF_HIDDEN = 1024
REDUCED_ATTN_INNER = 224
REDUCED_FF_HIDDEN = 1088

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
        "geometry": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "chunk_size": CHUNK_SIZE,
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
        "scale_up_authorized": False,
    }
