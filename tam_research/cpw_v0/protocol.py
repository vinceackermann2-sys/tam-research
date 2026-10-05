from __future__ import annotations

ISSUE = 1245
CLASSIFICATION = "CPW_V0_MULTI_PREDICTOR_BRAIN_INSPIRED_PREREGISTRATION"
SOURCE_SHA = "5ec655e2efdf7a1cea97456007d21d4fe0ff2ace"
BRANCH = "experiment/cpw-v0-brain-predictive-workspace-20261005"

SMOKE_SEED = 1_260_001
REPLICATION_SEEDS = (1_260_101, 1_260_102, 1_260_103)
RESULT_ROOT = "/vol/cpw-v0/multi-predictor-workspace"

VOCAB_SIZE = 50_257
D_MODEL = 256
N_LAYERS = 15
N_HEADS = 8
MAX_SEQ_LEN = 1024
FF_MULT = 4

ATTENTION_INNER = 128
WORLD_STATE_SIZE = 64
SEQUENCE_PREDICTOR_RANK = 96
MEMORY_PREDICTOR_RANK = 64
ROUTER_EXPERTS = 4
ROUTER_TOP_K = 2
MEMORY_HORIZON = 4
AUXILIARY_WEIGHT = 0.05
WORKSPACE_MIX = 0.25

SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2
SMOKE_TOKENS = 1_000_000
REPLICATION_TOKENS = 5_000_000

TRANSFORMER_PARAMETERS = 24_940_288
MAX_PARAMETER_MISMATCH_FRACTION = 0.001
MIN_THROUGHPUT_RATIO = 0.50

POSITIVE_CLASSIFICATION = "CPW_V0_POSITIVE_SIGNAL"
STOP_CLASSIFICATION = "CPW_V0_STOP_OR_REDESIGN"


def protocol_manifest() -> dict[str, object]:
    return {
        "issue": ISSUE,
        "classification": CLASSIFICATION,
        "source_sha": SOURCE_SHA,
        "branch": BRANCH,
        "seeds": {
            "smoke": SMOKE_SEED,
            "replication": list(REPLICATION_SEEDS),
        },
        "result_root": RESULT_ROOT,
        "architecture": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "attention_inner": ATTENTION_INNER,
            "world_state_size": WORLD_STATE_SIZE,
            "sequence_predictor_rank": SEQUENCE_PREDICTOR_RANK,
            "memory_predictor_rank": MEMORY_PREDICTOR_RANK,
            "router_experts": ROUTER_EXPERTS,
            "router_top_k": ROUTER_TOP_K,
            "memory_horizon": MEMORY_HORIZON,
            "workspace_mix": WORKSPACE_MIX,
            "biological_drives": False,
            "persistent_reward_maximizer": False,
        },
        "training": {
            "seq_len": SEQ_LEN,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "smoke_tokens": SMOKE_TOKENS,
            "replication_tokens": REPLICATION_TOKENS,
            "auxiliary_weight": AUXILIARY_WEIGHT,
        },
        "breakthrough_claim_allowed": False,
        "scale_up_authorized": False,
    }
