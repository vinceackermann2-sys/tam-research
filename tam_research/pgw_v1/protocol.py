from __future__ import annotations

ISSUE = 1159
CLASSIFICATION = "PGW_V1_LOCAL_EVENT_WORKSPACE_PREREGISTRATION"

SOURCE_SHA = "5fd8101a88da76040ca8c879b891154979387cb4"
SOURCE_TREE = "2d66f8fb52d1b3f54a412519d04a89a23f0ba4a3"

SMOKE_SEED = 1_160_001
REPLICATION_SEEDS = (1_160_101, 1_160_102, 1_160_103)
RESULT_ROOT = "/vol/pgw-v1/local-event-workspace"
TRIGGER_TITLE = "[modal-pgw-v1-25m-1159]"

MODEL_SCALE = "25m"
VOCAB_SIZE = 50_257
D_MODEL = 256
N_LAYERS = 15
N_HEADS = 8
MAX_SEQ_LEN = 1024

LOCAL_ATTN_INNER = 224
CHUNK_SIZE = 64
WORKSPACE_LAYERS = (4, 9, 14)
WORKSPACE_DIM = 128
WORKSPACE_SLOTS = 4
WORKSPACE_HEADS = 4
PREDICTOR_RANK = 40
WORKSPACE_FF_RANK = 46
SELECTED_EVENTS = 4

SMOKE_TOKENS = 1_000_000
REPLICATION_TOKENS = 10_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2

EXPECTED_PARAMETERS = 24_940_288
EXPECTED_SELECTED_FRACTION = SELECTED_EVENTS / CHUNK_SIZE
MIN_PGW_THROUGHPUT_RATIO = 0.50

POSITIVE_CLASSIFICATION = "PGW_V1_25M_POSITIVE_SIGNAL"
STOP_CLASSIFICATION = "PGW_V1_25M_STOP_OR_REDESIGN"


def protocol_manifest() -> dict[str, object]:
    return {
        "issue": ISSUE,
        "classification": CLASSIFICATION,
        "source_sha": SOURCE_SHA,
        "source_tree": SOURCE_TREE,
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "result_root": RESULT_ROOT,
        "trigger_title": TRIGGER_TITLE,
        "model_scale": MODEL_SCALE,
        "architecture": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "local_attn_inner": LOCAL_ATTN_INNER,
            "chunk_size": CHUNK_SIZE,
            "workspace_layers": list(WORKSPACE_LAYERS),
            "workspace_dim": WORKSPACE_DIM,
            "workspace_slots": WORKSPACE_SLOTS,
            "workspace_heads": WORKSPACE_HEADS,
            "predictor_rank": PREDICTOR_RANK,
            "workspace_ff_rank": WORKSPACE_FF_RANK,
            "selected_events": SELECTED_EVENTS,
        },
        "smoke_tokens": SMOKE_TOKENS,
        "replication_tokens": REPLICATION_TOKENS,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "expected_parameters": EXPECTED_PARAMETERS,
        "min_pgw_throughput_ratio": MIN_PGW_THROUGHPUT_RATIO,
        "expected_selected_fraction": EXPECTED_SELECTED_FRACTION,
        "ordinary_next_token_ce_only": True,
        "episodic_memory": False,
        "auxiliary_prediction_loss": False,
        "future_chunk_routing": False,
        "breakthrough_claim_allowed": False,
    }
