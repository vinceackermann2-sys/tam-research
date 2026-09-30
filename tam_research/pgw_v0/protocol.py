from __future__ import annotations

ISSUE = 1145
CLASSIFICATION = "PGW_V0_25M_PREDICTIVE_GLOBAL_WORKSPACE_PREREGISTRATION"

SOURCE_SHA = "3fd7229531ae2fa99c3a98f34118eecb34014112"
SOURCE_TREE = "0495665a13c7ab6447279eb2ec0e070bb27d29e8"

SMOKE_SEED = 1_145_001
REPLICATION_SEEDS = (1_145_101, 1_145_102, 1_145_103)
RESULT_ROOT = "/vol/pgw-v0/issue-1145"
TRIGGER_TITLE = "[modal-pgw-v0-25m-1145]"

MODEL_SCALE = "25m"
VOCAB_SIZE = 50_257
D_MODEL = 256
N_LAYERS = 15
N_HEADS = 8
MAX_SEQ_LEN = 1024

STATE_SIZE = 64
WORKSPACE_DIM = 164
WORKSPACE_SLOTS = 4
WORKSPACE_HEADS = 4
SEGMENT_SIZE = 16
SELECTED_EVENTS = 4
AUTO_RANK = 40

SMOKE_TOKENS = 1_000_000
REPLICATION_TOKENS = 10_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2

PARAMETER_MISMATCH_LIMIT = 0.001
MIN_PGW_THROUGHPUT_RATIO = 0.50
EXPECTED_SELECTED_FRACTION = SELECTED_EVENTS / SEGMENT_SIZE

POSITIVE_CLASSIFICATION = "PGW_V0_25M_POSITIVE_SIGNAL"
STOP_CLASSIFICATION = "PGW_V0_25M_STOP_OR_REDESIGN"


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
            "state_size": STATE_SIZE,
            "workspace_dim": WORKSPACE_DIM,
            "workspace_slots": WORKSPACE_SLOTS,
            "workspace_heads": WORKSPACE_HEADS,
            "segment_size": SEGMENT_SIZE,
            "selected_events": SELECTED_EVENTS,
            "auto_rank": AUTO_RANK,
        },
        "smoke_tokens": SMOKE_TOKENS,
        "replication_tokens": REPLICATION_TOKENS,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "parameter_mismatch_limit": PARAMETER_MISMATCH_LIMIT,
        "min_pgw_throughput_ratio": MIN_PGW_THROUGHPUT_RATIO,
        "expected_selected_fraction": EXPECTED_SELECTED_FRACTION,
        "ordinary_next_token_ce_only": True,
        "episodic_memory": False,
        "auxiliary_prediction_loss": False,
        "future_token_routing": False,
    }
