from __future__ import annotations

ISSUE = 1181
CLASSIFICATION = "PGW_V2_MECHANISM_ATTRIBUTION_PANEL_PREREGISTRATION"

SOURCE_SHA = "5884d6043bb7802395cc89c55aa0f3374c47b95e"
SOURCE_TREE = "13b49115983e17b04f069e8f82608378f7996779"

SMOKE_SEED = 1_180_001
REPLICATION_SEEDS = (1_180_101, 1_180_102, 1_180_103)
ROUTING_CONTROL_SEED = 1_180_999
RESULT_ROOT = "/vol/pgw-v2/mechanism-panel-v1"

ARMS = (
    "transformer",
    "mean_read_predictive",
    "token_read_predictive",
    "token_read_fixed_random",
    "token_read_recency",
    "no_workspace",
)

FIXED_RANDOM_POSITIONS = {
    4: (29, 45, 55, 60),
    9: (9, 40, 62, 63),
    14: (23, 26, 30, 56),
}
RECENCY_POSITIONS = (60, 61, 62, 63)

SMOKE_TOKENS = 1_000_000
REPLICATION_TOKENS = 10_000_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 2

EXPECTED_PARAMETERS = 24_940_288
EXPECTED_SELECTED_FRACTION = 0.0625
EFFECT_THRESHOLD_NLL = 0.005


def protocol_manifest() -> dict[str, object]:
    return {
        "issue": ISSUE,
        "classification": CLASSIFICATION,
        "source_sha": SOURCE_SHA,
        "source_tree": SOURCE_TREE,
        "smoke_seed": SMOKE_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "routing_control_seed": ROUTING_CONTROL_SEED,
        "result_root": RESULT_ROOT,
        "arms": list(ARMS),
        "fixed_random_positions": {
            str(k): list(v) for k, v in FIXED_RANDOM_POSITIONS.items()
        },
        "recency_positions": list(RECENCY_POSITIONS),
        "smoke_tokens": SMOKE_TOKENS,
        "replication_tokens": REPLICATION_TOKENS,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "expected_parameters": EXPECTED_PARAMETERS,
        "expected_selected_fraction": EXPECTED_SELECTED_FRACTION,
        "effect_threshold_nll": EFFECT_THRESHOLD_NLL,
        "breakthrough_claim_allowed": False,
        "scale_up_authorized": False,
    }
