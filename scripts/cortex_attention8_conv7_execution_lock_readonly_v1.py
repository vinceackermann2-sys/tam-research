"""Read-only Conv7 source and pilot contract; no execution pathway."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = "research/reduced_attention/attention8_conv7_execution_lock_readonly_v1.json"
PROPOSAL_PATH = "research/reduced_attention/attention8_conv7_engineering_proposal_v1.json"
MODEL_NAMES = ("fresh_transformer", "reduced_attention_dense_v2_attention8", "attention8_causal_depthwise_conv7")
PARAMETERS = (101803520, 101795328, 101803536)
# Anchor the reviewed manifest outside its own mutable hash map. Updating this
# digest or source set requires a separate reviewed successor, not rehashing a
# changed proposal to make the current validation pass.
EXPECTED_LOCK_SHA256 = "07392393728872ed5c9d37fea33bd9ff9d53b31ab50f5112efb51cbfe19880a3"
EXPECTED_SOURCE_PATHS = frozenset({
    "tam_research/train.py", "tam_research/models.py", "tam_research/data.py",
    "architectures/cortex_s/reduced_attention_200m_panel_v2.py",
    "architectures/cortex_s/reduced_attention_100m_v2_attention8.py",
    "architectures/cortex_s/attention8_causal_conv7_v1.py",
    PROPOSAL_PATH,
    "scripts/cortex_attention8_conv7_readiness_v1.py",
    "tam_research/cortex_attention8_conv7_engineering_cpu_kernel_v1.py",
    "tam_research/cortex_attention8_conv7_data_stream_parity_v1.py",
    "tam_research/cortex_attention8_conv7_dispatch_dryrun_v1.py",
    "tam_research/cortex_attention8_conv7_journal_simulation_v1.py",
    "tam_research/cortex_attention8_conv7_engineering_rehearsal_v1.py",
})


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def check_execution_lock(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    raw = (root / LOCK_PATH).read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != EXPECTED_LOCK_SHA256:
        raise RuntimeError("protected source-lock manifest drift")
    lock = json.loads(raw)
    if (lock.get("schema") != "cortex.attention8.conv7.read_only_execution_lock.v1" or
        lock.get("classification") != "ZERO_GPU_REVIEW_ONLY_NO_PAID_AUTHORITY" or
        lock.get("reference_repo") != "vinceackermann2-sys/tam-research"):
        raise RuntimeError("wrong read-only execution lock identity")
    hashes = lock.get("trusted_source_blobs")
    if not isinstance(hashes, dict) or set(hashes) != EXPECTED_SOURCE_PATHS:
        raise RuntimeError("exact protected source paths required")
    for path, expected in hashes.items():
        if (not isinstance(path, str) or path.startswith("/") or ".." in Path(path).parts or
            not isinstance(expected, str) or len(expected) != 40):
            raise RuntimeError("invalid source lock entry")
        if blob_sha((root / path).read_bytes()) != expected:
            raise RuntimeError("protected source drift: " + path)

    proposal = json.loads((root / PROPOSAL_PATH).read_text(encoding="utf-8"))
    if proposal["stage"] != "ZERO_GPU_PROPOSAL_NOT_PREREGISTERED":
        raise RuntimeError("proposal is not an unapproved draft")
    if (not isinstance(proposal["authority"], dict) or
        any(value is not False for value in proposal["authority"].values())):
        raise RuntimeError("proposal unexpectedly grants paid authority")
    p = proposal["draft_engineering_pilot"]
    if lock["model_names"] != list(MODEL_NAMES) or lock["model_parameters"] != list(PARAMETERS):
        raise RuntimeError("model names/counts changed")
    if p["comparators"] != lock["model_names"]:
        raise RuntimeError("three-model comparator identity drift")
    for field, expected in zip(("transformer_parameters","attention8_parameters","attention8_conv7_parameters"),PARAMETERS):
        if proposal["model_geometry"][field] != expected:
            raise RuntimeError("parameter contract drift: " + field)

    fields = {
        "model_steps": "per_model_optimizer_steps",
        "tokens_per_step": "tokens_per_step",
        "tokens_per_model": "per_model_token_exposures",
        "full_horizon_steps": "full_horizon_optimizer_steps",
        "full_horizon_tokens": "full_horizon_token_exposures",
        "warmup_steps": "warmup_steps_full_horizon",
        "sequence_length": "sequence_length",
        "micro_batch": "micro_batch",
        "gradient_accumulation": "gradient_accumulation",
        "final_eval_batches": "final_evaluation_batches",
        "eval_batch_size": "evaluation_batch_size",
        "lr_peak": "learning_rate",
        "adamw_betas": "betas",
        "weight_decay": "weight_decay",
        "grad_clip": "grad_clip",
    }
    b = lock["pilot"]
    for key, original in fields.items():
        if b[key] != p[original]:
            raise RuntimeError("pilot training/evaluation schedule drift: " + key)
    if (b["model_steps"], b["full_horizon_steps"], b["warmup_steps"], b["tokens_per_step"]) != (3052, 30518, 610, 65536):
        raise RuntimeError("full-horizon training schedule must be frozen")
    if b["micro_batch"] * b["sequence_length"] * b["gradient_accumulation"] != b["tokens_per_step"]:
        raise RuntimeError("training microbatch geometry mismatch")
    if b["model_steps"] * b["tokens_per_step"] != b["tokens_per_model"]:
        raise RuntimeError("200M training exposure mismatch")
    if b["full_horizon_steps"] * b["tokens_per_step"] != b["full_horizon_tokens"]:
        raise RuntimeError("full training exposure mismatch")
    if b["final_eval_batches"] * b["eval_batch_size"] * b["sequence_length"] != b["final_eval_tokens"]:
        raise RuntimeError("final heldout tokens mismatch")
    if (not p["same_training_seed_per_model"] or not p["shared_heldout_batch_stream"] or
        p["engineering_seed"] is not None or p["strict_max_gpu_spend_usd"] is not None):
        raise RuntimeError("seed/evaluation parity or authority drift")
    for key, field in (("train","train_sha256"),("val","val_sha256"),("metadata","metadata_sha256")):
        if lock["corpus_expected_sha256"][key] != p[field]:
            raise RuntimeError("expected corpus SHA changed: " + key)
    if lock["frozen_failed_scientific_result_sha256"] != proposal["source"]["scientific_result_sha256"]:
        raise RuntimeError("frozen failed science lineage drift")
    gates = proposal["draft_screen_gates"]
    for key, source_key in (
        ("nll_delta_vs_transformer_max","transformer_comparison_max_nll_delta"),
        ("nll_improvement_vs_parent_min","attention8_parent_min_quality_improvement_nll"),
        ("tps_vs_transformer_min","transformer_min_training_tokens_per_second_ratio"),
    ):
        if lock["screen_thresholds"][key] != gates[source_key]:
            raise RuntimeError("quality/throughput gate drift: " + key)
    if lock["consumed_scientific_seeds"] != [60231, 60232]:
        raise RuntimeError("consumed scientific seed lineage drift")

    required = {
        "modal_account":"primary", "gpu_class":"H100 80GB",
        "approved":False, "seed":None, "max_spend_usd":None, "job_id":None,
        "one_shot":True, "retries":0, "resume":False,
        "real_reservation_created":False, "remote_volume_verified":False,
        "paid_launcher_present":False, "paid_dispatch_permitted":False,
        "scientific_replication_permitted":False,
        "scale_250m_5b_permitted":False, "breakthrough_claim_permitted":False,
    }
    if lock["execution"] != required:
        raise RuntimeError("execution policy must be deny-only")
    if p["modal_account"] != required["modal_account"] or p["gpu_class"] != required["gpu_class"]:
        raise RuntimeError("modal account/gpu drift")
    return {
        "classification": "CONV7_EXECUTION_LOCK_CPU_PASS_GPU_BLOCKED",
        "source_lock_sha256": manifest_sha256,
        "source_blobs_checked": len(hashes),
        "three_model_token_exposures": 3 * b["tokens_per_model"],
        "heldout_tokens_per_model": b["final_eval_tokens"],
        "real_corpus_bytes_verified": False,
        "real_modal_volume_reservation_verified": False,
        "paid_dispatch_permitted": False,
        "seed_consumed": False,
        "scientific_evidence": False,
        "breakthrough_claim_permitted": False,
    }


if __name__ == "__main__":
    print(json.dumps(check_execution_lock(), sort_keys=True))
