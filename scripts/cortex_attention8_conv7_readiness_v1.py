"""Read-only, zero-GPU readiness audit for the unapproved Conv7 engineering proposal.

This file has no Modal runner, no GPU calls, no seed reservation and no writes.
A PASS here explicitly blocks paid execution until a separately approved protocol.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from architectures.cortex_s.attention8_causal_conv7_v1 import prototype_contract
from tam_research.train import cosine_lr

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BLOBS = {
    "tam_research/train.py": "101a18c3cd8d0be23f01e9251a9b76b7a7be9787",
    "tam_research/models.py": "90e77bcbaa39a20d336628acc60db26b05c17895",
    "architectures/cortex_s/reduced_attention_100m_v2_attention8.py": "d16e5e901f0ac5b6d52e434c3467308de9701874",
    "architectures/cortex_s/reduced_attention_200m_panel_v2.py": "2f954a4f40d681279efd9f6635d71192a0f9ab1e",
    "architectures/cortex_s/attention8_causal_conv7_v1.py": "8c8ad5437378a08f996589b2fb7c967dcbefdb92",
    "research/reduced_attention/attention8_conv7_engineering_proposal_v1.json": "fc403bdcd78c9351fd6c928d40b49d045265d5e6",
}


def _git_blob_sha(content: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()


def check_readiness(root: Path = REPO_ROOT) -> dict[str, Any]:
    source_hashes: dict[str, str] = {}
    for path, expected in EXPECTED_BLOBS.items():
        payload = (root / path).read_bytes()
        actual = _git_blob_sha(payload)
        if actual != expected:
            raise RuntimeError(f"source-lock mismatch: {path}")
        source_hashes[path] = actual

    proposal = json.loads(
        (root / "research/reduced_attention/attention8_conv7_engineering_proposal_v1.json").read_text(encoding="utf-8")
    )
    if proposal["stage"] != "ZERO_GPU_PROPOSAL_NOT_PREREGISTERED":
        raise RuntimeError("unexpected proposal stage")
    if not all(value is False for value in proposal["authority"].values()):
        raise RuntimeError("a paid authority flag changed")
    p = proposal["draft_engineering_pilot"]
    if p["engineering_seed"] is not None or p["strict_max_gpu_spend_usd"] is not None:
        raise RuntimeError("proposal unexpectedly contains paid allocation parameters")
    if p["modal_account"] != "primary":
        raise RuntimeError("unexpected workspace")
    if p["comparators"] != [
        "fresh_transformer",
        "reduced_attention_dense_v2_attention8",
        "attention8_causal_depthwise_conv7",
    ]:
        raise RuntimeError("comparison identity drift")
    if p["full_horizon_optimizer_steps"] != 30518 or p["per_model_optimizer_steps"] != 3052:
        raise RuntimeError("optimizer horizon drift")
    if p["per_model_optimizer_steps"] * p["tokens_per_step"] != 200015872:
        raise RuntimeError("engineering exposure geometry drift")
    if p["full_horizon_optimizer_steps"] * p["tokens_per_step"] != 2000027648:
        raise RuntimeError("full horizon geometry drift")
    if p["sequence_length"] != 512 or p["micro_batch"] != 64 or p["gradient_accumulation"] != 2:
        raise RuntimeError("batch/sequence mismatch")
    if p["final_evaluation_batches"] * p["evaluation_batch_size"] * p["sequence_length"] != 819200:
        raise RuntimeError("evaluation geometry mismatch")
    if p["lr_schedule"] != "cosine full-horizon prefix" or p["warmup_steps_full_horizon"] != 610:
        raise RuntimeError("invalid learning-rate horizon")
    if not p["shared_heldout_batch_stream"] or not p["same_training_seed_per_model"]:
        raise RuntimeError("pair comparability missing")
    c = prototype_contract()
    if c["status"] != "CPU_PROTOTYPE_ONLY_UNTRAINED":
        raise RuntimeError("prototype unexpectedly marked trained")
    if c["expected_parameters"] != 101803536 or c["parameter_delta"] != 16:
        raise RuntimeError("parameter contract drift")
    if c["attention_layers_one_based"] != [3, 6, 9, 12, 15, 18, 21, 24]:
        raise RuntimeError("attention schedule drift")
    if any(c.get(flag) is not False for flag in
           ("science_seed_assigned", "training_authorized", "gpu_authorized",
            "replication_authorized", "250m_5b_authorized", "breakthrough_claim_allowed")):
        raise RuntimeError("prototype unexpectedly authorized")
    full = p["full_horizon_optimizer_steps"]
    warmup = p["warmup_steps_full_horizon"]
    peak = p["learning_rate"]
    last_step = p["per_model_optimizer_steps"] - 1
    correct = cosine_lr(last_step, full, warmup, peak)
    short = cosine_lr(last_step, last_step + 1, int((last_step + 1) * p["warmup_ratio_full_horizon"]), peak)
    if not (9.8 < correct / short < 10):
        raise RuntimeError("cosine prefix drift")
    return {
        "classification": "CONV7_THREE_WAY_CPU_READINESS_PASS_PAID_EXECUTION_BLOCKED",
        "source_locked": True,
        "git_blobs": source_hashes,
        "architecture_cpu_contract_verified": True,
        "comparators": p["comparators"],
        "per_model_optimizer_steps": 3052,
        "per_model_token_exposures": 200015872,
        "full_horizon_optimizer_steps": 30518,
        "full_horizon_lr_at_pilot_endpoint": correct,
        "incorrect_short_horizon_lr_at_pilot_endpoint": short,
        "engineering_seed_assigned": False,
        "strict_gpu_spend_cap_assigned": False,
        "gpu_launch_permitted": False,
        "writes_performed": False,
        "gpu_allocated": False,
        "seed_consumed": False,
    }


if __name__ == "__main__":
    print(json.dumps(check_readiness(), sort_keys=True))
