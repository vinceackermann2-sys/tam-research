"""Read-only Colab Pro readiness check for the unapproved Conv7 pilot.
No GPU, research seed, dataset retrieval, issue creation, or job execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = "research/reduced_attention/attention8_conv7_engineering_proposal_v1.json"
LOCK = "research/reduced_attention/attention8_conv7_execution_lock_readonly_v1.json"
WORKER = "scripts/colab_pro_issue_worker.py"
GUIDE = "docs/COLAB_PRO_CHATGPT_BRIDGE.md"
COMPUTE_GUIDE = "docs/COLAB_COMPUTE.md"
CONV7_JOB = "colab_jobs/cortex_attention8_conv7_engineering_v1.py"

PINNED = {
    PROPOSAL: "fc403bdcd78c9351fd6c928d40b49d045265d5e6",
    LOCK: "c4f2f346a4e4c304b0ce5df192fdb5d8d762c0f2",
    WORKER: "0d65f93be1c9d17650c94b3285da6ea092100379",
    GUIDE: "633b3ce9761a82b48a898f2013de398e4cfe3d03",
}
EXPECTED_MODELS = [
    "fresh_transformer",
    "reduced_attention_dense_v2_attention8",
    "attention8_causal_depthwise_conv7",
]


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def audit_colab_migration(root: Path = ROOT) -> dict:
    root = Path(root)
    for path, sha in PINNED.items():
        source = root / path
        if source.is_symlink() or not source.is_file() or git_blob(source.read_bytes()) != sha:
            raise RuntimeError("protected Colab/Conv7 source changed: " + path)

    if not (root / COMPUTE_GUIDE).is_file():
        raise RuntimeError("Colab compute guide missing")
    guide = (root / GUIDE).read_text(encoding="utf-8")
    worker = (root / WORKER).read_text(encoding="utf-8")
    for anchor in ("Colab Pro", "colab-approved", "colab_jobs/", "not a hard", "Modal"):
        if anchor.lower() not in guide.lower():
            raise RuntimeError("Colab bridge documentation drift: " + anchor)
    for anchor in (
        "SCRIPT_PATTERN = re.compile",
        "colab_jobs/",
        "issues?state=open&labels=colab-approved",
        "max-seconds must be 30..300",
        "default=180,",
        'default="CPU,T4"',
        "os.O_EXCL",
    ):
        if anchor not in worker:
            raise RuntimeError("Colab issue worker guard changed: " + anchor)

    proposal = json.loads((root / PROPOSAL).read_text(encoding="utf-8"))
    lock = json.loads((root / LOCK).read_text(encoding="utf-8"))
    pilot = proposal["draft_engineering_pilot"]
    if proposal["stage"] != "ZERO_GPU_PROPOSAL_NOT_PREREGISTERED":
        raise RuntimeError("Conv7 proposal no longer an unapproved draft")
    if any(value is not False for value in proposal["authority"].values()):
        raise RuntimeError("unexpected paid proposal authority")
    if pilot["comparators"] != EXPECTED_MODELS:
        raise RuntimeError("model comparison changed")
    if pilot["per_model_optimizer_steps"] != 3052 or pilot["per_model_token_exposures"] != 200015872:
        raise RuntimeError("200M pilot geometry changed")
    if pilot["engineering_seed"] is not None or pilot["strict_max_gpu_spend_usd"] is not None:
        raise RuntimeError("seed or budget populated without new authorization")
    if lock["classification"] != "ZERO_GPU_REVIEW_ONLY_NO_PAID_AUTHORITY":
        raise RuntimeError("historical read-only lock changed")
    old = lock["execution"]
    if old["modal_account"] != "primary" or old["gpu_class"] != "H100 80GB":
        raise RuntimeError("historical compute identity changed; inspect new design")
    if old["paid_dispatch_permitted"] is not False or old["approved"] is not False:
        raise RuntimeError("historical lock now permits paid execution")
    if (root / CONV7_JOB).exists():
        raise RuntimeError("Conv7 standalone job added; a new source-bound runner audit required")

    return {
        "classification": "CONV7_COLAB_MIGRATION_CPU_AUDIT_PASS_PAID_EXECUTION_BLOCKED",
        "protected_blobs_checked": len(PINNED),
        "model_names": EXPECTED_MODELS,
        "historical_modal_lock_not_colab_authority": True,
        "colab_worker_default_gpu_classes": ["CPU", "T4"],
        "colab_worker_default_timeout_sec": 180,
        "colab_worker_max_timeout_sec": 300,
        "real_colab_dataset_verified": False,
        "real_colab_worker_live_status_known": False,
        "real_colab_training_done": False,
        "seed_reserved": False,
        "paid_execution_permitted": False,
        "scientific_evidence": False,
        "breakthrough_claim_permitted": False,
        "blockers": [
            "historical_modal_h100_draft_not_executable_on_colab",
            "colab_conv7_standalone_job_missing",
            "frozen_train_and_validation_corpus_not_verified_in_colab",
            "worker_max_300_seconds_not_suitable_without_new_reviewed_runtime_plan",
            "colab_compute_unit_hard_spending_ceiling_not_verified",
            "equal_accelerator_class_and_hardware_benchmark_fairness_not_verified",
            "unique_new_engineering_seed_job_id_and_explicit_gpu_authorization_missing",
            "local_worker_current_online_state_unknown",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(audit_colab_migration(), sort_keys=True))
