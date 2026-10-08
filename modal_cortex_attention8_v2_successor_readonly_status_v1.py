from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import modal

SOURCE_SHA = "27a3a15cb8456523798163f4cfa5a829844477ed"
SOURCE_TREE = "027e0c1bbc845783d547f935f378889d298c9dab"
HARNESS_SHA = "5d9e1a170d9bbb93e4a02de6a47467049df10d55"
SCIENTIFIC_SEED = 60232
ROOT = Path("/vol/cortex-s-v0/100m-2b/reduced-attention-v2-attention8-pair1-successor-v1")
MODELS = ("fresh_transformer", "reduced_attention_dense_v2_attention8")

app = modal.App("cortex-attention8-v2-successor-readonly-status-v1")
volume = modal.Volume.from_name("tam-research-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")

def _load(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    data = path.read_bytes()
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("invalid JSON object")
    for key, expected in (("source_sha", SOURCE_SHA), ("source_tree", SOURCE_TREE), ("harness_sha", HARNESS_SHA)):
        if key in payload and payload[key] != expected:
            raise ValueError(f"frozen {key} mismatch: {path}")
    if "pair_seed" in payload and payload["pair_seed"] != SCIENTIFIC_SEED:
        raise ValueError(f"seed mismatch: {path}")
    return payload, hashlib.sha256(data).hexdigest()

def _summary(payload: dict[str, Any] | None, sha: str | None) -> dict[str, Any]:
    return {"exists": payload is not None, "sha256": sha, "status": payload.get("status") if payload else None, "classification": payload.get("classification") if payload else None}

@app.function(image=image, cpu=2, memory=2048, timeout=5 * 60, retries=0, volumes={"/vol": volume})
def inspect_primary_read_only() -> str:
    volume.reload()
    report: dict[str, Any] = {
        "classification": "ATTENTION8_V2_SUCCESSOR_PRIMARY_READONLY_STATUS_V1",
        "account": "primary", "gpu_allocated": False, "writes_performed": False,
        "new_seed_consumed": False, "scientific_seed": SCIENTIFIC_SEED,
        "source_sha": SOURCE_SHA, "source_tree": SOURCE_TREE, "harness_sha": HARNESS_SHA,
        "result_root_exists": ROOT.is_dir(),
    }
    if not ROOT.is_dir():
        report["stage"] = "NO_RESULT_ROOT"
        return json.dumps(report, sort_keys=True)
    reservation, rsha = _load(ROOT / "PAIR1_DISPATCH_RESERVED.json")
    gate, gsha = _load(ROOT / "ZERO_GPU_GATE.json")
    report["reservation"] = _summary(reservation, rsha)
    report["reservation"]["seed_consumed"] = reservation.get("pair_seed_consumed") if reservation else None
    report["reservation"]["h100_allocation_started"] = reservation.get("h100_allocation_started") if reservation else None
    report["reservation"]["account"] = reservation.get("selected_modal_account") if reservation else None
    report["zero_gpu_gate"] = _summary(gate, gsha)
    report["models"] = {}
    for model in MODELS:
        attempt, asha = _load(ROOT / model / "ATTEMPT_STARTED.json")
        result, msha = _load(ROOT / model / "RESULT.json")
        details: dict[str, Any] = {"attempt": _summary(attempt, asha), "result": _summary(result, msha)}
        if result:
            training = result.get("training") or {}
            metrics = training.get("final_eval") or {}
            details["metrics"] = {"final_nll": metrics.get("nll"),
                                  "training_tps": training.get("training_tokens_per_second"),
                                  "optimizer_steps": training.get("optimizer_steps"),
                                  "tokens": training.get("full_batch_token_exposures")}
            details["error_type"] = result.get("error_type")
        report["models"][model] = details
    final, fsha = _load(ROOT / "RESULT.json")
    report["final_result"] = _summary(final, fsha)
    if final:
        report["final_result"]["comparison"] = final.get("comparison")
        report["final_result"]["seed_consumed"] = final.get("pair_seed_consumed")
    report["stage"] = "TERMINAL" if final else "RESERVED" if reservation else "ZERO_GPU_GATE_ONLY" if gate else "ROOT_PRESENT_NO_MARKERS"
    return json.dumps(report, sort_keys=True, separators=(",", ":"))

@app.local_entrypoint()
def main() -> None:
    print("ATTENTION8_V2_READONLY_STATUS=" + inspect_primary_read_only.remote(), flush=True)
